# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""BulkTensor ownership and typed region reconstruction."""

import gc

import pytest
import torch

from transformer_engine.pytorch.tensor import BulkTensor, BulkTensorRegion
import transformer_engine_torch as tex


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return request.param


def test_regions_and_lifetime(device):
    bulk = BulkTensor.empty(
        [(3, 5), (2, 3), (), (0, 7)],
        [torch.uint8, torch.float32, torch.int64, torch.bfloat16],
        device=device,
        alignments=[16, 16, 16, 16],
    )
    tensors = bulk.unpack()
    for index, (tensor, region) in enumerate(zip(tensors, bulk.layout)):
        assert tensor.shape == region.shape
        assert tensor.dtype == region.dtype
        tensor.fill_(index + 1)
        if tensor.numel():
            assert tensor.data_ptr() == bulk.buffer.data_ptr() + region.offset
            assert tensor.untyped_storage().data_ptr() == bulk.buffer.untyped_storage().data_ptr()
        else:
            assert tensor.data_ptr() == 0
    assert bulk.buffer._version == 3
    del bulk
    gc.collect()
    for index, tensor in enumerate(tensors):
        torch.testing.assert_close(tensor, torch.full_like(tensor, index + 1))


def test_restored_buffer(device):
    bulk = BulkTensor.empty([(7,), (3, 5)], [torch.int32, torch.float32], device=device)
    tensors = bulk.unpack()
    tensors[0].copy_(torch.arange(7, dtype=torch.int32, device=device))
    tensors[1].copy_(torch.arange(15, dtype=torch.float32, device=device).reshape(3, 5))
    layout = bulk.layout
    saved = bulk.buffer.cpu().clone().to(device)
    restored = BulkTensor(saved, layout).unpack()
    for actual, expected in zip(restored, tensors):
        torch.testing.assert_close(actual, expected)
    assert restored[0].data_ptr() != tensors[0].data_ptr()


def test_autograd_saved_buffer_hooks(device):
    packed = []

    class WithBulk(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            bulk = BulkTensor.empty([x.shape], [x.dtype], device=x.device)
            bulk.unpack()[0].copy_(x)
            ctx.layout = bulk.layout
            ctx.save_for_backward(bulk.buffer)
            return x.square()

        @staticmethod
        def backward(ctx, grad):
            (buffer,) = ctx.saved_tensors
            (x,) = BulkTensor(buffer, ctx.layout).unpack()
            return 2 * x * grad

    def pack(tensor):
        packed.append((tensor.dtype, tensor.shape))
        return tensor.cpu().clone(), tensor.device

    def unpack(saved):
        tensor, original_device = saved
        return tensor.to(original_device)

    x = torch.arange(12, device=device, dtype=torch.float32).requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        WithBulk.apply(x).sum().backward()
    assert packed == [(torch.uint8, torch.Size([48]))]
    torch.testing.assert_close(x.grad, 2 * x.detach())


@pytest.mark.parametrize(
    "region, message",
    [
        (BulkTensorRegion(-1, (1,), torch.uint8, 1), "exceeds"),
        (BulkTensorRegion(16, (5,), torch.float32, 4), "exceeds"),
        (BulkTensorRegion(1, (1,), torch.float32, 4), "unaligned"),
        (BulkTensorRegion(0, (-1,), torch.uint8, 1), "invalid"),
        (BulkTensorRegion(0, (2**62, 8), torch.uint8, 1), "overflowing"),
        (BulkTensorRegion(0, (1,), torch.float32, 2), "alignment"),
        (BulkTensorRegion(0, (1,), torch.uint8, 3), "alignment"),
    ],
)
def test_native_validation(region, message, device):
    with pytest.raises(RuntimeError, match=message):
        tex.unpack_bulk_tensor(torch.empty(32, dtype=torch.uint8, device=device), (region,))


def test_buffer_validation(device):
    buffer = torch.empty(16, dtype=torch.uint8, device=device)
    for bad in (buffer[1:], buffer[::2], buffer.view(4, 4), buffer.float()):
        with pytest.raises(ValueError, match="buffer"):
            BulkTensor(bad, ())
        with pytest.raises(RuntimeError, match="buffer"):
            tex.unpack_bulk_tensor(bad, ())
    assert BulkTensor.empty([], [], device=device).unpack() == []


def test_empty_and_scalar(device):
    bulk = BulkTensor.empty([(0, 2), (), (3, 0)], [torch.float32] * 3, device=device)
    first, scalar, last = bulk.unpack()
    assert first.data_ptr() == last.data_ptr() == 0
    assert scalar.shape == ()
    scalar.fill_(7)
    assert scalar.item() == 7


@pytest.fixture(scope="module")
def bulk_ops():
    if not hasattr(torch.library, "custom_op"):
        pytest.skip("torch.library.custom_op is unavailable")
    calls = []

    @torch.library.custom_op("te_bulk_tensor_test::pack", mutates_args=())
    def _pack(x: torch.Tensor) -> torch.Tensor:
        bulk = BulkTensor.empty([x.shape], [x.dtype], device=x.device)
        bulk.unpack()[0].copy_(x)
        return bulk.buffer

    @_pack.register_fake
    def _pack_fake(x):
        return BulkTensor.empty([x.shape], [x.dtype], device=x.device).buffer

    @torch.library.custom_op("te_bulk_tensor_test::read", mutates_args=())
    def _read(buffer: torch.Tensor, rows: int) -> torch.Tensor:
        calls.append(1)
        layout = (BulkTensorRegion(0, (rows,), torch.float32, 4),)
        return BulkTensor(buffer, layout).unpack()[0].clone()

    @_read.register_fake
    def _read_fake(buffer, rows):
        return torch.empty(rows, dtype=torch.float32, device=buffer.device)

    return _pack, _read, calls


@pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
def test_compiled_base_boundary(mode, device, bulk_ops):
    pack, read, calls = bulk_ops
    torch._dynamo.reset()

    def operation(x):
        return read(pack(x), x.numel()) + 1

    compiled = torch.compile(operation, fullgraph=True, mode=mode, dynamic=True)
    for size in (17, 33):
        invocations = []
        for value in range(6):
            if device == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            x = torch.full((size,), value, dtype=torch.float32, device=device)
            actual = compiled(x)
            torch.testing.assert_close(actual, x + 1)
            del actual
            invocations.append(len(calls))
        if device == "cuda" and mode == "reduce-overhead":
            assert invocations[-1] == invocations[-2] == invocations[-3]


def test_custom_op_contract(device, bulk_ops):
    pack, read, _ = bulk_ops
    x = torch.randn(17, device=device)
    torch.library.opcheck(pack, (x,))
    torch.library.opcheck(read, (pack(x), x.numel()))
