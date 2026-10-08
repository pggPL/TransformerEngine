# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NCCL communicator borrowing across PyTorch backend interfaces."""

import ctypes
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformer_engine.pytorch.distributed import get_nccl_comm_ptr


@pytest.fixture
def borrow_env(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    barrier = Mock()
    monkeypatch.setattr(dist, "barrier", barrier)
    return barrier


@pytest.mark.parametrize("interface", ["comm_ptr", "_comm_ptr"])
@pytest.mark.parametrize("backend_name", ["nccl", "nccl2", "nccl-lazy"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_borrow_nccl_comm(borrow_env, interface, backend_name, wrapped):
    pointer = 0x123456789ABC
    group = Mock()

    def get_pointer(_self=None):
        borrow_env.assert_called_once_with(group=group, device_ids=[3])
        return pointer

    getter = property(get_pointer) if interface == "comm_ptr" else get_pointer
    backend = type("Backend", (), {interface: getter, "name": lambda _: backend_name})()
    if wrapped:
        backend = SimpleNamespace(wrapped_pg=SimpleNamespace(wrapped_pg=backend))
    group._get_backend.return_value = backend
    assert get_nccl_comm_ptr(group) == pointer
    group._get_backend.assert_called_once_with(torch.device("cuda", 3))


@pytest.mark.parametrize("pointer", [0, -1, None, True, "123", 1.5])
def test_invalid_nccl_comm(borrow_env, pointer):
    backend = SimpleNamespace(name=lambda: "nccl", comm_ptr=pointer, _comm_ptr=Mock())
    group = Mock()
    group._get_backend.return_value = backend
    with pytest.raises(RuntimeError, match="invalid communicator pointer"):
        get_nccl_comm_ptr(group)
    backend._comm_ptr.assert_not_called()


def test_missing_nccl_getter(borrow_env):
    group = Mock()
    group._get_backend.return_value = SimpleNamespace(name=lambda: "nccl2")
    with pytest.raises(RuntimeError, match="does not expose a NCCL communicator pointer"):
        get_nccl_comm_ptr(group)


def test_non_nccl_backend(borrow_env):
    group = Mock()
    group._get_backend.return_value = SimpleNamespace(name=lambda: "gloo", comm_ptr=123)
    with pytest.raises(RuntimeError, match="Expected a NCCL CUDA backend"):
        get_nccl_comm_ptr(group)
    borrow_env.assert_not_called()


def test_getter_error_propagates(borrow_env):
    class Backend:
        name = lambda _: "nccl"
        _comm_ptr = Mock()

        @property
        def comm_ptr(self):
            raise RuntimeError("communicator aborted")

    group = Mock()
    group._get_backend.return_value = Backend()
    with pytest.raises(RuntimeError, match="communicator aborted"):
        get_nccl_comm_ptr(group)
    Backend._comm_ptr.assert_not_called()


def _run_nccl_rank(rank, world_size, store_path, debug, legacy):
    torch.cuda.set_device(rank)
    dist.set_debug_level(dist.DebugLevel.DETAIL if debug else dist.DebugLevel.OFF)
    backend_name = "nccl-legacy" if legacy and hasattr(dist, "ProcessGroupNCCL2") else "nccl"
    dist.init_process_group(
        backend_name,
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(seconds=60),
    )
    graph = None
    try:
        group = dist.group.WORLD
        backend = group._get_backend(torch.device("cuda", rank))
        if debug:
            assert hasattr(backend, "wrapped_pg")
            backend = backend.wrapped_pg
        if not hasattr(type(backend), "comm_ptr") and not hasattr(backend, "_comm_ptr"):
            with pytest.raises(RuntimeError, match="does not expose a NCCL communicator pointer"):
                get_nccl_comm_ptr(group)
            return

        comm = ctypes.c_void_p(get_nccl_comm_ptr(group))
        nccl = ctypes.CDLL("libnccl.so.2")
        for name, expected in [
            ("ncclCommCount", world_size),
            ("ncclCommUserRank", rank),
            ("ncclCommCuDevice", rank),
        ]:
            query = getattr(nccl, name)
            query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            query.restype = ctypes.c_int
            value = ctypes.c_int()
            assert query(comm, ctypes.byref(value)) == 0
            assert value.value == expected

        nccl.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        nccl.ncclAllReduce.restype = ctypes.c_int
        data = torch.full((32,), rank + 1, dtype=torch.float32, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        # NCCL float32 = 7, sum = 0.
        assert nccl.ncclAllReduce(data.data_ptr(), data.data_ptr(), 32, 7, 0, comm, stream) == 0
        torch.testing.assert_close(data, torch.full_like(data, world_size * (world_size + 1) / 2))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stream = torch.cuda.current_stream().cuda_stream
            assert nccl.ncclAllReduce(data.data_ptr(), data.data_ptr(), 32, 7, 0, comm, stream) == 0
        for _ in range(3):
            data.fill_(rank + 1)
            graph.replay()
            torch.testing.assert_close(
                data, torch.full_like(data, world_size * (world_size + 1) / 2)
            )
    finally:
        del graph
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_native_nccl_comm(tmp_path, world_size, debug, legacy):
    if not dist.is_nccl_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires NCCL and {world_size} CUDA devices")
    if debug and not dist.is_gloo_available():
        pytest.skip("DEBUG DETAIL requires Gloo")
    mp.spawn(
        _run_nccl_rank,
        args=(world_size, str(tmp_path / "store"), debug, legacy),
        nprocs=world_size,
    )
