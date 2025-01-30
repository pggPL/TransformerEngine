# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8Quantizer

import nvdlfw_inspect.api as nvinspect_api

from utils import reset_debug_log, check_debug_log

try:
    import transformer_engine
    import transformer_engine_torch as tex
except (ImportError, ModuleNotFoundError):
    print("Could not find TransformerEngine package.")
    exit(1)


def test_transformer_engine_no_config(feature_dirs):
    reset_debug_log()
    nvinspect_api.initialize("", feature_dirs=feature_dirs)
    try:

        tensor = torch.rand(24, 2046).cuda()

        # FP8 enabled - true by the default
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="fprop")


        # use process tensor - False by default
        assert not nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.attn.qkv", tensor=tensor, gemm="fprop", tensor_name="activation"
        )

        tensor_copy = tensor.clone()

        # save tensor for stats - it does nothing
        nvinspect_api.transformer_engine.look_at_tensor_before_process(
            "decoder.1.attn.qkv", tensor=tensor, gemm="fprop", tensor_name="activation"
        )
        assert (tensor_copy == tensor).all()

        # save tensor for stats - it does nothing
        nvinspect_api.transformer_engine.look_at_tensor_after_the_process(
            "decoder.1.attn.qkv", tensor=tensor, gemm="fprop", tensor_name="activation"
        )
        assert (tensor_copy == tensor).all()

    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_disable_fp8_gemm(configs_dir, feature_dirs):
    reset_debug_log()
    try:
        nvinspect_api.initialize(configs_dir + "disable_fp8_gemms.yaml", feature_dirs=feature_dirs)

        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="fprop")
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="dgrad")
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="wgrad")

        # caching
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="fprop")
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="dgrad")
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="wgrad")

    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_disable_fp8_layer(configs_dir, feature_dirs):
    reset_debug_log()
    try:
        nvinspect_api.initialize(configs_dir + "disable_fp8_layer.yaml", feature_dirs=feature_dirs)

        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.mlp.fc1", gemm="fprop")
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.mlp.fc1", gemm="wgrad")
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.mlp.fc1", gemm="dgrad")
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="fprop")
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="wgrad")
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="dgrad")

    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_per_tensor_scaling(configs_dir, feature_dirs):
    reset_debug_log()
    try:

        nvinspect_api.initialize(configs_dir + "per_tensor_scaling.yaml", feature_dirs=feature_dirs)

        tensor = torch.rand(24, 2046).cuda()

        # check use_process_tensor
        assert nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="fprop", tensor_name="activation"
        )
        assert nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="fprop", tensor_name="weight"
        )
        assert nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor_name="gradient"
        )
        assert not nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor_name="weight"
        )
        assert not nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="wgrad", tensor_name="gradient"
        )
        assert not nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="wgrad", tensor_name="activation"
        )

        # check process_tensor

        default_quantizer1 = Float8Quantizer(
            scale=torch.tensor([1]).cuda(),
            amax=torch.tensor([0]).cuda(),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        default_quantizer2 = Float8Quantizer(
            scale=torch.tensor([1]).cuda(),
            amax=torch.tensor([0]).cuda(),
            fp8_dtype=tex.DType.kFloat8E5M2,
        )

        output1 = nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1",
            gemm="fprop",
            tensor=tensor,
            tensor_name="activation",
            default_quantizer=default_quantizer1,
        )
        assert type(output1) == Float8Tensor
        assert output1._fp8_dtype == tex.DType.kFloat8E4M3
        # assert check_debug_log("decoder.1.mlp.fc1: Feature=PerTensorScaling, API=process_tensor: fprop, activation: Per Tensor Scaling")

        output2 = nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1",
            gemm="dgrad",
            tensor=tensor,
            tensor_name="gradient",
            default_quantizer=default_quantizer2,
        )
        assert type(output2) == Float8Tensor
        assert output2._fp8_dtype == tex.DType.kFloat8E5M2
        # assert check_debug_log("decoder.1.mlp.fc1: Feature=PerTensorScaling, API=process_tensor: dgrad, gradient: Per Tensor Scaling")

        output3 = nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1",
            gemm="wgrad",
            tensor=tensor,
            tensor_name="gradient",
            default_quantizer=default_quantizer2,
        )
        assert (tensor == output3).all()
        # assert not check_debug_log("decoder.1.mlp.fc1: Feature=PerTensorScaling, API=process_tensor: wgrad, gradient: Per Tensor Scaling")
        # assert not check_debug_log("decoder.1.mlp.fc1: Feature=PerTensorScaling, API=process_tensor: wgrad, activation: Per Tensor Scaling")
        reset_debug_log()

        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc4",
            gemm="fprop",
            tensor=tensor,
            tensor_name="activation",
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        # assert not check_debug_log("decoder.1.mlp.fc4: Feature=PerTensorScaling, API=process_tensor: fprop, activation: Per Tensor Scaling")
    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_fake_quant(configs_dir, feature_dirs):
    reset_debug_log()
    try:
        nvinspect_api.initialize(
            configs_dir + "fake_quantization_config.yaml", feature_dirs=feature_dirs
        )

        tensor = torch.rand(24, 2046).cuda()

        # use_process_tensor
        nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="fprop", tensor_name="activation"
        )
        # assert check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: fprop, activation: QDQ")

        nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor_name="gradient"
        )
        # assert check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: dgrad, gradient: QDQ")

        # process_tensor
        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1", gemm="fprop", tensor=tensor, tensor_name="activation"
        )
        # assert check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: fprop, activation: QDQ")

        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor=tensor, tensor_name="gradient"
        )
        # assert check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: dgrad, gradient: QDQ")

        reset_debug_log()

        # FP8 enabled - out of config
        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1", gemm="fprop", tensor=tensor, tensor_name="weight"
        )
        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor=tensor, tensor_name="weight"
        )
        torch.testing.assert_close(
            tensor,
            nvinspect_api.transformer_engine.process_tensor(
                "decoder.1.mlp.fc1", gemm="wgrad", tensor=tensor, tensor_name="gradient"
            ),
        )
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.fc2", gemm="wgrad")
        # caching
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.fc2", gemm="wgrad")

        # assert not check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: fprop, weight: QDQ")
        # assert not check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: dgrad, weight: QDQ")
        # assert not check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: wgrad, gradient: QDQ")
        reset_debug_log()

        # Disabling FP8 - in config
        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.moe.fc1", gemm="fprop", tensor=tensor, tensor_name="activation"
        )
        # assert check_debug_log("decoder.1.mlp.moe.fc1: Feature=FakeQuantFp8, API=process_tensor: fprop, activation: QDQ")

        # Disabling FP8 - out of config
        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor=tensor, tensor_name="weight"
        )
        # assert not check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: dgrad, weight: QDQ")
    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_feature_medley(configs_dir, feature_dirs):
    reset_debug_log()
    try:
        nvinspect_api.initialize(
            configs_dir + "tranformer_engine_feature_medley.yaml", feature_dirs=feature_dirs
        )

        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.mlp.fc1", gemm="fprop")
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="dgrad")
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="wgrad")

        tensor = torch.rand(24, 2046).cuda()
        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1", gemm="fprop", tensor=tensor, tensor_name="activation"
        )

        default_quantizer = Float8Quantizer(
            scale=torch.tensor([1]).cuda(),
            amax=torch.tensor([0]).cuda(),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.attn.qkv",
            gemm="dgrad",
            tensor=tensor,
            tensor_name="gradient",
            default_quantizer=default_quantizer,
        )
        # assert check_debug_log("decoder.1.attn.qkv: Feature=PerTensorScaling, API=process_tensor: dgrad, gradient: Per Tensor Scaling") and not check_debug_log("decoder.1.attn.proj: Feature=PerTensorScaling, API=process_tensor: dgrad, gradient: Per Tensor Scaling")

        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1", gemm="wgrad", tensor=tensor, tensor_name="gradient"
        )
        # assert check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: wgrad, gradient: QDQ") and not check_debug_log("decoder.1.mlp.fc2: Feature=FakeQuantFp8, API=process_tensor: wgrad, gradient: QDQ")

        reset_debug_log()

        # Disabling FP8
        torch.testing.assert_close(
            tensor,
            nvinspect_api.transformer_engine.process_tensor(
                "decoder.1.mlp.fc1", gemm="fprop", tensor=tensor, tensor_name="activation"
            ),
        )
        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1", gemm="wgrad", tensor=tensor, tensor_name="gradient"
        )

        # assert not check_debug_log("decoder.1.attn.qkv: Feature=PerTensorScaling, API=process_tensor: dgrad, gradient: Per Tensor Scaling")
        # assert check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: wgrad, gradient: QDQ")

    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_statistics_collection(configs_dir, feature_dirs):
    nvinspect_api.initialize(
        config_file=configs_dir + "stats_collection_test_config.yaml",
        feature_dirs=feature_dirs,
        default_logging_enabled=False,
    )

    tensor = torch.randn((100, 100, 5))
    tensor_fp8 = Float8Tensor(
        data=tensor.to(torch.uint8),
        fp8_scale_inv=torch.ones([1]).cuda,
        fp8_dtype=tex.DType.kFloat8E4M3,
        shape=tensor.shape,
        dtype=torch.float32,
    )

    def log():
        from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS

        return STATS_BUFFERS.log_stats()

    def assert_empty():
        stats = log()
        assert len(stats) == 0

    # TE tensor stats --
    nvinspect_api.transformer_engine.look_at_tensor_before_process(
        "decoder.1.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=200
    )
    stats = log()
    assert stats[("decoder.1.mlp.fc1", "activation", "cur_amax", 200)] == tensor.abs().max()

    nvinspect_api.transformer_engine.look_at_tensor_before_process(
        "decoder.1.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=201
    )
    assert_empty()
    nvinspect_api.transformer_engine.look_at_tensor_before_process(
        "decoder.2.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=200
    )
    assert_empty()
    nvinspect_api.transformer_engine.look_at_tensor_before_process(
        "decoder.1.mlp.fc1", tensor=tensor, tensor_name="gradient", iteration=200
    )
    assert_empty()

    expected_underflows = (tensor_fp8._data == 0).sum() * 100 / (100 * 100 * 5)
    expected_overflows = (tensor_fp8._data == 255).sum() * 100 / (100 * 100 * 5)

    # TE FP8 tensor stats --
    nvinspect_api.transformer_engine.look_at_tensor_after_process(
        "decoder.1.mlp.fc1", tensor=tensor_fp8, tensor_name="gradient", iteration=200
    )
    stats = log()
    assert stats[("decoder.1.mlp.fc1", "gradient", "underflows%", 200)] == expected_underflows
    assert stats[("decoder.1.mlp.fc1", "gradient", "overflows%", 200)] == expected_overflows

    nvinspect_api.transformer_engine.look_at_tensor_after_process(
        "decoder.1.mlp.fc1", tensor=tensor_fp8, tensor_name="activation", iteration=201
    )
    assert_empty()
    nvinspect_api.transformer_engine.look_at_tensor_after_process(
        "decoder.2.mlp.fc1", tensor=tensor_fp8, tensor_name="gradient", iteration=200
    )
    assert_empty()

    # Second config in same yaml
    tensor = torch.rand((100, 100, 5))
    nvinspect_api.transformer_engine.look_at_tensor_before_process(
        "decoder.6.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=200
    )
    stats = log()
    stats_names = [x[3] for x in stats.keys()]
    all(s in stats_names for s in ["cur_amax", "dynamic_range", "mean", "std", "l1_norm"])
    assert stats[("decoder.6.mlp.fc1", "activation", "mean", 200)] == tensor.mean()

    nvinspect_api.transformer_engine.look_at_tensor_before_process(
        "decoder.7.mlp.fc1", tensor=tensor, tensor_name="weight", iteration=200
    )
    stats = log()
    stats_names = [x[3] for x in stats.keys()]
    all(s in stats_names for s in ["mean", "std", "l1_norm", "min", "max"])
    assert stats[("decoder.7.mlp.fc1", "weight", "max", 200)] == tensor.max()

    nvinspect_api.transformer_engine.look_at_tensor_before_process(
        "decoder.7.mlp.fc1", tensor=tensor, tensor_name="weight", iteration=201
    )
    assert_empty()

    nvinspect_api.end_debug()
    transformer_engine.debug.debug_state.TEDebugState.reset()
    reset_debug_log()


def test_statistics_multi_run(configs_dir, feature_dirs):
    nvinspect_api.initialize(
        config_file=configs_dir + "stats_collection_test_config.yaml",
        feature_dirs=feature_dirs,
        default_logging_enabled=False,
    )

    def feed(tensor, tensor_fp8):
        nvinspect_api.transformer_engine.look_at_tensor_before_process(
            "decoder.5.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=1
        )
        nvinspect_api.transformer_engine.look_at_tensor_after_process(
            "decoder.5.mlp.fc1", tensor=tensor_fp8, tensor_name="activation", iteration=1
        )

    def log_stats():
        from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS

        return STATS_BUFFERS.log_stats()

    def fp8_tensor(t):
        return Float8Tensor(
            data=t.to(torch.uint8).cuda(),
            fp8_scale_inv=torch.ones([1]).cuda(),
            fp8_dtype=tex.DType.kFloat8E4M3,
            shape=t.shape,
            dtype=torch.float32,
        )

    shape = [1024, 1024]
    tensors = [torch.randn(shape) for _ in range(2)]
    tensors_fp8 = [fp8_tensor(tensors[i]) for i in range(2)]

    feed(tensors[0], tensors_fp8[0])
    feed(tensors[1], tensors_fp8[1])
    stats1 = log_stats()

    tensor2 = torch.cat((tensors[0], tensors[1])).cuda()
    fp8tensor2 = fp8_tensor(tensor2)
    feed(tensor2, fp8tensor2)
    stats2 = log_stats()

    assert len(stats1.keys()) > 0
    for k in stats1.keys():
        print(k, stats1[k], stats2[k])
        torch.testing.assert_close(stats1[k], stats2[k])


if __name__ == "__main__":
    pass
