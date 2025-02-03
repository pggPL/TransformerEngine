# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8Quantizer

import nvdlfw_inspect.api as nvinspect_api

try:
    import transformer_engine
    import transformer_engine_torch as tex
except (ImportError, ModuleNotFoundError):
    print("Could not find TransformerEngine package.")
    exit(1)


def test_transformer_engine_no_config(feature_dirs):
    nvinspect_api.initialize("", feature_dirs=feature_dirs)
    try:

        tensor = torch.rand(24, 2046).cuda()

        # FP8 enabled - true by the default
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="fprop")

        # use process tensor - False by default
        assert not nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.attn.qkv", tensor=tensor, gemm="fprop", tensor_name="activation"
        )

        # use look at tensor before process - False by default
        assert not nvinspect_api.transformer_engine.use_look_at_tensor_before_process(
            "decoder.1.attn.qkv",  gemm="fprop", tensor_name="activation"
        )

        # use look at tensor after the process - False by default
        assert not nvinspect_api.transformer_engine.look_at_tensor_after_the_process(
            "decoder.1.attn.qkv", gemm="fprop", tensor_name="activation"
        )

    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_disable_fp8_gemm(configs_dir, feature_dirs):
    try:
        nvinspect_api.initialize(configs_dir + "disable_fp8_gemms.yaml", feature_dirs=feature_dirs)

        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="fprop", iteration=0)
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="dgrad", iteration=0)
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="wgrad", iteration=0)

        # caching
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="fprop", iteration=0)
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="dgrad", iteration=0)
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="wgrad", iteration=0)

    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_disable_fp8_layer(configs_dir, feature_dirs):
    try:
        nvinspect_api.initialize(configs_dir + "disable_fp8_layer.yaml", feature_dirs=feature_dirs)

        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.mlp.fc1", gemm="fprop", iteration=0)
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.mlp.fc1", gemm="wgrad", iteration=0)
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.mlp.fc1", gemm="dgrad", iteration=0)
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="fprop", iteration=0)
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="wgrad", iteration=0)
        assert not nvinspect_api.transformer_engine.fp8_gemm("decoder.1.attn.qkv", gemm="dgrad", iteration=0)

    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_per_tensor_scaling(configs_dir, feature_dirs):
    try:

        nvinspect_api.initialize(configs_dir + "per_tensor_scaling.yaml", feature_dirs=feature_dirs)

        tensor = torch.rand(24, 2046).cuda()

        # check use_process_tensor
        assert nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="fprop", tensor_name="activation", iteration=0
        )
        assert nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="fprop", tensor_name="weight", iteration=0
        )
        assert nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor_name="gradient", iteration=0
        )
        assert not nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor_name="weight", iteration=0
        )
        assert not nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="wgrad", tensor_name="gradient", iteration=0
        )
        assert not nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="wgrad", tensor_name="activation", iteration=0
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
            layer_name="decoder.1.mlp.fc1",
            gemm="fprop",
            tensor_name="activation",
            default_quantizer=default_quantizer1,
            iteration=0,
            tensor=tensor
        )
        assert type(output1) == Float8Tensor
        assert output1._fp8_dtype == tex.DType.kFloat8E4M3

        output2 = nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1",
            gemm="dgrad",
            tensor=tensor,
            tensor_name="gradient",
            default_quantizer=default_quantizer2,
            iteration=0
        )
        assert type(output2) == Float8Tensor
        assert output2._fp8_dtype == tex.DType.kFloat8E5M2

        assert not nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1",
            gemm="wgrad",
            tensor=tensor,
            tensor_name="gradient",
            default_quantizer=default_quantizer2,
            iteration=0
        )

        assert not nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc4",
            gemm="fprop",
            tensor=tensor,
            tensor_name="activation",
            fp8_dtype=tex.DType.kFloat8E4M3,
            iteration=0
        )
    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_fake_quant(configs_dir, feature_dirs):
    try:
        nvinspect_api.initialize(
            configs_dir + "fake_quantization_config.yaml", feature_dirs=feature_dirs
        )

        tensor = torch.rand(24, 2046).cuda()

        # use_process_tensor
        assert nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="fprop", tensor_name="activation", iteration=0
        )

        assert nvinspect_api.transformer_engine.use_process_tensor(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor_name="gradient", iteration=0
        )

        # process_tensor
        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1", gemm="fprop", tensor=tensor, tensor_name="activation", iteration=0
        )

        nvinspect_api.transformer_engine.process_tensor(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor=tensor, tensor_name="gradient", iteration=0
        )

        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.fc2", gemm="wgrad", iteration=0)
        # caching
        assert nvinspect_api.transformer_engine.fp8_gemm("decoder.1.fc2", gemm="wgrad", iteration=0)
    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_statistics_collection(configs_dir, feature_dirs):
    try:
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
            "decoder.1.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=200, rowwise=True
        )
        stats = log()
        assert stats[("decoder.1.mlp.fc1", "activation", "cur_amax", 200)] == tensor.abs().max()

        nvinspect_api.transformer_engine.look_at_tensor_before_process(
            "decoder.1.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=201, rowwise=True
        )
        assert_empty()
        nvinspect_api.transformer_engine.look_at_tensor_before_process(
            "decoder.2.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=200, rowwise=True
        )
        assert_empty()
        nvinspect_api.transformer_engine.look_at_tensor_before_process(
            "decoder.1.mlp.fc1", tensor=tensor, tensor_name="gradient", iteration=200, rowwise=True
        )
        assert_empty()

        expected_underflows = (tensor_fp8._data == 0).sum() * 100 / (100 * 100 * 5)
        expected_overflows = (tensor_fp8._data == 255).sum() * 100 / (100 * 100 * 5)

        # TE FP8 tensor stats --
        nvinspect_api.transformer_engine.look_at_tensor_after_process(
            "decoder.1.mlp.fc1", tensor=tensor_fp8, tensor_name="gradient", iteration=200, rowwise=True
        )
        stats = log()
        assert stats[("decoder.1.mlp.fc1", "gradient", "underflows%", 200)] == expected_underflows
        assert stats[("decoder.1.mlp.fc1", "gradient", "overflows%", 200)] == expected_overflows

        nvinspect_api.transformer_engine.look_at_tensor_after_process(
            "decoder.1.mlp.fc1", tensor=tensor_fp8, tensor_name="activation", iteration=201, rowwise=True
        )
        assert_empty()
        nvinspect_api.transformer_engine.look_at_tensor_after_process(
            "decoder.2.mlp.fc1", tensor=tensor_fp8, tensor_name="gradient", iteration=200, rowwise=True
        )
        assert_empty()

        # Second config in same yaml
        tensor = torch.rand((100, 100, 5))
        nvinspect_api.transformer_engine.look_at_tensor_before_process(
            "decoder.6.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=200, rowwise=True
        )
        stats = log()
        stats_names = [x[3] for x in stats.keys()]
        all(s in stats_names for s in ["cur_amax", "dynamic_range", "mean", "std", "l1_norm"])
        assert stats[("decoder.6.mlp.fc1", "activation", "mean", 200)] == tensor.mean()

        nvinspect_api.transformer_engine.look_at_tensor_before_process(
            "decoder.7.mlp.fc1", tensor=tensor, tensor_name="weight", iteration=200, rowwise=True
        )
        stats = log()
        stats_names = [x[3] for x in stats.keys()]
        all(s in stats_names for s in ["mean", "std", "l1_norm", "min", "max"])
        assert stats[("decoder.7.mlp.fc1", "weight", "max", 200)] == tensor.max()

        nvinspect_api.transformer_engine.look_at_tensor_before_process(
            "decoder.7.mlp.fc1", tensor=tensor, tensor_name="weight", iteration=201, rowwise=True
        )
        assert_empty()

    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()


def test_statistics_multi_run(configs_dir, feature_dirs):
    try:
        nvinspect_api.initialize(
            config_file=configs_dir + "stats_collection_test_config.yaml",
            feature_dirs=feature_dirs,
            default_logging_enabled=False,
        )

        def feed(tensor, tensor_fp8):
            nvinspect_api.transformer_engine.look_at_tensor_before_process(
                "decoder.5.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=1,
                rowwise=True
            )
            nvinspect_api.transformer_engine.look_at_tensor_after_process(
                "decoder.5.mlp.fc1", tensor=tensor_fp8, tensor_name="activation", iteration=1,
                rowwise=True
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
    finally:
        nvinspect_api.end_debug()
        transformer_engine.debug.debug_state.TEDebugState.reset()



if __name__ == "__main__":
    pass
