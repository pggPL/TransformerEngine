# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

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

    tensor = torch.rand(24, 2046).cuda()

    # FP8 enabled
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="fprop", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="dgrad", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="wgrad", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_enabled_for_layer("decoder.1.mlp.fc1", fp8_enabled=True)["ret"]

    output = nvinspect_api.transformer_engine.process_tensor("decoder.1.attn.qkv", tensor=tensor, gemm="fprop", tensor_name="activation", fp8_enabled=True)["tensor"]
    torch.testing.assert_close(output, tensor)

    # If FP8 enabled -> Delayed scaling enabled by default
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.attn.proj", gemm="dgrad", tensor_name="gradient", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="dgrad", tensor_name="weight", fp8_enabled=True)["ret"]

    # No FP8
    assert not nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="wgrad", tensor_name="activation", fp8_enabled=False)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.qkv", gemm="fprop", fp8_enabled=False)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_enabled_for_layer("decoder.1.mlp.fc1", fp8_enabled=False)["ret"]

    output_no_fp8 = nvinspect_api.transformer_engine.process_tensor("decoder.1.attn.qkv", tensor=tensor, gemm="fprop", tensor_name="gradient", fp8_enabled=False)["tensor"]
    torch.testing.assert_close(output_no_fp8, tensor)

    nvinspect_api.end_debug()
    transformer_engine.debug.debug_state.DebugLayerState.reset()


def test_disable_fp8_gemm(configs_dir, feature_dirs):
    print(configs_dir)
    reset_debug_log()
    nvinspect_api.initialize(configs_dir + "disable_fp8_gemms.yaml",
                               feature_dirs=feature_dirs)
    
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="fprop", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="dgrad", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="wgrad", fp8_enabled=True)["ret"]

    # caching
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="fprop", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="dgrad", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="wgrad", fp8_enabled=True)["ret"]

    # If FP8 gemm enabled -> Delayed scaling enabled by default
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.attn.proj", gemm="dgrad", tensor_name="gradient", fp8_enabled=True)["ret"]
    # caching
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.attn.proj", gemm="dgrad", tensor_name="gradient", fp8_enabled=True)["ret"]

    # Should call the default is_fp8_delayed_scaling_enabled
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="dgrad", tensor_name="weight", fp8_enabled=True)["ret"]

    # Disabling FP8
    assert not nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="wgrad", tensor_name="activation", fp8_enabled=False)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.qkv", gemm="fprop", fp8_enabled=False)["ret"]

    nvinspect_api.end_debug()
    transformer_engine.debug.debug_state.DebugLayerState.reset()


def test_disable_fp8_layer(configs_dir, feature_dirs):
    reset_debug_log()

    nvinspect_api.initialize(configs_dir + "disable_fp8_layer.yaml", 
                               feature_dirs=feature_dirs)
    
    assert nvinspect_api.transformer_engine.is_fp8_enabled_for_layer("decoder.1.mlp.fc1", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_enabled_for_layer("decoder.1.attn.qkv", fp8_enabled=True)["ret"]

    # Disabling FP8
    assert not nvinspect_api.transformer_engine.is_fp8_enabled_for_layer("decoder.1.mlp.fc1", fp8_enabled=False)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_enabled_for_layer("decoder.1.attn.qkv", fp8_enabled=False)["ret"]

    nvinspect_api.end_debug()
    transformer_engine.debug.debug_state.DebugLayerState.reset()


def test_per_tensor_scaling(configs_dir, feature_dirs):
    reset_debug_log()

    nvinspect_api.initialize(configs_dir + "per_tensor_scaling.yaml", 
                               feature_dirs=feature_dirs)
    
    tensor = torch.rand(24, 2046).cuda()

    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="fprop", tensor=tensor, tensor_name="activation", fp8_enabled=True, fp8_dtype=tex.DType.kFloat8E4M3)
    assert check_debug_log("decoder.1.mlp.fc1: Feature=PerTensorScaling, API=process_tensor: fprop, activation: Per Tensor Scaling")

    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="dgrad", tensor=tensor, tensor_name="gradient", fp8_enabled=True, fp8_dtype=tex.DType.kFloat8E5M2)
    assert check_debug_log("decoder.1.mlp.fc1: Feature=PerTensorScaling, API=process_tensor: dgrad, gradient: Per Tensor Scaling")

    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="wgrad", tensor=tensor, tensor_name="gradient", fp8_enabled=True, fp8_dtype=tex.DType.kFloat8E5M2)
    assert not check_debug_log("decoder.1.mlp.fc1: Feature=PerTensorScaling, API=process_tensor: wgrad, gradient: Per Tensor Scaling")
    assert not check_debug_log("decoder.1.mlp.fc1: Feature=PerTensorScaling, API=process_tensor: wgrad, activation: Per Tensor Scaling")

    # FP8, in config
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.fc2", gemm="fprop", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.fc2", gemm="dgrad", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="fprop", tensor_name="activation", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="dgrad", tensor_name="gradient", fp8_enabled=True)["ret"]

    # caching
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.fc2", gemm="fprop", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.fc2", gemm="dgrad", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="fprop", tensor_name="activation", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="dgrad", tensor_name="gradient", fp8_enabled=True)["ret"]
    
    # Not in config
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.3.fc2", gemm="wgrad", tensor_name="activation", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.3.fc2", gemm="dgrad", tensor_name="weight", fp8_enabled=True)["ret"]
    
    # No FP8
    assert not nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc1", gemm="dgrad", tensor_name="gradient", fp8_enabled=False)["ret"]

    reset_debug_log()
    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc4", gemm="fprop", tensor=tensor, tensor_name="activation", fp8_enabled=True, fp8_dtype=tex.DType.kFloat8E4M3)
    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc4", gemm="dgrad", tensor=tensor, tensor_name="gradient", fp8_enabled=True, fp8_dtype=tex.DType.kFloat8E5M2)
    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc4", gemm="wgrad", tensor=tensor, tensor_name="gradient", fp8_enabled=True, fp8_dtype=tex.DType.kFloat8E5M2)

    assert not check_debug_log("decoder.1.mlp.fc4: Feature=PerTensorScaling, API=process_tensor: fprop, activation: Per Tensor Scaling")

    nvinspect_api.end_debug()
    transformer_engine.debug.debug_state.DebugLayerState.reset()


def test_fake_quant(configs_dir, feature_dirs):
    reset_debug_log()

    nvinspect_api.initialize(configs_dir + "fake_quantization_config.yaml",
                               feature_dirs=feature_dirs)

    tensor = torch.rand(24, 2046).cuda()

    # FP8 enabled - in config
    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="fprop", tensor=tensor, fp8_enabled=True, tensor_name="activation")
    assert check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: fprop, activation: QDQ")

    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="dgrad", tensor=tensor, fp8_enabled=True, tensor_name="gradient")
    assert check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: dgrad, gradient: QDQ")

    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="wgrad", tensor_name="activation", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="wgrad", tensor_name="gradient", fp8_enabled=True)["ret"]

    # caching
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="wgrad", tensor_name="activation", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="wgrad", tensor_name="gradient", fp8_enabled=True)["ret"]

    # Should call the default is_fp8_delayed_scaling_enabled
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="dgrad", tensor_name="gradient", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="dgrad", tensor_name="weight", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc2", gemm="fprop", tensor_name="weight", fp8_enabled=True)["ret"]

    assert not nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.fc2", gemm="fprop", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.fc2", gemm="dgrad", fp8_enabled=True)["ret"]

    reset_debug_log()

    # FP8 enabled - out of config
    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="fprop", tensor=tensor, fp8_enabled=True, tensor_name="weight")
    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="dgrad", tensor=tensor, fp8_enabled=True, tensor_name="weight")
    torch.testing.assert_close(tensor, nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="wgrad", tensor=tensor, fp8_enabled=True, tensor_name="gradient")["tensor"])
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.fc2", gemm="wgrad", fp8_enabled=True)["ret"]
    # caching
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.fc2", gemm="wgrad", fp8_enabled=True)["ret"]
    
    assert not check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: fprop, weight: QDQ")
    assert not check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: dgrad, weight: QDQ")
    assert not check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: wgrad, gradient: QDQ")
    reset_debug_log()
    
    # Disabling FP8 - in config
    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.moe.fc1", gemm="fprop", tensor=tensor, fp8_enabled=False, tensor_name="activation")
    assert check_debug_log("decoder.1.mlp.moe.fc1: Feature=FakeQuantFp8, API=process_tensor: fprop, activation: QDQ")
    
    # Disabling FP8 - out of config
    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="dgrad", tensor=tensor, fp8_enabled=False, tensor_name="weight")
    assert not check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: dgrad, weight: QDQ")

    nvinspect_api.end_debug()
    transformer_engine.debug.debug_state.DebugLayerState.reset()


def test_feature_medley(configs_dir, feature_dirs):
    reset_debug_log()
    nvinspect_api.initialize(configs_dir + "tranformer_engine_feature_medley.yaml",
                               feature_dirs=feature_dirs)
    
    assert not nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.mlp.fc1", gemm="fprop", fp8_enabled=True)["ret"]
    assert check_debug_log("decoder.1.mlp.fc1: Feature=DisableFp8Gemm, API=is_fp8_gemm_enabled: fprop: FP8 GEMM: False")
    assert nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="dgrad", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_gemm_enabled("decoder.1.attn.qkv", gemm="wgrad", fp8_enabled=True)["ret"]
    assert check_debug_log("ecoder.1.attn.qkv: Feature=FakeQuantFp8, API=is_fp8_gemm_enabled: wgrad: FP8 GEMM: False")

    # Should call the default is_fp8_delayed_scaling_enabled
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc1", gemm="fprop", tensor_name="weight", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc1", gemm="dgrad", tensor_name="gradient", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc1", gemm="dgrad", tensor_name="weight", fp8_enabled=True)["ret"]
    # Should call the default is_fp8_delayed_scaling_enabled
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc1", gemm="wgrad", tensor_name="activation", fp8_enabled=True)["ret"]
    # Should call the default is_fp8_delayed_scaling_enabled
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc1", gemm="wgrad", tensor_name="gradient", fp8_enabled=True)["ret"]

    # caching
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc1", gemm="wgrad", tensor_name="gradient", fp8_enabled=True)["ret"]
    assert nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc1", gemm="wgrad", tensor_name="activation", fp8_enabled=True)["ret"]
    assert not nvinspect_api.transformer_engine.is_fp8_delayed_scaling_enabled("decoder.1.fc1", gemm="dgrad", tensor_name="gradient", fp8_enabled=True)["ret"]

    tensor = torch.rand(24, 2046).cuda()
    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="fprop", tensor=tensor, fp8_enabled=True, tensor_name="activation")

    nvinspect_api.transformer_engine.process_tensor("decoder.1.attn.qkv", gemm="dgrad", tensor=tensor, fp8_enabled=True, tensor_name="gradient", fp8_dtype=tex.DType.kFloat8E4M3)
    assert check_debug_log("decoder.1.attn.qkv: Feature=PerTensorScaling, API=process_tensor: dgrad, gradient: Per Tensor Scaling") and not check_debug_log("decoder.1.attn.proj: Feature=PerTensorScaling, API=process_tensor: dgrad, gradient: Per Tensor Scaling")

    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="wgrad", tensor=tensor, fp8_enabled=True, tensor_name="gradient")
    assert check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: wgrad, gradient: QDQ") and not check_debug_log("decoder.1.mlp.fc2: Feature=FakeQuantFp8, API=process_tensor: wgrad, gradient: QDQ")

    reset_debug_log()

    # Disabling FP8
    torch.testing.assert_close(tensor, nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="fprop", tensor=tensor, fp8_enabled=False, tensor_name="activation")["tensor"])
    try:
        # This should fail since trying to use per tensor scaling when fp8_enabled=False.
        tensor, nvinspect_api.transformer_engine.process_tensor("decoder.1.attn.qkv", gemm="dgrad", tensor=tensor, fp8_enabled=False, tensor_name="gradient", fp8_dtype=tex.DType.kFloat8E4M3)
    except AssertionError:
        pass
    nvinspect_api.transformer_engine.process_tensor("decoder.1.mlp.fc1", gemm="wgrad", tensor=tensor, fp8_enabled=False, tensor_name="gradient")

    assert not check_debug_log("decoder.1.attn.qkv: Feature=PerTensorScaling, API=process_tensor: dgrad, gradient: Per Tensor Scaling")
    assert check_debug_log("decoder.1.mlp.fc1: Feature=FakeQuantFp8, API=process_tensor: wgrad, gradient: QDQ")

    nvinspect_api.end_debug()
    transformer_engine.debug.debug_state.DebugLayerState.reset()


def test_statistics_collection(configs_dir, feature_dirs):
    nvinspect_api.initialize(config_file=configs_dir + "stats_collection_test_config.yaml",
                               feature_dirs=feature_dirs, default_logging_enabled=False)

    tensor = torch.randn((100, 100, 5))
    tensor_fp8 = tensor.to(torch.uint8)

    def log(forward):
        return transformer_engine.debug.features.utils.stats_buffer.STATS_BUFFERS.log_stats(forward)

    def assert_empty():
        stats = log(forward=True)
        assert len(stats) == 0
        stats = log(forward=False)
        assert len(stats) == 0


    # TE tensor stats -- 
    nvinspect_api.transformer_engine.save_stats_for_logging("decoder.1.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=200)
    stats = log(forward=True)
    assert stats[("decoder.1.mlp.fc1", "activation", "cur_amax", 200)] == tensor.abs().max()

    nvinspect_api.transformer_engine.save_stats_for_logging("decoder.1.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=201)
    assert_empty()
    nvinspect_api.transformer_engine.save_stats_for_logging("decoder.2.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=200)
    assert_empty()
    nvinspect_api.transformer_engine.save_stats_for_logging("decoder.1.mlp.fc1", tensor=tensor, tensor_name="gradient", iteration=200)
    assert_empty()

    expected_underflows = (tensor_fp8 == 0).sum() * 100 / (100 * 100 * 5)
    expected_overflows = (tensor_fp8 == 255).sum() * 100 / (100 * 100 * 5)

    # TE FP8 tensor stats -- 
    nvinspect_api.transformer_engine.save_fp8_stats_for_logging("decoder.1.mlp.fc1", tensor=tensor_fp8, tensor_name="gradient", iteration=200)
    stats = log(forward=True)
    assert len(stats) == 0 # gradient is logged during backward
    stats = log(forward=False)
    assert stats[("decoder.1.mlp.fc1", "gradient", "underflows%", 200)] == expected_underflows
    assert stats[("decoder.1.mlp.fc1", "gradient", "overflows%", 200)] == expected_overflows

    nvinspect_api.transformer_engine.save_fp8_stats_for_logging("decoder.1.mlp.fc1", tensor=tensor_fp8, tensor_name="activation", iteration=201)
    assert_empty()
    nvinspect_api.transformer_engine.save_fp8_stats_for_logging("decoder.2.mlp.fc1", tensor=tensor_fp8, tensor_name="gradient", iteration=200)
    assert_empty()


    # Second config in same yaml
    tensor = torch.rand((100, 100, 5))
    nvinspect_api.transformer_engine.save_stats_for_logging("decoder.6.mlp.fc1", tensor=tensor, tensor_name="activation", iteration=200)
    stats = log(forward=True)
    stats_names = [x[3] for x in stats.keys()]
    all(s in stats_names for s in ["cur_amax", "dynamic_range", "mean", "std", "l1_norm"])
    assert stats[("decoder.6.mlp.fc1", "activation", "mean", 200)] == tensor.mean()


    nvinspect_api.transformer_engine.save_stats_for_logging("decoder.7.mlp.fc1", tensor=tensor, tensor_name="weight", iteration=200)
    stats = log(forward=True)
    stats_names = [x[3] for x in stats.keys()]
    all(s in stats_names for s in ["mean", "std", "l1_norm", "min", "max"])
    assert stats[("decoder.7.mlp.fc1", "weight", "max", 200)] == tensor.max()


    nvinspect_api.transformer_engine.save_stats_for_logging("decoder.7.mlp.fc1", tensor=tensor, tensor_name="weight", iteration=201)
    assert_empty()

    nvinspect_api.end_debug()
    transformer_engine.debug.debug_state.DebugLayerState.reset()
    reset_debug_log()

if __name__ == "__main__":
    pass