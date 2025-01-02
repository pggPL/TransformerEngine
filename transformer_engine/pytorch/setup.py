# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script for TE pytorch extensions."""

# pylint: disable=wrong-import-position,wrong-import-order

import sys
import os
import shutil
from pathlib import Path

import setuptools
from torch.utils.cpp_extension import BuildExtension

from importlib.metadata import version as get_pkg_version
from importlib.metadata import PackageNotFoundError
from packaging.version import Version as PkgVersion

try:
    import torch  # pylint: disable=unused-import
except ImportError as e:
    raise RuntimeError("This package needs Torch to build.") from e


current_file_path = Path(__file__).parent.resolve()
build_tools_dir = current_file_path.parent.parent / "build_tools"
if bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))) or os.path.isdir(build_tools_dir):
    build_tools_copy = current_file_path / "build_tools"
    if build_tools_copy.exists():
        shutil.rmtree(build_tools_copy)
    shutil.copytree(build_tools_dir, build_tools_copy)


from build_tools.build_ext import get_build_ext
from build_tools.utils import copy_common_headers
from build_tools.te_version import te_version
from build_tools.pytorch import setup_pytorch_extension


os.environ["NVTE_PROJECT_BUILDING"] = "1"
CMakeBuildExtension = get_build_ext(BuildExtension)


if __name__ == "__main__":
    # Extensions
    common_headers_dir = "common_headers"
    copy_common_headers(current_file_path.parent, str(current_file_path / common_headers_dir))
    ext_modules = [
        setup_pytorch_extension(
            "csrc", current_file_path / "csrc", current_file_path / common_headers_dir
        )
    ]

    # FA for blackwell.
    try:
        fa_version = PkgVersion(get_pkg_version("flash-attn"))
    except PackageNotFoundError:
        fa_version = "unknown"
    if fa_version != PkgVersion("2.4.2.dev0"):
        import subprocess

        fa_path = current_file_path.parent.parent / "3rdparty/flashattn_internal"
        subprocess.check_call([sys.executable, "-m", "pip", "install", fa_path])

    # Configure package
    setuptools.setup(
        name="transformer_engine_torch",
        version=te_version(),
        description="Transformer acceleration library - Torch Lib",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        install_requires=["torch"],
        tests_require=["numpy", "torchvision"],
    )
    if any(x in sys.argv for x in (".", "sdist", "bdist_wheel")):
        shutil.rmtree(common_headers_dir)
        shutil.rmtree("build_tools")
