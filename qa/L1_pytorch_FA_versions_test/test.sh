# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

: ${TE_PATH:=/opt/transformerengine}

FA_versions=(2.1.1 2.3.0 2.4.0.post1 2.4.1 2.5.7 2.6.3 3.0.0b1)
device_compute_capability=$(CUDA_VISIBLE_DEVICES=0 deviceQuery | grep -oP 'CUDA Capability Major/Minor version number:\s*\K[0-9]+\.[0-9]+')
apt-get update; apt-get install bc

if (( $(echo "$device_compute_capability >= 10.0" | bc -l) )); then
  echo "Skipping FA version tests ..."
  exit 0
else
  echo "Running fused attention tests for FA versions: ${FA_versions[*]}"
fi

pip install pytest==8.2.1
for fa_version in "${FA_versions[@]}"
do
  if [ "${fa_version}" \< "3.0.0" ]
  then
    pip install flash-attn==${fa_version}
  else
    pip install "git+https://github.com/Dao-AILab/flash-attention.git#egg=flashattn-hopper&subdirectory=hopper"
    python_path=`python -c "import site; print(site.getsitepackages()[0])"`
    mkdir -p $python_path/flashattn_hopper
    wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/hopper/flash_attn_interface.py
  fi
  NVTE_TORCH_COMPILE=0 pytest -v -s $TE_PATH/tests/pytorch/fused_attn/test_fused_attn.py
done
