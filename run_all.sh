#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

torchrun --nproc_per_node=2 train.py --config config/case90.yaml
torchrun --nproc_per_node=2 train.py --config config/case60.yaml
torchrun --nproc_per_node=2 train.py --config config/case30.yaml
