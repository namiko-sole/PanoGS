#!/bin/bash

set -e

CHECKPOINTS_DIR="$(cd "$(dirname "$0")" && pwd)/checkpoints"
mkdir -p "$CHECKPOINTS_DIR"

download() {  # download <repo_id> <local_name> [extra args...]
    local repo=$1 name=$2
    shift 2
    local dest="$CHECKPOINTS_DIR/$name"
    if [ -e "$dest" ]; then
        echo "[skip] $dest already exists"
    else
        echo "[download] $repo -> $dest"
        huggingface-cli download "$repo" --local-dir "$dest" "$@"
    fi
}

download stabilityai/stable-diffusion-xl-base-1.0 stabilityai/stable-diffusion-xl-base-1.0
download h94/IP-Adapter h94/IP-Adapter --include "sdxl_models/*"
download diffusers/controlnet-canny-sdxl-1.0 diffusers/controlnet-canny-sdxl-1.0

echo "Done. Checkpoints are in $CHECKPOINTS_DIR"
