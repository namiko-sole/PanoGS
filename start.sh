#!/bin/bash
set -e

# ===============================Configuration=============================== #
SOURCE=data_2dgs/output/playroom
MODEL_DIR=data_2dgs/output/my_scene
STYLE_IMG=style_data/indoor/room3.jpg
PREP_DIR=preprocess/my_scene
RENDER_DIR=renders/my_scene
GPU=0
PROMPT=""
ITERS=30000
# =========================================================================== #

PLY="$MODEL_DIR/point_cloud/iteration_$ITERS/point_cloud.ply"
CAMERAS_JSON="$MODEL_DIR/cameras.json"

echo "=========================================="
echo "[1/3] Reconstruct the scene (2DGS)"
echo "=========================================="
if [ -f "$PLY" ]; then
    echo "[skip] found $PLY"
else
    CUDA_VISIBLE_DEVICES=$GPU \
    python 2d_gaussian_splatting/train.py \
        -s "$SOURCE" \
        -m "$MODEL_DIR" \
        --iterations "$ITERS"
fi

echo "=========================================="
echo "[2/3] Stylize (headless)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=$GPU \
python run_stylization.py \
    "$MODEL_DIR" \
    -s "$SOURCE" \
    --cameras_json "$CAMERAS_JSON" \
    --style_img "$STYLE_IMG" \
    --prep_dir "$PREP_DIR" \
    --prompt "$PROMPT" \
    --iterations "$ITERS"

echo "=========================================="
echo "[3/3] Render training views"
echo "=========================================="
CUDA_VISIBLE_DEVICES=$GPU \
python render_training_views.py \
    --prep_dir "$PREP_DIR" \
    -s "$SOURCE" \
    --output_dir "$RENDER_DIR"

echo "=========================================="
echo "Done."
echo "  stylized model : $PREP_DIR/cam_*/styled/scene_styled.ply"
echo "  rendered views : $RENDER_DIR"
echo "=========================================="
