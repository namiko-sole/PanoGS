# CUDA_VISIBLE_DEVICES=0 \
# python -u viewer.py \
# data_2dgs/output/drjohnson/point_cloud/iteration_30000/point_cloud.ply \
# -s data_3dgs/db/drjohnson \
# --cameras_json data_2dgs/output/drjohnson/cameras.json \
# --style_img style_data/indoor/room5.jpg \
# --prep_dir preprocess/drjohnson_focus_cnn_room5 \
# --port 8087

# CUDA_VISIBLE_DEVICES=1 \
# python -u viewer.py \
# data_2dgs/output/drjohnson/point_cloud/iteration_30000/point_cloud.ply \
# -s data_3dgs/db/drjohnson \
# --cameras_json data_2dgs/output/drjohnson/cameras.json \
# --style_img style_data/indoor/office_room.jpg \
# --prep_dir preprocess/drjohnson_focus_cnn_office_room \
# --port 8088


CUDA_VISIBLE_DEVICES=6 \
python -u viewer_fast_optimize.py \
/nas1/hyh22/backup/PanoGS/data_2dgs/output/dl3dv_8cb2e/point_cloud/iteration_30000/point_cloud.ply \
-s /nas1/hyh22/backup/PanoGS/data_3dgs/DL3DV-10K-Benchmark/8cb2e97d26a639f05a571476240a8fa86988e6853f0f13cc05830d1578002aad/gaussian_splat_lowres \
--cameras_json /nas1/hyh22/backup/PanoGS/data_2dgs/output/dl3dv_8cb2e/cameras.json \
--style_img /nas1/hyh22/backup/PanoGS/style_data/cyber.jpg \
--prep_dir preprocess/dl3dv_8cb2e_cyber_center \
--port 8060

CUDA_VISIBLE_DEVICES=5 \
python -u viewer_big_room.py \
/nas1/hyh22/backup/PanoGS/data_big_room_816e9_2dgs_optimized/output/point_cloud/iteration_30000/point_cloud.ply \
-s /nas1/hyh22/backup/PanoGS/data_big_room_816e9_2dgs_optimized \
--cameras_json /nas1/hyh22/backup/PanoGS/data_big_room_816e9_2dgs_optimized/output/cameras.json \
--style_img /nas1/hyh22/backup/PanoGS/style_data/indoor/room3.jpg \
--prep_dir preprocess/manual_816e9_big_room_minorsplit_optimized \
--port 8022
