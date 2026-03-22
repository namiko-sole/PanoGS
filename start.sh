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
python -u viewer.py \
/nas1/hyh22/backup/PanoGS/colmap_spliter/output_data_big_room_d918a_2dgs_submodels/room_1/colmap_submodel/output/point_cloud/iteration_30000/point_cloud.ply \
-s /nas1/hyh22/backup/PanoGS/colmap_spliter/output_data_big_room_d918a_2dgs_submodels/room_1/colmap_submodel \
--cameras_json /nas1/hyh22/backup/PanoGS/colmap_spliter/output_data_big_room_d918a_2dgs_submodels/room_1/colmap_submodel/output/cameras.json \
--style_img /nas1/hyh22/backup/PanoGS/style_data/indoor/room3.jpg \
--prep_dir preprocess/d918a_r1_cnn_room3 \
--port 8090

CUDA_VISIBLE_DEVICES=4 \
python -u viewer.py \
/nas1/hyh22/backup/PanoGS/colmap_spliter/output_rooms_d918a/room_1/colmap_submodel/output/point_cloud/iteration_30000/point_cloud.ply \
-s /nas1/hyh22/backup/PanoGS/colmap_spliter/output_rooms_d918a/room_1/colmap_submodel \
--cameras_json /nas1/hyh22/backup/PanoGS/colmap_spliter/output_rooms_d918a/room_1/colmap_submodel/output/cameras.json \
--style_img /nas1/hyh22/backup/PanoGS/style_data/indoor/room3.jpg \
--prep_dir preprocess/manual_d918a_r1_cnn_room3_adaptive_mini_tvloss0.1 \
--port 8092
