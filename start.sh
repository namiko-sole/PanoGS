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


CUDA_VISIBLE_DEVICES=2 \
python -u viewer.py \
/nas1/hyh22/backup/PanoGS/colmap_spliter/output_rooms_d918a/room_5/colmap_submodel/output/point_cloud/iteration_30000/point_cloud.ply \
-s /nas1/hyh22/backup/PanoGS/colmap_spliter/output_rooms_d918a/room_5/colmap_submodel \
--cameras_json /nas1/hyh22/backup/PanoGS/colmap_spliter/output_rooms_d918a/room_5/colmap_submodel/output/cameras.json \
--style_img /nas1/hyh22/backup/PanoGS/style_data/indoor/room3.jpg \
--prep_dir preprocess/manual_d918a_r5_cnn_room3_adaptive_mini_tvloss1e-3_pretrain100_strength0.5_nodynamic \
--port 8060

CUDA_VISIBLE_DEVICES=5 \
python -u viewer_big_room.py \
/nas1/hyh22/backup/PanoGS/data_big_room_d918a_2dgs/output/point_cloud/iteration_30000/point_cloud.ply \
-s /nas1/hyh22/backup/PanoGS/data_big_room_d918a_2dgs/ \
--cameras_json /nas1/hyh22/backup/PanoGS/data_big_room_d918a_2dgs/output/cameras.json \
--style_img /nas1/hyh22/backup/PanoGS/style_data/indoor/room3.jpg \
--prep_dir preprocess/manual_d918a_big_room_cnn_room3_tvloss0.1_mpointcloud \
--port 8092
