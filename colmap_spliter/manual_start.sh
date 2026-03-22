CUDA_VISIBLE_DEVICES=5 python colmap_spliter/manual_split_viewer.py \
  --model-path /nas1/hyh22/backup/PanoGS/data_big_room_d918a_2dgs/output/point_cloud/iteration_30000/point_cloud.ply \
  --input /nas1/hyh22/backup/PanoGS/data_big_room_d918a_2dgs \
  --output /nas1/hyh22/backup/PanoGS/colmap_spliter/output_rooms_d918a \
  --host 0.0.0.0 \
  --port 8090