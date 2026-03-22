python split_rooms.py --input /nas1/hyh22/backup/PanoGS/data_big_room_d918a_2dgs --output /nas1/hyh22/backup/PanoGS/colmap_spliter/output_data_big_room_d918a_2dgs

# Recomended parameters for tuning:
python split_rooms.py --input /nas1/hyh22/backup/PanoGS/data_big_room_d918a_2dgs --output /nas1/hyh22/backup/PanoGS/colmap_spliter/output_data_big_room_d918a_2dgs_submodels --min-shared-points 30 --edge-threshold 0.015 --knn 20 --min-room-size 20

bash colmap_spliter/adaptive_split.sh \
  --input-list colmap_spliter/dataset_list.example.txt \
  --output-root /nas1/hyh22/backup/PanoGS/colmap_spliter/batch_outputs \
  --target-rooms 0