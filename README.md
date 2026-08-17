# sphereformer-ros
An ros implementation of SphereFormer

## DBSCAN vehicle clustering

`testv5_road_veh_DBSCAN.py` runs SphereFormer inference on a live ROS point
cloud, applies DBSCAN to points classified as cars, and publishes an
axis-aligned bounding box for each cluster.

Before running it, set `checkpoint_path` in
`testv5_road_veh_DBSCAN.py` to the location of the SemanticKITTI model
checkpoint. The script requires a CUDA-capable GPU and the dependencies in
`sphereformer.yaml`, including scikit-learn.

Run each command in a separate terminal from the repository root:

```bash
roscore
rosbag play /path/to/input.bag
python testv5_road_veh_DBSCAN.py
rviz
```

The script subscribes to `/lidar_tc/velodyne_points` and publishes:

- `/colored_points` (`sensor_msgs/PointCloud2`): the segmented point cloud.
- `/bounding_boxes` (`visualization_msgs/MarkerArray`): green DBSCAN cluster
  bounding boxes.

In RViz, add a **PointCloud2** display for `/colored_points` and a
**MarkerArray** display for `/bounding_boxes`, then set the fixed frame to the
frame used by the input point cloud. The DBSCAN defaults are `eps=1` and
`min_samples=50`; adjust them in `testv5_road_veh_DBSCAN.py` for different
point-cloud densities. If your lidar uses a different topic, change the
subscriber topic near the bottom of the script.

## 01/16/2024 update Roadway detection (test_ros_v3.py)

Category 8 is detected as the roadway and marked with red points. Roughly 50 meters ahead and behind the ego vehicle.

```
roscore
rosbag play [.bag filepath]
python test_ros_v3.py
rviz
```
![alt text](https://github.com/ava-share/sphereformer-ros/blob/main/road_detection.png)



## 11/16/2023 update

**real-time inference**
```
roscore
rosbag play [.bag filepath]
python test_ros_v2.py
rviz
```
![alt text](https://github.com/ava-share/sphereformer-ros/blob/main/vis_Rviz.png)



---------------

## Step 1: construct a data organization consitant with SemanticKITTI

```
roscore
rosbag play [.bag filepath]
python pcd2bin_label_ros.py
```

## Step 2: Run inference code

```
python test.py

```

## Step 3: visualize the segmented result with open3d (optional)

```
python visulize.py

```

## An example of the lidar segmentation result
![alt text](https://github.com/ava-share/sphereformer-ros/blob/main/segmented_result_with_normalized_intensity.png)

### And a closer look
![alt text](https://github.com/ava-share/sphereformer-ros/blob/main/closer_look.png)




