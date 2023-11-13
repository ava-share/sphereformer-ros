# sphereformer-ros
An ros implementation of SphereFormer

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


