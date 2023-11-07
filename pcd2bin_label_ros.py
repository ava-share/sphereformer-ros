#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import os

def convert_pcd_to_bin(np_points, bin_filename, reflectance_value=0.3):
    # Add a column for reflectance
    reflectance = np.full((np_points.shape[0], 1), reflectance_value)
    np_points_with_reflectance = np.hstack((np_points, reflectance))

    # Save to .bin file
    np_points_with_reflectance.astype(np.float32).tofile(bin_filename)
    print(f"Converted to {bin_filename}")

def create_fake_label_file(bin_filename, label_folder):
    # Determine the number of points
    num_points = os.path.getsize(bin_filename) // (4 * 4)  # 4 floats per point, 4 bytes per float

    # Create a zero-valued array for labels
    labels = np.zeros(num_points, dtype=np.uint32)

    # Construct the corresponding .label file path
    label_filename = os.path.splitext(os.path.basename(bin_filename))[0] + '.label'
    label_path = os.path.join(label_folder, label_filename)

    # Save to .label file
    labels.tofile(label_path)
    print(f"Created fake label file {label_path}")

# Convert a PointCloud2 message to a PCD format
def pointcloud2_to_pcd(point_cloud2_msg):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Read points from the PointCloud2 message
    gen = pc2.read_points(point_cloud2_msg, skip_nans=True, field_names=("x", "y", "z"))
    points = np.array(list(gen))

    # Transform the points to Open3D usable format
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def pointcloud_callback(point_cloud2_msg):
    # Get current time to create a unique filename
    timestamp = rospy.Time.now()

    # Convert the PointCloud2 message to PCD and then to numpy array
    pcd = pointcloud2_to_pcd(point_cloud2_msg)
    np_points = np.asarray(pcd.points)

    # Create a unique filename for the .bin file
    output_filename = str(timestamp) + '.bin'
    bin_filename = os.path.join(bin_folder, output_filename)

    # Call the conversion function to create the .bin file
    convert_pcd_to_bin(np_points, bin_filename)

    # Call the function to create a fake label file
    create_fake_label_file(bin_filename, label_folder)

if __name__ == '__main__':
    rospy.init_node('pcd_to_bin_converter', anonymous=True)
    bin_folder = '/media/avresearch/DATA/SphereFormer/test_output_from_ROSbag/dataset/sequences/00/velodyne'
    label_folder = '/media/avresearch/DATA/SphereFormer/test_output_from_ROSbag/dataset/sequences/00/labels'

    # Make sure the bin and label folders exist
    if not os.path.exists(bin_folder):
        os.makedirs(bin_folder)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    rospy.Subscriber("/lidar_tc/velodyne_points", PointCloud2, pointcloud_callback)
    rospy.spin()
