import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def load_bin(file_name):
    point_cloud = np.fromfile(file_name, dtype=np.float32)
    return point_cloud.reshape(-1, 4)

def load_label(file_name):
    labels = np.fromfile(file_name, dtype=np.int32)
    return labels & 0xFFFF  # Get the semantic label

def visualize_semantic_segmentation(point_cloud, labels):
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for label, color in color_map.items():
        colors[labels == label] = color
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # xyz
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)  # Convert to [0,1]
    
    o3d.visualization.draw_geometries([pcd])


def visualize_semantic_segmentation_with_legend(point_cloud, point_labels, label_descriptions, color_map):
    colors = np.zeros((point_labels.shape[0], 3), dtype=np.uint8)
    for label_id, color in color_map.items():
        colors[point_labels == label_id] = color
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # xyz
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)  # Convert to [0,1]
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name='Point Cloud Visualization')

    # Create a legend for the labels after the point cloud window is closed
    legend_labels = [label_descriptions[label_id] for label_id in sorted(label_descriptions.keys())]
    legend_colors = [color_map[label_id] for label_id in sorted(color_map.keys())]

    # Normalize colors to [0, 1] for matplotlib
    legend_colors_normalized = np.array(legend_colors) / 255.0

    # Plot legend
    plt.figure(figsize=(8, 6))
    for i, (color, label) in enumerate(zip(legend_colors_normalized, legend_labels)):
        plt.fill_between([0, 1], i - 0.5, i + 0.5, color=color)
        plt.text(1.1, i, label, verticalalignment='center', fontsize=9)
    plt.xlim(0, 2)
    plt.ylim(-1, len(legend_labels))
    plt.axis('off')

    # Use plt.show() without block=False to ensure it shows up and stays
    plt.show(block=False)




# labels = {
#     0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus",
#     15: "motorcycle", 16: "on-rails", 18: "truck", 20: "other-vehicle",
#     30: "person", 31: "bicyclist", 32: "motorcyclist", 40: "road", 44: "parking",
#     48: "sidewalk", 49: "other-ground", 50: "building", 51: "fence", 52: "other-structure",
#     60: "lane-marking", 70: "vegetation", 71: "trunk", 72: "terrain", 80: "pole", 
#     81: "traffic-sign", 99: "other-object", 252: "moving-car", 253: "moving-bicyclist",
#     254: "moving-person", 255: "moving-motorcyclist", 256: "moving-on-rails", 257: "moving-bus",
#     258: "moving-truck", 259: "moving-other-vehicle"
# }

# color_map = {
#     0: [0, 0, 0], 1: [0, 0, 255], 10: [245, 150, 100], 11: [245, 230, 100], 13: [250, 80, 100], 15: [150, 60, 30], 
#     16: [255, 0, 0], 18: [180, 30, 80], 20: [255, 0, 0], 30: [30, 30, 255], 31: [200, 40, 255], 32: [90, 30, 150], 
#     40: [255, 0, 255], 44: [255, 150, 255], 48: [75, 0, 75], 49: [75, 0, 175], 50: [0, 200, 255], 51: [50, 120, 255],
#     52: [0, 150, 255], 60: [170, 255, 150], 70: [0, 175, 0], 71: [0, 60, 135], 72: [80, 240, 150], 80: [150, 240, 255],
#     81: [0, 0, 255], 99: [255, 255, 50], 252: [245, 150, 100], 256: [255, 0, 0], 253: [200, 40, 255], 254: [30, 30, 255],
#     255: [90, 30, 150], 257: [250, 80, 100], 258: [180, 30, 80], 259: [255, 0, 0]
# }

### 14 = 51 -> fence; 12 = 49 -> other-ground
labels = {
    0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus",
    15: "motorcycle", 16: "on-rails", 18: "truck", 20: "other-vehicle",
    30: "person", 31: "bicyclist", 32: "motorcyclist", 40: "road", 44: "parking",
    48: "sidewalk", 12: "other-ground", 50: "building", 14: "fence", 52: "other-structure",
    60: "lane-marking", 70: "vegetation", 71: "trunk", 72: "terrain", 80: "pole", 
    81: "traffic-sign", 99: "other-object", 252: "moving-car", 253: "moving-bicyclist",
    254: "moving-person", 255: "moving-motorcyclist", 256: "moving-on-rails", 257: "moving-bus",
    258: "moving-truck", 259: "moving-other-vehicle"
}

color_map = {
    0: [0, 0, 0], 1: [0, 0, 255], 10: [245, 150, 100], 11: [245, 230, 100], 13: [250, 80, 100], 15: [150, 60, 30], 
    16: [255, 0, 0], 18: [180, 30, 80], 20: [255, 0, 0], 30: [30, 30, 255], 31: [200, 40, 255], 32: [90, 30, 150], 
    40: [255, 0, 255], 44: [255, 150, 255], 48: [75, 0, 75], 12: [75, 0, 175], 50: [0, 200, 255], 14: [50, 120, 255],
    52: [0, 150, 255], 60: [170, 255, 150], 70: [0, 175, 0], 71: [0, 60, 135], 72: [80, 240, 150], 80: [150, 240, 255],
    81: [0, 0, 255], 99: [255, 255, 50], 252: [245, 150, 100], 256: [255, 0, 0], 253: [200, 40, 255], 254: [30, 30, 255],
    255: [90, 30, 150], 257: [250, 80, 100], 258: [180, 30, 80], 259: [255, 0, 0]
}



# Load point cloud and labels

# point_cloud_file = "/media/avresearch/DATA/SphereFormer/test_data/dataset/sequences/00/velodyne/1681488505942115.bin"
# label_file = "/media/avresearch/DATA/SphereFormer/segmented/segmented0.label"

# point_cloud_file = "/media/avresearch/DATA/SphereFormer/test_data/dataset/sequences/00/velodyne/1681488653430824.bin"
# label_file = "/media/avresearch/DATA/SphereFormer/segmented/segmented1456.label"

point_cloud_file = "/media/avresearch/DATA/SphereFormer/test_output_from_ROSbag/dataset/sequences/00/velodyne/1699891948167198657.bin"
label_file = "/media/avresearch/DATA/SphereFormer/segmented/segmented1489.label"
points = load_bin(point_cloud_file)
point_labels = load_label(label_file)  # Renamed for clarity

# Visualize
# visualize_semantic_segmentation(points, labels)
visualize_semantic_segmentation_with_legend(points, point_labels, labels, color_map)

