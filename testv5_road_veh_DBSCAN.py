import os
import time
import random
import numpy as np
import argparse

import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import spconv.pytorch as spconv
import yaml
from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.logger import get_logger
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, Constant
from util.nuscenes import nuScenes
from util.semantic_kitti_ros import SemanticKITTI
from util.waymo import Waymo
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import struct

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField
import ros_numpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

lim_x = [3, 200]
lim_y = [-10, 10]
lim_z = [-5, 10]

semkitti_dataset = SemanticKITTI(split='val')

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/semantic_kitti/semantic_kitti_unet32_spherical_transformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def crop_pointcloud(pointcloud):
    mask = np.where((pointcloud[:, 0] >= lim_x[0]) & (pointcloud[:, 0] <= lim_x[1]) & (pointcloud[:, 1] >= lim_y[0]) & (pointcloud[:, 1] <= lim_y[1]) & (pointcloud[:, 2] >= lim_z[0]) & (pointcloud[:, 2] <= lim_z[1]))
    pointcloud = pointcloud[mask]
    return pointcloud

def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main():
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        print(args)
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou, current_marker_ids
    args, best_iou = argss, 0
    current_marker_ids = set()  # Initialize the marker IDs set
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # get model
    if args.arch == 'unet_spherical_transformer':
        from model.unet_spherical_transformer import Semantic as Model

        args.patch_size = np.array([args.voxel_size[i] * args.patch_size for i in range(3)]).astype(np.float32)
        window_size = args.patch_size * args.window_size
        window_size_sphere = np.array(args.window_size_sphere)
        model = Model(input_c=args.input_c,
                      m=args.m,
                      classes=args.classes,
                      block_reps=args.block_reps,
                      block_residual=args.block_residual,
                      layers=args.layers,
                      window_size=window_size,
                      window_size_sphere=window_size_sphere,
                      quant_size=window_size / args.quant_size_scale,
                      quant_size_sphere=window_size_sphere / args.quant_size_scale,
                      rel_query=args.rel_query,
                      rel_key=args.rel_key,
                      rel_value=args.rel_value,
                      drop_path_rate=args.drop_path_rate,
                      window_size_scale=args.window_size_scale,
                      grad_checkpoint_layers=args.grad_checkpoint_layers,
                      sphere_layers=args.sphere_layers,
                      a=args.a,
                      )
    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))

    checkpoint_path = '/media/avresearch/DATA/SphereFormer/model_semantic_kitti.pth'
    checkpoint = torch.load(checkpoint_path)

    # Remove 'module.' prefix from keys
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}

    # Load the modified state_dict to the model
    model.load_state_dict(new_state_dict)

    model = model.cuda()  # Move the model to GPU if you're using one

    # set optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "transformer_block" not in n and p.requires_grad],
            "lr": args.base_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "transformer_block" in n and p.requires_grad],
            "lr": args.base_lr * args.transformer_lr_scale,
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())

    if main_process():
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    # set loss func
    class_weight = args.get("class_weight", None)
    class_weight = torch.tensor(class_weight).cuda() if class_weight is not None else None
    if main_process():
        logger.info("class_weight: {}".format(class_weight))
        logger.info("loss_name: {}".format(args.get("loss_name", "ce_loss")))
    criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=args.ignore_label, reduction='none' if args.loss_name == 'focal_loss' else 'mean').cuda()

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print(type(checkpoint['state_dict']))

            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    val_transform = None
    args.use_tta = getattr(args, "use_tta", False)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Ensure the model is in evaluation mode
    model.eval()

    def ros_callback(msg):
        seg_points, output_labels = inference_from_ros_message(msg, model)

        # Filter points with label == 0 (cars)
        car_indices = np.where(output_labels.cpu().numpy() == 0)
        car_points = seg_points[car_indices]

        # Convert structured array to 2D array for DBSCAN
        if len(car_points) == 0:
            car_points_2d = np.empty((0, 3))
        else:
            car_points_2d = np.array([list(point) for point in car_points[['x', 'y', 'z']]])

        # Run DBSCAN clustering on car points if there are any car points
        if car_points_2d.shape[0] > 0:
            clustering = DBSCAN(eps=1, min_samples=50).fit(car_points_2d)
            cluster_labels = clustering.labels_

            # Create bounding boxes for each cluster
            unique_labels = set(cluster_labels)
            bounding_boxes = []
            for label in unique_labels:
                if label == -1:
                    continue  # Ignore noise points
                cluster_indices = np.where(cluster_labels == label)
                cluster_points = car_points_2d[cluster_indices]

                xmin, ymin, zmin = np.min(cluster_points, axis=0)
                xmax, ymax, zmax = np.max(cluster_points, axis=0)
                bounding_boxes.append(((xmin, ymin, zmin), (xmax, ymax, zmax)))
        else:
            bounding_boxes = []

        # Publish bounding boxes to RViz
        publish_bounding_boxes(bounding_boxes, msg.header)

        # Convert labels to colors
        colors = label_to_color(output_labels.cpu().numpy())

        # Create a new PointCloud2 message with colored points
        header = msg.header
        seg = ros_numpy.msgify(PointCloud2, seg_points)
        colored_cloud = create_colored_pointcloud2(seg, colors, header)

        # Publish the colored point cloud
        pub.publish(colored_cloud)



    def publish_bounding_boxes(bounding_boxes, header):
        global current_marker_ids
        marker_array = MarkerArray()
        new_marker_ids = set()

        for i, (min_point, max_point) in enumerate(bounding_boxes):
            marker_id = i  # Assign a unique marker ID
            new_marker_ids.add(marker_id)
            marker = Marker()
            marker.header = header
            marker.ns = "bounding_boxes"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = (min_point[0] + max_point[0]) / 2
            marker.pose.position.y = (min_point[1] + max_point[1]) / 2
            marker.pose.position.z = (min_point[2] + max_point[2]) / 2
            marker.scale.x = max_point[0] - min_point[0]
            marker.scale.y = max_point[1] - min_point[1]
            marker.scale.z = max_point[2] - min_point[2]
            marker.color.a = 0.5  # Transparency
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        # Add DELETE action for markers that are no longer present
        for marker_id in current_marker_ids - new_marker_ids:
            delete_marker = Marker()
            delete_marker.header = header
            delete_marker.ns = "bounding_boxes"
            delete_marker.id = marker_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        bounding_box_pub.publish(marker_array)
        current_marker_ids = new_marker_ids

    rospy.init_node('pointcloud_inference', anonymous=True)

    global pub, bounding_box_pub
    pub = rospy.Publisher("/colored_points", PointCloud2, queue_size=1)
    bounding_box_pub = rospy.Publisher("/bounding_boxes", MarkerArray, queue_size=1)
    rospy.Subscriber("/lidar_tc/velodyne_points", PointCloud2, ros_callback, queue_size=1)

    rospy.spin()

def convert_to_binary_format(np_points, intensities):
    # Concatenate intensity values with xyz points
    np_points_with_intensity = np.hstack((np_points, intensities.reshape(-1, 1)))
    binary_points = np_points_with_intensity.astype(np.float32).tobytes()
    labels = np.zeros(np_points_with_intensity.shape[0], dtype=np.uint32)
    binary_labels = labels.tobytes()
    return np.asarray(binary_points), binary_labels

def collation_fn_voxelmean(batch):
    coords, xyz, feats, labels, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)
    accmulate_points_num = 0
    offset = []
    for i in range(len(coords)):
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]
        offset.append(accmulate_points_num)
    coords = torch.cat(coords)
    xyz = torch.cat(xyz)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    offset = torch.IntTensor(offset)
    inds_recons = torch.cat(inds_recons)
    return coords, xyz, feats, labels, offset, inds_recons

def inference_from_ros_message(ros_msg, model):
    pc = ros_numpy.numpify(ros_msg)
    pcd = np.zeros((pc.shape[0], 4))
    pcd[:, 0] = pc['x']
    pcd[:, 1] = pc['y']
    pcd[:, 2] = pc['z']
    np_points = np.asarray(pcd)
    np_points = crop_pointcloud(np_points)
    p_points = np_points[:, 0:3]
    intensities = np_points[:, 3]
    binary_points, binary_labels = convert_to_binary_format(p_points, intensities)
    processed_data = semkitti_dataset.process_live_data(binary_points, binary_labels)
    batch_data = [processed_data]
    (coord, xyz, feat, target, offset, inds_reverse) = collation_fn_voxelmean(batch_data)
    inds_reverse = inds_reverse.to('cuda:0', non_blocking=True)
    offset_ = offset.clone()
    offset_[1:] = offset_[1:] - offset_[:-1]
    batch = torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0).long()
    coord = torch.cat([batch.unsqueeze(-1), coord], -1)
    spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
    batch = batch.cuda(non_blocking=True)
    sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, 1)
    assert batch.shape[0] == feat.shape[0]
    with torch.no_grad():
        output = model(sinput, xyz, batch)
        output = output[inds_reverse, :]
        output = output.max(1)[1]
    points = np.zeros(np_points.shape[0], dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32)])
    points['x'] = np_points[:, 0]
    points['y'] = np_points[:, 1]
    points['z'] = np_points[:, 2]
    return points, output

def label_to_color(labels):
    color_map = {
        0: [0, 0, 255], 1: [0, 0, 0], 10: [0, 0, 0], 11: [0, 0, 0], 13: [0, 0, 0], 15: [0, 0, 0],
        16: [0, 0, 0], 18: [0, 0, 0], 20: [0, 0, 0], 30: [0, 0, 0], 31: [0, 0, 0], 8: [255, 0, 0],
        40: [0, 0, 0], 44: [0, 0, 0], 48: [0, 0, 0], 12: [0, 0, 0], 50: [0, 0, 0], 14: [0, 0, 0],
        52: [0, 0, 0], 60: [0, 0, 0], 70: [0, 0, 0], 71: [0, 0, 0], 72: [0, 0, 0], 80: [0, 0, 0],
        81: [0, 0, 0], 99: [0, 0, 0], 252: [0, 0, 0], 256: [0, 0, 0], 253: [0, 0, 0], 254: [0, 0, 0],
        255: [0, 0, 0], 257: [0, 0, 0], 258: [0, 0, 0], 259: [0, 0, 0]
    }
    default_color = color_map[8]
    colors = np.array([color_map.get(label, default_color) for label in labels])
    return colors

def create_colored_pointcloud2(original_cloud, colors, header):
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1)
    ]
    new_points = []
    for point, color in zip(pc2.read_points(original_cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True), colors):
        rgb_packed = struct.pack('BBBB', color[2], color[1], color[0], 255)
        rgb_float = struct.unpack('f', rgb_packed)[0]
        new_points.append([point[0], point[1], point[2], point[3], rgb_float])
    colored_cloud = pc2.create_cloud(header, fields, new_points)
    return colored_cloud

if __name__ == '__main__':
    main()

