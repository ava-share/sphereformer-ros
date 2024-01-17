import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import zlib
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


from functools import partial
import pickle
import yaml
from torch_scatter import scatter_mean




from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.logger import get_logger
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, Constant
from util.nuscenes import nuScenes
from util.semantic_kitti_ros import SemanticKITTI
from util.waymo import Waymo


#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import os
import struct
from sensor_msgs.msg import PointField



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


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return  not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
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
    global args, best_iou
    args, best_iou = argss, 0
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

    model = model.cuda() # Move the model to GPU if you're using one
    
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
            #permute torch tensor of model weight
            
            

            
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


    # Define the ROS callback function within main_worker
    def ros_callback(msg):
        # Process the ROS message and perform inference
        output_labels = inference_from_ros_message(msg, model)

        # Convert labels to colors
        colors = label_to_color(output_labels.cpu().numpy())

        # Create a new PointCloud2 message with colored points
        header = msg.header  # Use the same header as the input message
        colored_cloud = create_colored_pointcloud2(msg, colors, header)

        # Publish the colored point cloud
        pub.publish(colored_cloud)


    # ROS initialization
    rospy.init_node('pointcloud_inference', anonymous=True)

    global pub
    pub = rospy.Publisher("/colored_points", PointCloud2, queue_size=10)
    rospy.Subscriber("/lidar_tc/velodyne_points", PointCloud2, ros_callback)

    # Spin to keep the script for exiting
    rospy.spin()


# Convert a PointCloud2 message to a PCD format and also extract intensity
def pointcloud2_to_pcd(point_cloud2_msg):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Read points and intensity from the PointCloud2 message
    gen = pc2.read_points(point_cloud2_msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))

    points_and_intensity = np.array(list(gen))
    points = points_and_intensity[:, :3]  # Extract xyz points
    raw_intensities = points_and_intensity[:, 3]  # Extract raw intensity values

    # Normalize intensities by dividing each by 256
    normalized_intensities = raw_intensities / 256.0

    # Transform the points to Open3D usable format
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, normalized_intensities

def convert_to_binary_format(np_points, intensities):
    # Concatenate intensity values with xyz points
    np_points_with_intensity = np.hstack((np_points, intensities.reshape(-1, 1)))

    # Convert numpy arrays to binary format
    binary_points = np_points_with_intensity.astype(np.float32).tobytes()

    # Create dummy binary labels
    labels = np.zeros(np_points_with_intensity.shape[0], dtype=np.uint32)
    binary_labels = labels.tobytes()

    return binary_points, binary_labels


def collation_fn_voxelmean(batch):
    """
    :param batch:
    :return:   coords_batch: N x 4 (x,y,z,batch)

    """
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
    # Convert the ROS message to the format expected by the model
    # This involves extracting and processing the point cloud data
   # Get current time to create a unique filename
    timestamp = rospy.Time.now()

    # Convert the PointCloud2 message to PCD and then to numpy array
    pcd, intensities = pointcloud2_to_pcd(ros_msg)
    np_points = np.asarray(pcd.points)

    # Convert to binary format
    binary_points, binary_labels = convert_to_binary_format(np_points, intensities)

    # Process the binary data using SemanticKITTI
    semkitti_dataset = SemanticKITTI(split='val')  # Instantiate your SemanticKITTI class
    processed_data = semkitti_dataset.process_live_data(binary_points, binary_labels)

    # # Print the type and shape of the processed_data
    # print("Type of processed_data:", type(processed_data))
    # if isinstance(processed_data, tuple) or isinstance(processed_data, list):
    #     for i, data in enumerate(processed_data):
    #         print(f"Element {i}: Type - {type(data)}, Shape - {data.shape if hasattr(data, 'shape') else 'N/A'}")

        
    batch_data = [processed_data]
    (coord, xyz, feat, target, offset, inds_reverse) = collation_fn_voxelmean(batch_data)
    # print(len(batch_data),"batch_data length")  
    # print(coord.shape,"coord shape")
    # print(xyz.shape,"xyz shape")
    # print(feat.shape,"feat shape")
    # print(target.shape,"target shape")
    # print(offset.shape,"offset shape")
    # print(inds_reverse.shape,"inds_reverse shape")
    # print(inds_reverse.dtype,"inds_reverse dtype")

    # print(inds_reverse.device, 'inds_reverse device')
    # # print(model.device, 'model device')
    # print(">>>>>>>>>>>>>>>>>>>>>Before inference>>>>>>>>>>>>>>>>> ")

    # inds_reverse = inds_reverse.cuda(non_blocking=True)
    inds_reverse = inds_reverse.to('cuda:0',non_blocking=True)
    # print(inds_reverse.device, 'inds_reverse device')




    offset_ = offset.clone()
    offset_[1:] = offset_[1:] - offset_[:-1]
    batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

    coord = torch.cat([batch.unsqueeze(-1), coord], -1)
    spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)

    coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
    batch = batch.cuda(non_blocking=True)

    batch_size_val = 1
    sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, batch_size_val)

        
    assert batch.shape[0] == feat.shape[0]
    
    with torch.no_grad():
        output = model(sinput, xyz, batch)
        output = output[inds_reverse, :]

                
        output = output.max(1)[1] #output classes of each point [N, 1]
        print(output.shape,"output shape") #[N, 1]
        print(output[:10],"sample output")
        # print(coord.shape,"coord shape") #[N, 4]
        # print(xyz.shape,"xyz shape") #[N, 3]
        # print(feat.shape,"feat shape") #[N, 4]
        # print(batch.shape,"batch shape") #[N, 1]
        # print(target.shape,"target shape") #[N, 1]

    # Return the inference result or perform additional post-processing
    return output



def label_to_color(labels):
    # Define a simple colormap, you can expand this as needed
    # color_map = {
    #     0: [0, 0, 0], 1: [0, 0, 255], 10: [245, 150, 100], 11: [245, 230, 100], 13: [250, 80, 100], 15: [150, 60, 30], 
    #     16: [255, 0, 0], 18: [180, 30, 80], 20: [255, 0, 0], 30: [30, 30, 255], 31: [200, 40, 255], 8: [90, 30, 150], 
    #     40: [255, 0, 255], 44: [255, 150, 255], 48: [75, 0, 75], 12: [75, 0, 175], 50: [0, 200, 255], 14: [50, 120, 255],
    #     52: [0, 150, 255], 60: [170, 255, 150], 70: [0, 175, 0], 71: [0, 60, 135], 72: [80, 240, 150], 80: [150, 240, 255],
    #     81: [0, 0, 255], 99: [255, 255, 50], 252: [245, 150, 100], 256: [255, 0, 0], 253: [200, 40, 255], 254: [30, 30, 255],
    #     255: [90, 30, 150], 257: [250, 80, 100], 258: [180, 30, 80], 259: [255, 0, 0]
    # }
    color_map = {
        0: [0, 0, 0], 1: [0, 0, 0], 10: [0, 0, 0], 11: [0, 0, 0], 13: [0, 0, 0], 15: [0, 0, 0], 
        16: [0, 0, 0], 18: [0, 0, 0], 20: [0, 0, 0], 30: [0, 0, 0], 31: [0, 0, 0], 8: [255, 0, 0], 
        40: [0, 0, 0], 44: [0, 0, 0], 48: [0, 0, 0], 12: [0, 0, 0], 50: [0, 0, 0], 14: [0, 0, 0],
        52: [0, 0, 0], 60: [0, 0, 0], 70: [0, 0, 0], 71: [0, 0, 0], 72: [0, 0, 0], 80: [0, 0, 0],
        81: [0, 0, 0], 99: [0, 0, 0], 252: [0, 0, 0], 256: [0, 0, 0], 253: [0, 0, 0], 254: [0, 0, 0],
        255: [0, 0, 0], 257: [0, 0, 0], 258: [0, 0, 0], 259: [0, 0, 0]
    }
   # Default color for unknown labels - color of label 0
    default_color = color_map[0]  

    colors = np.array([color_map.get(label, default_color) for label in labels])
    return colors


def create_colored_pointcloud2(original_cloud, colors, header):
    """
    Create a PointCloud2 message with colored points.
    Args:
    - original_cloud: The original PointCloud2 message.
    - colors: An Nx3 array of RGB colors for N points.
    - header: Header for the new PointCloud2 message.
    Returns:
    - colored_cloud: A new PointCloud2 message with colored points.
    """
    assert len(colors) == len(list(pc2.read_points(original_cloud))), "Number of colors must match number of points"

    # Define new fields for the colored point cloud
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1)
    ]

    new_points = []
    for point, color in zip(pc2.read_points(original_cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True), colors):
        rgb_packed = struct.pack('BBBB', color[2], color[1], color[0], 255)  # Colors in BGR order + alpha
        rgb_float = struct.unpack('f', rgb_packed)[0]
        new_points.append([point[0], point[1], point[2], point[3], rgb_float])

    colored_cloud = pc2.create_cloud(header, fields, new_points)
    return colored_cloud




if __name__ == '__main__':

    main()