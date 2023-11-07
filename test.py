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


from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collation_fn_voxelmean_tta ,collate_fn_limit, collation_fn_voxelmean
from util.logger import get_logger
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, Constant
from util.nuscenes import nuScenes
from util.semantic_kitti import SemanticKITTI
from util.waymo import Waymo

import open3d as o3d
import open3d.ml.tf as ml3d
import open3d.ml as _ml3d


from open3d.ml.vis import Visualizer, LabelLUT
from open3d.ml.datasets import SemanticKITTI as SemanticKITTI_dataset
from open3d.ml.tf.pipelines import SemanticSegmentation


from functools import partial
import pickle
import yaml
from torch_scatter import scatter_mean
import spconv.pytorch as spconv

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

    
    
    if args.data_name == 'semantic_kitti':
        train_data = SemanticKITTI(args.data_root, 
            voxel_size=args.voxel_size, 
            split='train', 
            return_ref=True, 
            label_mapping=args.label_mapping, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            scale_params=[0.95,1.05], 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            elastic_aug=False, 
            elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
            ignore_label=args.ignore_label, 
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )

    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    collate_fn = partial(collate_fn_limit, max_batch_points=args.max_batch_points, logger=logger if main_process() else None)
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=args.workers,
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=True, 
        collate_fn=collate_fn
    )

    val_transform = None
    args.use_tta = getattr(args, "use_tta", False)
    
    if args.data_name == 'semantic_kitti':
        val_data = SemanticKITTI(data_path=args.data_root, 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )

    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("val_data samples: '{}'".format(len(val_data)))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    else:
        val_sampler = None
        
    if getattr(args, "use_tta", False):
        val_loader = torch.utils.data.DataLoader(val_data, 
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers, 
            pin_memory=True, 
            sampler=val_sampler, 
            collate_fn=collation_fn_voxelmean_tta
        )
    else:
        val_loader = torch.utils.data.DataLoader(val_data, 
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers,
            pin_memory=True, 
            sampler=val_sampler, 
            collate_fn=collation_fn_voxelmean
        )

    # set scheduler
    if args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if args.val:
        if args.use_tta:
            validate_tta(val_loader, model, criterion)
        else:
            # validate(val_loader, model, criterion)
            validate_distance(val_loader, model, criterion)
        exit()



def focal_loss(output, target, class_weight, ignore_label, gamma, need_softmax=True, eps=1e-8):
    mask = (target != ignore_label)
    output_valid = output[mask]
    if need_softmax:
        output_valid = F.softmax(output_valid, -1)
    target_valid = target[mask]
    p_t = output_valid[torch.arange(output_valid.shape[0], device=target_valid.device), target_valid] #[N, ]
    class_weight_per_sample = class_weight[target_valid]
    focal_weight_per_sample = (1.0 - p_t) ** gamma
    loss = -(class_weight_per_sample * focal_weight_per_sample * torch.log(p_t + eps)).sum() / (class_weight_per_sample.sum() + eps)
    return loss



def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluationn 1 >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    loss_name = args.loss_name

    model.eval()
    end = time.time()
    for i, batch_data in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        (coord, xyz, feat, target, offset, inds_reconstruct) = batch_data
        inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]
        
        with torch.no_grad():
            output = model(sinput, xyz, batch)
            output = output[inds_reconstruct, :]
        
            # if loss_name == 'focal_loss':
            #     loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            # elif loss_name == 'ce_loss':
            #     loss = criterion(output, target)
            # else:
            #     raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc

def validate_tta(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation 2 >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    loss_name = args.loss_name

    model.eval()
    end = time.time()
    import tqdm
    for i, batch_data_list in enumerate(val_loader):

        data_time.update(time.time() - end)
        #print(len(batch_data_list),"len(batch_data_list)") #1
        with torch.no_grad():
            output = 0.0
            for batch_data in batch_data_list:

                (coord, xyz, feat, target, offset, inds_reconstruct) = batch_data
                inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)
                
                #print(pts.shape,"pts shape")
                offset_ = offset.clone()
                offset_[1:] = offset_[1:] - offset_[:-1]
                batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

                coord = torch.cat([batch.unsqueeze(-1), coord], -1)
                spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
            
                coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
                batch = batch.cuda(non_blocking=True)

                sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size)

                assert batch.shape[0] == feat.shape[0]
                
                output_i = model(sinput, xyz, batch)
                output_i = F.softmax(output_i[inds_reconstruct, :], -1)
                output = output + output_i
            
            output = output / len(batch_data_list)
            
            # if loss_name == 'focal_loss':
            #     loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            # elif loss_name == 'ce_loss':
            #     loss = criterion(output, target)
            # else:
            #     raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]

        print(output.shape,"output shape") #[N, 1]
        print(coord.shape,"coord shape") #[N, 4]
        print(xyz.shape,"xyz shape") #[N, 3]
        print(feat.shape,"feat shape") #[N, 4]
        print(batch.shape,"batch shape") #[N, 1]
        print(batch_data[0].shape,"batch_data[0] shape") #[N, 3]
        print(target.shape,"target shape") #[N, 1]
        print(output)
        #print maximum value of output
        print(torch.max(output))
        print(torch.min(output))
        print(target)
        pts=xyz[inds_reconstruct]
        print(pts.shape,"pts shape")
        print(output.shape,"output shape......")

        Save=True
        if Save:
            pathSave="/media/avresearch/DATA/SphereFormer/segmented/segmented"+str(i)+".label"
            print(pathSave)
            outputSave = output.cpu().numpy()
            outputSave = outputSave.reshape(-1)
            outputSave = outputSave.astype(np.int32)
            outputSave.tofile(pathSave)
            print("######saved#######")

        
        
        kitti_labels=SemanticKITTI_dataset.get_label_to_names()
        v = Visualizer()
        lut = LabelLUT()
   
        #lut.add_label("unlabeled", 0)
        lut.add_label("car",0)
        lut.add_label("bicycle", 1)
        lut.add_label("bus", 4)
        lut.add_label("motorcycle", 2)
        lut.add_label("truck", 3)
        lut.add_label("person", 5)
        lut.add_label("bicyclist", 6)
        lut.add_label("motorcyclist", 7)
        lut.add_label("road", 8) #lane marking
        lut.add_label("parking", 9)
        lut.add_label("sidewalk", 10)
        lut.add_label("other-ground", 11)
        lut.add_label("building", 12)
        lut.add_label("fence", 13)
        lut.add_label("vegetation", 14)
        lut.add_label("trunk", 15)
        lut.add_label("terrain", 16)
        lut.add_label("pole", 17)
        lut.add_label("traffic-sign", 18)
        '''lut.add_label("other-object", 99)
        lut.add_label("moving-car", 252)
        lut.add_label("moving-bicyclist", 253)
        lut.add_label("moving-person", 254)
        lut.add_label("moving-motorcyclist", 255)
        lut.add_label("moving-on-rails", 256)
        lut.add_label("moving-bus", 257)
        lut.add_label("moving-truck", 258)
        lut.add_label("moving-other-vehicle", 259)
        '''

        

        Visualize=True
        if Visualize:
            dataset=ml3d.datasets.SemanticKITTI(dataset_path='/media/avresearch/DATA/SphereFormer/SemanticKITTI')
            data=[
                {
                "name": "SemanticKITTI",
                "points": pts.cpu().numpy(),
                "labels": target.cpu().numpy(),#this is the ground truth
                "pred": output.cpu().numpy()#this is the prediction

            }
            ]
            
            v.set_lut("labels", lut)
            v.set_lut("pred", lut)
            v.visualize(data)
            #show predictions live and go to next frame within 1 second
            # pipeline=SemanticSegmentation.run(data)




        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation 2 <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate_distance(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation 3 >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # For validation on points with different distance
    intersection_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    union_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    target_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]

    torch.cuda.empty_cache()

    loss_name = args.loss_name

    model.eval()
    end = time.time()
    for i, batch_data in enumerate(val_loader):
        print(i,"i")
        data_time.update(time.time() - end)
    
        (coord, xyz, feat, target, offset, inds_reverse) = batch_data
        inds_reverse = inds_reverse.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
        #print(type(sinput),"sinput shape") #[N, 4] #<class 'spconv.pytorch.core.SparseConvTensor'>
        identity=sinput.features #<class 'torch.Tensor'>
        print((identity.shape),"identity shape") #[N, 4] #<class 'spconv.pytorch.core.SparseConvTensor'>
        
        
        
        assert batch.shape[0] == feat.shape[0]
        
        with torch.no_grad():
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
        
            # if loss_name == 'focal_loss':
            #     loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            # elif loss_name == 'ce_loss':
            #     loss = criterion(output, target)
            # else:
            #     raise ValueError("such loss {} not implemented".format(loss_name))

        #print(output.shape,"output shape") #[N, C=19 in semanticKITTI]
        output = output.max(1)[1] #output classes of each point [N, 1]
        #print(type(output)) #<class 'torch.Tensor'>
        #print coordinate of each point
        print(output.shape,"output shape") #[N, 1]
        print(coord.shape,"coord shape") #[N, 4]
        print(xyz.shape,"xyz shape") #[N, 3]
        print(feat.shape,"feat shape") #[N, 4]
        print(batch.shape,"batch shape") #[N, 1]
        print(batch_data[0].shape,"batch_data[0] shape") #[N, 3]
        print(target.shape,"target shape") #[N, 1]
        #save output as .label file as semanticKITTI format
        Save=True
        if Save:
            pathSave="/media/avresearch/DATA/SphereFormer/segmented/segmented"+str(i)+".label"
            print(pathSave)
            num=str(i)
            os.path.join(pathSave, num)
            outputSave = output.cpu().numpy()
            outputSave = outputSave.reshape(-1)
            outputSave = outputSave.astype(np.int32)
            outputSave.tofile(pathSave)
            print("######saved#######")


        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        r = torch.sqrt(feat[:, 0] ** 2 + feat[:, 1] ** 2 + feat[:, 2] ** 2)
        print(r.shape,"r shape") #[N, 1]
        r = r[inds_reverse]
        print(r.shape,"r shape") #[N, 1]
        # For validation on points with different distance
        masks = [r <= 20, (r > 20) & (r <= 50), r > 50]
        #print(masks[0].shape,"masks[0] shape") #[N, 1]

        for ii, mask in enumerate(masks):
            #print(output.shape,"output[mask] shape") #[N, 1]
            intersection, union, tgt = intersectionAndUnionGPU(output[mask], target[mask], args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(tgt)
            intersection, union, tgt = intersection.cpu().numpy(), union.cpu().numpy(), tgt.cpu().numpy()
            intersection_meter_list[ii].update(intersection), union_meter_list[ii].update(union), target_meter_list[ii].update(tgt)

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        print("----------------------------------------")

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    iou_class_list = [intersection_meter_list[i].sum / (union_meter_list[i].sum + 1e-10) for i in range(3)]
    accuracy_class_list = [intersection_meter_list[i].sum / (target_meter_list[i].sum + 1e-10) for i in range(3)]
    mIoU_list = [np.mean(iou_class_list[i]) for i in range(3)]
    mAcc_list = [np.mean(accuracy_class_list[i]) for i in range(3)]
    allAcc_list = [sum(intersection_meter_list[i].sum) / (sum(target_meter_list[i].sum) + 1e-10) for i in range(3)]

    if main_process():

        metrics = ['close', 'medium', 'distant']
        for ii in range(3):
            logger.info('Val result_{}: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(metrics[ii], mIoU_list[ii], mAcc_list[ii], allAcc_list[ii]))
            for i in range(args.classes):
                logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class_list[ii][i], accuracy_class_list[ii][i]))

        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation 3<<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()