# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

# WandB configuration
import wandb
import random


def get_args_parser():
    """
    Creates an argument parser for the transformer detector.

    Returns:
        argparse.ArgumentParser: The argument parser with the following arguments:
            --lr (float): Learning rate. Default is 1e-4.
            --lr_backbone (float): Learning rate for the backbone. Default is 1e-5.
            --batch_size (int): Batch size. Default is 2.
            --weight_decay (float): Weight decay. Default is 1e-4.
            --epochs (int): Number of epochs. Default is 300.
            --lr_drop (int): Epoch to drop learning rate. Default is 200.
            --clip_max_norm (float): Gradient clipping max norm. Default is 0.1.
            --frozen_weights (str): Path to the pretrained model. If set, only the mask head will be trained. Default is None.
            --backbone (str): Name of the convolutional backbone to use. Default is 'resnet50'.
            --dilation (bool): If true, replaces stride with dilation in the last convolutional block (DC5).
            --position_embedding (str): Type of positional embedding to use on top of the image features. Choices are 'sine' and 'learned'. Default is 'sine'.
            --enc_layers (int): Number of encoding layers in the transformer. Default is 6.
            --dec_layers (int): Number of decoding layers in the transformer. Default is 6.
            --dim_feedforward (int): Intermediate size of the feedforward layers in the transformer blocks. Default is 2048.
            --hidden_dim (int): Size of the embeddings (dimension of the transformer). Default is 256.
            --dropout (float): Dropout applied in the transformer. Default is 0.1.
            --nheads (int): Number of attention heads inside the transformer's attentions. Default is 8.
            --num_queries (int): Number of query slots. Default is 100.
            --pre_norm (bool): If true, applies pre-normalization in the transformer.
            --masks (bool): If true, trains the segmentation head.
            --no_aux_loss (bool): If true, disables auxiliary decoding losses (loss at each layer).
            --set_cost_class (float): Class coefficient in the matching cost. Default is 1.
            --set_cost_bbox (float): L1 box coefficient in the matching cost. Default is 5.
            --set_cost_giou (float): GIoU box coefficient in the matching cost. Default is 2.
            --mask_loss_coef (float): Coefficient for the mask loss. Default is 1.
            --dice_loss_coef (float): Coefficient for the dice loss. Default is 1.
            --bbox_loss_coef (float): Coefficient for the bounding box loss. Default is 5.
            --giou_loss_coef (float): Coefficient for the GIoU loss. Default is 2.
            --eos_coef (float): Relative classification weight of the no-object class. Default is 0.1.
            --dataset_file (str): Name of the dataset file. Default is 'coco'.
            --coco_path (str): Path to the COCO dataset.
            --coco_panoptic_path (str): Path to the COCO panoptic dataset.
            --remove_difficult (bool): If true, removes difficult examples from the dataset.
            --output_dir (str): Path where to save the output. Default is '' (no saving).
            --device (str): Device to use for training/testing. Default is 'cuda'.
            --seed (int): Random seed. Default is 42.
            --resume (str): Path to resume from checkpoint. Default is ''.
            --start_epoch (int): Epoch to start training from. Default is 0.
            --eval (bool): If true, evaluates the model.
            --num_workers (int): Number of data loading workers. Default is 2.
            --world_size (int): Number of distributed processes. Default is 1.
            --dist_url (str): URL used to set up distributed training. Default is 'env://'.
    """
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


# WandB configuration
# Initialize the parser
parser = get_args_parser()
args = parser.parse_args()
# Log the parameters to WandB
# # Start a new WandB run and log all args parameters
# wandb.init(
#     project="DETR",
#     config=vars(args)  # Convert args Namespace to dictionary
# )
# Start a new WandB run and log the parameters with specific names
wandb.init(
    project="DETR",
    config={
        "learning_rate": args.lr,
        "learning_rate_backbone": args.lr_backbone,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "lr_drop": args.lr_drop,
        "clip_max_norm": args.clip_max_norm,
        "frozen_weights": args.frozen_weights,
        "backbone": args.backbone,
        "dilation": args.dilation,
        "position_embedding": args.position_embedding,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "dim_feedforward": args.dim_feedforward,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "nheads": args.nheads,
        "num_queries": args.num_queries,
        "pre_norm": args.pre_norm, #check here
        "masks": args.masks,
        "aux_loss": args.aux_loss, #check here
        "set_cost_class": args.set_cost_class,
        "set_cost_bbox": args.set_cost_bbox,
        "set_cost_giou": args.set_cost_giou,
        "mask_loss_coef": args.mask_loss_coef,
        "dice_loss_coef": args.dice_loss_coef,
        "bbox_loss_coef": args.bbox_loss_coef,
        "giou_loss_coef": args.giou_loss_coef,
        "eos_coef": args.eos_coef,
        "dataset_file": args.dataset_file,
        "coco_path": args.coco_path,
        "coco_panoptic_path": args.coco_panoptic_path,
        "remove_difficult": args.remove_difficult,
        "output_dir": args.output_dir,
        "device": args.device,
        "seed": args.seed,
        "resume": args.resume,
        "start_epoch": args.start_epoch,
        "eval": args.eval,
        "num_workers": args.num_workers,
        "world_size": args.world_size,
        "dist_url": args.dist_url
    }
)


def main(args):
    """
    Main function to train and evaluate the model.

    Args:
        args (Namespace): Arguments for training and evaluation.

    The function performs the following steps:
    1. Initializes distributed mode if specified.
    2. Sets the random seed for reproducibility.
    3. Builds the model, criterion, and postprocessors.
    4. Moves the model to the specified device.
    5. Wraps the model with DistributedDataParallel if in distributed mode.
    6. Sets up the optimizer and learning rate scheduler.
    7. Builds the training and validation datasets and data loaders.
    8. Loads frozen weights if specified.
    9. Resumes from a checkpoint if specified.
    10. Evaluates the model if in evaluation mode.
    11. Trains the model for the specified number of epochs.
    12. Saves checkpoints and evaluation results periodically.
    13. Logs training and evaluation statistics.
    14. Prints the total training time.
    """
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, # All key-value pairs in train_stats are added to log_stats with the key prefixed by 'train_'
                     **{f'test_{k}': v for k, v in test_stats.items()}, # All key-value pairs in test_stats are added to log_stats with the key prefixed by 'test_'
                                                                        # ** is used to unpack the dictionary into key-value pairs
                     'epoch': epoch, # Add the epoch number to log_stats
                     'n_parameters': n_parameters} # Add the number of parameters to log_stats
        # Log metrics to WandB
        wandb.log(log_stats)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
