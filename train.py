import argparse
import os
from pyexpat import model


models = ['vit_tiny','vit_small','vit_base','xcit','deit_tiny','deit_small']

parser = argparse.ArgumentParser('DINO', add_help=False)
parser.add_argument('--model',default ='vit_small',type=str,choices= models,help="Name of the model. Default vit_small")
parser.add_argument('--patch_size',default=16,type=int,help="VIT square patch sizes. Default 16")
parser.add_argument('--out_dim',default=65536,type=int,help="Dimensionality of the DINO head output, bigger valus for complex tax. Default 65536")
parser.add_argument('--norm_last_layer',default=True,type=bool,help="Whether or not normilize the last layer of DINO head. Increases Training stability. Default True")
parser.add_argument('--momentum_teacher',default=0.996,type=float,help="Base EMA paramater used for teacher parameter uptade. The value increased to 1 during training with Cosine Schedule. Bigger value for smaller batch-size. Default 0.996")
parser.add_argument('--use_bn_in_head',default=False,type=bool,help="Wheter or not use batch normilization in projection head. Default False")


# Temperature teacher parameters
parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
    help="""Initial value for the teacher temperature: 0.04 works well in most cases.
    Try decreasing it if the training loss does not decrease.""")
parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
    of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
    starting with the default value of 0.04 and increase this slightly if needed.""")
parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
    help='Number of warmup epochs for the teacher temperature (Default: 30).')

# Training/Optimization parameters
parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
    to use half precision for training. Improves training time and memory requirements,
    but can provoke instability and slight decay of performance. We recommend disabling
    mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
    weight decay. With ViT, a smaller value at the beginning of training works well.""")
parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
    weight decay. We use a cosine schedule for WD and using a larger decay by
    the end of training improves performance for ViTs.""")
parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
    gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
    help optimization for larger ViT architectures. 0 for disabling.""")
parser.add_argument('--batch_size_per_gpu', default=64, type=int,
    help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
    during which we keep the output layer fixed. Typically doing so during
    the first epoch helps training. Try increasing this value if the loss does not decrease.""")
parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
    linear warmup (highest LR used during training). The learning rate is linearly scaled
    with the batch size, and specified here for a reference batch size of 256.""")
parser.add_argument("--warmup_epochs", default=10, type=int,
    help="Number of epochs for the linear learning-rate warm up.")
parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
    end of optimization. We use a cosine LR schedule with linear warmup.""")
parser.add_argument('--optimizer', default='adamw', type=str,
    choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

# Multi-crop parameters
parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
    help="""Scale range of the cropped image before resizing, relatively to the origin image.
    Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
    recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
    local views to generate. Set this parameter to 0 to disable multi-crop training.
    When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
    help="""Scale range of the cropped image before resizing, relatively to the origin image.
    Used for small local view cropping of multi-crop.""")

# Misc
parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
    help='Please specify path to the ImageNet training data.')
parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
parser.add_argument('--seed', default=0, type=int, help='Random seed.')
parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    distributed training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")





def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
