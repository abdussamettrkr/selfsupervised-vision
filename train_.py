from pickletools import optimize
from turtle import st
import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torchvision import models as torchvision_models
from torchvision.datasets import ImageFolder


import models.vit.vit as vits
from models.vit.vit import DINOHead
from losses.loss import DINOLoss
from dataset.imageDataset import ImageDataset
from dataset.augmentation import *


from pathlib import Path
import numpy as np
from tqdm import tqdm
import time, math, sys, json, argparse, os, datetime, logging, yaml
from utils import Logger
import utils 




parser = argparse.ArgumentParser('DINO', add_help=False)
parser.add_argument("--exp-name", required=True, type=str, help="Experiment name")
parser.add_argument("--data-path", required=True, type=str, help="Root path of image datasets")
parser.add_argument("--epochs",required=True,type=int)
parser.add_argument("--batch-size",required=True,type=int)
parser.add_argument("--num-workers",default=4   ,type=int)
parser.add_argument("--work-path",default = os.path.dirname(os.path.abspath(__file__)),type=str)
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
parser.add_argument("--save-freq", default=5, type=int)
parser.add_argument("--config-file",required=True,type=str)

args = parser.parse_args()


#Prepare logging
experiment_dir_path = os.path.join(args.work_path,"experiments",args.exp_name)
if not os.path.exists(experiment_dir_path):
    os.mkdir(experiment_dir_path)
logger = Logger(
    log_file_name= os.path.join(experiment_dir_path,"log.txt"),
    log_level=logging.DEBUG,
    logger_name="DINO",
).get_log()


def train(train_loader, nets: list, criterion, 
        optimizer, epoch,schedulers: list,
        model_config,fp16_scaler =None):
    
    student,teacher = nets
    lr_sch,wd_sch,momentum_sch = schedulers

    start = time.time()
    student.train()


    
    end = time.time()

    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1,args.epochs))
    for it, (images) in enumerate(pbar :=tqdm(train_loader)):
        data_time = time.time() - end

        it = len(train_loader)*epoch + it

        images = [img.cuda() for img in images]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            model_time = time.time()            
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher                        
            student_output = student(images)            
            model_time = time.time() - model_time
            loss = criterion(student_output, teacher_output, epoch)

        optimizer.zero_grad()

        if fp16_scaler is None:
            loss.backward()
            if model_config['clip_grad']:
                param_norms = utils.clip_gradients(student, model_config.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                    args.freeze_last_layer)

            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if model_config['clip_grad']:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, model_config['clip_grad'])
            utils.cancel_gradients_last_layer(epoch, student,
                                              model_config['freeze_last_layer'])
            fp16_scaler.step(optimizer)
            fp16_scaler.update()


        ##Update teacher weights with EMA
        with torch.no_grad():
            m = momentum_sch[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        pbar.set_postfix({'loss':loss.item(),'model':model_time,'data':data_time})
        end = time.time() 



    





def main():    
    with open(args.config_file) as stream:
        model_config = yaml.safe_load(stream)

    utils.set_random_seeds(41)
    

    transform = DataAugmentationDINO(
        global_crops_scale=model_config['global_crops_scale'],
        local_crops_number=model_config['local_crops_number'],
        local_crops_scale=model_config['local_crops_scale'])
    

    dataset = ImageDataset(args.data_path,transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,     
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f"== Data loaded: there are {len(dataset)} training images.")

    student,teacher = get_models(model_config)

    
    dino_loss = DINOLoss(
        model_config['out_dim'],
        model_config['local_crops_number'] + 2,  # total number of crops = 2 global crops + local_crops_number
        model_config['warmup_teacher_temp'],
        model_config['teacher_temp'],
        model_config['warmup_teacher_temp_epochs'],
        model_config['epochs'],
    ).cuda()

    params_groups = utils.get_params_groups(student)
    if model_config['optimizer'] == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif model_config['optimizer'] == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif model_config['optimizer'] == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches


    fp16_scaler = None
    if model_config['use_fp16']:        
        fp16_scaler = torch.cuda.amp.GradScaler()

    
        # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        model_config['lr'] * args.batch_size / 256.,  # linear scaling rule
        model_config['min_lr'],
        model_config['epochs'], len(dataset),
        warmup_epochs=model_config['warmup_epochs'],
    )
    wd_schedule = utils.cosine_scheduler(
        model_config['weight_decay'],
        model_config['weight_decay_end'],
        model_config['epochs'], len(dataset),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(model_config['momentum_teacher'], 1,
                                               args.epochs, len(data_loader))
    logger.info(f"== Loss, optimizer and schedulers ready.")

    # resume from a checkpoint
    last_epoch = -1
    best_loss = 0
    if args.resume:
        ckpt_file = os.path.join('experiments',args['exp_name'],'last.pth')
        if os.path.exists(ckpt_file):
            best_loss, last_epoch = utils.load_checkpoint(ckpt_file,student,
                                            teacher,optimizer=optimizer)


    logger.info("            =======  Training  =======\n")
    for epoch in range(last_epoch + 1, args.epochs):
        train(data_loader,[student,teacher],
            dino_loss,optimizer,epoch,
            [lr_schedule,wd_schedule,momentum_schedule],model_config,fp16_scaler)


def get_models(config):
    student = None
    embed_dim = None
    #utils.init_distributed_mode(args)
    if config['arch'] in vits.__dict__.keys():

        student = vits.__dict__[config['arch']](
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path_rate'],  # stochastic depth
        )
        teacher = vits.__dict__[config['arch']](patch_size=config['patch_size'])
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif config['arch'] in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', config['arch'],
                                 pretrained=False, drop_path_rate=config['drop_path_rate'])
        teacher = torch.hub.load('facebookresearch/xcit:main', config['arch'], pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif config['arch'] in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[config['arch']]()
        teacher = torchvision_models.__dict__[config['arch']]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {config['arch']}")


    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        config['out_dim'],
        use_bn=config['use_bn_in_head'],
        norm_last_layer=config['norm_last_layer'],
    ))

    teacher = utils.MultiCropWrapper(teacher, DINOHead(
        embed_dim, 
        config['out_dim'],
        use_bn=config['use_bn_in_head']),
    )    
    student, teacher = student.cuda(), teacher.cuda()


    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {config['arch']} network.")

    return student,teacher


if __name__ == "__main__":
    main()