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
parser.add_argument("--batch-size",required=True,tpye=int)
parser.add_argument("--work-path",default = os.path.dirname(os.path.abspath(__file__)),type=str)
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
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



    train_loss = 0
    correct = 0
    total = 0
    end = time.time()

    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1,args.epochs))
    for it, (images) in enumerate(pbar :=tqdm(train_loader)):
        data_time = end -time.time()

        it = len(train_loader)*epoch + it

        images = [img.cuda() for img in images]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_time = time.time()            
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            teacher_time = time.time() - teacher_time
            student_time = time.time()
            student_output = student(images)
            student_time = time.time() - student_time
            loss = criterion(student_output, teacher_output, epoch)

        optimizer.zero_grad()

        if fp16_scaler is None:
            loss.backward()
            if model_config.clip_grad:
                param_norms = utils.clip_gradients(student, model_config.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                    args.freeze_last_layer)

            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if model_config.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()


        ##Update teacher weights with respect to EMA
        with torch.no_grad():
            m = momentum_sch[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        pbar.set_postfix({'loss':loss.item(),'student':student_time,'teacher':teacher_time,'data':data_time})
        end = time.time() 

    





def main():    

    with open(args.config_file) as stream:
        model_config = yaml.safe_load(stream)


    utils.set_random_seeds(args.seed)

    transform = DataAugmentationDINO(
        global_crops_scale=args.global_crops_scale,
        local_crops_number=args.local_crops_number,
        local_crops_scale=args.local_crops_scale)
    

    dataset = ImageDataset(args.data_path,transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,     
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f"== Data loaded: there are {len(dataset)} training images.")

    student,teacher,teacher_without_ddp = utils.get_models(args)

    
    dino_loss = DINOLoss(
        model_config.out_dim,
        model_config.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        model_config.warmup_teacher_temp,
        model_config.teacher_temp,
        model_config.warmup_teacher_temp_epochs,
        model_config.epochs,
    ).cuda()

    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches


    fp16_scaler = None
    if args.use_fp16:        
        fp16_scaler = torch.cuda.amp.GradScaler()

    
        # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        model_config.lr * args.batch_size / 256.,  # linear scaling rule
        model_config.min_lr,
        model_config.epochs, len(dataset),
        warmup_epochs=model_config.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        model_config.weight_decay,
        model_config.weight_decay_end,
        model_config.epochs, len(dataset),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(model_config.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    logger.info(f"== Loss, optimizer and schedulers ready.")

    # resume from a checkpoint
    last_epoch = -1
    best_loss = 0
    if args.resume:
        ckpt_file = os.path.join('experiments',args.exp_name,'last.pth')
        if os.path.exists(ckpt_file):
            best_loss, last_epoch = utils.load_checkpoint(ckpt_file,student,
                                            teacher,optimizer=optimizer)


    logger.info("            =======  Training  =======\n")
    for epoch in range(last_epoch + 1, args.epochs):
        train(data_loader,[student,teacher],
            dino_loss,optimizer,epoch,
            [lr_schedule,wd_schedule,momentum_schedule],model_config,fp16_scaler)


if __name__ == "__main__":
    main()