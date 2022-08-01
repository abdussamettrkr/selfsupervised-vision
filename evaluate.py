import sklearn
import torch
from torchvision import transforms as T

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import models.vit.vit as vits
import models.resnet.resnet as resnets
from models.base import DINOHead
from dataset.augmentation import *
from dataset.imageDataset import ImageDataset

import utils 

from tqdm import tqdm
import numpy as np
rng = np.random.default_rng(2021)

import argparse, os, yaml, math

parser = argparse.ArgumentParser('DINO evaluation', add_help=True)
parser.add_argument("--exp-name", required=True, type=str, help="Experiment name")
parser.add_argument("--train-path", required=True, type=str, help="Root path of training image datasets")
parser.add_argument("--val-path", required=True, type=str, help="Root path of validation image datasets")
parser.add_argument("--num-workers",default=4,type=int)
parser.add_argument("--work-path",default = os.path.dirname(os.path.abspath(__file__)),type=str)
parser.add_argument("--train-percent",default=1.0,type=float)
parser.add_argument("--model",default="student",choices=['student','teacher'],type=str)
parser.add_argument("--config-file",required=True,type=str)
args = parser.parse_args()

experiment_dir_path = os.path.join(args.work_path,"experiments",args.exp_name)

np.random.seed(42)

def run_knn():
    with open(args.config_file) as stream:
        model_config = yaml.safe_load(stream)

    student,teacher = get_models(model_config)

    # resume from a checkpoint
    last_epoch = -1
    best_loss = math.inf

    ckpt_file = os.path.join(experiment_dir_path,'400.pth.tar')
    if os.path.exists(ckpt_file):
        utils.load_checkpoint(ckpt_file,student,teacher)
    else:
        print("File:{} does not exists. Please check experiment name.".format(ckpt_file))
        exit(1)

    if args.model == 'student':
        model = student
    else:
        model = teacher

    model = model.backbone
    model.eval()

    train_loader = get_dataloader(model_config,args.train_path)
    val_loader = get_dataloader(model_config,args.val_path)
    
    if os.path.exists('./rand.npy'):        
        print('loaded')
        with open('rand.npy', 'rb') as f:
            rand_arr = np.load(f)
    else:
        with open('rand.npy', 'wb') as f:
            rand_arr = rng.random(len(train_loader))
            np.save(f, rand_arr)
    
    
    
    train_outs = []        
    train_y = []

    val_outs = []
    val_y = []

    class_counts = np.zeros(10)
    
    print(" === Evaluating === ")
    
    for it, (images,labels) in enumerate(pbar :=tqdm(train_loader)):        
        if rand_arr[it] > args.train_percent:
            continue

        images = torch.stack([img.cuda() for img in images])        

        out = model(images).squeeze().detach().cpu().numpy()
        labels = labels.squeeze().detach().cpu().numpy()

        train_outs.append(out)        
        train_y.append(labels)
        class_counts[labels] += 1




        
    for it, (images,labels) in enumerate(pbar :=tqdm(val_loader)):
        images = torch.stack([img.cuda() for img in images])

        out = model(images).squeeze().detach().cpu().numpy()
        labels = labels.squeeze().detach().cpu().numpy()

        val_outs.append(out)
        val_y.append(labels)


    estimator = KNeighborsClassifier()
    estimator.fit(train_outs,train_y)

    pred_val = estimator.predict(val_outs)    
    acc = accuracy_score(val_y, pred_val)

    print("Accuracy {}".format(acc))
    
    return acc



        


def get_dataloader(model_config,data_path):

    transform = T.Compose([
        T.Resize(256, interpolation=3),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    

    dataset = ImageDataset(data_path,transform=transform,returnLabels=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,             
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return data_loader



def get_models(config):
    student = None
    embed_dim = None    
    if config['arch'] in vits.__dict__.keys():
        student = vits.__dict__[config['arch']](
            patch_size=config['patch_size'],
            drop_path_rate=config['drop_path_rate'],  # stochastic depth
        )
        teacher = vits.__dict__[config['arch']](patch_size=config['patch_size'])
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif config['arch'] in resnets.__dict__.keys():
        student = resnets.__dict__[config['arch']]()
        teacher = resnets.__dict__[config['arch']]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {config['arch']}")

    print('embed_dim',embed_dim)

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

    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {config['arch']} network.")

    return student,teacher





if __name__ == '__main__':
    run_knn()




