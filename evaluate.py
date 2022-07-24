import sklearn
import torch
from torchvision import transforms as T

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import models.vit.vit as vits
import models.resnet.resnet as resnets
from models.base import DINOHead
from dataset.augmentation import *
from dataset.imageDataset import ImageDataset

import utils 

from tqdm import tqdm

import argparse, os, yaml, math

parser = argparse.ArgumentParser('DINO evaluation', add_help=True)
parser.add_argument("--exp-name", required=True, type=str, help="Experiment name")
parser.add_argument("--train-path", required=True, type=str, help="Root path of training image datasets")
parser.add_argument("--val-path", required=True, type=str, help="Root path of validation image datasets")
parser.add_argument("--num-workers",default=4,type=int)
parser.add_argument("--work-path",default = os.path.dirname(os.path.abspath(__file__)),type=str)
parser.add_argument("--config-file",required=True,type=str)
args = parser.parse_args()

experiment_dir_path = os.path.join(args.work_path,"experiments",args.exp_name)

def run_knn():
    with open(args.config_file) as stream:
        model_config = yaml.safe_load(stream)

    student,teacher = get_models(model_config)

    # resume from a checkpoint
    last_epoch = -1
    best_loss = math.inf

    ckpt_file = os.path.join(experiment_dir_path,'best.pth.tar')
    if os.path.exists(ckpt_file):
        utils.load_checkpoint(ckpt_file,student,teacher)
    else:
        print("File:{} does not exists. Please check experiment name.".format(ckpt_file))
        exit(1)

    student = student.backbone
    student.eval()
    teacher.eval()    

    train_loader = get_dataloader(model_config,args.train_path)
    val_loader = get_dataloader(model_config,args.val_path)
        
    student_train_outs = []    
    teacher_train_outs = []
    train_y = []

    student_val_outs = []
    teacher_val_outs = []
    val_y = []

    
    print(" === Evaluating === ")
    for it, (images,labels) in enumerate(pbar :=tqdm(train_loader)):
        images = torch.stack([img.cuda() for img in images])        

        s_out = student(images).squeeze().detach().cpu().numpy()
        t_out = teacher(images).squeeze().detach().cpu().numpy()
        labels = labels.squeeze().detach().cpu().numpy()

        student_train_outs.append(s_out)        
        teacher_train_outs.append(t_out)
        train_y.append(labels)



        
    for it, (images,labels) in enumerate(pbar :=tqdm(val_loader)):
        images = torch.stack([img.cuda() for img in images])

        s_out = student(images).squeeze().detach().cpu().numpy()
        t_out = teacher(images).squeeze().detach().cpu().numpy()
        labels = labels.squeeze().detach().cpu().numpy()

        student_val_outs.append(s_out)
        teacher_val_outs.append(t_out)

        val_y.append(labels)


    estimator_student = KNeighborsClassifier()
    estimator_teacher = KNeighborsClassifier()
    estimator_student.fit(student_train_outs,train_y)
    estimator_teacher.fit(teacher_train_outs,train_y)

    pred_val = estimator_student.predict(student_val_outs)    
    student_acc = accuracy_score(val_y, pred_val)
    pred_val = estimator_teacher.predict(teacher_val_outs)
    teacher_acc = accuracy_score(val_y, pred_val)

    print("Student accuracy {}, Teacher accuracy {}".format(student_acc,teacher_acc))
    
    return student_acc,teacher_acc



        


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
        shuffle=True,
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




