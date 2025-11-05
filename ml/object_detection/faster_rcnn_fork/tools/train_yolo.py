import torch
import argparse
import os
import numpy as np
import yaml
import random
from yolo.model import YOLO
from yolo.loss import yolo_v1_loss
from tqdm import tqdm
from dataset.voc_yolo import VocYoloDatasetAdapter
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    voc = VocYoloDatasetAdapter('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'],
                     grid_size=model_config["grid_size"],
                     im_size=model_config["im_size"])
    train_dataset = DataLoader(voc,
                               batch_size=1,
                               shuffle=True,
                               num_workers=4)
    
    yolo = YOLO(
        num_classes=voc.num_classes, 
        num_anchors=model_config["num_anchors"],
        grid_size=model_config["grid_size"]
    )
    yolo.train()
    yolo.to(device)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=filter(lambda p: p.requires_grad,
                                              yolo.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1

    for i in range(num_epochs):
        yolo_losses = []
        optimizer.zero_grad()
        
        for im, target, fname, yolo_targets in tqdm(train_dataset):
            im = im.float().to(device)
            yolo_targets = yolo_targets.float().to(device)
            predictions = yolo(im)
            
            loss = yolo_v1_loss(targets=yolo_targets, predictions=predictions)
            
            yolo_losses.append(loss)
            loss = loss / acc_steps
            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1
        print('Finished epoch {}'.format(i))
        optimizer.step()
        optimizer.zero_grad()
        torch.save(yolo.state_dict(), os.path.join(train_config['task_name'],
                                                                train_config['ckpt_name']))
        loss_output = ''
        loss_output += 'yolo loss : {:.4f}'.format(np.mean(yolo_losses))
        print(loss_output)
        scheduler.step()
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn training')
    parser.add_argument('--config', dest='config_path',
                        default='config/yolo.yaml', type=str)
    args = parser.parse_args()
    train(args)
