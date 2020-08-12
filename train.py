import wandb
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from data import get_loader
from utils import clip_gradient, adjust_lr


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=10, help='epoch number')
#parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
#parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--model_id', type=str, required=True, help='required unique id for trained model name')
parser.add_argument('--resume', type=str, default='', help='path to resume model training from checkpoint')
parser.add_argument('--wandb', type=bool, default=False, help='enable wandb tracking model training')
parser.add_argument('--wandb_resume', type=str, default='', help='enable wandb resume previus training')
opt = parser.parse_args()

model_id = opt.model_id
resume_model_path = opt.resume
resume_wandb_model = opt.wandb_resume
WANDB_EN = opt.wandb
if WANDB_EN:
	run = wandb.init(entity="albytree", project="cpd-train", resume=True)
	run.save()
		
# Add all parsed config in one line
if WANDB_EN:
	if not resume_wandb_model:
		wandb.config.update(opt)
tot_epochs = opt.epoch
print("Training Info")
print("EPOCHS: {}".format(opt.epoch))
print("LEARNING RATE: {}".format(opt.lr))
print("BATCH SIZE: {}".format(opt.batchsize))
print("TRAIN SIZE: {}".format(opt.trainsize))
print("CLIP: {}".format(opt.clip))
print("USING ResNet backbone: {}".format(opt.is_ResNet))
print("DECAY RATE: {}".format(opt.decay_rate))
print("DECAY EPOCH: {}".format(opt.decay_epoch))
print("MODEL ID: {}".format(opt.model_id))


# build models
if opt.is_ResNet:
	model = CPD_ResNet()
else:
	model = CPD_VGG()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
# If no previous training, 0 epochs passed
last_epoch = 0
if WANDB_EN:
	if resume_wandb_model:
		print("Loading previous trained model:"+resume_wandb_model)
		#wandb.restore(resume_wandb_model)
		resume_wandb_model_path = './models/CPD_VGG/'+resume_wandb_model
		checkpoint = torch.load(resume_wandb_model_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		last_epoch = checkpoint['epoch']
		last_loss = checkpoint['loss']
else:
	if resume_model_path:
		print("Loading previous trained model:"+resume_model_path)
		checkpoint = torch.load(resume_model_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		last_epoch = checkpoint['epoch']
		last_loss = checkpoint['loss']

dataset_name = 'MSRA_B'
image_root = '../../DATASETS/TRAIN/'+dataset_name+'/im/'
gt_root = '../../DATASETS/TRAIN/'+dataset_name+'/gt/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
print("Total step per epoch: {}".format(total_step))

CE = torch.nn.BCEWithLogitsLoss()

####################################################################################################

def train(train_loader, model, optimizer, epoch):
	model.train()
	for i, pack in enumerate(train_loader, start=1):
		optimizer.zero_grad()
		images, gts = pack
		images = Variable(images)
		gts = Variable(gts)
		images = images.cuda()
		gts = gts.cuda()

		atts, dets = model(images)
		loss1 = CE(atts, gts)
		loss2 = CE(dets, gts)
		loss = loss1 + loss2
		loss.backward()

		clip_gradient(optimizer, opt.clip)
		optimizer.step()
		if WANDB_EN:
			wandb.log({'Loss': loss})
		#if i % 400 == 0 or i == total_step:
		if i % 100 == 0 or i == total_step:
			print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
				  format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data))

	# Save model and optimizer training data
	trained_model_data = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch,
		'loss': loss
	}

	if opt.is_ResNet:
		save_path = 'models/CPD_Resnet/'
	else:
		save_path = 'models/CPD_VGG/'

	if not os.path.exists(save_path):
		print("Making trained model folder [{}]".format(save_path))
		os.makedirs(save_path)
	#if (epoch+1) % 5 == 0:
	torch_model_ext = '.pth'
	wandb_model_ext = '.h5'
	#model_unique_id = '.%d' % epoc
	model_unique_id = model_id+'_'+'ep'+'_'+'%d' % epoch
	trained_model_name = 'CPD_train' 
	save_full_path_torch = save_path + trained_model_name + '_' + model_unique_id + torch_model_ext 
	save_full_path_wandb = save_path + trained_model_name + '_' + model_unique_id + wandb_model_ext
	#if not os.listdir(save_path):
	if os.path.exists(save_full_path_torch):
		print("Torch model with name ["+save_full_path_torch+"] already exists!")
		answ = raw_input("Do you want to replace it? [y/n] ")
		if("y" in answ):
			torch.save(trained_model_data, save_full_path_torch) 
			print("Saved torch model in "+save_full_path_torch)
	else:
			torch.save(trained_model_data, save_full_path_torch) 
			print("Saved torch model in "+save_full_path_torch)

	if WANDB_EN:
		if os.path.exists(save_full_path_wandb):	
			print("Wandb model with name ["+save_full_path_wandb+"] already exists!")
			answ = raw_input("Do you want to replace it? [y/n] ")
			if("y" in answ):
				wandb.save(save_full_path_wandb)
				print("Saved wandb model in "+save_full_path_wandb)
		else:
				wandb.save(save_full_path_wandb)
				print("Saved wandb model in "+save_full_path_wandb)


####################################################################################################

print("Training on dataset: "+dataset_name)
print("Train images path: "+image_root)
print("Train gt path: "+gt_root)
print("Let's go!")

if WANDB_EN:
	wandb.watch(model, log="all")
#if last_epoch != 0:
#	last_epoch++
# Range goes from 1 to last number included
# so as ex. with epoch=1 it's range(1,1) = [] empty.
# Adding +1 so range(1,1+1)=range(1,2)=[1] so 1 epoch.
for epoch in range(last_epoch+1, tot_epochs+1):
	adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
	train(train_loader, model, optimizer, epoch)
print("TRAINING DONE!")
