import pdb
import torch
from torch.cuda.amp import GradScaler, autocast
from configs.resnet import configuration
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
from exp_utils.setup_exp import set_exp
from exp_utils.utils import CustomMultiStepLR
import sys
from models import get_models
from data import get_dataset
import pandas as pd
from math import sqrt
from torch.optim.optimizer import Optimizer
import copy
from pyhessian import hessian  # Hessian computation
from functorch import make_functional, vmap, vjp, jvp, jacrev, make_functional_with_buffers
import numpy as np

SUBSET = 500
JAC = False
HESS = True


def compute_jacobian(loader, model):
    jacs = []
    for i, train_subs_x in enumerate(tqdm(loader)):
        train_subs_x = train_subs_x[0]
        fmodel, params, buffers = make_functional_with_buffers(model)

        def predict(params, buffers, x):
            return fmodel(params, buffers, x)
        (compute_batch_jacobian,) = vmap(jacrev(predict, argnums=(0, )),
                                         in_dims=(None, None, 0))(params, buffers, train_subs_x)
        jac = torch.norm(torch.cat([j.detach().cpu().flatten(
            1) for j in compute_batch_jacobian], 1), dim=1)
        jacs.append(jac.detach().cpu())
        compute_batch_jacobian = 0
        jac = 0
    return torch.cat(jacs).mean().item()


def evaluate(model, loaders, loss_fn, context='test'):
    model.eval()
    with torch.no_grad():
        total_correct, total_num, loss = 0., 0., 0.
        for ims, labs in tqdm(loaders[context], leave=False):
            with autocast():
                # + model(torch.fliplr(ims))) / 2.  Test-time augmentation
                out = (model(ims))
                loss += loss_fn(out, labs).detach().cpu().item()
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
    return 1 - total_correct / total_num, loss / total_num

def eval(model, loaders_eval, loss_fn, ep, aim_writer, context='SDE'): 
    test_err, test_loss = evaluate(
        model, loaders_eval, loss_fn, context='test')
    train_err, train_loss = evaluate(
        model, loaders_eval, loss_fn, context='train')
    l2 = sum([(param**2).sum()
             for param in model.parameters()]).detach().cpu().item()
    aim_writer.track({'Test Error': test_err, 'Test Loss': 0.5*config.wd*l2+test_loss,
                     'Test Cross Entropy': test_loss}, context={'subset': context}, step=ep)
    aim_writer.track({'Train Error': train_err, 'Train Loss': 0.5*config.wd*l2+train_loss,
                     'Train Cross Entropy': train_loss, 'L2': sqrt(l2), 'reg': 0.5*config.wd*l2}, context={'subset': context}, step=ep)

def compute_jac(loaders, model, ep, aim_writer, context='SDE'):
    ############## JACOBIAN COMPUTATION ################
    print("============> COMPUTING JACOBIAN <============")
    jac_norm = compute_jacobian(loaders['train'], model)

    aim_writer.track({'J_norm': jac_norm},  context={'subset': context}, step=ep)


def compute_hess(loaders, model, loss, ep, top_eig=False, aim_writer=None, context='SDE'):
    print("============> COMPUTING HESSIAN <============")
    hessian_comp = hessian(
        model, loss, dataloader=loaders['train'], cuda=True)
    trace = np.mean(hessian_comp.trace())
    if top_eig:
        eigenvalues, _ = hessian_comp.eigenvalues()
        aim_writer.track({'trace_H': trace , 'top_eig':  eigenvalues[-1]},  context={'subset': context}, step=ep)

    else:
        aim_writer.track({'trace_H': trace},  context={'subset': context}, step=ep)


def train(model, loaders, loaders_eval, loss_fn, opt, scaler, ep, aim_writer, ema=None, ma=False, p_radius=None):
    model.train()
    if ma:
        model_ma = copy.deepcopy(model).eval()
    for ims, labs in tqdm(loaders['train'], leave=False):
        opt.zero_grad(set_to_none=True)
        with autocast():
            out = model(ims)
            train_loss = loss_fn(out, labs)

        scaler.scale(train_loss).backward()
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            model_ema.moving_average(model, 0.999)
        if ma:
            with torch.no_grad():
                for param1, param2 in zip(model_ma.parameters(), model.parameters()):
                    param1.data += param2.data
    if p_radius is not None:
        with torch.no_grad():
            norm = torch.norm(
                torch.cat([p.reshape(-1) for p in model.parameters() if p.requires_grad]), p=2)
            for param in model.parameters():
                if param.requires_grad:
                    param.div_(norm/p_radius)
    if ma:
        with torch.no_grad():
            for param in model_ma.parameters():
                param /= len(loaders['train'])
        ma_test_err, ma_test_loss = evaluate(model_ma, loaders_eval, loss_fn)
        aim_writer.track({'Test Error': ma_test_err, 'Test Cross Entropy': ma_test_loss}, context={
                         'subset': 'ma'}, step=ep)


# --------------------------------------------------------------------------------
# SETUP EXPERIMENT
# --------------------------------------------------------------------------------
config = configuration()
device, aim_writer = set_exp(config)
if config.dataset == 'cifar100':    
    n_cls = 100 # TODO: take it from dataset
else: 
    n_cls = 10
model = get_models.get_model(config.model, n_cls, config.half_prec, get_dataset.shapes_dict[config.dataset], config.model_width,
                             batch_norm=config.batch_norm, freeze_last_layer=False, learnable_bn=True).to(memory_format=torch.channels_last).cuda()
model_ema = copy.deepcopy(model).eval()
l2 = sum([(param**2).sum()
         for param in model.parameters()]).detach().cpu().item()
print(l2)
start_epoch = 0
opt = SGD(model.parameters(), lr=config.lr,
          momentum=config.momentum, weight_decay=config.wd)
if config.lr_flow is not None:
    gamma_lr = config.lr_flow/config.lr
else:
    gamma_lr = config.lr_gamma_decay

scaler = GradScaler()
loss_fn = CrossEntropyLoss()
loaders = get_dataset.create_dataloaders(
    config.dataset, config.no_data_augm, config.batch_size, device)
loaders_eval = get_dataset.create_dataloaders(
    config.dataset, config.no_data_augm, 1000, device)
h_j_loaders = get_dataset.create_dataloaders(
    'cifar10_5k', config.no_data_augm, 50, device)

# --------------------------------------------------------------------------------
# TRAINING
# --------------------------------------------------------------------------------
with tqdm(range(start_epoch, config.epochs), desc=f'Epochs', unit='epoch') as tepoch:
    for ep in tepoch:
        tepoch.set_description(f"Epoch {ep}")
        train(model, loaders, loaders_eval, loss_fn, opt, scaler, ep, aim_writer, ema=model_ema, ma=False)
        eval(model, loaders_eval, loss_fn, ep, aim_writer, context='SDE')
        aim_writer.track({'learning rate': opt.param_groups[0]['lr'], 'lambda': opt.param_groups[0]['weight_decay']}, context={
                         'subset': 'SDE'}, step=ep)
        get_models.bn_update(loaders['train'], model_ema)
        eval(model_ema, loaders_eval, loss_fn, ep, aim_writer, context='ema')
# --------------------------------------------------------------------------------
# GRADIENT FLOW
# --------------------------------------------------------------------------------
        if ep % 2 == 0:
            model2 = copy.deepcopy(model)
            opt_ft = SGD(model2.parameters(), lr=config.lr_flow,
                         momentum=config.momentum, weight_decay=config.wd_flow)  # Check if you need WD here

            with tqdm(range(ep, ep+config.flow_steps), desc=f'Flow Epochs', unit='epoch', leave=False) as tepoch2:
                for ep2 in tepoch2:
                    tepoch2.set_description(f"Flow {ep}")
                    train(model2, loaders, loaders_eval, loss_fn, opt_ft, scaler, ep2, aim_writer, p_radius=None)
                # I am only evaluating the last point of thefolow to make it faster 
                eval(model2, loaders_eval, loss_fn, ep, aim_writer, context=f'flow_{ep}')
            
            get_models.save_model(
                ep, model, opt, config.out_dir, 'ckpt_SDE')
            get_models.save_model(
                ep, model_ema, opt, config.out_dir, 'ckpt_ema')
            get_models.save_model(
                ep, model2, opt, config.out_dir, 'ckpt_flow')
            
            if JAC:
                compute_jac(h_j_loaders, model2, ep=ep, aim_writer=aim_writer, context=f'flow')
            if HESS:
                compute_hess(h_j_loaders, model2, loss_fn, ep=ep, aim_writer=aim_writer, context=f'flow')