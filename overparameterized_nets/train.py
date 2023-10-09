import torch
from torch.cuda.amp import GradScaler, autocast
from configs.config import configuration
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm
from exp_utils.setup_exp import set_exp
from exp_utils.utils import CustomMultiStepLR
from models import get_models
from data import get_dataset
from math import sqrt
import copy

def evaluate(model, loaders, loss_fn, context='test'):
    model.eval()
    with torch.no_grad():
        total_correct, total_num, loss = 0., 0., 0.
        for ims, labs in tqdm(loaders[context], leave=False):
            with autocast():
                out = (model(ims))
                loss += loss_fn(out, labs).detach().cpu().item()*ims.shape[0]
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
    return 1 - total_correct / total_num, loss / total_num


def train(model, loaders, loaders_eval, loss_fn, opt, scaler, ep, aim_writer, context, ema=None, ma=False, p_radius=None):
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
    return test_err, train_err, 0.5*config.wd*l2+test_loss, 0.5*config.wd*l2+train_loss, test_loss, train_loss, sqrt(l2)


# --------------------------------------------------------------------------------
# SETUP EXPERIMENT
# --------------------------------------------------------------------------------
config = configuration()
device, aim_writer = set_exp(config)
if config.dataset == 'cifar100':    
    n_cls = 100
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

scheduler = CustomMultiStepLR(opt, [int(
    config.epochs*config.first_decay)], gamma_lr=gamma_lr, gamma_wd=config.wd_gamma_decay)
scaler = GradScaler()
loss_fn = CrossEntropyLoss()
loaders = get_dataset.create_dataloaders(
    config.dataset, config.no_data_augm, config.batch_size, device)
loaders_eval = get_dataset.create_dataloaders(
    config.dataset, config.no_data_augm, 1000, device)

with tqdm(range(start_epoch, config.epochs), desc=f'Epochs', unit='epoch') as tepoch:
    for ep in tepoch:
        tepoch.set_description(f"Epoch {ep}")
        test_err, train_err, test_loss, train_loss, test_CE, train_CE, l2_sde = train(
            model, loaders, loaders_eval, loss_fn, opt, scaler, ep, aim_writer, context='SDE', ema=model_ema, ma=False, p_radius = config.radius)
        scheduler.step()
        aim_writer.track({'learning rate': opt.param_groups[0]['lr'], 'lambda': opt.param_groups[0]['weight_decay']}, context={
                         'subset': 'SDE'}, step=ep)
        get_models.bn_update(loaders['train'], model_ema)
        ema_train_err, ema_train_loss = evaluate(
            model_ema, loaders_eval, loss_fn, context='train')
        ema_test_err, ema_test_loss = evaluate(
            model_ema, loaders_eval, loss_fn, context='test')
        model_ema.eval()
        ema_l2 = sqrt(sum([(param**2).sum()
                      for param in model_ema.parameters()]).detach().cpu().item())

        aim_writer.track({'Test Error': ema_test_err, 'Test Cross Entropy': ema_test_loss, 'Train Error': ema_train_err, 'Train Cross Entropy': ema_train_loss, 'L2': ema_l2}, context={
                         'subset': 'ema'}, step=ep)

        tepoch.set_postfix(Train_CE=train_CE,
                           Train_e=train_err, Test_e=test_err)
