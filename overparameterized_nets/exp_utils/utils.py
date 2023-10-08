import torch.optim.lr_scheduler as lr_scheduler
import torch 
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer


class CustomMultiStepLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma_lr=0.1, gamma_wd=0.1, last_epoch=-1):
        self.milestones = milestones
        self.gamma_lr = gamma_lr
        self.gamma_wd = gamma_wd
        self.base_wds = [group['weight_decay'] for group in optimizer.param_groups]
        super(CustomMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr_factor = 1.0
        wd_factor = 1.0
        for milestone in self.milestones:
            if self.last_epoch >= milestone:
                lr_factor *= self.gamma_lr
                wd_factor *= self.gamma_wd
        return [base_lr * lr_factor for base_lr in self.base_lrs], [base_wd * wd_factor for base_wd in self.base_wds]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr, wd = self.get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr[i]
            param_group['weight_decay'] = wd[i]

class SGLD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0, dampening=0,
                 nesterov=False, noise_scale=1.0, decay=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        noise_scale=noise_scale, decay=decay)
        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, epoch, closure=None,):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)
                if group['nesterov']:
                    d_p = d_p.add(group['momentum'], d_p)
                p.data.add_(-group['lr'], d_p)
                if epoch < group['decay']:
                    p.data.add_(group['noise_scale'] * torch.randn_like(p.data))
        return loss

class Ntk_noise_loss(torch.nn.Module):
    def __init__(self, scale,decay):
        super(Ntk_noise_loss, self).__init__()
        self.scale = scale
        self.decay = decay

    def forward(self, outputs, targets, epoch):
        if epoch >= self.decay:
            cross_entropy_loss = F.cross_entropy(outputs, targets)
            return cross_entropy_loss, cross_entropy_loss
        else: 
            cross_entropy_loss = F.cross_entropy(outputs, targets)
            noise = Normal(torch.zeros(outputs.shape),torch.ones(outputs.shape))
            e = self.scale*noise.sample().cuda()
            mean_squared_error_loss = F.mse_loss(outputs, outputs.detach() + e)
            loss = cross_entropy_loss + mean_squared_error_loss
        return loss, cross_entropy_loss

from functorch import make_functional_with_buffers, vmap, grad

def compute_loss_stateless_model (fmodel,params, buffers, batch_x, batch_y,loss_fn):
    predictions = fmodel(params, buffers, batch_x) 
    loss = loss_fn(predictions, batch_y)
    return loss

def compute_jacobian_norm(model,loss_fn,batch_x,batch_y): 
    fmodel, params, buffers = make_functional_with_buffers(model)
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
