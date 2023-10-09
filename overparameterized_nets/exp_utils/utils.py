import torch.optim.lr_scheduler as lr_scheduler


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