from torch.optim.lr_scheduler import ReduceLROnPlateau


class CustomReduceLROnPlateau(ReduceLROnPlateau):

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        super(CustomReduceLROnPlateau, self).__init__(optimizer, mode, factor, patience, verbose, threshold,
                                                      threshold_mode, cooldown, min_lr, eps)
        self.num_of_reductions = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                self.num_of_reductions += 1
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))