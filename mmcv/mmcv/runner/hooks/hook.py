# Copyright (c) Open-MMLab. All rights reserved.
from mmcv.utils import Registry

HOOKS = Registry('hook')


class Hook:

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def before_train_epoch(self, runner):
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        self.before_iter(runner)

    def before_val_iter(self, runner):
        self.before_iter(runner)

    def after_train_iter(self, runner):
        self.after_iter(runner)

    def after_val_iter(self, runner):
        self.after_iter(runner)

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        if hasattr(runner, 'data_loader'):
            return(runner.inner_iter + 1 == len(runner.data_loader))
        elif hasattr(runner, 'data_loader_t'):
            return runner.inner_iter + 1 == max(len(runner.data_loader_s), len(runner.data_loader_t))
        elif hasattr(runner, 'data_loader_s'):
            return runner.inner_iter + 1 == len(runner.data_loader_s)
