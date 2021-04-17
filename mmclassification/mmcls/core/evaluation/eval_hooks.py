import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=True, classwise=False, **eval_kwargs):
        #if not isinstance(dataloader, DataLoader):
        #    raise TypeError('dataloader must be a pytorch DataLoader, but got'
        #                    f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.by_epoch = by_epoch
        self.classwise = classwise
    """
    def before_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)
    """
    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, classwise=self.classwise, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

class OfficeEvalHook(EvalHook):
    def __init__(self,
                 dataloader,
                 interval=1,
                 by_epoch=True,
                 **eval_kwargs):
        super(OfficeEvalHook, self).__init__(dataloader, interval=1, by_epoch=True, **eval_kwargs)

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        results = []
        results.append(single_gpu_test(runner.model, self.dataloader[0], show=False))
        results.append(single_gpu_test(runner.model, self.dataloader[1], show=False))
        results.append(single_gpu_test(runner.model, self.dataloader[2], show=False))
        self.evaluate(runner, results)
    def evaluate(self, runner, results):
        eval_res = []
        datasetdict = {0: 'amazon', 1: 'webcam', 2: 'dslr'}
        for i in range(3):
            eval_res.append(self.dataloader[i].dataset.evaluate(
                results[i], logger=runner.logger, classwise=self.classwise, **self.eval_kwargs))
            for name, val in eval_res[i].items():
                runner.log_buffer.output[datasetdict[i] + name] = val
            runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=True,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

class DAEvalHook(Hook):
    """Evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=True, classwise=False, **eval_kwargs):
        #if not isinstance(dataloader, DataLoader):
        #    raise TypeError('dataloader must be a pytorch DataLoader, but got'
        #                    f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.by_epoch = by_epoch
        self.classwise = classwise
    """
    def before_train_epoch(self, runner):
        from mmcls.apis import da_single_gpu_test
        #results_s = da_single_gpu_test(runner.model, self.dataloader[0], show=False)
        results_s = None
        results_t = da_single_gpu_test(runner.model, self.dataloader[1], show=False)
        self.evaluate(runner, results_s, results_t)
    """
    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import da_single_gpu_test
        #results_s = da_single_gpu_test(runner.model, self.dataloader[0], show=False)
        results_s = None
        results_t = da_single_gpu_test(runner.model, self.dataloader[1], show=False)
        self.evaluate(runner, results_s, results_t)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import da_single_gpu_test
        runner.log_buffer.clear()
        #results_s = da_single_gpu_test(runner.model, self.dataloader[0], show=False)
        results_s = None
        results_t = da_single_gpu_test(runner.model, self.dataloader[1], show=False)
        self.evaluate(runner, results_s, results_t)


    def evaluate(self, runner, results_s, results_t):
        if results_s is not None:
            eval_res_s, _ = self.dataloader[0].dataset.evaluate(
                results_s, logger=runner.logger, classwise=self.classwise, **self.eval_kwargs)
            for name, val in eval_res_s.items():
                runner.log_buffer.output['source_' + name] = val
            runner.log_buffer.ready = True
        eval_res_t, _ = self.dataloader[1].dataset.evaluate(
            results_t, logger=runner.logger, classwise=self.classwise, **self.eval_kwargs)
        for name, val in eval_res_t.items():
            runner.log_buffer.output['target_' + name] = val
        runner.log_buffer.ready = True


class DADistEvalHook(DAEvalHook):
    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=True,
                 **eval_kwargs):
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import da_multi_gpu_test
        results_s = da_multi_gpu_test(
            runner.model,
            self.dataloader[0],
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        results_t = da_multi_gpu_test(
            runner.model,
            self.dataloader[1],
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)

        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results_s, results_t)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import da_multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

