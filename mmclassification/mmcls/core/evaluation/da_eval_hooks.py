import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class DAEvalHook(Hook):
    """Evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=True, **eval_kwargs):
        #if not isinstance(dataloader, DataLoader):
        #    raise TypeError('dataloader must be a pytorch DataLoader, but got'
        #                    f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.by_epoch = by_epoch
        import pdb
        pdb.set_trace()
    def before_train_epoch(self, runner):
        import pdb
        pdb.set_trace()
        from mmcls.apis import single_gpu_test
        #results = single_gpu_test(runner.model, self.dataloader, show=False)
        #self.evaluate(runner, results)
        results = []

        results.append(single_gpu_test(runner.model, self.dataloader[0], show=False))
        results.append(single_gpu_test(runner.model, self.dataloader[1], show=False))
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        results = []
        
        results.append(single_gpu_test(runner.model, self.dataloader[0], show=False))
        results.append(single_gpu_test(runner.model, self.dataloader[1], show=False))
        self.evaluate(runner, results)
    """
    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)
    """
    def evaluate(self, runner, results):
        eval_res = []
        datasetdict = {0: 'source', 1: 'target'}
        for i in range(2):
            eval_res.append(self.dataloader[i].dataset.evaluate(
                results[i], logger=runner.logger, **self.eval_kwargs)
            for name, val in eval_res.items():
                runner.log_buffer.output[datasetdict[i] + '_' + name] = val
            runner.log_buffer.ready = True

class OfficeEvalHook(DAEvalHook):
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
                results[i], logger=runner.logger, **self.eval_kwargs))
            for name, val in eval_res[i].items():
                runner.log_buffer.output[datasetdict[i] + name] = val
            runner.log_buffer.ready = True


class DADistEvalHook(DAEvalHook):
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
        #if not isinstance(dataloader, DataLoader):
        #    raise TypeError('dataloader must be a pytorch DataLoader, but got '
        #                    f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import multi_gpu_test
        results = []
        results.append(multi_gpu_test(
            runner.model,
            self.dataloader[0],
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect))
        results.append(multi_gpu_test(
            runner.model,
            self.dataloader[1],
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect))
        results.append(multi_gpu_test(
            runner.model,
            self.dataloader[2],
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect))

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
