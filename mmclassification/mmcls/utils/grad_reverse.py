from torch.autograd.function import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        #ctx.save_for_backward(result)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        #pdb.set_trace()
        #result, = ctx.saved_tensors
        return (grad_output * (1))
