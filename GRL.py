from torch.autograd import Function
import torch

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        # 返回一个与x相同的tensor
        # https://blog.csdn.net/gongxifacai_believe/article/details/121278968
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 梯度反转
        output = grad_output.neg() * ctx.alpha

        return output, None