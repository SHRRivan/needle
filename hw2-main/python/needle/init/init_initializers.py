import math
from .init_basic import *

# 适用于 Sigmoid 或 Tanh 激活函数: 主要用于解决梯度消失和梯度爆炸问题。
# 目标是使输入和输出的方差相等，保持激活信号的尺度不变
def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION

# 适用于 ReLU 及其变体: 由于 ReLU 激活函数容易导致部分神经元的输出为零，因此需要另一种初始化方法以确保前向传递的信号不变。
# 目标是使激活信号在前向传递时保持其动态范围，这对于深层网络尤其重要，防止各层信号的无效化（例如，过度缩放或扩展）。
def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # gain = math.sqrt(2)
    # bound = gain * math.sqrt(3 / fan_in)

    # one step way is:
    bound = math.sqrt(6 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2 / fan_in)
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION