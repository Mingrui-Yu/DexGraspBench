import torch
import time

# 维度设定
n = 100  # 输入维度
m = 50  # 输出维度

# 随机初始化函数 f: R^n -> R^m
A = torch.randn(m, n)
b = torch.randn(m)


def func(x):
    return A @ x + b  # Linear function, Jacobian is constant A


# 手动 Jacobian（恒定值）
def manual_jacobian(x):
    return A  # Since the function is linear


# 测试输入
x = torch.randn(n, requires_grad=True)

# -------- 自动求导 ----------
start = time.time()
J_auto = torch.autograd.functional.jacobian(func, x)
end = time.time()
print(f"[autograd] Time taken: {(end - start) * 1000:.3f} ms")

# -------- 手动求导 ----------
start = time.time()
J_manual = manual_jacobian(x)
end = time.time()
print(f"[manual]   Time taken: {(end - start) * 1000:.3f} ms")

# 校验结果一致性
assert torch.allclose(J_auto, J_manual, atol=1e-6)
