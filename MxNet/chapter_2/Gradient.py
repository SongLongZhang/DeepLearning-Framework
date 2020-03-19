from mxnet import autograd, nd


#
x = nd.arange(4).reshape((4, 1))
x.attach_grad()
with autograd.record():
    y = 2 * nd.dot(x.T, x)
y.backward()

assert (x.grad - 4 * x).norm().asscalar() == 0
print(x.grad)

#训练模式和预测模式
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())