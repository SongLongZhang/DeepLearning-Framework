from mxnet import nd
import numpy as np


#创建NDArray
x = nd.arange(12)
print(x)
print(x.shape)
print(x.size)

x = x.reshape((3, 4))
print(x)

print(nd.zeros((3,4)))
print(nd.ones((2,3,4)))

#运算
X = nd.arange(12).reshape((3, 4))
Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X+Y)
print(X*Y)
print(X/Y)
print(Y.exp())
print(X.norm().asscalar())
print(X.sum())
Z = nd.concat(X, Y, dim=0), nd.concat(X, Y, dim=1)
print(Z)

#广播机制
A = nd.arange(3).reshape((3, 1))
B = nd.arange(2).reshape((1, 2))
print(A+B)

#索引
print(X[1:3])
X[1:2, :] = 12
print(X)

#运算的内存开销
before = id(Y)
Y = Y + X
print(id(Y) == before)#False

Z = Y.zeros_like()
before = id(Z)
Z[:] = X + Y
print(id(Z) == before)#True

#NDArray和NumPy相互变换
P = np.ones((2, 3))
D = nd.array(P)
print(D)
print(D.asnumpy())
