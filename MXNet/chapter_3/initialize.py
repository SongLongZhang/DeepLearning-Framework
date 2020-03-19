
from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = nd.random.uniform(shape=(2, 20))
Y = net(X)
print(net[0].params)
print(type(net[0].params))

print(net[0].params['dense0_weight'])
print(net[0].weight)
print(net[0].weight.data())
print(net.collect_params())

net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
print(net[0].weight.data()[0])


net.initialize(init=init.Constant(1), force_reinit=True)
print(net[0].weight.data()[0])

net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[0].weight.data()[0])



