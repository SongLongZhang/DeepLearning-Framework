from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
nd.save('x', x)

x2 = nd.load('x')
print(x2)

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = nd.random.uniform(shape=(2, 20))
Y = net(X)
filename = 'mlp.params'
net.save_parameters(filename)

net2 = MLP()
net2.load_parameters(filename)

Y2 = net2(X)
print(Y2 == Y)