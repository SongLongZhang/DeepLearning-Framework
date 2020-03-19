from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):

    def __init__(self, **kwargs):

        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # 隐藏层
        self.output = nn.Dense(10)  # 输出层


    def forward(self, x):
        return self.output(self.hidden(x))

X = nd.random.uniform(shape=(2, 20))
net = MLP()
net.initialize()
print(net(X))



class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):

        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x



net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
print(net(X))



