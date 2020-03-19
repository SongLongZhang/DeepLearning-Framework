from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss, nn
import os
import subprocess
import time


class Benchmark():
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))


with Benchmark('Workloads are queued.'):
    x = nd.random.uniform(shape=(2000, 2000))
    y = nd.dot(x, x).sum()

with Benchmark('Workloads are finished.'):
    print('sum =', y)


with Benchmark():
    y = nd.dot(x, x)
    y.wait_to_read()

#waitall
with Benchmark():
    y = nd.dot(x, x)
    z = nd.dot(x, x)
    nd.waitall()
#asnumpy
with Benchmark():
    y = nd.dot(x, x)
    y.asnumpy()

with Benchmark('synchronous.'):
    for _ in range(1000):
        y = x + 1
        y.wait_to_read()

with Benchmark('asynchronous.'):
    for _ in range(1000):
        y = x + 1
    nd.waitall()
