import collections
from d2l import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile

def download_imdb(data_dir='./d2l/data'):
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)

download_imdb()

def read_imdb(folder='train'):  # 本函数已保存在d2lzh包中方便以后使用
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('../data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

train_data, test_data = read_imdb('train'), read_imdb('test')
#预处理数据
def get_tokenized_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

def get_vocab_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5)

vocab = get_vocab_imdb(train_data)
print('# words in vocab:', len(vocab))

def preprocess_imdb(data, vocab):  # 本函数已保存在d2lzh包中方便以后使用
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels

batch_size = 64
train_set = gdata.ArrayDataset(*preprocess_imdb(train_data, vocab))
test_set = gdata.ArrayDataset(*preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)


for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
print('#batches:', len(train_iter))

class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.T)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.decoder(encoding)
        return outs

embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)
glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)

net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')



lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence), ctx=d2l.try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'

predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])

