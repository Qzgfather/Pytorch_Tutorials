from utils import build_dataset, build_iterator
from importlib import import_module
from models import TextRNN
import tqdm
from models.TextRNN import Config
import os
import pickle as pkl
import torch
from utils import DatasetIterater
import numpy as np

x = TextRNN
dataset = 'AFF'
embedding = 'random'
UNK, PAD = '<UNK>', '<PAD>'
MAX_VOCAB_SIZE = 10000  # 词表长度限制
pad_size = 32
config = x.Config(dataset, embedding)


class MyConfig(Config):
    def __init__(self):
        super(MyConfig, self).__init__(dataset, embedding)
        self.batch_size = 1


# vocab, train_data, dev_data, test_data = build_dataset(config, False)
# print(len(train_data))

# train_iter = build_iterator(train_data, config)
# for i in train_iter:
#     print(i[0][0][0], '\n', i[0][1], '\n', i[1])
#     break
# quit()


def build_dataset(str_data, ues_word=False):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    vocab = pkl.load(open('AFF/data/vocab.pkl', 'rb'))
    print(f"Vocab size: {len(vocab)}")

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def load_dataset(lines, pad_size=32):
        contents = []
        lin = lines.strip()
        content, label = lin.split('\t')
        words_line = []
        token = tokenizer(content)
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        contents.append((words_line, int(label), seq_len))
        return contents

    data = load_dataset(lines=str_data, pad_size=pad_size)
    return data


def get_data(data_):
    data = DatasetIterater(data_, 1, 'cuda')
    return data


model = torch.load('checkpoint/TextRNN.pth')
model = model.to('cuda')


def get_result(input_data):
    data = build_dataset(str_data=input_data)
    for i in get_data(data):
        with torch.no_grad():
            result = model(i[0])
            index = int(np.argmax(result.cpu().numpy()[0]))
            # print(result)
            print(index)
            print(classes[index]+'---相关！')


classes = ['身体不舒服', '游戏', '哥哥', '不高兴', '睡觉', '学习', '焦虑', '道歉', '疑问', '时政']
while True:
    data_ = input("请输入句子：")
    data_ = data_ + '\t0'
    get_result(data_)

