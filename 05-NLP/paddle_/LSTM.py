import paddle.fluid.layers as layers
import paddle.fluid as fluid

emb_dim = 256
vocab_size = 10000
data = fluid.layers.data(name='x',
                         shape=[-1, 100, 1],
                         dtype='int64')
print(data)

emb = fluid.layers.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)
print(emb)
batch_size = 1
max_len = 100
dropout_prob = 0.2
hidden_size = 150
num_layers = 1
init_h = layers.fill_constant([num_layers, batch_size, hidden_size], 'float32', 0.0)
init_c = layers.fill_constant([num_layers, batch_size, hidden_size], 'float32', 0.0)
print(init_h, init_c)
rnn_out, last_h, last_c = layers.lstm(emb, init_h, init_c, max_len, hidden_size, num_layers, dropout_prob=dropout_prob)
print(rnn_out.shape)  # (-1, 100, 150)
print(last_h.shape)  # (1, 20, 150)
print(last_c.shape)  # (1, 20, 150)
