import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


def cnn(x, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "cnn", reuse=tf.AUTO_REUSE):
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __pooling__(x)
        x = activation(x)
        return x


def __cnn_cell__(x, hidden_size, kernel_size, stride_size):
    x = tf.layers.conv1d(inputs=x,
                         filters=hidden_size,
                         kernel_size=kernel_size,
                         strides=stride_size,
                         padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x


def __pooling__(x):
    return tf.reduce_max(x, axis=-2)


def pcnn(x, mask, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __piecewise_pooling__(x, mask)
        x = activation(x)
        return x


def __piecewise_pooling__(x, mask):
    mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    mask = tf.nn.embedding_lookup(mask_embedding, mask)
    hidden_size = x.shape[-1]
    x = tf.reduce_max(tf.expand_dims(mask * 100, 2) + tf.expand_dims(x, 3), axis=1) - 100
    return tf.reshape(x, [-1, hidden_size * 3])


def __rnn_cell__(hidden_size, cell_name='lstm'):
    if isinstance(cell_name, list) or isinstance(cell_name, tuple):
        if len(cell_name) == 1:
            return __rnn_cell__(hidden_size, cell_name[0])
        cells = [__rnn_cell__(hidden_size, c) for c in cell_name]
        return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    if cell_name.lower() == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    elif cell_name.lower() == 'gru':
        return tf.contrib.rnn.GRUCell(hidden_size)
    raise NotImplementedError


def rnn(x, length, hidden_size=230, cell_name='lstm', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
        cell = __rnn_cell__(hidden_size, cell_name)
        _, states = tf.nn.dynamic_rnn(cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-rnn')
        if isinstance(states, tuple):
            states = states[0]
        return states


def birnn(x, length, hidden_size=230, cell_name='lstm', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "birnn", reuse=tf.AUTO_REUSE):
        fw_cell = __rnn_cell__(hidden_size, cell_name)
        bw_cell = __rnn_cell__(hidden_size, cell_name)
        _, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=length, dtype=tf.float32,
                                                    scope='dynamic-bi-rnn')
        fw_states, bw_states = states
        if isinstance(fw_states, tuple):
            fw_states = fw_states[0]
            bw_states = bw_states[0]
        return tf.concat([fw_states, bw_states], axis=1)


def encoder(x, mask ,is_trainIng, word_embed_size, is_pcnn=True):
    if is_pcnn:
        if is_trainIng:
            sentence_embedding = pcnn(x, mask, hidden_size=word_embed_size, keep_prob=0.5)
        else:
            sentence_embedding = pcnn(x, mask, hidden_size=word_embed_size)
    else:
        if is_trainIng:
            sentence_embedding = cnn(x, mask, hidden_size=word_embed_size, keep_prob=0.5)
        else:
            sentence_embedding = cnn(x, mask, hidden_size=word_embed_size)
    return sentence_embedding


def pos_embedding(pos1, pos2, var_scope=None, pos_embedding_dim=5, max_length=128):
    with tf.variable_scope(var_scope or "pos_embedding", reuse=tf.AUTO_REUSE):
        pos_tot = max_length * 2
        pos1_embedding = tf.get_variable("pos1_embedding", [pos_tot, pos_embedding_dim], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
        pos2_embedding = tf.get_variable("pos2_embedding", [pos_tot, pos_embedding_dim], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
        input_pos1 = tf.nn.embedding_lookup(pos1_embedding, pos1)
        input_pos2 = tf.nn.embedding_lookup(pos2_embedding, pos2)
        x = tf.concat([input_pos1, input_pos2], axis=-1)
        return x
