# https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/

from tensorflow import matmul, math, cast, float32
from tensorflow.keras.layers import Layer, Dense
from keras.backend import softmax

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.attention_map = None
    # end init

    def call(self, queries, keys, values, d_k, mask=None):
        # queries times keys transposed - scaled
        scores = matmul(queries, keys, transpose_b=True)/math.sqrt(cast(d_k, float32))

        # apply mask to the attention scores
        if mask is not None:
            scores += -1e9*mask
        
        # compute the score weights with softmax
        self.attention_map = softmax(scores)

        # compute attention
        return matmul(self.attention_map, values)
    # end call
# end class DotProductAttention

class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()
        self.heads = h # number of heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_q = Dense(d_k) # learned query input weights
        self.W_k = Dense(d_k) # learned key input weights
        self.W_v = Dense(d_v) # learned value input weights
        self.W_o = Dense(d_model) # learned multi-head output weights
    # end init