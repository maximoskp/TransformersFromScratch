# https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/

from tensorflow import matmul, reshape, shape, transpose, math, cast, float32, zeros
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

# https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/

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

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # spreading to all heads- (batch_size, heads, seq_length, -1)
            x = reshape( x, shape=(shape(x)[0], shape(x)[1], heads, -1) )
            x = transpose(x, perm=(0,2,1,3))
        else:
            # concatenate all heads together
            x = transpose(x, perm=(0,2,1,3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x
    # end reshape_tensor

    def call(self, queries, keys ,values, mask=None):
        # reshape to prepare for multihead paralellization (batch_size, heads, seq_length, -1)
        q_reshaped = self.reshape_tensor( self.W_q(queries), self.heads, True )
        k_reshaped = self.reshape_tensor( self.W_k(keys), self.heads, True )
        v_reshaped = self.reshape_tensor( self.W_v(values), self.heads, True )
        # apply attention to all heads
        o_reshaped = self.attention( q_reshaped, k_reshaped, v_reshaped, self.d_k, mask )
        # rearrange to concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # apply the final linear layer
        return self.W_o(output)
    # end call
# end class MultiHeadAttention

# LOCKING ======================================

class LockingMultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(LockingMultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()
        self.heads = h # number of heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_q = Dense(d_k) # learned query input weights
        self.W_k = Dense(d_k) # learned key input weights
        self.W_v = Dense(d_v) # learned value input weights
        self.W_o = Dense(d_model) # learned multi-head output weights
    # end init

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # spreading to all heads- (batch_size, heads, seq_length, -1)
            x = reshape( x, shape=(shape(x)[0], shape(x)[1], heads, -1) )
            x = transpose(x, perm=(0,2,1,3))
        else:
            # concatenate all heads together
            x = transpose(x, perm=(0,2,1,3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x
    # end reshape_tensor

    def call(self, queries, keys ,values, mask=None, lock=False):
        # reshape to prepare for multihead paralellization (batch_size, heads, seq_length, -1)
        if lock:
            return zeros( shape( queries) )
        else:
            q_reshaped = self.reshape_tensor( self.W_q(queries), self.heads, True )
            k_reshaped = self.reshape_tensor( self.W_k(keys), self.heads, True )
            v_reshaped = self.reshape_tensor( self.W_v(values), self.heads, True )
            # apply attention to all heads
            o_reshaped = self.attention( q_reshaped, k_reshaped, v_reshaped, self.d_k, mask )
            # rearrange to concatenated form
            output = self.reshape_tensor(o_reshaped, self.heads, False)
            # apply the final linear layer
            return self.W_o(output)
    # end call
# end class LockingMultiHeadAttention