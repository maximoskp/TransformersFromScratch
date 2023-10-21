from transformer.attention import DotProductAttention, MultiHeadAttention
from numpy import random

h = 8
d_k = 64
d_v = 64
d_model = 512
batch_size = 64

input_seq_length = 5

queries = random.random( (batch_size, input_seq_length, d_k) )
keys = random.random( (batch_size, input_seq_length, d_k) )
values = random.random( (batch_size, input_seq_length, d_k) )

multihead = MultiHeadAttention(h, d_k, d_v, d_model)
multihead.compile()

output = multihead( queries, keys, values )

print(output.shape)