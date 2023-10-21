from transformer.attention import DotProductAttention
from numpy import random

d_k = 64
d_v = 64
batch_size = 64

input_seq_length = 5

queries = random.random( (batch_size, input_seq_length, d_k) )
keys = random.random( (batch_size, input_seq_length, d_k) )
values = random.random( (batch_size, input_seq_length, d_k) )

attention = DotProductAttention()

a = attention(queries, keys, values, d_k)
m = attention.attention_map

print(a.shape)
print(m.shape)