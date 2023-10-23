import tensorflow as tf
from tensorflow import convert_to_tensor, string
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
from tensorflow.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from transformer.positional_encoding import PositionalEmbeddingFixedWeights

output_sequence_length = 5
vocab_size = 10

sentences = [['I am a robot'], ['you too robot']]
sentence_data = Dataset.from_tensor_slices(sentences)

# create the vectorization layer
vectorize_layer = TextVectorization( output_sequence_length=output_sequence_length, max_tokens=vocab_size )

# adapt layer to available dictionary
vectorize_layer.adapt( sentence_data )

# convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)

# use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)

# print('Vocabulary:', vectorize_layer.get_vocabulary())
# print('Vecrorized words:', vectorized_words)

# apply embeddings
output_length = 6
word_embedding_layer = Embedding( vocab_size, output_length )
embedded_words = word_embedding_layer(vectorized_words)
# random initialization of the Embedding layer, means different result any time you run it
# print(embedded_words)

pos_emb = PositionalEmbeddingFixedWeights(output_length, vocab_size, output_length)
pos_emb_output = pos_emb(vectorized_words)

# print('pos_emb_output:', pos_emb_output)

# example with larger sentences and visualisation
technical_phrase = "to understand machine learning algorithms you need" +\
                   " to understand concepts such as gradient of a function "+\
                   "Hessians of a matrix and optimization etc"
wise_phrase = "patrick henry said give me liberty or give me death "+\
              "when he addressed the second virginia convention in march"

total_vocabulary = 200
sequence_length = 20
final_output_len = 50
phrase_vectorization_layer = TextVectorization(
                  output_sequence_length=sequence_length,
                  max_tokens=total_vocabulary)

# Learn the dictionary
phrase_vectorization_layer.adapt([technical_phrase, wise_phrase])
# Convert all sentences to tensors
phrase_tensors = convert_to_tensor([technical_phrase, wise_phrase], 
                                   dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_phrases = phrase_vectorization_layer(phrase_tensors)

fixed_weights_embedding_layer = PositionalEmbeddingFixedWeights(sequence_length, 
                                                        total_vocabulary,
                                                        final_output_len)

fixed_embedding = fixed_weights_embedding_layer(vectorized_phrases)


fig = plt.figure(figsize=(15, 5))    
title = ["Tech Phrase", "Wise Phrase"]
for i in range(2):
    ax = plt.subplot(1, 2, 1+i)
    matrix = tf.reshape(fixed_embedding[i, :, :], (sequence_length, final_output_len))
    cax = ax.matshow(matrix)
    plt.gcf().colorbar(cax)   
    plt.title(title[i], y=1.2)
fig.suptitle("Fixed Weight Embedding from Attention is All You Need")
plt.show()