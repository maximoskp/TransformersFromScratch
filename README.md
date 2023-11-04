# TransformersFromScratch
All the implementations are based on the code from Machine Learning Mastery book: Building Transformer Models with Attention.

https://machinelearningmastery.com/transformer-models-with-attention/

Parts of the code can be found in the following links, even though following the code from the book makes more sense.

Attention:
https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/

Multihead attention
https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/

Encoder:
https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/

Decoder:
https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras/

Joining the Encoder and the Decoder
https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/

Training the transformer
https://machinelearningmastery.com/training-the-transformer-model/

## LockingDecoderModel

This is a new model that will be examined as a pretrained GPT-style decoder that is plugable to any kind of encoder input, either transformer encoder (e.g., pretrained with masked language modeling), or any other encoder-style architecture (e.g., a pretrained encoder from a pretrained (VQ / variational) autoencoder).

The LockingDecoderModel class (in *transformer.models*) entails locking versions of the underlying constituent components, i.e., a LockingDecoder layer and a LockingDecoderLayer layer (in *transformer.layers*) and a LockingMultiHeadAttention (in *transformers.attention*).