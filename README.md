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

The novelty in this model is that it employs the cross-attention mechanism, which already exists in the original tranformer decoder architecture, in any encoder input. This will, assumingly, a strong feature, since it enables the pretrained decoder to start attending to any given input, rergardles of the form of the encoded input (i.e., it can be embeddings of a sentence from a BERT-style decoder, or embeddings by an autoencoder encoder). While the locking decoder is trained, the cross-attention part is *locked*, i.e., the output is produces is a zeros tensor of the proper size. The subsequent add-normalize layer after cross-attention, simply echoes the output of the pre-cross-attention layer, effectively bypassing the cross-attention.

The goal is to enable few-shot learning of "captioning" input coming from any incoming encoder. This may require very low dimensionality in the key, value but also the query weights of cross-attention, but that is something that needs to be empirically tested.

The **LockingDecoderModel** class (in **transformer.models**) entails locking versions of the underlying constituent components, i.e., a **LockingDecoder** layer and a **LockingDecoderLayer** layer (in **transformer.layers**) and a **LockingMultiHeadAttention** (in **transformers.attention**).