from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from transformer.layers import Encoder, Decoder
import numpy as np

class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)
        self.model_last_layer = Dense(dec_vocab_size)
    # end init

    def padding_mask(self, input):
        mask = math.equal(input, 0)
        mask = cast(mask, float32)
        # shape needs to match attention maps
        return mask[:, newaxis, newaxis, :]
    # end padding_mask

    def lookahead_mask(self, shape):
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)
        return mask
    # end lookahead_mask

    def call(self, encoder_input, decoder_input, training):
        enc_padding_mask = self.padding_mask(encoder_input)
        
        dec_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_padding_mask, dec_in_lookahead_mask)

        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)
        model_output = self.model_last_layer(decoder_output)
        return model_output
    # end call