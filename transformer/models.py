from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from transformer.layers import Encoder, Decoder

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
# end class TransformerModel

class EncoderModel(Model):
    def __init__(self, enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super(EncoderModel, self).__init__(**kwargs)
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)
    # end init

    def padding_mask(self, input):
        mask = math.equal(input, 0)
        mask = cast(mask, float32)
        # shape needs to match attention maps
        return mask[:, newaxis, newaxis, :]
    # end padding_mask

    def call(self, encoder_input, training):
        enc_padding_mask = self.padding_mask(encoder_input)
        return self.encoder(encoder_input, enc_padding_mask, training)
    # end call
# end class EncoderModel

class DecoderModel(Model):
    def __init__(self, dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super(DecoderModel, self).__init__(**kwargs)
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

    def call(self, decoder_input, encoder_output, enc_padding_mask, training):
        # if no encoder input is provided, encoder_output and enc_padding_mask should be set to trivial values
        dec_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_padding_mask, dec_in_lookahead_mask)

        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)
        model_output = self.model_last_layer(decoder_output)
        return model_output
    # end call
# end class DecoderModel

class TransformerEncDecModel(Model):
    def __init__(self, encoder_model, decoder_model, **kwargs):
        super(TransformerEncDecModel, self).__init__(**kwargs)
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
    # end init

    def padding_mask(self, input):
        mask = math.equal(input, 0)
        mask = cast(mask, float32)
        # shape needs to match attention maps
        return mask[:, newaxis, newaxis, :]
    # end padding_mask

    def call(self, encoder_input, decoder_input, training):
        enc_padding_mask = self.padding_mask(encoder_input)
        encoder_output = self.encoder_model(encoder_input, training)
        model_output = self.decoder_model(decoder_input, encoder_output, enc_padding_mask, training)
        return model_output
    # end call
# end class TransformerEncDecModel