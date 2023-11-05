# huggingface datasets
from datasets import load_dataset
# huggingface BertTokenizer
# https://nlpiation.medium.com/how-to-use-huggingfaces-transformers-pre-trained-tokenizers-e029e8d6d1fa
# BertTokenizer class docs - scroll down on this page:
# https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/bert
from transformers import BertTokenizer, GPT2TokenizerFast
# https://keras.io/examples/nlp/masked_language_modeling/
from numpy.random import shuffle
from tensorflow import convert_to_tensor, int64
import numpy as np

# TODO: 
# 1. Create a WikipediaDataset super class that loads the wikitext dataset
# during initialization. This class could include the create_tokenizer
# function and get the type of the tokenizer as input, e.g BertTokenizer.
# 
# 2. Create a generator that feeds data randomly from the entire wikitext 
# dataset. The generator should be able to prepare data for each batch.

class MLMWikipediaDataset:
    def __init__(self,**kwargs):
        super(MLMWikipediaDataset, self).__init__(**kwargs)
        self.n_sentences = 10000
        self.train_split = 0.9
        self.max_sentence_length = 50
    # end init

    def create_tokenizer(self, sentences_list):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer_results = self.tokenizer( sentences_list, padding='max_length', truncation=True , max_length=self.max_sentence_length )
    # end create_tokenizer

    # based on https://keras.io/examples/nlp/masked_language_modeling/
    def get_masked_input_and_labels(self, vectorized_texts):
        # 15% BERT masking
        inp_mask = np.random.rand(*vectorized_texts.shape) < 0.15
        # Do not mask special tokens
        for token_id in self.tokenizer.all_special_ids:
            inp_mask[vectorized_texts == token_id] = False
        # Set targets to -1 by default, it means ignore
        sample_weights = np.zeros(vectorized_texts.shape, dtype=np.float32)
        # Set labels for masked tokens
        sample_weights[inp_mask] = 1

        # Prepare input
        vectorized_texts_masked = np.copy(vectorized_texts)
        # Set input to [MASK] which is the last token for the 90% of tokens
        # This means leaving 10% unchanged
        inp_mask_2mask = inp_mask & (np.random.rand(*vectorized_texts.shape) < 0.90)
        vectorized_texts_masked[
            inp_mask_2mask
        ] = self.tokenizer.mask_token_id  # mask token is the last in the dict

        # Set 10% to a random token
        random_tokens_to_select = np.arange( self.tokenizer.vocab_size )
        idx_to_delete = [np.where(random_tokens_to_select == x)[0][0]  for x in self.tokenizer.all_special_ids]
        random_tokens_to_select = np.delete( random_tokens_to_select, idx_to_delete )
        inp_mask_2random = inp_mask_2mask & (np.random.rand(*vectorized_texts.shape) < 1 / 9)
        vectorized_texts_masked[inp_mask_2random] = np.random.choice(
            random_tokens_to_select, inp_mask_2random.sum()
        )
        # y_labels would be same as encoded_texts i.e input tokens
        y_labels = np.copy(vectorized_texts)
        return vectorized_texts_masked, y_labels, sample_weights
    # end get_masked_input_and_labels

    def __call__(self, **kwargs):
        # ['wikitext-103-v1', 'wikitext-2-v1', 'wikitext-103-raw-v1', 'wikitext-2-raw-v1']
        wikitext = load_dataset('wikitext', 'wikitext-103-raw-v1')
        texts = np.array(wikitext['train']['text'][:2*self.n_sentences])
        large_sentences = []
        tmp_idx = 0
        while len(large_sentences) < self.n_sentences and tmp_idx < len(texts):
            s = texts[tmp_idx]
            tmp_idx += 1
            if len(s) > 5 and '=' not in s:
                large_sentences.append( s )
        clean_dataset = np.array([s.lower() + '.' for l in large_sentences for s in l.split('.')])
        # create tokenizer
        self.create_tokenizer( list(clean_dataset) )
        dataset = np.array(self.tokenizer_results['input_ids'])
        # shuffle and get training data
        shuffle(dataset) # no return
        train = dataset[:int(self.n_sentences * self.train_split)]
        # Prepare tokenizer for the encoder input
        enc_seq_length = self.max_sentence_length
        enc_vocab_size = self.tokenizer.vocab_size + 1
        trainX, trainY, mask_weights = self.get_masked_input_and_labels(train)
        
        return trainX, trainY, mask_weights, train, enc_seq_length, enc_vocab_size
    # end call
# end class MLMWikipediaDataset

class GPTWikipediaDataset:
    def __init__(self,**kwargs):
        super(GPTWikipediaDataset, self).__init__(**kwargs)
        self.n_sentences = 100000
        self.train_split = 0.9
        self.max_sentence_length = 50
    # end init

    def create_tokenizer(self, sentences_list):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer_results = self.tokenizer( sentences_list, padding='max_length', truncation=True , max_length=self.max_sentence_length )
    # end create_tokenizer

    # based on https://keras.io/examples/nlp/masked_language_modeling/
    def get_word_shift_input_output(self, vectorized_texts):
        trainX = vectorized_texts[:, :-1]
        trainY = vectorized_texts[:, 1:]
        return trainX, trainY
    # end get_masked_input_and_labels

    def __call__(self, **kwargs):
        # ['wikitext-103-v1', 'wikitext-2-v1', 'wikitext-103-raw-v1', 'wikitext-2-raw-v1']
        wikitext = load_dataset('wikitext', 'wikitext-103-raw-v1')
        texts = np.array(wikitext['train']['text'][:2*self.n_sentences])
        large_sentences = []
        tmp_idx = 0
        while len(large_sentences) < self.n_sentences and tmp_idx < len(texts):
            s = texts[tmp_idx]
            tmp_idx += 1
            if len(s) > 5 and '=' not in s:
                large_sentences.append( s )
        clean_dataset = np.array([s.lower() + '.' for l in large_sentences for s in l.split('.')])
        # create tokenizer
        self.create_tokenizer( list(clean_dataset) )
        dataset = np.array(self.tokenizer_results['input_ids'])
        # shuffle and get training data
        shuffle(dataset) # no return
        train = dataset[:int(self.n_sentences * self.train_split)]
        # Prepare tokenizer for the encoder input
        dec_seq_length = self.max_sentence_length
        dec_vocab_size = self.tokenizer.vocab_size + 1
        padding_token = self.tokenizer.pad_token_id
        trainX, trainY = self.get_word_shift_input_output(train)
        
        return trainX, trainY, dec_seq_length, dec_vocab_size, padding_token
    # end call
# end class GPTWikipediaDataset