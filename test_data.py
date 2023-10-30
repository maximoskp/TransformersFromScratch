from datatools.mlm_pretraining import MLMWikipediaDataset

mlmWikipediaDataset = MLMWikipediaDataset()

trainX, trainY, mask_weights, train, enc_seq_length, enc_vocab_size = mlmWikipediaDataset()