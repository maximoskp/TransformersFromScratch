from datatools.mlm_pretraining import MLMWikipediaDataset

mlmWikipediaDataset = MLMWikipediaDataset()

trainX, trainY, mask_weights, train, enc_seq_length, enc_vocab_size = mlmWikipediaDataset()

print('trainX:', trainX.shape)
print('trainY:', trainY.shape)
print('mask_weights:', mask_weights.shape)