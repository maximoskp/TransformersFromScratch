from keras.losses import sparse_categorical_crossentropy
import pickle

with open('debug/model_true_output.pickle', 'rb') as handle:
    d = pickle.load(handle)

true_output = d['true_output']
model_output = d['model_output']

loss = sparse_categorical_crossentropy(true_output, model_output, from_logits=True)

print('true_output: ', true_output)
print('model_output: ', model_output)
print('loss: ', loss)