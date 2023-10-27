from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64, sqrt
from keras.losses import sparse_categorical_crossentropy
from transformer.models import TransformerModel
from datatools.text import PrepareDataset
from time import time
import numpy as np

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack

# Define the training parameters
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    # end init

    def __call__(self, step_num):
        arg1 = 1./sqrt(cast(step_num, float32))
        arg2 = cast(step_num, float32)/(self.warmup_steps*sqrt(cast(step_num, float32)))
        return (self.d_model ** -0.5)*math.minimum(arg1, arg2)
    # end call
# end class LRScheduler

optimizer = Adam( LRScheduler(d_model), beta_1, beta_2, epsilon )

dataset = PrepareDataset()
trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('data/english-german-both.pkl')
# Prepare the dataset batches
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

# Create model
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

def loss_fcn(target, prediction):
    # padded values not included in loss computation
    padding_mask = math.logical_not( equal(target, 0) )
    padding_mask = cast(padding_mask, float32)
    # compute loss on masked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True)*padding_mask
    return reduce_sum(loss)/reduce_sum(padding_mask)
# end loss_fcn

def accuracy_fcn(target, prediction):
    # padded values not included in loss computation
    padding_mask = math.logical_not( equal(target, 0) )
    # find equals and apply mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)
    # cast results to 32bit floats
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)
    # compute accuracy on masked values
    return reduce_sum(accuracy)/reduce_sum(padding_mask)
# end accuracy_fcn

# include metrics monitoring
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')

# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

# speeding up the training process with eager execution of the training step
# @function
def train_step(encoder_input, decoder_input, decoder_output):
    print('--'*20)
    with GradientTape() as tape:
        # run prediction
        prediction = training_model(encoder_input, decoder_input, training=True)
        print('training_model.trainable_weights has nan 0:', np.isnan(training_model.trainable_weights[0]).any())
        print('prediction has nan:', np.isnan(prediction).any())
        # compute loss and accuracy
        loss = loss_fcn(decoder_output, prediction)
        print('loss has nan:', np.isnan(loss).any())
        accuracy = accuracy_fcn(decoder_output, prediction)
    # get gradient
    gradients = tape.gradient(loss, training_model.trainable_weights)
    print('gradients has nan:', np.isnan(gradients[0]).any())
    # print('gradients has nan:', math.is_nan(gradients).any())
    # update trainable parameters
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))
    # print('training_model.trainable_weights has nan 2:', type(training_model.trainable_weights[0]))
    print('training_model.trainable_weights has nan 2:', np.isnan(training_model.trainable_weights[0]).any())
    # update mean values
    train_loss(loss)
    train_accuracy(accuracy)
    # end context manager
# end train_step

for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    print("\nStart of epoch %d" % (epoch + 1))
    start_time = time()
    for step, (train_batchX, train_batchY) in enumerate(train_dataset):
        # define encoder/decoder inputs/outputs
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]
        train_step(encoder_input, decoder_input, decoder_output)
        if step % 2 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    # end for step
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))
    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))
print("Total time taken: %.2fs" % (time() - start_time))