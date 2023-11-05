from tensorflow.keras.optimizers.legacy import Adam # faster on M1 macs?
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, bool, int32, int64, GradientTape, TensorSpec, function, int64, sqrt
from keras.losses import sparse_categorical_crossentropy
from transformer.models import GPTDecoderWrapper, LockingDecoderModel
from datatools.text_pretraining import GPTWikipediaDataset
from time import time

# Define the model parameters
h = 1  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 2  # Number of layers in the encoder stack

# Define the training parameters
epochs = 30
batch_size = 16
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

# prepare data
gptWikipediaDataset = GPTWikipediaDataset()
trainX, trainY, dec_seq_length, dec_vocab_size, padding_token = gptWikipediaDataset()

train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)
        self.d_model = cast(d_model, float32)
        self.warmup_steps = cast(warmup_steps, float32)
    # end init

    def __call__(self, step_num):
        step_num = cast(step_num, float32)
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5)*math.minimum(arg1, arg2)
    # end call
# end class LRScheduler

optimizer = Adam( LRScheduler(d_model), beta_1, beta_2, epsilon )

# Create models
decoder = LockingDecoderModel(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
training_model = GPTDecoderWrapper(decoder)

# for step, (train_batchX, train_batchY, weights_batch) in enumerate(train_dataset):
#     print('train_batchX.shape', train_batchX.shape)
#     print('train_batchY.shape', train_batchY.shape)
#     print('weights_batch.shape', weights_batch.shape)
#     prediction = training_model(train_batchX, training=True)
#     print('prediction.shape', prediction.shape)

def loss_fcn(model_output, true_output, padding_token):
    # padded values not included in loss computation
    padding_mask = math.logical_not( equal(true_output, padding_token) )
    padding_mask = cast(padding_mask, float32)
    loss = sparse_categorical_crossentropy(true_output, model_output, from_logits=True)*padding_mask
    return reduce_sum(loss)/reduce_sum(padding_mask)
# end loss_fcn

def accuracy_fcn(model_output, true_output, padding_token):
    # padded values not included in loss computation
    padding_mask = math.logical_not( equal(true_output, padding_token) )
    # find equals and apply mask
    accuracy = equal(true_output, cast(argmax(model_output, axis=2), int64))
    accuracy = math.logical_and(padding_mask, accuracy)
    # cast results to 32bit floats
    padding_mask = cast(padding_mask, float32)
    # cast results to 32bit floats
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
def train_step(train_X, train_Y, padding_token):
    with GradientTape() as tape:
        # run prediction
        prediction = training_model(train_X, None, None, training=True, lock=True)
        # compute loss and accuracy
        loss = loss_fcn(prediction, train_Y, padding_token)
        accuracy = accuracy_fcn(prediction, train_Y, padding_token)
    # get gradient
    gradients = tape.gradient(loss, training_model.trainable_weights)
    # update trainable parameters
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))
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
        masked_input, unmasked_output = train_batchX, train_batchY
        train_step(masked_input, unmasked_output, padding_token)
        if step % 10 == 0:
            print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    # end for step
    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))
    # Save a checkpoint after every five epochs
    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))
print("Total time taken: %.2fs" % (time() - start_time))