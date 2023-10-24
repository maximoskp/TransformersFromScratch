from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from keras.losses import sparse_categorical_crossentropy
from model import TransformerModel
from prepare_dataset import PrepareDataset
from time import time
 
 
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
        arg1 = step_num ** -0.5
        arg2 = step_num*(self.warmup_steps ** -1.5)
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