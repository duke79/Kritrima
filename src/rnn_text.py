import logging
import re
import sys

import numpy
import os

# Meta settings
TRAIN_OVER_TEST = True
EPOCH = 50
BATCH = 64
GENERATED_LENGTH = 50
MODEL_FILE_PREFIX = os.path.basename(__file__)

# Create logger
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)  # 'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'
log = logging.getLogger(__name__)

# Load text data
log.info("Loading Keras...")
import tensorflow
from keras import Sequential
from keras.utils import np_utils

data_dir = os.path.join(os.path.dirname(os.path.abspath(os.curdir)), "data")
alices_adventures = os.path.join(data_dir, "Alice\'s Adventures in Wonderland, by Lewis Carroll - gutenberg.org.txt")

log.info("Fetching the text from " + alices_adventures)
raw_text = open(alices_adventures, encoding="utf-8").read()
raw_text = raw_text.lower()

# Create character to integer dictionary
log.info("Preparing data...")
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Create patterns (dataX), each a 100 characters long slice of text
# dataY is the 101th character for each pattern
len_text = len(raw_text)
len_vocab = len(chars)

seq_length = 100
dataX = []
dataY = []
for i in range(0, len_text - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

len_patterns = len(dataX)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len_patterns, seq_length, 1))
# pyplot.imshow(X)

# normalize
X = X / float(len_vocab)
# one-hot encode the output variable
Y = np_utils.to_categorical(dataY)

# define the LSTM model
log.info("Creating model...")
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


def train():
    log.info("Checking GPU...")
    sess = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))
    log.info("Training model...")
    # define the checkpoint
    filepath = MODEL_FILE_PREFIX + "-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, Y, epochs=EPOCH, batch_size=BATCH, callbacks=callbacks_list)


def test():
    log.info("Loading model weights...")
    p = re.compile(MODEL_FILE_PREFIX + '-([0-9]*)-([0-9]*\.[0-9]*)\.hdf5', re.IGNORECASE)
    model_file = None
    last_loss = 9999999999999.9
    files = os.listdir(os.curdir)
    for file in files:
        if file[:19] == MODEL_FILE_PREFIX:
            m = p.match(file)
            loss = float(m.group(2))
            if not model_file or loss < last_loss:
                last_loss = loss
                model_file = file
    model.load_weights(model_file)

    log.info("Testing generation...")
    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    log.info("Seed:")
    log.info("\"" + ''.join([int_to_char[value] for value in pattern]) + "\"")
    # generate characters
    for i in range(GENERATED_LENGTH):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")


if TRAIN_OVER_TEST:
    train()
else:
    test()
