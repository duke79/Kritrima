import logging
import re
import sys

import numpy
import os

# Meta settings
from src import output_dir

TRAIN_OVER_TEST = True
EPOCH = 100
BATCH = 256
SEQUENCE_LENGTH = 200
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


def get_text():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(os.curdir)), "data")
    alices_adventures = os.path.join(data_dir,
                                     "Alice\'s Adventures in Wonderland, by Lewis Carroll - gutenberg.org.txt")

    log.info("Fetching the text from " + alices_adventures)
    raw_text = open(alices_adventures, encoding="utf-8").read()
    raw_text = raw_text.lower()
    return raw_text


def prepare_data(text):
    # Create character to integer dictionary
    log.info("Preparing data...")
    chars = sorted(list(set(text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # Create patterns (dataX), each a 100 characters long slice of text
    # dataY is the 101th character for each pattern
    len_text = len(text)
    len_vocab = len(chars)

    dataX = []
    dataY = []
    for i in range(0, len_text - SEQUENCE_LENGTH, 1):
        seq_in = text[i:i + SEQUENCE_LENGTH]
        seq_out = text[i + SEQUENCE_LENGTH]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    len_patterns = len(dataX)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (len_patterns, SEQUENCE_LENGTH, 1))
    # pyplot.imshow(X)

    # normalize
    X = X / float(len_vocab)
    # one-hot encode the output variable
    Y = np_utils.to_categorical(dataY)
    return X, Y, dataX, dataY, char_to_int, int_to_char, len_vocab


def create_model(x, y):
    # define the LSTM model
    log.info("Creating model...")
    model = Sequential()
    model.add(LSTM(256, input_shape=(tuple(x.shape[1:])), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def train(x, y):
    model = create_model(x, y)
    log.info("Checking GPU...")
    sess = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))
    log.info("Training model...")
    # define the checkpoint
    filepath = os.path.join(output_dir, MODEL_FILE_PREFIX + "-{epoch:02d}-{loss:.4f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(x, y, epochs=EPOCH, batch_size=BATCH, callbacks=callbacks_list)


def test(x, y, data_x, data_y):
    model = create_model(x, y)
    log.info("Loading model weights...")
    p = re.compile(MODEL_FILE_PREFIX + '-([0-9]*)-([0-9]*\.[0-9]*)\.hdf5', re.IGNORECASE)
    model_file = None
    last_loss = 9999999999999.9
    files = os.listdir(os.path.join(output_dir, os.curdir))
    for file in files:
        file_prefix = file[:len(MODEL_FILE_PREFIX) + 1]
        if file_prefix == MODEL_FILE_PREFIX + "-":
            m = p.match(file)
            loss = float(m.group(2))
            if not model_file or loss < last_loss:
                last_loss = loss
                model_file = file
    model.load_weights(os.path.join(output_dir, model_file))

    log.info("Testing generation...")
    # pick a random seed
    start = numpy.random.randint(0, len(data_x) - 1)
    pattern = data_x[start]
    log.info("Seed:")
    log.info("\"" + ''.join([int_to_char[value] for value in pattern]) + "\"")
    # generate characters
    for i in range(GENERATED_LENGTH):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        srt = prediction.argsort()
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")


raw_text = get_text()
X, Y, data_X, data_Y, char_to_int, int_to_char, len_vocab = prepare_data(raw_text)
if TRAIN_OVER_TEST:
    train(X, Y)
else:
    test(X, Y, data_X, data_Y)
