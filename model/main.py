import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import argparse
import sys
from keras import backend as K
#print K.tensorflow_backend._get_available_gpus()

class main:
    def __init__(self,file_name):
        """
        :param file_name: article sent for training
        """
        self.file_name=file_name

    def preprocessing(self):
        """
        preprocess the file
        :return: X, y
        """
        raw_text=open(self.file_name).read()
        raw_text=raw_text.lower()
        chars=sorted(list(set(raw_text)))
        char_to_int = dict((c, i) for i, c in enumerate(chars)) # one hot encoding embedding, can replace it with glove/word2vec embeddings also
        n_chars = len(raw_text)
        n_vocab = len(chars)
        print "Total Characters: ", n_chars
        print "Total Vocab: ", n_vocab
        seq_length = 100
        dataX = []
        dataY = []
        for i in range(0, n_chars - seq_length, 1):
            seq_in = raw_text[i:i + seq_length]
            seq_out = raw_text[i + seq_length]
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])
        n_patterns = len(dataX)
        print "Total Patterns: ", n_patterns
        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
        # normalize
        X = X / float(n_vocab)
        # one hot encode the output variable
        y = np_utils.to_categorical(dataY)

        return X,y

    def models(self,X,y):
        """
        :return: model
        """
        # define the LSTM model
        model = Sequential()
        model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def fit(self,filepath="./weights/weights.hdf5",epochs=20,batch_size=128):
        X,y=self.preprocessing()
        model=self.models(X,y)
        #filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

    def generate(self,weightfile="./weights/weights.hdf5",number=100):
        raw_text = open(self.file_name).read()
        raw_text = raw_text.lower()
        # create mapping of unique chars to integers, and a reverse mapping
        chars = sorted(list(set(raw_text)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        int_to_char = dict((i, c) for i, c in enumerate(chars))
        # summarize the loaded data
        n_chars = len(raw_text)
        n_vocab = len(chars)
        # prepare the dataset of input to output pairs encoded as integers
        seq_length = 100
        dataX = []
        dataY = []
        for i in range(0, n_chars - seq_length, 1):
            seq_in = raw_text[i:i + seq_length]
            seq_out = raw_text[i + seq_length]
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])
        n_patterns = len(dataX)
        print "Total Patterns: ", n_patterns
        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
        # normalize
        X = X / float(n_vocab)
        # one hot encode the output variable
        y = np_utils.to_categorical(dataY)
        # define the LSTM model
        model = self.models(X,y)
        model.load_weights(weightfile)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # pick a random seed
        start = numpy.random.randint(0, len(dataX) - 1)
        pattern = dataX[start]
        print "Seed:"
        print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
        # generate characters
        for i in range(number):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(n_vocab)
            prediction = model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_char[index]
            seq_in = [int_to_char[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        print "\nDone."



if __name__ == '__main__':
    #filename = raw_input('Enter a file name: ')
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="./data/pride_prejudice.txt",
                        help='Pass file name for training purpose')
    parser.add_argument('--train_flag', type=bool, default=False,
                        help='Set Flag to True for Training Purpose')
    
    parser.add_argument('--epochs', type=int, default=50, help='Set nunber of epochs Training Purpose')

    parser.add_argument('--test_flag', type=bool, default=True, help='Set Flag to True for Testing Purpose')
    
    parser.add_argument('--number_random', type=int, default=10000, help='Number of random characters')
    args = parser.parse_args()

    filename=args.filename
    m=main(filename)
    epochs=args.epochs
    if(args.train_flag):
        m.fit(epochs=epochs)
    if(args.test_flag):
        m.generate("./weights/weights.hdf5",number=args.number_random) # number to generate random words


