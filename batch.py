#import scipy.io as sio
from hyper_parameters import *
from scipy import signal
import numpy as np
import csv
import random as rd

# Set the path to directory "data" containing .csv files
data_path = '../../data/'

'''Call this function in main to create train and test .csv files from heartbeats_data.csv'''
def shuffle_csv(num_test):
    data_loca = data_path + 'heartbeats_data.csv'
    line_list = open(data_loca).readlines()
    rd.shuffle(line_list)
    num_lines = len(line_list)
    train_lines = line_list[:num_lines - num_test]
    test_lines = line_list[num_lines - num_test:]

    write_file = open(data_path+'train_data.csv', 'wb')
    for line in train_lines:
        write_file.write(line)
    write_file.close()
    print "train_data.csv created!"
    write_file = open(data_path+'test_data.csv', 'wb')
    for line in test_lines:
        write_file.write(line)
    write_file.close()
    print "test_data.csv created!"

class Batch():
    def __init__(self):
        self.train_offset = 0
        self.test_offset = 0
        self.train_x, self.train_y, self.train_names = \
                                   self.load_table(data_path+"train_data.csv")
        self.train_size = np.shape(self.train_x)[0]


        self.test_x, self.test_y, self.test_names = \
                                   self.load_table(data_path+"test_data.csv")
        self.test_size = np.shape(self.test_x)[0]

    def load_table(self, file_path):
        csv_file = open(file_path)
        csv_reader = csv.reader(csv_file)
        names = []
        x = []
        y = []
        plot = 0
        for row in csv_reader:
            names.append(row[0])
            f, t, Sxx =signal.spectrogram(np.asarray(map(float, row[2:])),nperseg=64,noverlap = 32)
            if plot == 0:
                shape = np.shape(Sxx)
                plot+=1
            shape1 = np.shape(Sxx)
            if np.array_equal(np.array([33,280]),shape1):
                x.append(Sxx)
                y.append(self.class_to_onehot(row[1]))

        x = np.array(x)
        y = np.array(y)
        x = np.expand_dims(x, axis=3)
        y = np.squeeze(y)
        return x, y, names

    def class_to_onehot(self, symbol):
        onehot = np.zeros(1)
        if symbol == 'N':
            onehot[0] = 0
        elif symbol == 'A':
            onehot[0] = 1
        elif symbol == 'O':
            onehot[0] = 2
        elif symbol == '~':
            onehot[0] = 3
        else:
            print "unknown symbol: "+symbol
        return onehot

    def minibatch(self,is_Train):
        if is_Train:
            indexes = np.random.choice(self.train_size,FLAGS.train_batch_size)
            return self.train_x[indexes],self.train_y[indexes]
        else:
            indexes = np.random.choice(self.test_size,FLAGS.test_batch_size)
            return self.test_x[indexes],self.test_y[indexes]
