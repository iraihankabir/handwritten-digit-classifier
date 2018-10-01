from mlxtend.data import loadlocal_mnist
import numpy as np


class Dataset:

    def __init__(self, input_size, batch_size, length, test_length):
        self.size       = input_size
        self.batch_size = batch_size
        self.length     = length

        self.train_data = None
        self.test_data  = None
        self.train_label= None
        self.test_label = None

        # epoch controlling variables
        self.current_indx = 0
        self.current_epoch = 0
        self.total_batch_per_epoch = int(self.length/self.batch_size)
        self.test_data_count = test_length

        # Set datasets
        self.set_train_dataset()
        self.set_test_dataset()

    def set_train_dataset(self):
        # Data source
        image  = 'data/train-images.idx3-ubyte'
        label  = 'data/train-labels.idx1-ubyte'
        x, y = loadlocal_mnist(
            images_path=image, 
            labels_path=label
        )
        self.train_data = x
        self.train_label = np.array([self.get_label(l) for l in y])

    def set_test_dataset(self):
        image = 'data/t10k-images.idx3-ubyte'
        label = 'data/t10k-labels.idx1-ubyte'
        x, y = loadlocal_mnist(
            images_path=image, 
            labels_path=label
        )
        self.test_data  = x
        self.test_label = [np.array(self.get_label(l)) for l in y]

    @property
    def get_next(self):
        if self.current_indx+1 >= self.total_batch_per_epoch:
            self.current_indx = 0
            self.current_epoch += 1
        batch_input = self.train_data[self.current_indx:(self.current_indx+self.batch_size)]
        batch_label = self.train_label[self.current_indx:(self.current_indx+self.batch_size)]
        self.current_indx += 1

        return [batch_input, batch_label]

    def shuffle_data(self, data, label):
        combind_data = list(zip(data, label))
        random.shuffle(combind_data)
        _x, _y = list(zip(*combind_data))
        _x = np.array(x)
        _y = np.array(y)
        return [_x, _y]

    @staticmethod
    def get_label(y):
        arr = np.array(np.zeros((10, 1)), dtype=np.int32)
        arr[y]=1
        return arr
