from tensorflow.examples.tutorials.mnist import input_data
from mnist_tutorials import softmax, cnn

MNIST_DATA = input_data.read_data_sets("MNIST_data/", one_hot=True)

def main():
    #softmax_precision = softmax(MNIST_DATA)
    #print(softmax_precision)
    cnn(MNIST_DATA)

main()