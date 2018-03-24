import numpy as np
import tensorflow as tf 
from sklearn.utils import shuffle


class MLP:
    """
    Implement of Multiple-Layer_Perceptron for binary classification.

    n_input  : int, number of input features
    n_hidden : list, number of neurons for each hidden layer. e.g.:[2,3] means there are two hidden layers,
               the first layer has 2 neurons, and the second layer has 3 neurons.
    n_classes: number of classes for classification
    config   : config file including setup for models

    """
    def __init__(self, n_input, n_hidden, n_classes, config):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.config = config

        # construct weights and biases
        # If there are 2 hidden layers, then self.weights looks like {'h0': tf.Variable,  'h1': tf.Variable,  'out': tf.Variable}, 
        # self.biases looks like {'b0': tf.Variable,  'b1': tf.Variable,  'out': tf.Variable}
        self.weights = {}
        self.biases = {}
        for i in range(len(self.n_hidden)):
            if i == 0:
                self.weights['h0'] = tf.Variable(tf.random_normal([self.n_input, self.n_hidden[0]]))
            else:
                self.weights['h{0}'.format(i)] = tf.Variable( tf.random_normal( [self.n_hidden[i-1], self.n_hidden[i]] ) )
            self.biases['b{0}'.format(i)] = tf.Variable( tf.random_normal( [self.n_hidden[i]] ) )
        self.weights['out'] = tf.Variable( tf.random_normal( [self.n_hidden[-1], self.n_classes] ) )
        self.biases['out'] = tf.Variable( tf.random_normal( [self.n_classes] ) )

        # build model
        self.build_model()
        # init saver
        self.saver = tf.train.Saver()


    def build_model(self):
        # define placeholder
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # construct hidden layers
        self.hidden_layers = {}
        for i in range(len(self.n_hidden)):
            # layer_0 = x * W_0 + b_0
            if i == 0:
                self.hidden_layers['layer0'] = tf.add( tf.matmul( self.x, self.weights['h0'] ), self.biases['b0'] )
            # layer_i = layer_(i-1) * W_i + b_i
            else:
                self.hidden_layers['layer{0}'.format(i)] = tf.add( tf.matmul( self.hidden_layers['layer{0}'.format(i-1)], self.weights['h{0}'.format(i)] ), self.biases['b{0}'.format(i)] )
            # layer_i = relu(layer_i)
            self.hidden_layers['layer{0}'.format(i)] = tf.nn.relu( self.hidden_layers['layer{0}'.format(i)] )
        # layer_out = layer_(n-1) * W_out + b_out
        self.out_layer = tf.matmul( self.hidden_layers['layer{0}'.format(len(self.n_hidden)-1)],
                                    self.weights['out'] ) + self.biases['out']

        # define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out_layer, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate']).minimize(self.cost)
        self.init = tf.global_variables_initializer()
        return


    def train(self, X, Y):
        with tf.Session() as sess:
            # init
            sess.run(self.init)
            # train
            for epoch in range(self.config['training_epochs']):
                avg_cost = 0.
                total_batch = int(len(X) / self.config['batch_size'])
                X_batches = np.array_split(X, total_batch)
                Y_batches = np.array_split(Y, total_batch)
                # loop over batches for training
                for i in range(total_batch):
                    batch_x, batch_y = X_batches[i], Y_batches[i]
                    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x,
                                                                  self.y: batch_y})
                    avg_cost += c / total_batch
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            # save model
            self.saver.save(sess, '../models/{0}/mlp'.format(self.config['model_name']))
        return


    def evaluate(self, X, Y):
        with tf.Session() as sess:
            # restore trained model
            saver = tf.train.import_meta_graph( '../models/{0}/mlp.meta'.format( self.config['model_name'] ) )
            saver.restore( sess, tf.train.latest_checkpoint( '../models/{0}'.format( self.config['model_name'] ) ) )

            # calculate accuracy
            y_pred_prob = tf.nn.softmax(self.out_layer)
            correct_prediction = tf.equal(tf.argmax(y_pred_prob, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({self.x: X, self.y: Y}))
        return





def test():
    # generate data
    X_s = np.random.normal(loc=[0., 0.], scale=[1., 1.], size=[10000, 2])
    X_b = np.random.normal(loc=[10., 10.], scale=[1., 1.], size=[10000, 2])
    y_s = np.zeros(10000) + 1
    y_b = np.zeros(10000)
    X = np.concatenate([X_s, X_b])
    y = np.concatenate([y_s, y_b])
    Y = np.array([y, -(y-1)]).T
    X, Y = shuffle(X, Y)

    # define config json and parameters
    config = {'model_name': 'mlp_0321',
              'learning_rate': 0.001,
              'training_epochs': 10,
              'batch_size': 100}
    n_input = 2
    n_hidden = [3, 3]
    n_classes = 2

    # train model
    mlp = MLP(n_input, n_hidden, n_classes, config)
    mlp.train(X, Y)
    # test model
    mlp.evaluate(X, Y)

    return


if __name__ == '__main__':
    #test()
    import sys
    import time
    from os import path
    lib_path = path.dirname(path.dirname(path.abspath(__file__)))
    sys.path.append(lib_path)
    from preprocess.PreProcess import PreProcess, load_data
    from util.resampling import binary_downsampling, binary_upsampling

    print 'Loading data and preprocessing ......'
    start_time = time.time()
    process_path = {'imputer': '../preprocess/imputer.pkl', 'scaler': '../preprocess/scaler.pkl', 'encoder': '../preprocess/encoder.pkl'}
    X, y, w = load_data('../data/train.csv', process_path, fit=False)
    # resample
    X_train, y_train = binary_upsampling(X, y)
    Y_train = np.array([y_train, -(y_train-1)]).T
    end = time.time()
    print 'Done. Took {} seconds.'.format(end - start_time)
    print

    print 'Training ......'
    start = time.time()
    # train
    config = {'model_name': 'mlp_0321',
              'learning_rate': 0.001,
              'training_epochs': 10,
              'batch_size': 100}
    mlp = MLP(X_train.shape[1], [100, 100], 2, config)
    mlp.train(X_train, Y_train)
    end = time.time()
    print 'Done. Took {} seconds.'.format(end - start)
    print

    mlp.evaluate(X_train, Y_train)
    end = time.time()
    print 'Complete. Took total of {} seconds.'.format(end - start)
    print

