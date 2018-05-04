import sys
from os import path
lib_path = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(lib_path)
from util.metric import estimate_metric

import numpy as np
import os
import tensorflow as tf 
from sklearn.utils import shuffle


class EarlyStoppingHandler:
    """
    A handler to deal with early stopping criterion.
    Run update() every evaluation batch, and get stop() to indicate whether to stop training.

    """
    def __init__(self, tole_step=1000):
        self.curr_metric = 0.
        self.best_metric = 0.001

        self.tole_step = tole_step
        self.step = 0

    def update(self, curr):
        self.curr_metric = curr
        if self.best_metric < self.curr_metric:
            self.best_metric = self.curr_metric
            self.step = 0
        else:
            self.step = self.step + 1

    def stop(self):
        return (self.step > self.tole_step)


class MLP:
    """
    Implement of Multiple-Layer-Perceptron for binary classification.

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
        # If there are 2 hidden layers, then self.weights looks like 
        # {'h0': tf.Variable,  'h1': tf.Variable,  'out': tf.Variable}, 
        # self.biases looks like {'b0': tf.Variable,  'b1': tf.Variable,  'out': tf.Variable}
        tf.reset_default_graph()
        self.weights = {}
        self.biases = {}
        for i in range(len(self.n_hidden)):
            if i == 0:
                self.weights['h0'] = tf.get_variable('h0', shape=[self.n_input, self.n_hidden[0]],
                                                     initializer=self.weight_initializer(self.config['weight_init']) )
            else:
                self.weights['h{0}'.format(i)] = tf.get_variable('h{0}'.format(i), 
                                                    shape=[self.n_hidden[i-1], self.n_hidden[i]], 
                                                    initializer=self.weight_initializer(self.config['weight_init']) )
            self.biases['b{0}'.format(i)] = tf.get_variable('b{0}'.format(i), 
                                                    shape=[self.n_hidden[i]], 
                                                    initializer=self.weight_initializer(self.config['bias_init']) )
        self.weights['out'] = tf.get_variable('wout', shape=[self.n_hidden[-1], self.n_classes], 
                                              initializer=self.weight_initializer(self.config['weight_init']) )
        self.biases['out'] = tf.get_variable('bout', shape=[self.n_classes], 
                                             initializer=self.weight_initializer(self.config['bias_init']) )

        # build model
        self.build_model()
        # build visualizer tensorboard
        self.build_tensorboard()
        # init saver
        self.saver = tf.train.Saver()
        self.save_path = '../models/{0}/mlp'.format(self.config['model_name'])


    def weight_initializer(self, method):
        if method == 'he':
            return tf.keras.initializers.he_normal()
        elif method == 'xavier':
            return tf.contrib.layers.xavier_initializer()
        else:
            return tf.truncated_normal_initializer()


    def build_model(self):
        # define placeholder
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])
        self.is_training = tf.placeholder("bool", shape=None)

        # construct hidden layers
        self.hidden_layers = {}
        for i in range(len(self.n_hidden)):
            if i == 0:
                self.hidden_layers['layer0'] = tf.add( tf.matmul( self.x, self.weights['h0'] ), self.biases['b0'] )
            else:
                self.hidden_layers['layer{0}'.format(i)] = tf.add( tf.matmul( self.hidden_layers['layer{0}'.format(i-1)], 
                                                                   self.weights['h{0}'.format(i)] ), self.biases['b{0}'.format(i)] )
            # batch normalization
            self.hidden_layers['layer{0}'.format(i)] = tf.keras.layers.BatchNormalization()(self.hidden_layers['layer{0}'.format(i)],
                                                                                            training=self.is_training)
            self.hidden_layers['layer{0}'.format(i)] = tf.nn.relu( self.hidden_layers['layer{0}'.format(i)] )
        self.out_layer = tf.matmul( self.hidden_layers['layer{0}'.format(len(self.n_hidden)-1)],
                                    self.weights['out'] ) + self.biases['out']

        # define metrics, loss and optimizer
        self.y_pred = tf.nn.softmax(self.out_layer)
        self.train_auc, self.train_op = tf.metrics.auc(self.y[:,0], self.y_pred[:,0])
        self.validate_auc, self.validate_op = tf.metrics.auc(self.y[:,0], self.y_pred[:,0])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out_layer, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate']).minimize(self.cost)
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        return


    def build_tensorboard(self):
        # design tensorboard
        self.train_merged_summary = []
        self.validate_merged_summary = []
        grads = tf.gradients(self.cost, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))

        with tf.name_scope("train"):
            self.train_merged_summary.append( tf.summary.scalar("loss", self.cost) )
            self.train_merged_summary.append( tf.summary.scalar("auc", self.train_auc) )
            for grad, var in grads:
                print var.name
                self.train_merged_summary.append( tf.summary.histogram(var.name.replace(':','_')+'/gradient', grad) )
            
        with tf.name_scope("validate"):
            self.validate_merged_summary.append( tf.summary.scalar("loss", self.cost) )
            self.validate_merged_summary.append( tf.summary.scalar("auc", self.validate_auc) )
            for grad, var in grads:
                self.validate_merged_summary.append( tf.summary.histogram(var.name.replace(':','_')+'/gradient', grad) )
        self.train_merged_summary = tf.summary.merge(self.train_merged_summary)
        self.validate_merged_summary = tf.summary.merge(self.validate_merged_summary)
        return


    def train(self, X, Y, X_validate, Y_validate):
        stop_handler = EarlyStoppingHandler(tole_step=self.config['early_stopping'])
        with tf.Session() as sess:
            # init
            sess.run(self.init)
            train_writer = tf.summary.FileWriter(self.save_path + '/train', graph=tf.get_default_graph()) 
            validate_writer = tf.summary.FileWriter(self.save_path + '/validate')
            # train
            epoch = 0
            while epoch < self.config['training_epochs']:
                epoch = epoch + 1
                # prepare batches
                total_batch = int(len(X) / self.config['batch_size'])
                X_batches = np.array_split(X, total_batch)
                Y_batches = np.array_split(Y, total_batch)
                # batch training
                for i in range(total_batch):
                    batch_x, batch_y = X_batches[i], Y_batches[i]
                    _, c = sess.run([self.optimizer, self.cost],
                                    feed_dict={self.x: batch_x,
                                               self.y: batch_y,
                                               self.is_training: True})
                     
                    # update metrics every 100 batches
                    if i % 100 == 0:
                        summary, train_auc_, _ = sess.run([self.train_merged_summary, self.train_auc, self.train_op],
                                                          feed_dict={self.x: batch_x,
                                                                     self.y: batch_y,
                                                                     self.is_training: True})
                        train_writer.add_summary(summary, epoch * total_batch + i)
                        summary, validate_auc_, _ = sess.run([self.validate_merged_summary, self.validate_auc, self.validate_op], 
                                                             feed_dict={self.x: X_validate,
                                                                        self.y: Y_validate,
                                                                        self.is_training: False})
                        validate_writer.add_summary(summary, epoch * total_batch + i)
                        
                        print validate_auc_
                        stop_handler.update(validate_auc_)
                        # early stopping
                        if(stop_handler.stop()):
                            self.saver.save(sess, self.save_path)
                            return
            # save model
            self.saver.save(sess, self.save_path)
        return


    def evaluate(self, X, Y):
        with tf.Session() as sess:
            # restore trained model
            saver = tf.train.import_meta_graph( self.save_path + '.meta' )
            saver.restore( sess, tf.train.latest_checkpoint('../models/mlp') )

            # calculate accuracy
            y_pred_prob = tf.nn.softmax(self.out_layer)
            correct_prediction = tf.equal(tf.argmax(y_pred_prob, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            y_pred = y_pred_prob.eval({self.x: X, self.y: Y, self.is_training: False})
        return y_pred
            





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
    mlp.train(X, Y, X, Y)
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
    from util.metric import estimate_metric
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print 'Loading data and preprocessing ......'
    start_time = time.time()
    process_path = {'imputer': '../preprocess/imputer.pkl', 'scaler': '../preprocess/scaler.pkl', 'encoder': '../preprocess/encoder.pkl'}
    X, y, w = load_data('../data/train.csv', process_path, fit=False)
    X_validate, y_validate, w_validate = load_data('../data/validate.csv', process_path, fit=False)
    # resample
    X_train, y_train, w_train = binary_upsampling(X, y, w)
    Y_train = np.array([y_train, -(y_train-1)]).T
    Y_validate = np.array([y_validate, -(y_validate-1)]).T
    end = time.time()
    print 'Input Shapes : '
    print 'X_train: {0}    Y_train: {1}    w_train: {2}'.format(X_train.shape, Y_train.shape, w_train.shape)
    print 'X_validate: {0}    Y_validate: {1}    w_validate: {2}'.format(X_validate.shape, Y_validate.shape, w_validate.shape)
    print 'Done. Took {} seconds.'.format(end - start_time)
    print

    print 'Training ......'
    start = time.time()
    # train
    config = {'model_name': 'mlp_0427',
              'learning_rate': 0.001,
              'training_epochs': 20,
              'batch_size': 16,
              'weight_init': 'normal',
              'bias_init': 'normal',
              'early_stopping': 0.001}

    mlp = MLP(X_train.shape[1], [10, 10], 2, config)
    mlp.train(X_train, Y_train, X_validate, Y_validate)
    end = time.time()
    print 'Done. Took {} seconds.'.format(end - start)
    print
    
    print 'Evaluating ......'
    y_pred_train = mlp.evaluate(X_train, Y_train)[:,0]
    result_train = estimate_metric(y_pred_train, y_train)
    end = time.time()
    print result_train
    print
    
    X_validate, y_validate, w_validate = load_data('../data/validate.csv', process_path, fit=False)
    Y_validate = np.array([y_validate, -(y_validate-1)]).T
    y_pred_validate = mlp.evaluate(X_validate, Y_validate)[:,0]
    result_validate = estimate_metric(y_pred_validate, y_validate)
    print result_validate
    print
    print 'Complete. Took total of {} seconds.'.format(end - start)
    print

