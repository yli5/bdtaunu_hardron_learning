import os
import numpy as np
import sys
import time
import pickle

from os import path
lib_path = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(lib_path)
from preprocess.PreProcess import PreProcess, load_data
from util.resampling import binary_downsampling, binary_upsampling
from util.metric import estimate_metric

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.INFO)

# Global variables for hyperparameters
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 2
EVAL_FREQUENCY = 5000  # Number of steps between evaluations.

def get_data(data, fit=True):
    # Preprocessor
    process_path = {'imputer': '../preprocess/imputer.pkl', 'scaler': '../preprocess/scaler.pkl', 'encoder': '../preprocess/encoder.pkl'}
    processor = PreProcess()
    X, y, w = load_data(data, process_path, fit=fit)
    if fit:
        # Resample to balance labels
        X, Y, _ = binary_upsampling(X, y, w)
    else:
        Y = y
    # Return labels as 1D array
    assert X.shape[0] == Y.shape[0]
    return X, Y.reshape((len(Y),1))

class RegularizedLinearClassifier:
    def __init__(self, sess, n_features, n_classes=2, config=None):
        self.sess = sess
        self.n_features = n_features
        self.n_classes = n_classes
        if not config:
            self.config = {'model_name': 'logre',
                           'learning_rate': 0.2,
                           'training_epochs': 200,
                           'l1_reg': 0,
                           'l2_reg': 0,
                           'save_history': True}
        else:
            self.config = config

        self.build_model()
        self.saver = tf.train.Saver()

        # History for plotting
        self.history = {'loss': [],
            'norm_of_weights': [],
            'gradient_norm': [],
            'learning_rate': [],
            'result_train': [],
            'result_val': []}

    def build_model(self):
        self.features = tf.placeholder(
            shape=[None, self.n_features],
            dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.weights = tf.Variable(tf.random_normal([self.n_features, 1], 
                                                    stddev=0.1,
                                                    seed=SEED))
        self.bias = tf.Variable(tf.zeros([1, 1]))

        self.logits = tf.add(tf.matmul(self.features, self.weights), self.bias)
        self.regularization = (self.config['l1_reg'] 
                               * tf.reduce_mean(tf.abs(self.weights)) 
                               + self.config['l2_reg'] 
                               * tf.nn.l2_loss(self.weights))
        self.unregularized_loss = (tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.logits, labels = self.labels))) 
        self.loss = self.unregularized_loss + self.regularization

        # Set up learning rate decay (step decay)
        batch = tf.Variable(0, dtype=tf.float32)
        self.learning_rate = tf.train.exponential_decay(
            self.config['learning_rate'], # Base learning rate.
            batch * BATCH_SIZE,           # Current index into the dataset.
            300000,                       # Decay step.
            0.8,                          # Decay rate.
            staircase=True)

        # Define optimization algorithm
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                    0.5)
        #self.optimizer = tf.train.GradientDescentOptimizer(
        #    config['learning_rate'])
        #self.optimizer = tf.train.AdagradOptimizer(
        #    config['learning_rate'])
        self.gradient = self.optimizer.compute_gradients(self.loss, 
                                                         var_list=self.weights)
        self.minimizer = self.optimizer.minimize(self.loss, 
                                                 global_step=batch)
        self.init = tf.global_variables_initializer()
        return

    def train(self, X, Y, X_val=None, Y_val=None):
        start_time = time.time()
        train_size = X.shape[0]
        self.sess.run(self.init)

        for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            x_batch = X[offset:(offset + BATCH_SIZE), ...]
            y_batch = Y[offset:(offset + BATCH_SIZE)]
            feed_dict = {self.features: x_batch, self.labels: y_batch}
            _ = self.sess.run(self.minimizer,
                              feed_dict = feed_dict)
            if EVAL_FREQUENCY != -1 and step % EVAL_FREQUENCY == 0:
                l, lr, g, predictions = self.sess.run([self.unregularized_loss, 
                                                       self.learning_rate,
                                                       self.gradient,
                                                       tf.sigmoid(self.logits)],
                                                      feed_dict={self.features: X,
                                                                 self.labels: Y})
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('Loss: %.3f, learning rate: %.6f' % (l, lr))
                result_train = estimate_metric(predictions, Y.ravel())
                print('AUC: %.3f' % result_train['auc'])
                if X_val is not None and Y_val is not None:
                    predictions_val = self.sess.run(tf.sigmoid(self.logits),
                                                    feed_dict={self.features: X_val})
                    result_val = estimate_metric(predictions_val, Y_val.ravel())
                    if self.config['save_history']:
                        self.history['result_val'].append(result_val)
                    print('Validation AUC: %.3f' % result_val['auc'])
                if self.config['save_history']:
                    self.history['loss'].append(l)
                    self.history['norm_of_weights'].append(self.sess.run(tf.norm(self.weights)))
                    self.history['gradient_norm'].append(self.sess.run(tf.norm(g[0][0])))
                    self.history['learning_rate'].append(lr)
                    self.history['result_train'].append(result_train)
                sys.stdout.flush()
        # save model
        self.saver.save(self.sess, '../models/{0}/logre'.format(self.config['model_name']))
        return

    def predict(self, X):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(self.init)
            # restore trained model
            importer = tf.train.import_meta_graph('../models/{0}/logre.meta'.format(self.config['model_name']))
            importer.restore(sess, tf.train.latest_checkpoint('../models/{0}'.format(self.config['model_name'])))
            # calculate accuracy
            predictions = tf.sigmoid(self.logits).eval({self.features: X})
        return predictions

def main(_):
    skip_training = False
    dump_results = True

    # Load training data
    print 'Loading training data and preprocessing ......'
    start_time = time.time()
    training_data = '../data/train.csv'
    X_train, Y_train = get_data(training_data)
    end = time.time()
    print 'Done. Took {} seconds.'.format(end - start_time)
    print

    # Load validation data
    print 'Loading validation data and preprocessing ......'
    start = time.time()
    validation_data = '../data/validate.csv'
    X_val, Y_val = get_data(validation_data, False)
    end = time.time()
    print 'Done. Took {} seconds.'.format(end - start)
    print

    # Instantiate and train model
    n_features = X_train.shape[1]
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    l2_regularizations = [0., 0.5, 1.]
#    learning_rates = [0.5]
#    l2_regularizations = [0.]
    results = []

    with tf.device('/gpu:0'):
        for learning_rate in learning_rates:
            for l2_reg in l2_regularizations:
                tf.reset_default_graph()
                with tf.Session() as sess:
#                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#                with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#                with tf.Session(config=tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))) as sess:
                    # Define the model
                    print('Instantiating a model with learning rate = {} '
                        'and l2_reg = {} ......').format(
                        learning_rate, l2_reg)
                    start = time.time()
                    config = {'model_name': 'logre_{}_{}'.format(learning_rate, 
                                                                 l2_reg),
                              'learning_rate': learning_rate,
                              'training_epochs': 1,
                              'l1_reg': 0,
                              'l2_reg': l2_reg,
                              'save_history': True}
                    estimator = RegularizedLinearClassifier(sess, n_features, 
                        config = config)
                    end = time.time()
                    print 'Done. Took {} seconds.'.format(end - start)
                    print

                    # Train
                    if skip_training:
                        print 'Skipping training.'
                        print
                    else:
                        print 'Training ......'
                        start = time.time()
                        estimator.train(X_train, Y_train, X_val, Y_val)
                        end = time.time()
                        print 'Done. Took {} seconds.'.format(end - start)
                        print

                    # Evaluate
                    print 'Evaluating on train data ......'
                    start = time.time()
                    y_pred_train = estimator.predict(X_train)#, Y_train)
                    result_train = estimate_metric(y_pred_train, Y_train.ravel())
                    print result_train
                    print

                    print 'Evaluating on validation data ......'
                    start = time.time()
                    y_pred_val = estimator.predict(X_val)#, Y_val)
                    result_val = estimate_metric(y_pred_val, Y_val.ravel())

                    print result_val
                    print

                    # Construct return values
                    result = {'model': estimator.config['model_name'],
                              'loss': estimator.history['loss'],
                              'w_norm': estimator.history['norm_of_weights'],
                              'g_norm': estimator.history['gradient_norm'],
                              'learning_rate': estimator.history['learning_rate'],
                              'result_train': estimator.history['result_train'],
                              'result_val': estimator.history['result_val']}
                    results.append(result)
                    end = time.time()
                    print 'Done. Took {} seconds.'.format(end - start)
                    print

#    # Output final results
#    for i, learning_rate in enumerate(learning_rates):
#        for j, l2_reg in enumerate(l2_regularizations):
#            result = results[len(l2_regularizations)*i+j]
#            print ('Learning rate = {}, '
#                   'L2 regularization = {}, '
#                   'train_err = {:.4f}, '
#                   'val_err = {:.4f}').format(learning_rate,
#                                              l2_reg,
#                                              result['train_error'][-1],
#                                              result['val_error'][-1])
#            print
    if dump_results:
        print 'Dumping results ......'
        print
        with open('logre_results_momentum.pkl', 'wb') as pkl:
            pickle.dump(results, pkl)
    end = time.time()
    print 'Session complete. Took total of {} seconds.'.format(
        end - start_time)
    print

if __name__ == '__main__':
  tf.app.run(main=main)
