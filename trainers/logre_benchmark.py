import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.INFO)
from sklearn.utils import shuffle


def load_data(data, for_learning=True):
    # Adapter
    adapter = LearningDataAdapter(for_learning)
    adapter.adapt_file(data)
    X_num, X_cat = adapter.X_num, adapter.X_cat
    w, y = adapter.w, adapter.y
    # Preprocessor
    processor = PreProcess()
    processor.fit(X_num, X_cat,
                  {'imputer': '../preprocess/imputer.pkl',
                   'scaler': '../preprocess/scaler.pkl',
                   'encoder': '../preprocess/encoder.pkl'})
    X = processor.transform(X_num, X_cat)
    # Resample to balance labels
    X, Y = binary_upsampling(X, y)
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
            'gradient_norm': []}

    def build_model(self):
        self.features = tf.placeholder(
            shape=[None, self.n_features],
            dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        #self.weights = tf.Variable(tf.zeros([self.n_features, 1]))
        tf.set_random_seed(1)
        self.weights = tf.Variable(0.01*tf.random_normal([self.n_features, 1]))
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
        self.optimizer = tf.train.GradientDescentOptimizer(
            config['learning_rate'])
        #self.optimizer = tf.train.AdagradOptimizer(
        #    config['learning_rate'])
        self.minimizer = self.optimizer.minimize(self.loss)
        self.gradient = self.optimizer.compute_gradients(self.loss)
        self.init = tf.global_variables_initializer()
        return

    def train(self, X, Y):
        self.sess.run(self.init)
        for epoch in range(self.config['training_epochs']):
            avg_cost = 0.
            # No batch training; train over entire dataset
            _, c, gradient, unreg_c = self.sess.run([self.minimizer, self.loss, 
                                                     self.gradient, self.unregularized_loss], 
                                                    feed_dict = {self.features: X,
                                                                 self.labels: Y})
            if (epoch + 1) % 10 == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.6f}".format(c))
            if self.config['save_history']:
                self.history['loss'].append(unreg_c)
                self.history['norm_of_weights'].append(self.sess.run(tf.norm(self.weights)))
                self.history['gradient_norm'].append(self.sess.run(tf.norm(gradient[0][0])))

        # save model
        self.saver.save(sess, '../models/{0}/logre'.format(self.config['model_name']))
        return

    def evaluate(self, X, Y):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(self.init)
            # restore trained model
            importer = tf.train.import_meta_graph('../models/{0}/logre.meta'.format(self.config['model_name']))
            importer.restore(sess, tf.train.latest_checkpoint('../models/{0}'.format(self.config['model_name'])))
            # calculate accuracy
            predictions = tf.round(tf.sigmoid(self.logits))
            n_correct = tf.cast(tf.equal(predictions, Y), tf.float32)
            accuracy = tf.reduce_mean(n_correct).eval({self.features: X, self.labels: Y})
        return accuracy

    def predict():
        return


if __name__ == '__main__':
    import sys
    import time
    import pickle
    from os import path
    lib_path = path.dirname(path.dirname(path.abspath(__file__)))
    sys.path.append(lib_path)
    from preprocess.PreProcess import PreProcess
    from preprocess.LearningDataAdapter import LearningDataAdapter
    from util.resampling import binary_downsampling, binary_upsampling

    skip_training = False
    dump_results = False

    # Load training data
    print 'Loading training data and preprocessing ......'
    start_time = time.time()
    training_data = '../data/train.csv'
    X_train, Y_train = load_data(training_data)
    n_features = X_train.shape[1]
    end = time.time()
    print 'Done. Took {} seconds.'.format(end - start_time)
    print

    # Load validation data
    print 'Loading validation data and preprocessing ......'
    start = time.time()
    validation_data = '../data/validate.csv'
    X_val, Y_val = load_data(validation_data)
    end = time.time()
    print 'Done. Took {} seconds.'.format(end - start)
    print

    # Instantiate and train model
    #learning_rates = [0.01, 0.05, 0.1, 0.5]
    #l2_regularizations = [0., 0.5, 1.]
    learning_rates = [0.5]
    l2_regularizations = [0.]
    results = []

    with tf.device('/gpu:0'):
        for learning_rate in learning_rates:
            for l2_reg in l2_regularizations:
                tf.reset_default_graph()
#                with tf.Session() as sess:
#                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
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
                        estimator.train(X_train, Y_train)
                        end = time.time()
                        print 'Done. Took {} seconds.'.format(end - start)
                        print

                    # Evaluate
                    print 'Evaluating on validation data ......'
                    start = time.time()
                    accuracy = estimator.evaluate(X_val, Y_val)

                    # Construct return values
                    result = {'model': estimator.config['model_name'],
                              'accuracy': accuracy,
                              'loss': estimator.history['loss'],
                              'w_norm': estimator.history['norm_of_weights'],
                              'g_norm': estimator.history['gradient_norm']}
                    results.append(result)
                    end = time.time()
                    print 'Done. Took {} seconds.'.format(end - start)
                    print

    # Output results
    for i, learning_rate in enumerate(learning_rates):
        for j, l2_reg in enumerate(l2_regularizations):
            result = results[len(l2_regularizations)*i+j]
            print ('Learning rate = {}, '
                   'L2 regularization = {}, '
                   'accuracy = {:.4f}').format(learning_rate,
                                               l2_reg,
                                               result['accuracy'])
            print
    if dump_results:
        print 'Dumping results ......'
        print
        with open('logre_results_sgd.pkl', 'wb') as pkl:
            pickle.dump(results, pkl)
    end = time.time()
    print 'Session complete. Took total of {} seconds.'.format(
        end - start_time)
    print
