#import numpy as np
import tensorflow as tf
#from sklearn.utils import shuffle
tf.logging.set_verbosity(tf.logging.INFO)


def load_data(data):
    # Adapter
    adapter = LearningDataAdapter(for_learning=True)
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
    # Create a dictionary of features
    assert X.shape[0] == Y.shape[0]
    X_dict = {}
    for i, feature in enumerate(X.T):
        X_dict[str(i)] = feature
    feature_columns = [
        tf.feature_column.numeric_column(k) for k in X_dict.keys()]

    return X_dict, Y, feature_columns



def input_fn(X, Y):
    return tf.estimator.inputs.numpy_input_fn(
        x=X,
        y=Y,
        shuffle=False)



if __name__ == '__main__':
    import sys
    import time
    from os import path
    lib_path = path.dirname(path.dirname(path.abspath(__file__)))
    sys.path.append(lib_path)
    from preprocess.PreProcess import PreProcess
    from preprocess.LearningDataAdapter import LearningDataAdapter
    from util.resampling import binary_downsampling, binary_upsampling

    with tf.Session() as sess:
        # Load training data
        print 'Loading training data and preprocessing ......'
        start_time = time.time()
        training_data = '../data/train.csv'
        X_train, Y_train, feature_columns = load_data(training_data)
        train_input_fn = input_fn(X_train, Y_train)
        end = time.time()
        print 'Done. Took {} seconds.'.format(end - start_time)
        print

        # Load validation data
        print 'Loading validation data and preprocessing ......'
        start = time.time()
        validation_data = '../data/validate.csv'
        X_val, Y_val, _ = load_data(validation_data)
        validation_input_fn = input_fn(X_val, Y_val)
        end = time.time()
        print 'Done. Took {} seconds.'.format(end - start)
        print

        # Instantiate and train model
        #learning_rates = [0.01, 0.05, 0.1, 0.5]
        #l2_regularizations = [0., 0.5, 1.]
        learning_rates = [0.05]
        l2_regularizations = [0.5]
        results = []
        for learning_rate in learning_rates:
            for l2_reg in l2_regularizations:
                print('Training with learning rate = {}'
                      'and l2_reg = {} ......').format(
                    learning_rate, l2_reg)
                start = time.time()
                model_dir = '../models/logre_{}_{}'.format(
                    learning_rate, l2_reg)
                config = tf.estimator.RunConfig(model_dir=model_dir)
                estimator = tf.estimator.LinearClassifier(
                    feature_columns=feature_columns,
                    #            weight_column = weights,
                    optimizer=tf.train.FtrlOptimizer(
                        learning_rate=learning_rate,
                        l2_regularization_strength=l2_reg),
                    config=config)

                estimator.train(input_fn=train_input_fn)
                end = time.time()
                print 'Done. Took {} seconds.'.format(end - start)
                print

                # Evaluate
                print 'Evaluating on validation data ......'
                start = time.time()
                result = estimator.evaluate(input_fn=validation_input_fn)
                results.append(result)
                end = time.time()
                print 'Done. Took {} seconds.'.format(end - start)
                print
        for i, learning_rate in enumerate(learning_rates):
            for j, l2_reg in enumerate(l2_regularizations):
                result = results[len(l2_regularizations)*i+j]
                print ('Learning rate = {},'
                       'L2 regularization = {},'
                       'accuracy = {}, AUC = {}').format(learning_rate,
                                                         l2_reg,
                                                         result['accuracy'],
                                                         result['auc'])
                print
        end = time.time()
        print 'Session complete. Took total of {} seconds.'.format(
            end - start_time)
        print
