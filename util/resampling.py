import numpy as np 
from sklearn.utils import shuffle, resample

def binary_downsampling(X, y, pos_label=1, neg_label=0, n_samples=None):
    """
    Perform down-sampling to X and y. 

    X:         np.array, input features
    y:         1-D np.array, labels
    pos_label: notation of positive label, default 1.
    neg_label: notation of negative label, default 0.
    n_samples: number of resampled events. If it is none, we use the minimum value of
               the number of positive and negative events.

    """
    n_pos = y[y==pos_label].shape[0]
    n_neg = y[y==neg_label].shape[0]
    X_s, y_s = X[y==pos_label], y[y==pos_label]
    X_b, y_b = X[y==neg_label], y[y==neg_label]

    if(n_pos < n_neg):
        X_b, y_b = shuffle(X_b, y_b, n_samples=n_pos)
    if(n_pos > n_neg):
        X_s, y_s = shuffle(X_s, y_s, n_samples=n_neg)
    
    X_sample = np.concatenate([X_s, X_b])
    y_sample = np.concatenate([y_s, y_b])
    X_sample, y_sample = shuffle(X_sample, y_sample)
    return X_sample, y_sample


def binary_upsampling(X, y, pos_label=1, neg_label=0, n_samples=None):
    """
    Perform up-sampling to X and y.

    X:         np.array, input features
    y:         1-D np.array, labels
    pos_label: notation of positive label, default 1.
    neg_label: notation of negative label, default 0.
    n_samples: number of resampled events. If it is none, we use the maximum value of
               the number of positive and negative events.
    """
    n_pos = y[y==pos_label].shape[0]
    n_neg = y[y==neg_label].shape[0]
    X_s, y_s = X[y==pos_label], y[y==pos_label]
    X_b, y_b = X[y==neg_label], y[y==neg_label]

    if(n_pos > n_neg):
        X_b, y_b = resample(X_b, y_b, n_samples=n_pos)
    if(n_pos < n_neg):
        X_s, y_s = resample(X_s, y_s, n_samples=n_neg)
        
    X_sample = np.concatenate([X_s, X_b])
    y_sample = np.concatenate([y_s, y_b])
    X_sample, y_sample = shuffle(X_sample, y_sample)
    return X_sample, y_sample



if __name__ == '__main__':
    X = np.array([[1,1,1],[0,0,0],[1,1,1],[0,0,0],[1,1,1]])
    y = np.array([1, 0, 1, 0, 1])
    print 'X: {0}'.format(X)
    print 'y: {0}'.format(y)

    print 'Down sampling:'
    X_d, y_d = binary_downsampling(X, y)
    print 'X_d: {0}'.format(X_d)
    print 'y_d: {0}'.format(y_d)

    print 'Up sampling:'
    X_u, y_u = binary_upsampling(X, y)
    print 'X_u: {0}'.format(X_u)
    print 'y_u: {0}'.format(y_u)
