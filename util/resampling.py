import numpy as np 
from sklearn.utils import shuffle, resample

def binary_downsampling(X, y, w, pos_label=1, neg_label=0, n_samples=None):
    """
    Perform down-sampling to X and y. 

    X        : np.array, input features
    y        : 1-D np.array, labels
    w        : weights of each event
    pos_label: notation of positive label, default 1.
    neg_label: notation of negative label, default 0.
    n_samples: number of resampled events. If it is none, we use the minimum value of
               the number of positive and negative events.

    """
    n_pos = y[y==pos_label].shape[0]
    n_neg = y[y==neg_label].shape[0]
    X_s, y_s, w_s = X[y==pos_label], y[y==pos_label], w[y==pos_label]
    X_b, y_b, w_b = X[y==neg_label], y[y==neg_label], w[y==neg_label]
    
    # downsample negative data
    if(n_pos < n_neg):
        sample_idx = np.random.choice(y_b.shape[0], size=n_pos, p=w_b/np.sum(w_b))
        X_b, y_b, w_b = X_b[sample_idx], y_b[sample_idx], w_b[sample_idx]
    # downsample positive data
    if(n_pos > n_neg):
        sample_idx = np.random.choice(y_s.shape[0], size=n_neg, p=w_s/np.sum(w_s))
        X_s, y_s, w_s = X_s[sample_idx], y_s[sample_idx], w_s[sample_idx]
    # concatenate positive and negative data, then shuffle randomly.
    X_sample = np.concatenate([X_s, X_b])
    y_sample = np.concatenate([y_s, y_b])
    w_sample = np.concatenate([w_s, w_b])
    w_sample = w_sample / np.sum(w_sample)
    X_sample, y_sample, w_sample = shuffle(X_sample, y_sample, w_sample)
    return X_sample, y_sample, w_sample


def binary_upsampling(X, y, w, pos_label=1, neg_label=0, n_samples=None):
    """
    Perform up-sampling to X and y.

    X:         np.array, input features
    y:         1-D np.array, labels
    w        : weights of each event
    pos_label: notation of positive label, default 1.
    neg_label: notation of negative label, default 0.
    n_samples: number of resampled events. If it is none, we use the maximum value of
               the number of positive and negative events.
    """
    n_pos = y[y==pos_label].shape[0]
    n_neg = y[y==neg_label].shape[0]
    X_s, y_s, w_s = X[y==pos_label], y[y==pos_label], w[y==pos_label]
    X_b, y_b, w_b = X[y==neg_label], y[y==neg_label], w[y==neg_label]
    # upsample negative data
    if(n_pos > n_neg):
        sample_idx = np.random.choice(y_b.shape[0], size=n_pos, p=w_b/np.sum(w_b))
        X_b, y_b, w_b = X_b[sample_idx], y_b[sample_idx], w_b[sample_idx]
    # upsample positive data
    if(n_pos < n_neg):
        sample_idx = np.random.choice(y_s.shape[0], size=n_neg, p=w_s/np.sum(w_s))
        X_s, y_s, w_s = X_s[sample_idx], y_s[sample_idx], w_s[sample_idx]
    # concatenate positive and negative data, then shuffle randomly.
    X_sample = np.concatenate([X_s, X_b])
    y_sample = np.concatenate([y_s, y_b])
    w_sample = np.concatenate([w_s, w_b])
    w_sample = w_sample / np.sum(w_sample)
    X_sample, y_sample, w_sample = shuffle(X_sample, y_sample, w_sample)
    return X_sample, y_sample, w_sample



if __name__ == '__main__':
    X = np.array([[1,1,1],[0,0,0],[1,1,1],[0,0,0],[1,1,1]])
    y = np.array([1, 0, 1, 0, 1])
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
    print 'X: {0}'.format(X)
    print 'y: {0}'.format(y)
    print

    print 'Down sampling:'
    X_d, y_d, w_d = binary_downsampling(X, y, w)
    print 'X_d: {0}'.format(X_d)
    print 'y_d: {0}'.format(y_d)
    print 'w_d: {0}'.format(w_d)
    print

    print 'Up sampling:'
    X_u, y_u, w_u = binary_upsampling(X, y, w)
    print 'X_u: {0}'.format(X_u)
    print 'y_u: {0}'.format(y_u)
    print 'w_u: {0}'.format(w_u)
    print
