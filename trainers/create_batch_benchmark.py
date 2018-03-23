import sys
import time
import numpy as np
import bisect
import matplotlib.pyplot as plt

from os import path
lib_path = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(lib_path)
from preprocess.PreProcess import PreProcess, load_data
from util.resampling import binary_downsampling, binary_upsampling

def get_data(data, fit=True):
    # Preprocessor
    process_path = {'imputer': '../preprocess/imputer.pkl', 'scaler': '../preprocess/scaler.pkl', 'encoder': '../preprocess/encoder.pkl'}
    processor = PreProcess()
    X, y, w = load_data(data, process_path, fit=fit)
    # Resample to balance labels
    X, Y, W = binary_upsampling(X, y, w)
    # Return labels as 1D array
    assert X.shape[0] == Y.shape[0]
    return X, Y, W

def alias_setup(probs):
    K       = len(probs)
    q       = np.zeros(K)
    J       = np.zeros(K, dtype=np.int)
 
    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger  = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
 
    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
 
        J[small] = large
        q[large] = q[large] - (1.0 - q[small])
 
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
 
    return J, q
 
def alias_draw(J, q):
    K  = len(J)
 
    # Draw from the overall uniform mixture.
    kk = int(np.floor(np.random.rand()*K))
 
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
 

if __name__ == '__main__':
    # Load training data
    print 'Loading training data and preprocessing ......'
    start_time = time.time()
    training_data = '../data/train.csv'
    x, y, w = get_data(training_data)
    end = time.time()
    print 'Done. Took {} seconds.'.format(end - start_time)
    print

    # Parameters
    n = x.shape[0]
    BATCH_SIZE = 256
    NUM_EPOCHS = 1

    # Create batches by partitioning
#    print 'Batch sampling by partitioning ......'
#    start_time = time.time()
#    for step in xrange(int(NUM_EPOCHS * n) // BATCH_SIZE):
#        offset = (step * BATCH_SIZE) % (n - BATCH_SIZE)
#        x_batch = x[offset:(offset + BATCH_SIZE), ...]
#        y_batch = y[offset:(offset + BATCH_SIZE)]
#    end = time.time()
#    print 'Done. Took {} seconds.'.format(end - start_time)
#    print

    # Create batches by sampling by weights
    J, q = alias_setup(w)
    cs = np.cumsum(w)
#    s = np.cumsum(p)
    print 'Batch sampling by weights ......'
    print
    results = []
    results_part = []
    batch_sizes = [1024, 512, 256, 128, 64]

    def get_partition_idxs(cs, n_per_batch):
        '''
        cs: cumulative sum of probability vector
        n_per_batch: size of each batch
        '''
        idxs = []
        n = len(cs)
        approx_batch_weight = float(n_per_batch)/n
        for i in range(0, n/n_per_batch):
            idxs.append(bisect.bisect(cs, (i+1)*approx_batch_weight))
        return idxs

    batch_idxs = get_partition_idxs(cs, BATCH_SIZE)

    for BATCH_SIZE in batch_sizes:
        print 'Batch size = {}'.format(BATCH_SIZE)
        start_time = time.time()
        # Sampling according to weights
        prev_batch_end_idx = -1
        for step in xrange(int(NUM_EPOCHS * n) // BATCH_SIZE):

#           # Naive; very slow
#           rand_idxs = np.random.choice(n, size=BATCH_SIZE, 
#                                        replace=False, p=p)
#   
#           # Still too slow
#           r = np.random.rand(n)
#           rand_idxs = (s < r).sum()
#   
            # Faster
            rand_idxs = []
            for i in range(BATCH_SIZE):
                rand_idxs.append(bisect.bisect(cs, np.random.random() * cs[-1]))
#            
#           # Walker's alias method
#           rand_idxs = np.zeros(BATCH_SIZE)
#           for i in xrange(BATCH_SIZE):
#               rand_idxs[i] = alias_draw(J, q)
#           rand_idxs = rand_idxs.astype(np.int)
#
#            x_batch = x[rand_idxs]
#            y_batch = y[rand_idxs]
#
#            # Batch selection
#            batch_idx = batch_idxs[step % NUM_EPOCHS]
#            x_batch = x[prev_batch_end_idx+1:batch_idx, ...]
#            y_batch = y[prev_batch_end_idx+1:batch_idx]
#            if batch_idx == batch_idxs[-1]:
#                prev_batch_end_idx = -1
#            else:
#                prev_batch_end_idx = batch_idx
#
#            # Sequential selection
#            #start_idx = np.random.randint(n/BATCH_SIZE*BATCH_SIZE)
#            start_idx = 0
#            sum_w = 0.
#            batch_w = float(BATCH_SIZE) / n
#            i = start_idx
#            idxs = []
#            while sum_w < batch_w:
#                if np.random.random() < w[i]:
#                    sum_w += w[i]
#                    idxs.append(i)
#                elif i == start_idx:
#                    start_idx = i + 1
#                i += 1
#            x_batch = x[idxs, ...]
#            y_batch = y[idxs]
        end = time.time()
        results.append(end-start_time)
        print 'Done. Took {} seconds.'.format(end - start_time)
        print
        # Batch sample by partitioning
        start_time = time.time()
        for step in xrange(int(NUM_EPOCHS * n) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (n - BATCH_SIZE)
            x_batch = x[offset:(offset + BATCH_SIZE), ...]
            y_batch = y[offset:(offset + BATCH_SIZE)]
        end = time.time()
        results_part.append(end-start_time)
        print 'Done. Took {} seconds.'.format(end - start_time)
        print

    ax = plt.figure().gca()
    ax.plot(batch_sizes, results, color='r')
    ax2 = ax.twinx()
    ax2.plot(batch_sizes, results_part)
    plt.show()
