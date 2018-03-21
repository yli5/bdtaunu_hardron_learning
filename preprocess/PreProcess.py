import numpy as np 
import sklearn.preprocessing as preprocessing
from sklearn.externals import joblib

class PreProcess:
    """
    This class provides methods to transform, save and load preprocessing techniques for data before analysis.
    """

    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.encoder = None

    def fit(self, X_num, X_cat, save_paths):
        """
        Fit and save preprocessing functions using provided numerial and categorial data.

        X_num     : np.array with numerial elements
        X_cat     : np.array with categorial elements
        save_paths:  {'imputer': 'imputer_save_path',
                      'scaler' : 'scaler_save_path',
                      'encoder': 'encoder_save_path'}
        """
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(X_num)
        X_num_trans = imp.transform(X_num)
        joblib.dump(imp, save_paths['imputer'])

        scaler = preprocessing.StandardScaler(with_mean=True,with_std=True)
        scaler.fit(X_num_trans)
        joblib.dump(scaler, save_paths['scaler'])

        enc = preprocessing.OneHotEncoder(n_values='auto',sparse=False)
        enc.fit(X_cat)
        joblib.dump(enc, save_paths['encoder'])

        self.imputer = imp
        self.scaler = scaler
        self.encoder = enc

        return

    def load(self, paths):
        """
        Load preprocessing transformers

        paths: {'imputer': 'imputer_path',
                'scaler' : 'scaler_path',
                'encoder': 'encoder_path'}
        """
        self.imputer = joblib.load(paths['imputer'])
        self.scaler = joblib.load(paths['scaler'])
        self.encoder = joblib.load(paths['encoder'])
        return

    def transform(self, X_num, X_cat):
        """
        Transform X_num and X_cat using loaded imputer, scaler and encoder.
        Return transformed inputs.

        """
        if self.imputer:
            X_num = self.imputer.transform(X_num)
        if self.scaler:
            X_num = self.scaler.transform(X_num)
        if self.encoder:
            X_cat = self.encoder.transform(X_cat)
        return np.hstack((X_num, X_cat))

    

def test():
    # generate num data
    loc = np.array([1, 5, 10, 20, 100])
    scale = np.array([2, 4, 8, 16, 32])
    N = 100
    X_num = np.random.normal(loc=loc, scale=scale, size=[N, len(loc)])
    
    # generate cat data
    X_cat_1 = np.random.randint(0, 2, size=[N,1])
    X_cat_2 = np.random.randint(0, 3, size=[N,1])
    X_cat = np.hstack((X_cat_1, X_cat_2))

    processor = PreProcess()
    save_path = {'imputer': './imputer.pkl',
                 'scaler': './scaler.pkl',
                 'encoder': './encoder.pkl'}
    processor.fit(X_num, X_cat, save_path)
    processor.load(save_path)
    X = processor.transform(X_num, X_cat)

    print "X shape should be {0}, it is {1}".format([X_num.shape[0], X_num.shape[1]+5], X.shape)
    print "X column means: "
    print np.mean(X, axis=0)
    print "X column std: "
    print np.std(X, axis=0)
    return



if __name__ == '__main__':
   test()




