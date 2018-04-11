# This script aims to bootstrap train, validate and test csv files
# and store them in .hdf files. They will be used to train deep learning models.
import sys
import os
import numpy as np 
import pandas as pd 
import re



def generate_bootstrap(file_name, n, dir_name):
    """
    This function generates n bootstrap samples of df files and store in directory_path.
    file_name : Pandas.DataFrame file, file we will apply bootstrap on.
    n         : number of bootstrap samples
    dir_name  : bootstrap files store path

    """
    # check existence of file_name and dir_name
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if not os.path.isfile(file_name):
        raise NameError("file_name does not exist.")
    if n <= 0:
        print n
        raise ValueError("n should be positive integer.")
    
    save_name = dir_name + '/' + file_name.split('.')[-2] + '_bootstrap_{0}.hdf'
    df = pd.read_csv(file_name)
    for i in range(n):
        df_sample = df.sample(frac=1, replace=True, weights=df.weight)
        df_sample.to_hdf(save_name.format(i), 'bootstrap')
    return



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Bootstrap data files. ')
    parser.add_argument('--datafile_name', '-f', type=str,
                        help='csv file data name to apply bootstrap.')
    parser.add_argument('--number', '-n', type=int,
                        help='Number of generated bootstrap datasets. ')
    parser.add_argument('--dir_name', '-d', type=str,
                        help='Directory name to store bootstrap datasets. ')
    args = parser.parse_args()

    file_name = args.datafile_name
    n = args.number
    dir_name = args.dir_name
    print 'Bootstrapping ...... \n'
    generate_bootstrap(file_name, n, dir_name)
    print 'Done.  \n'

