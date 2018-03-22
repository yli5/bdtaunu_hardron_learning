import subprocess as sp
import tempfile
import re
import sys
import time

create_views_script = 'create_views.sql'
copy_data_ml_script = 'copy_data_ml.sql'
copy_data_explore_script = 'copy_data_explore.sql'
drop_views_script = 'drop_views.sql'

sample_map = {
    'explore' : 1,
    'train' : 2,
    'validate' : 3,
    'test' : 4
}


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Obtain machine learning sample. ')
    parser.add_argument('sample_type', type=str,
                        help='type of machine learning sample to obtain. One of '
                             'explore, train, validate, or test. ')
    parser.add_argument('--output_fname', '-o', type=str, required=True,
                        help='file name to copy to. ')
    parser.add_argument('--dbname', '-d', type=str, required=True,
                        help='database to connect to. ')
    args = parser.parse_args()

    start_all = time.time()

    print '+ Querying database \'{0}\'.\n'.format(args.dbname)
    sys.stdout.flush()

    # create views
    with tempfile.TemporaryFile(mode='w+b') as temp:
        sp.check_call(
            ['psql', '-q', '-d', args.dbname,
             '-v', 'sample_type={0}'.format(sample_map[args.sample_type]),
             '-f', create_views_script],
            stdout=temp)

    # copy records to file
    start = time.time()
    print '+ Copying rows to file \'{0}\'... '.format(args.output_fname)
    sys.stdout.flush()

    copy_data_script = copy_data_ml_script
    if args.sample_type == 'explore':
        copy_data_script = copy_data_explore_script

    with open(args.output_fname, 'w') as w:
        sp.check_call(
            ['psql', '-q', '-d', args.dbname,
            '-f', copy_data_script],
            stdout=w)
    end = time.time()
    print '  completed in {0} seconds. \n'.format(round(end-start, 2))

    # drop views
    with tempfile.TemporaryFile(mode='w+b') as temp:
        sp.check_call(
            ['psql', '-q', '-d', args.dbname, '-f', drop_views_script],
            stdout=temp)

    print '+ done. completed in {0} seconds. \n'.format(round(end-start_all, 2))
