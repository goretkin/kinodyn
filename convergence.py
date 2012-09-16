
from joblib import Parallel, delayed
import examplets_import
import pendulum as rrt_setup

import shelve
import numpy as np

import copy


folder = '/home/goretkin/diffruns'
n_rrts = 5
n_iters = 4000

TEST_IO = False #does no RRT iterations, just checks that all files can save (parallel joblib masks these errors until the end -- very painful)
TEST_RANDOM = False #prints out a bunch of random numbers from each parallel instance to sanity check things are being seeded properly
import logging
logging.getLogger().handlers[0].setLevel(logging.WARN)

import os.path

rrt_setup.rrt.viz_collided_paths = None #I suspect this takes up a lot a considerable chunk of memory

import signal

signal_file = 'diff_run_int.signal'

def run(pk,filename):
    f = None
    try:
        f = open(signal_file+'_','w')     #rename this file to stop running stuff
    finally:
        f.close()


    print 'starting {} with filename {}'.format(pk,filename)
    rrt = copy.deepcopy(rrt_setup.rrt)

    l = logging.getLogger('rrt{0:04}'.format(pk))
    log_file = logging.FileHandler('{}.log'.format(filename))
    log_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') )
    log_file.setLevel(logging.INFO)
    l.addHandler(log_file)
    rrt.logger = l
    rrt.logger.info('Starting in {}'.format(pk))
    
    if os.path.exists(filename):
        f = shelve.open(filename)
        if f.has_key('tree'):
            print '\tloading previous {} {}'.format(pk,filename)
            rrt.load(f,strict_consistency_check=False)
            print '\tdone loading {} {}'.format(pk,filename)
        f.close()
    elif os.path.exists(filename+'.temp'):
        f = shelve.open(filename+'.temp')

        if f.has_key('tree'):
            print '\tloading previous temp {} {}'.format(pk,filename+'.temp')
            rrt.load(f,strict_consistency_check=False)
            print '\tdone loading {} {}'.format(pk,filename)

        f.close()

    def hook(rrt_obj):
        print 'RRT {} improved solution.'.format(pk)

    rrt.improved_solution_hook = hook
    import numpy as np
    np.random.seed()
    if TEST_RANDOM:
        print np.array([rrt.sample() for i in range(3)])

    signal_break = False
    def signal_handler(signum,frame):
        global signal_break
        print 'Signal handler called with signal {}'.format(signum)
        signal_break = True
        
    signal.signal(signal.SIGINT,signal_handler)

    iters_of_last_save = 0
    iterations_after_first_solve = None
    remaining = np.inf if not TEST_IO else 0
    while not rrt.found_feasible_solution or iterations_after_first_solve < n_iters:
        if os.path.exists(signal_file):
            signal_break = True

        print 'RRT instance {} is at {}. Improved {} times. Break: {}'.format(pk,rrt.n_iters,len(rrt.cost_history),signal_break)

        if (rrt.n_iters - iters_of_last_save > 10000) or TEST_IO or signal_break:
            temp_shelve = shelve.open(filename+'.temp')
            iters_of_last_save = rrt.n_iters
            try:
                print 'RRT {} is saving temp file'.format(pk)
                rrt.save(temp_shelve)
            finally:
                temp_shelve.close()
        if TEST_IO or signal_break: break
        
        rrt.search(min(50,remaining))
        if rrt.found_feasible_solution:                
            i = rrt.cost_history[0][0]  #iteraction at which found first solution
            iterations_after_first_solve = rrt.n_iters-i
            remaining = n_iters-iterations_after_first_solve

    rrt.clean_nodes()

    f = shelve.open(filename)
    try:
        rrt.save(f)
    finally:
        f.close()
    log_file.close()
    print 'saved {} with filename {}'.format(pk,filename)
    return rrt.cost_history

a = Parallel(n_jobs=15,verbose=True)(delayed(run)(pk,fn) for (pk,fn) in [(d,folder+'/rrt_%04d.shelve'%(d)) for d in range(n_rrts)])
