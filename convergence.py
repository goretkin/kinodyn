import matplotlib as mpl
mpl.use('Agg')  #no GUI -- don't want to crash anything when running many simulations with no X server

from joblib import Parallel, delayed
import examplets_import

if False:
    import pendulum as rrt_setup
    folder = '/home/goretkin/Dropbox/kinodyn/experiments/pendulum_longtime'
else:
    import linear_ship_rrt_cost as rrt_setup
    folder = '/home/goretkin/Dropbox/kinodyn/experiments/double_integrator_fancy_cost'


import shelve
import numpy as np

import copy

import sys
sys.path.insert(0,folder) #allows domain file to be imported


n_rrts = 50
n_iters = 5000

TEST_IO = False #does no RRT iterations, just checks that all files can save (parallel joblib masks these errors until the end -- very painful)
TEST_RANDOM = False #prints out a bunch of random numbers from each parallel instance to sanity check things are being seeded properly
import logging
logging.getLogger().handlers[0].setLevel(logging.WARN)

import os.path

rrt_setup.rrt.viz_collided_paths = None #I suspect this takes up a lot a considerable chunk of memory

try:
    import domain_config
    domain_config.domain_configure_rrt(rrt_setup.rrt) #set parameters, etc
    print "domain_config loaded"
except ImportError:
    print "no domain_config file found"


import signal

signal_file = '_diff_run_int.signal'

def run(pk,folder):
    signal_filep = os.path.join(folder,signal_file)

    filename = folder+'/rrt_{0:04}.shelve'.format(pk)
    f = None
    try:
        f = open(signal_filep+'_','w')     #rename this file to stop running stuff
    finally:
        if f is None: f.close()
       
    print 'starting {} with filename {}'.format(pk,filename)

    if os.path.exists(signal_filep):
        print 'skipping {} {} due to break'.format(pk,filename)
        return

    

    rrt = copy.deepcopy(rrt_setup.rrt)
    rrt.special_convergence_sol_num = 0 #store this number in the RRT. bit weird. also will overwrite stuff on new runs FIXME
    rrt.special_pk = pk
    print 'sol num: {}'.format(rrt.special_convergence_sol_num)

    l = logging.getLogger('rrt{0:04}'.format(pk))
    log_file = logging.FileHandler('{}.log'.format(filename))
    log_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') )
    log_file.setLevel(logging.DEBUG)
    l.handlers = [] #remove all handlers -- this is necessary when re-running this file within the same interpreter session since the handlers persist (and you get duplicate log entries)
    l.addHandler(log_file)
    rrt.logger = l
    rrt.logger.info('Starting in {}'.format(pk))

    def hook(rrt):
        pk = rrt.special_pk
        sol_num = rrt.special_convergence_sol_num
        print 'RRT {} improved solution {}.'.format(pk,sol_num)
        fn = os.path.join(folder,"solution_{0:03}_{1:02}.shelve".format(pk,sol_num))
        print 'file name', fn
        s = shelve.open(fn)
        sol = rrt.best_solution_goal()  
        if sol is not None: #hook should not be called unless there is at least one solution, but we also call it to check file IO stuff
            upath = sol[2]
            s['best_solution'] = sol
            s['utraj'] = upath
            print 'saving RRT {} solution {}'.format(pk,sol_num)
        s.close()
        rrt.special_convergence_sol_num += 1

    rrt.improved_solution_hook = hook

    LOADED_RRT_FILE = False

    if os.path.exists(filename):
        f = shelve.open(filename,flag='r')
        if f.has_key('tree'):
            if os.path.exists(filename+'.temp'):
                f_temp = shelve.open(filename+'.temp',flag='r')
                if f['n_iters'] < f_temp['n_iters']:
                    #the temp version of this file has more iterations than the actual file, so load it instead
                    #(this means that we crashed before we made a checkpoint)
                    print '\tloading previous temp {} {}'.format(pk,filename+'.temp')        
                    rrt.load(f_temp,strict_consistency_check=False)
                    print '\tdone loading {} {}'.format(pk,filename)
                    LOADED_RRT_FILE = True
                else:
                    print '\ttemp file {} is obsolete'.format(filename+'.temp')
                f_temp.close()
            if not LOADED_RRT_FILE:
                print '\tloading previous {} {}'.format(pk,filename)
                rrt.load(f,strict_consistency_check=False)
                print '\tdone loading {} {}'.format(pk,filename)
        f.close()

    elif os.path.exists(filename+'.temp'):
        f = shelve.open(filename+'.temp',flag='r')

        if f.has_key('tree'):
            print '\tloading previous temp {} {}'.format(pk,filename+'.temp')
            rrt.load(f,strict_consistency_check=False)
            print '\tdone loading {} {}'.format(pk,filename)

        f.close()

    import numpy as np
    np.random.seed()
    if TEST_RANDOM:
        print np.array([rrt.sample() for i in range(3)])

    signal_break = False
    """
    def signal_handler(signum,frame):
        global signal_break
        print 'Signal handler called with signal {}'.format(signum)
        signal_break = True
        
    signal.signal(signal.SIGINT,signal_handler)
    """

    iters_of_last_save = rrt.n_iters    #we just loaded this.

    iterations_after_first_solve = None
    remaining = np.inf if not TEST_IO else 0

    while not rrt.found_feasible_solution or iterations_after_first_solve < n_iters:
        if os.path.exists(signal_filep):
            signal_break = True

        print 'RRT instance {} is at {}. Improved {} times. Break: {}'.format(pk,rrt.n_iters,len(rrt.cost_history),signal_break,remaining)

        if (rrt.n_iters - iters_of_last_save > 500) or TEST_IO or signal_break:
            temp_shelve = shelve.open(filename+'.temp')
            iters_of_last_save = rrt.n_iters
            try:
                print 'RRT {} is saving temp file'.format(pk)
                rrt.save(temp_shelve)
            finally:
                temp_shelve.close()
        if TEST_IO: hook(rrt)
        if TEST_IO or signal_break: break
        
        if rrt.found_feasible_solution:                
            i = rrt.cost_history[0][0]  #iteration at which found first solution
            iterations_after_first_solve = rrt.n_iters-i
            remaining = n_iters-iterations_after_first_solve
        print 'RRT instance {}  Remaining: {}'.format(pk,remaining)
        rrt.search(min(50,remaining))

    rrt.clean_nodes()

    f = shelve.open(filename)
    try:
        rrt.save(f)
    finally:
        f.close()
    log_file.close()
    print 'saved {} with filename {}'.format(pk,filename)
    return rrt.cost_history

a = Parallel(n_jobs=15,verbose=True)(delayed(run)(pk,fn) for (pk,fn) in [(d,folder) for d in range(n_rrts)])
