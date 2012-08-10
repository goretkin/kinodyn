"""allow imports from examplets directory"""

import sys
import os

_root_dir = os.path.dirname(os.path.realpath(__file__))

_root_dir = os.path.join(_root_dir,'examplets')
print 'adding to the python path',_root_dir
sys.path.insert(0, _root_dir)
