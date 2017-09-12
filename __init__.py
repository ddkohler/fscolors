#<<<<<<< HEAD
#from . import fscolors
from .fscolors import *
from . import cm
#from cm import *
#=======
### import ####################################################################


from __future__ import absolute_import, division, print_function, unicode_literals

from . import fscolors
from .fscolors import *
from .cm import *

import os as _os
import sys as _sys
_dir = _os.path.dirname(__file__)

try:
    import configparser as _ConfigParser  # python 3
except ImportError:
    import ConfigParser as _ConfigParser  # python 2


### version information #######################################################


# get config
_ini_path = _os.path.join(_dir, 'main.ini')
if _sys.version[0] == '3':
    _config = _ConfigParser.ConfigParser()
else:
    _config = _ConfigParser.SafeConfigParser()
_config.read(_ini_path)

# attempt get git sha
try:
    _HEAD_file = _os.path.join(_dir, '.git', 'logs', 'HEAD')
    with open(_HEAD_file) as _f:
        for _line in _f.readlines():
            _sha = _line.split(' ')[1]  # most recent commit is last
except:
    _sha = '0000000'

# version format: a.b.c.d
# a - major release
# b - minor release
# c - bugfix
# d - git sha key
__version__ = _config.get('main', 'version') + '.' + _sha[:7]
>>>>>>> fe0820c896175a8f8113a2eb99eeb34aaec7c0e2
