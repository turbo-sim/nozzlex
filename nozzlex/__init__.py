# Highlight exception messages
# https://stackoverflow.com/questions/25109105/how-to-colorize-the-output-of-python-errors-in-the-gnome-terminal/52797444#52797444
try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.FormattedTB(color_scheme='linux', call_pdb=False)


import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)



from .graphics import *

# Import subpackages
from . import duct_modeling
from . import functions
from . import functions_old
