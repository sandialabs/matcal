from .version import __version__

__all__ = [ "__version__"]

from . import core
from .core import *
__all__ += core.__all__

from . import dakota
from .dakota import *
__all__ += dakota.__all__

from . import full_field
from .full_field import *
__all__ += full_field.__all__

from . import sierra
from .sierra import *
__all__ += sierra.__all__

try:
    from site_matcal import *
except Exception as e:
    print("Warning no site matcal files found. Skipping site module imports...")
    print(f"Error caught:\n repr({e})")
