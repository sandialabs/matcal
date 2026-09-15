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

from .core.logger import initialize_matcal_logger as _initialize_matcal_logger
# Use a dedicated child logger (matcal.__init__) with its own stream handlers for
# messages emitted here. We deliberately do NOT attach handlers to the top-level
# "matcal" logger: it is the ancestor of every matcal.* module logger, and those
# module loggers already carry the matcal stream handlers and propagate to their
# ancestors. Attaching handlers to "matcal" would cause every matcal log record
# to be emitted twice.
_logger = _initialize_matcal_logger(__name__ + ".__init__")
import importlib.util as _importlib_util
if _importlib_util.find_spec("site_matcal") is None:
    # No site_matcal package on the path. This is expected for a standalone
    # install, so degrade gracefully and continue with core defaults.
    _logger.info("No 'site_matcal' package found. "
                 "Skipping site module imports and using core defaults.")
else:
    # site_matcal is installed: import it and let any error (a broken site
    # install, a missing site dependency, a bug in registration code) propagate
    # instead of being silently swallowed.
    import site_matcal as _site_matcal
    from site_matcal import *
    # Explicitly trigger site registration so failures are attributable and
    # occur at a controlled point rather than as an import side effect.
    if hasattr(_site_matcal, "register"):
        _site_matcal.register()
    del _site_matcal
del _logger, _initialize_matcal_logger, _importlib_util
