from .camel import CAMEL

import pkg_resources
__version__ = pkg_resources.get_distribution('camel-learn').version
__all__ = ["camel"]
