import os, tempfile
CACHE_PATH = os.path.join(tempfile.gettempdir(), "tinyrl")

from tinyrl.src import *

from tinyrl.src.utils import *