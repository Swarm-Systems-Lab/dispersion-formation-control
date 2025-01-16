"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

from .simulator_distr_new import *
from .simulator_distr import *
from .simulator import *

from .plots import *
from .animations import *

# import os
# import importlib

# def _import_modules():
#     # Dynamically import all .py files in the current folder
#     current_dir = os.path.join(os.path.dirname(__file__))
#     module_files = [
#         f[:-3] for f in os.listdir(current_dir)
#         if f.endswith(".py") and f != "__init__.py"
#     ]

#     # Import each module and expose its contents
#     for module_name in module_files:
#         module = importlib.import_module(f".{module_name}", package=__name__)
#         for attr_name in module.__all__:
#             # Avoid importing private attributes (those starting with '_')
#             if not attr_name.startswith("_"):      
#                 globals()[attr_name] = getattr(module, attr_name)
#                 print(attr_name,globals()[attr_name])
#                 __all__.append(attr_name)

# # Collect all names to expose
# __all__ = []  
# _import_modules()

# print(__all__)

# Delete __init__.py imports and declarations from global
# del os, importlib, _import_modules