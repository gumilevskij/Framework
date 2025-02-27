"""
Ideas :
*  recursive blocks           [by default]
* (order left hand side ?)    [by default]
* dependency across blocks
* dummy blocks that are basically substituted everywhere else
"""

import os, sys
import ruamel.yaml as yaml 
version = yaml.__version__

if getattr(sys, 'frozen', False):
    # we are running in a |PyInstaller| bundle
    DIR_PATH = sys._MEIPASS
else:
    DIR_PATH, this_filename = os.path.split(__file__)

DATA_PATH = os.path.join(DIR_PATH, "recipes.yaml")

with open(DATA_PATH) as f:
  
  if version >= '0.18':
      recipes = yaml.YAML(typ='safe').load(f)
  else:
      recipes = yaml.load(f, Loader=yaml.Loader)
