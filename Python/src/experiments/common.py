# ============================================
# common.py
# Simple script containing various setup cmds 
# that all the experiments need to run
# ============================================

# Lets us grab the class defintions from the parent directory
import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 