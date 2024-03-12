"""
    Runs all tests in this directory

    Note:
        Only test files with name: test*.py are executed

    Version:
        2024-03-12 DWW
"""

import os
from unittest import TestLoader, TextTestRunner

start = os.getcwd()
suite = TestLoader().discover(start)

TextTestRunner().run(suite)
