'''
Run tests as `pytest` in the command line.
'''

from setuptools import setup, find_packages
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

setup(
    name='garment',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
        ]
    },
)