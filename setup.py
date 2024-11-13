from setuptools import setup, find_packages
import sys
import os
from build_cpp import build_cpp_project

print("Starting C++ build process...")
try:
    module_path = build_cpp_project()
    print(f"Successfully built C++ module at: {module_path}")
except Exception as e:
    print(f"ERROR: Failed to build C++ module: {e}")
    # Exit with error status
    sys.exit(1)  # This will stop the installation process

print("Finding packages...")
packages = find_packages(where='tailorlang')
print(f"Found packages: {packages}")

setup(
    name='tailorlang',
    version='0.2',
    packages=packages,
    package_dir={'': 'tailorlang'},
    install_requires=[
        'numpy',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
        ]
    },
    package_data={
        'tailorlang': ['lib/*.so', 'lib/*.pyd'],
    },
    include_package_data=True,
)