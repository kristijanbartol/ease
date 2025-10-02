from setuptools import setup, find_packages
import sys
import os
from build_cpp import build_cpp_project

#print("Starting C++ build process...")
#try:
#    module_path = build_cpp_project()
#    print(f"Successfully built C++ module at: {module_path}")
#except Exception as e:
#    print(f"ERROR: Failed to build C++ module: {e}")
#    # Exit with error status
#    sys.exit(1)  # This will stop the installation process

print("Finding packages...")
packages = find_packages(where='.')#where='loom')
print(f"Found packages: {packages}")

setup(
    name='loom',
    version='0.2',
    packages=packages,
    #package_dir={'': 'loom'},
    install_requires=[
        'numpy',
        'scikit-sparse'
    ],
#    extras_require={
#        'dev': [
#            'pytest',
#            'pytest-cov',
#        ]
#    },
#    package_data={
#        'loom': ['lib/*.so', 'lib/*.pyd'],
#    },
    include_package_data=True,
)