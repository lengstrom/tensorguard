from setuptools import find_packages
import subprocess
from glob import glob

from distutils.core import setup, Extension

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='tensorguard',
      version='0.1',
      description='tensorguard',
      author='lengstrom',
      author_email='engstrom@mit.edu',
      url='https://github.com/lengstrom/tensorguard',
      license_files = ('LICENSE.txt',),
      packages=find_packages(),
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=['typeguard==2.13.3', 'termcolor'])
