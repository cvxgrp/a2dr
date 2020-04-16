from setuptools import setup, find_packages
import codecs
import os.path

# code for single sourcing versions
# reference: https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

# read the contents of your README file
# reference: https://packaging.python.org/guides/making-a-pypi-friendly-readme/
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='a2dr',
      version=get_version("a2dr/__init__.py"),
      description='A Python package for type-II Anderson accelerated Douglas-Rachford splitting (A2DR).',
      url='https://github.com/cvxgrp/a2dr',
      author='Anqi Fu, Junzi Zhang, Stephen Boyd',
      author_email='contact.a2dr.solver@gmail.com',
      license='Apache License, Version 2.0',
      packages=find_packages(),
      install_requires=['matplotlib',
                        'cvxpy >= 1.0.25',
                        'numpy >= 1.16',
                        'scipy >= 1.2.1'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      long_description=long_description,
      long_description_content_type='text/markdown')
