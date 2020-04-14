from setuptools import setup, find_packages

setup(name='a2dr',
      version='0.2',
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
      tests_require=['nose'])
