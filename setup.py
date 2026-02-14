#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except:
    from distutils.core import setup

import re

with open('README.rst', encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding="utf-8") as history_file:
    history = re.sub(r':py:class:`([^`]+)`', r'\1',
        history_file.read())

requirements = ['numpy', 'scipy', 'matplotlib', 'corner', 'optns', 'tqdm', 'ultranest', 'astropy']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*, !=3.6.*, !=3.7.*, !=3.8.*',
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    name='tinyxsf',
    packages=['tinyxsf'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JohannesBuchner/tinyXSF',
)
