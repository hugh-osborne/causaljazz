# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='causaljazz',
    version='0.0.1',
    description='A numerical solver for population dynamics in a causal framework.',
    long_description=readme,
    author='Hugh Osborne',
    author_email='hugh.osborne@gmail.com',
    url='https://github.com/hugh-osborne/causaljazz',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

