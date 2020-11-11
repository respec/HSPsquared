
import sys
import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

__version__ = '1.0.0'

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

README = open("./README.rst").read()

install_requires = [
    'pandas >= 0.17.1',
    'python-dateutil >= 2.1',
    'matplotlib',
    'dateparser',
    'tabulate',
    'docutils',
    'numba',
    'networkx',
    'pytables >=3.2',
]

setup(name='HSPsquared',
      version=__version__,
      description="Hydrological Simulation Program - Python",
      long_description=README,
      classifiers=[
          # Get strings from
          # http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Environment :: Console',
          'Environment :: Web Environment'
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      keywords='hydrology hydrological hydraulic hspf',
      author='RESPEC, Inc',
      author_email='',
      url='http://www.respec.com/product/hydrologic-simulation-program-python-hsp%C2%B2/',
      packages=['HSP2', 'HSP2tools'],
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      entry_points={
          'console_scripts':
              ['hsp2=HSP2.main:main']
      },
      test_suite='tests',
      python_requires='>=3.6',
      )
