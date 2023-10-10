# coding=utf-8
# Copyright 2022 The Fiddle-Config Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script for fiddle.

See https://github.com/google/fiddle for documentation.
"""

# pyformat: disable

import sys
from setuptools import find_packages
from setuptools import setup


_dct = {}
with open('fiddle/version.py', encoding='utf-8') as f:
  exec(f.read(), _dct)  # pylint: disable=exec-used
__version__ = _dct['__version__']

long_description = """
# Fiddle

Fiddle is a Python-first configuration library particularly well suited to ML
applications. Fiddle enables deep configurability of parameters in a program,
while allowing configuration to be expressed in readable and maintainable
Python code.

**Authors**: Dan Holtmann-Rice, Brennan Saeta, Sergio Guadarrama
"""

# pylint: disable=g-long-ternary
setup(
    name='fiddle',
    version=__version__,
    include_package_data=True,
    packages=find_packages(exclude=['docs']),  # Required
    package_data={'testdata': ['testdata/*.fiddle']},
    python_requires='>=3.8',
    install_requires=[
        'absl-py',
        'graphviz',
        'libcst',
        'typing-extensions',
    ],
    extras_require={
        'flags': [
            'absl-py',
            'etils[epath]',
        ],
        'testing': [
            'cloudpickle',
            'fiddle[flags]',
            'flax',
            'graphviz',
            'pytest',
            'pytype',
        ] + [
            'seqio',
            # Temporarily pin the TFDS version to avoid pip backtracking during
            # unit tests.
            'tfds_nightly>=4.9.2.dev202308090034',
        ] if sys.platform != 'darwin' else []
    },
    description='Fiddle: A Python-first configuration library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google/fiddle',
    author='The Fiddle Team',
    author_email='noreply@google.com',
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    project_urls={  # Optional
        'Documentation': 'https://github.com/google/fiddle/docs',
        'Bug Reports': 'https://github.com/google/fiddle/issues',
        'Source': 'https://github.com/google/fiddle',
    },
    license='Apache 2.0',
    keywords='fiddle python configuration machine learning'
)
