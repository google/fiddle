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

from setuptools import find_packages
from setuptools import setup

_VERSION = '0.2.2'

long_description = """
# Fiddle

Fiddle is a Python-first configuration library particularly well suited to ML
applications. Fiddle enables deep configurability of parameters in a program,
while allowing configuration to be expressed in readable and maintainable
Python code.

**Authors**: Dan Holtmann-Rice, Brennan Saeta, Sergio Guadarrama
"""

setup(
    name='fiddle',
    version=_VERSION,
    include_package_data=True,
    packages=find_packages(exclude=['docs']),  # Required
    package_data={'testdata': ['testdata/*.fiddle']},
    install_requires=[
        'libcst',
        'typing-extensions',
    ],
    extras_require={
        'testing': [
            'absl-py',
            'pytest',
            'pytype',
            'seqio-nightly',
        ]
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',

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
