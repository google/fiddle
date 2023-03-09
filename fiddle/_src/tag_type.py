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

"""Defines the core dataclasses for tagging.

Please see tagging.py for information about tagging APIs.
"""

# This module is separate so that we can import it from config.py, but then
# import config.py from tagging.py (and include user-facing APIs there).


class TagType(type):
  """All Fiddle tags are instances of this class.

  For defining Tags, we leverage Python's class definition and documentation
  syntax.

  See the documentation on `fdl.Tag` for instructions on how to use.
  """

  def __init__(cls, name, bases, dct):
    if '__doc__' not in dct:
      raise TypeError('You must provide a tag description with a docstring.')
    if '__qualname__' not in dct:
      raise TypeError('No `__qualname__` property found.')
    if '<' in dct['__qualname__']:
      raise TypeError('You cannot define a tag within a function or lambda.')
    super().__init__(name, bases, dct)

  def __call__(cls, *args, **kwds):
    raise TypeError('You cannot instantiate Fiddle tags (trying to instantiate '
                    f'{cls.name}); just use the type itself (no parenthesis) '
                    'when specifying tags on a TaggedValue(...) call, or call '
                    '`.new(default=)` to create a new `TaggedValue`.')

  @property
  def description(cls) -> str:
    """A string describing the semantics and intended usecases for this tag."""
    return cls.__doc__

  @property
  def name(cls) -> str:
    """A unique name for this tag."""
    return f'{cls.__module__}.{cls.__qualname__}'

  def __str__(cls) -> str:
    return f'#{cls.name}'

  def __repr__(cls) -> str:  # pylint: disable=invalid-repr-returned
    return cls.name


class TaggedValueNotFilledError(ValueError):
  """A TaggedValue was not filled when build() was called."""
