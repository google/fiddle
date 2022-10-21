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

"""Public API for tied values."""

from typing import Any, TypeVar

from fiddle import config as config_lib

TiedValue = config_lib.TiedValue

ValueT = TypeVar("ValueT")


def _identity(value: ValueT) -> ValueT:
  """Returns `value` modified."""
  return value


def new(value: ValueT) -> TiedValue[ValueT]:
  """Returns a new TiedValue (unless value is already a TiedValue).

  Args:
    value: Value to tie; if it is already a TiedValue it is returned unmodified.
  """
  return value if isinstance(value, TiedValue) else TiedValue(_identity, value)


def untie(parent: config_lib.Buildable, attribute: str) -> None:
  """Unties a configuration value assigned to a parent Buildable field.

  Args:
    parent: Parent buildable containing the tied value.
    attribute: Name of the attribute containing a tied value.

  Raises:
    TypeError: If the attribute in question did not actually contain a tied
      value.
  """
  attribute_value = parent.__arguments__[attribute]
  if isinstance(attribute_value, TiedValue):
    parent.__arguments__[attribute] = attribute_value.value
  else:
    raise TypeError(f"Expected a TiedValue but got {type(attribute_value)}")


def get_tied(parent: config_lib.Buildable, attribute: str) -> TiedValue[Any]:
  """Returns a TiedValue container from a parent Buildable field.

  By default, accessing the attribute via parent.<attribute> will return the
  value within the TiedValue container, so that the configuration has more
  uniform access (i.e. behaves similarly whether or not the value is tied). But
  occasionally one wants access to the tied value container, for example to
  share it with another field.

  Args:
    parent: Parent buildable containing the tied value.
    attribute: Name of the attribute containing a tied value.

  Raises:
    TypeError: If the attribute in question did not actually contain a tied
      value.
  """
  attribute_value = parent.__arguments__[attribute]
  if isinstance(attribute_value, TiedValue):
    return attribute_value
  else:
    raise TypeError(f"Expected a TiedValue but got {type(attribute_value)}")
