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

"""Helper functions for writing Daglish traversals.

These are kept in a separate file so they can reference both the core Daglish
implementation and Fiddle Buildable's.
"""

import ast
import re
from typing import Any, Callable, Type, Union

from fiddle._src import config as config_lib
from fiddle._src import daglish

_IMMUTABLE_OBJECT_IDS = set()


def register_immutable(value: Any) -> None:
  """Registers a certain type to be immutable."""
  _IMMUTABLE_OBJECT_IDS.add(id(value))


# Similar set of IDs of functions/types with immutable return values (for this
# case, maybe it's OK to replace IDs with just the functions/types?).
_FUNCTIONS_WITH_IMMUTABLE_RETURN_VALUES = set()


def register_function_with_immutable_return_value(
    fn_or_cls: Union[Type[Any], Callable[..., Any]]
) -> None:
  _FUNCTIONS_WITH_IMMUTABLE_RETURN_VALUES.add(id(fn_or_cls))


# TODO(b/285146396): Register the tensorflow dtypes as immutable.
def is_immutable(value: Any) -> bool:
  """Returns true if value is immutable from the point view of Fiddle.

  This method differs from `is_internable` in that NamedTuples and other tuple
  subclasses are included here.

  Frozen dataclasses are not considered as immutable here. Because dataclasses
  are not traversable by default, and will be handled like a generic object.
  The motivation for this decision is partly to encourage users to
  create fdl.Config out of dataclasses rather than have dataclass instances
  in the fdl.Config.

  Args:
    value: A candidate value to check for immutability.

  Returns:
    A bool indicating whether `value` is immutable.
  """
  return (
      daglish.is_internable(value)
      or (isinstance(value, tuple) and all(is_immutable(e) for e in value))
      or (id(value) in _IMMUTABLE_OBJECT_IDS)
  )


# TODO(b/285146396): Register the tensorflow dtypes as immutable.
def is_unshareable(value: Any) -> bool:
  """Returns true if value can be unshared and build an equivalent object.

  This is a slight generalization of is_immutable, for example we can have

  shared = fdl.Config(jax.nn.initializers.glorot_uniform, batch_axis=(0,))
  cfg.a.init = shared
  cfg.b.init = shared

  is_unshareable will return True for `shared`, indicating that we can replace
  the config with the following one,

  cfg2.a.init = fdl.Config(jax.nn.initializers.glorot_uniform, batch_axis=(0,))
  cfg2.b.init = fdl.Config(jax.nn.initializers.glorot_uniform, batch_axis=(0,))

  This differs in configuration API, because setting

  (cfg_or_cfg2).a.init.batch_axis = (1,)

  will set both initializers in cfg, but only the first initializer in cfg2.

  However, if we immediately call fdl.build(cfg), it is equivalent to
  fdl.build(cfg2), because the initializer is the same.

  For codegen, this will explictly classsify jax dtypes and initializers as
  immutable, so that they are not extracted as separate variables in generated
  code.

  Args:
    value: A candidate value to check for immutability.

  Returns:
    A bool indicating whether `value` is immutable.
  """
  return (
      daglish.is_internable(value)
      or (isinstance(value, tuple) and all(is_unshareable(e) for e in value))
      or (id(value) in _IMMUTABLE_OBJECT_IDS)
      or (
          isinstance(value, config_lib.Buildable)
          and id(config_lib.get_callable(value))
          in _FUNCTIONS_WITH_IMMUTABLE_RETURN_VALUES
      )
  )

_PATH_PART = re.compile(
    "(?:{})".format(
        "|".join([
            r"\.(?P<attr_name>[\w_]+)",
            # future improvement: support escape sequences.
            r"\[(?P<key>\d+|'[^']+'|\"[^\"]+\")\]",
        ])
    )
)


def parse_path(path: str) -> daglish.Path:
  """Parses a string path into a path with Attr and Key elements.

  Note that we can't distinguish between Index and Key or Attr and
  BuildableAttr. We might eventually remove that distinction from Daglish
  itself.

  Args:
    path: Path string. If appended to a symbol it should be a valid Python
      expression; in particular if the first element is an attribute, the path
      must start with ".attribute_name", not "attribute_name".

  Returns:
    Parsed Daglish path.
  """
  result = []
  current_idx = 0
  while current_idx < len(path):
    match = _PATH_PART.match(path, current_idx)
    if not match:
      raise ValueError(
          f"Could not parse {path!r} (starting at position {current_idx})"
      )
    match_dict = match.groupdict()
    if match_dict["attr_name"]:
      result.append(daglish.Attr(match_dict["attr_name"]))
    elif match_dict["key"]:
      result.append(daglish.Key(ast.literal_eval(match_dict["key"])))
    else:
      raise AssertionError(f"Unexpected regex match {match_dict}")
    current_idx = match.end(0)
  return tuple(result)
