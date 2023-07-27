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

"""Intermediate representation APIs for auto_config code generation.

In this approach we model something close to the output code, and make passes
over it to refine its structure.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict, List, Optional, Type

from fiddle import arg_factory
from fiddle import daglish
from fiddle._src.codegen import import_manager as import_manager_lib
from fiddle._src.codegen import namespace as namespace_lib
from fiddle._src.experimental import auto_config


@dataclasses.dataclass
class Name:
  """Represents the name of a variable/value/etc.

  Attributes:
    value: The string value of the name.
    is_generated: Whether the name is auto-generated.
    previous: Previous name, in case this name has been changed by a pass.
  """

  value: str
  is_generated: bool = True
  previous: Optional[Name] = None

  def __hash__(self):
    return id(self)

  def replace(self, new_name: str) -> None:
    """Mutable replacement method."""
    old_self = copy.copy(self)
    self.previous = old_self
    self.value = new_name

  def __str__(self):
    return self.value


@dataclasses.dataclass
class CodegenNode:
  """Base class for codegen nodes."""

  def __flatten__(self):
    names = [field.name for field in dataclasses.fields(self)]
    values = tuple(getattr(self, name) for name in names)
    return values, (names, type(self))

  def __path_elements__(self):
    names = [field.name for field in dataclasses.fields(self)]
    return tuple(daglish.Attr(name) for name in names)

  @classmethod
  def __unflatten__(cls, values, metadata):
    keys, typ = metadata
    return typ(**dict(zip(keys, values)))

  def __init_subclass__(cls):
    daglish.register_node_traverser(
        cls,
        flatten_fn=lambda x: x.__flatten__(),
        unflatten_fn=cls.__unflatten__,
        path_elements_fn=lambda x: x.__path_elements__(),
    )


@dataclasses.dataclass
class Parameter(CodegenNode):
  name: Name
  value_type: Optional[Type[Any]] = None


@dataclasses.dataclass
class BaseNameReference(CodegenNode):
  """Reference to a named symbol (mostly a base class for the below)."""

  name: Name


@dataclasses.dataclass
class VariableReference(BaseNameReference):
  """Reference to a variable or parameter."""


@dataclasses.dataclass
class ModuleReference(BaseNameReference):
  """Reference to an imported module."""


@dataclasses.dataclass
class BuiltinReference(BaseNameReference):
  """Reference to an imported module."""


@dataclasses.dataclass
class FixtureReference(BaseNameReference):
  """Reference to another fixture."""


@dataclasses.dataclass
class AttributeExpression(CodegenNode):
  """Reference to an attribute of another expression."""

  base: Any  # Wrapped expression, can involve VariableReference's
  attribute: str

  def __hash__(self):
    # Currently, some Pax (https://github.com/google/paxml) codegen involves
    # having AttributeExpression's as dict keys, as those keys are rewritten to
    # expressions. This function allows for that, but one shouldn't generally
    # assume equality as object identity here.
    return id(self)


@dataclasses.dataclass
class ParameterizedTypeExpression(CodegenNode):
  """Reference to a parameterized type like list[int]."""

  base_expression: Any  # Expression like BuiltinReference(Name("list"))
  param_expressions: List[Any]  # List of (positional) argument expressions


@dataclasses.dataclass
class ArgFactoryExpr(CodegenNode):
  """Represents a factory that should be interpreted as an argument factory.

  Inside Fiddle configs, we represent arg factories with fdl.ArgFactory, e.g.

  attention=fdl.ArgFactory(
    initializer=fdl.ArgFactory(nn.initializers.zeros, dtype='float16')
  )

  However in auto_config (and thus, we mean in normal Python code that just uses
  arg_factory but not Fiddle otherwise), the `initializer` factory can be any
  random callable, which can be pulled into a variable or `call`, etc., we just
  need a way of tagging it as such.
  """

  expression: Any  # Wrapped expression, can involve VariableReference's


@dataclasses.dataclass
class HistoryComments:
  """Brief assignment history, currently for buildables.

  Note: This is intentionally a plain dataclass and not traversed by daglish
  by default.
  """

  # For fdl.Config-like objects, attr name --> history string.
  per_field: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SymbolOrFixtureCall(CodegenNode):
  """Reference to a call of a library symbol/fixture, like MyEncoderLayer()."""

  symbol_expression: Any
  # Values for args can involve VariableReference's, Calls, etc.
  positional_arg_expressions: List[Any]
  arg_expressions: Dict[str, Any]
  history_comments: HistoryComments = dataclasses.field(
      default_factory=HistoryComments
  )

  def __post_init__(self):
    if isinstance(self.symbol_expression, str):
      raise TypeError(
          "Strings are no longer allowed in `symbol_expression`, please wrap"
          " with a name!"
      )


@dataclasses.dataclass
class WithTagsCall(CodegenNode):
  """Represents a call to auto_config.with_tags()."""

  tag_symbol_expressions: List[str]
  item_to_tag: Any


@dataclasses.dataclass
class FunctoolsPartialCall(SymbolOrFixtureCall):
  pass


@dataclasses.dataclass
class VariableDeclaration(CodegenNode):
  name: Name
  expression: Any  # Value that can involve VariableReference's


@dataclasses.dataclass
class FixtureFunction(CodegenNode):
  """Basic declaration of a function.

  Each auto_config function will have a name, and list of parameters. Its body
  will then be a list of variable declarations, followed by an output value
  in a `return` statement.
  """

  name: Name
  parameters: List[Parameter]
  variables: List[VariableDeclaration]
  output_value: Any  # Value that can involve VariableReference's
  return_type_annotation: Optional[Any] = None

  def __hash__(self):
    return id(self)

  def replace_with(self, other: FixtureFunction) -> None:
    self.name = other.name
    self.parameters = other.parameters
    self.variables = other.variables
    self.output_value = other.output_value


@dataclasses.dataclass
class CallInstance:
  """Represents a concrete function call.

  This is more of a dataflow node than an expression of calling a function, i.e.
  it represents a call to a function at a particular point in the auto_config
  fixture's execution.
  """

  fn: FixtureFunction
  parent: Optional[CallInstance]
  children: List[CallInstance]
  parameter_values: Dict[Name, Any]

  def __hash__(self) -> int:
    return id(self)

  def to_stack(self) -> List[CallInstance]:
    current = self
    result = [self]
    while current.parent is not None:
      current = current.parent
      result.append(current)
    return list(reversed(result))

  @arg_factory.supply_defaults
  def all_fixture_functions(
      self, seen=arg_factory.default_factory(set)
  ) -> List[FixtureFunction]:
    result = [] if self.fn in seen else [self.fn]
    for child in self.children:
      result.extend(child.all_fixture_functions(seen))
    return result


def _init_import_manager() -> import_manager_lib.ImportManager:
  return import_manager_lib.ImportManager(namespace=namespace_lib.Namespace())


@dataclasses.dataclass
class CodegenTask:
  """Encapsulates an entire task of code generation.

  Useful for combining dataflow and generated code.
  """

  original_config: Any
  top_level_call: CallInstance
  import_manager: import_manager_lib.ImportManager = dataclasses.field(
      default_factory=_init_import_manager
  )
  auto_config_fn: Any = auto_config.auto_config

  @property
  def global_namespace(self) -> namespace_lib.Namespace:
    return self.import_manager.namespace
