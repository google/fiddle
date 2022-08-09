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

"""Provides utilities for transforming builder functions into `fdl.Config`s.

This module defines the `auto_config` function (and associated helpers), which
can be used to convert an existing function that creates an object graph into a
function that creates a graph of `Config` and `Partial` objects. When the
resulting graph of `Config` and `Partial` objects is built via `fdl.build()`, it
will yield same object graph as the original function.
"""

import ast
import builtins
import contextlib
import dataclasses
import functools
import inspect
import textwrap
import types
from typing import Any, Callable

from fiddle import building
from fiddle import config
from fiddle import tagging
from fiddle.experimental import daglish

_CALL_HANDLER_ID = '__auto_config_call_handler__'
_CLOSURE_WRAPPER_ID = '__auto_config_closure_wrapper__'
_EMPTY_ARGUMENTS = ast.arguments(
    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
_BUILTINS = frozenset([
    builtin for builtin in builtins.__dict__.values()
    if inspect.isroutine(builtin) or inspect.isclass(builtin)
])
_DEBUG_MODE = False


@dataclasses.dataclass(frozen=True)
class AutoConfig:
  """A function wrapper for auto_config'd functions.

  In order to support auto_config'ing @classmethod's, we need to customize the
  descriptor protocol for the auto_config'd function. This simple wrapper type
  is designed to look like a simple `functool.wraps` wrapper, but implements
  custom behavior for bound methods.
  """
  func: Callable[..., Any]
  buildable_func: Callable[..., config.Buildable]

  def __post_init__(self):
    # Must copy-over to correctly implement "functools.wraps"-like
    # functionality.
    for name in ('__module__', '__name__', '__qualname__', '__doc__',
                 '__annotations__'):
      try:
        value = getattr(self.func, name)
      except AttributeError:
        pass
      else:
        object.__setattr__(self, name, value)

  def __call__(self, *args, **kwargs) -> Any:
    if _DEBUG_MODE:
      return self.func(*args, **kwargs)
    else:
      return building.build(self.as_buildable(*args, **kwargs))

  def as_buildable(self, *args, **kwargs) -> config.Buildable:
    return self.buildable_func(*args, **kwargs)

  def __get__(self, obj, objtype=None):
    return _BoundAutoConfig(self, obj)

  @property
  def __wrapped__(self):
    return self.func

  def __getattr__(self, name):
    # Pass through extra things on the thing we wrapped.
    return getattr(self.func, name)


@dataclasses.dataclass(frozen=True)
class _BoundAutoConfig:
  """An `AutoConfig` bound to an object.

  This parallels the function / bound method pair.
  """
  __slots__ = 'auto_config', 'obj'
  auto_config: AutoConfig
  obj: Any

  def __call__(self, *args, **kwargs) -> Any:
    ac: AutoConfig = self.auto_config  # pytype: disable=annotation-type-mismatch
    return ac.func(self.obj, *args, **kwargs)

  def as_buildable(self, *args, **kwargs) -> config.Buildable:
    ac: AutoConfig = self.auto_config  # pytype: disable=annotation-type-mismatch
    return ac.buildable_func(self.obj, *args, **kwargs)


def _returns_buildable(signature: inspect.Signature) -> bool:
  """Returns True iff the return annotation is a subclass of config.Buildable."""
  return (signature.return_annotation is not signature.empty and
          inspect.isclass(signature.return_annotation) and
          issubclass(signature.return_annotation, config.Buildable))


def _is_tag_new_method(fn_or_cls):
  """Returns True if fn_or_cls is `T.new`, where T is any subclass of Tag."""
  return (isinstance(fn_or_cls, types.MethodType) and
          fn_or_cls.__func__ is tagging.Tag.new.__func__ and  # pytype: disable=attribute-error
          issubclass(fn_or_cls.__self__, tagging.Tag))


def _is_tag_manipulation_function(fn_or_cls):
  tag_manipulation_functions = [
      config.add_tag, config.set_tags, config.remove_tag, config.clear_tags,
      config.get_tags
  ]
  return any(fn_or_cls is tag_fn for tag_fn in tag_manipulation_functions)


def _is_auto_config_eligible(fn_or_cls):
  """Helper to determine if `fn_or_cls` is eligible for auto-config."""
  try:
    signature = inspect.signature(fn_or_cls)
  except ValueError:
    signature = None

  try:
    _ = hash(fn_or_cls)
  except TypeError:
    has_hash = False
  else:
    has_hash = True

  is_buildable = (
      inspect.isclass(fn_or_cls) and issubclass(fn_or_cls, config.Buildable))

  return (
      # We can find a signature...
      signature is not None and
      # It's not a builtin function...
      not inspect.isbuiltin(fn_or_cls) and
      # It's not a builtin type like range()...
      (has_hash and fn_or_cls not in _BUILTINS) and
      # It's not a `fdl.Buildable` already...
      not is_buildable and
      # It's not a method...
      not inspect.ismethod(fn_or_cls) and
      # It's not an `auto_config`ed function...
      not hasattr(fn_or_cls, 'as_buildable') and
      # It's not a function returning a `fdl.Buildable`...
      not _returns_buildable(signature) and
      # It's not a Fiddle tagging function
      not _is_tag_new_method(fn_or_cls) and
      not _is_tag_manipulation_function(fn_or_cls)
  )  # pyformat: disable


def auto_config_call_handler(fn_or_cls, *args, **kwargs):
  """Handles calls in auto_config'ed functions.

  This intercepts calls in an auto-configed function, and determines whether the
  called `fn_or_cls` should be wrapped in a `Config` or `Partial`. If
  `fn_or_cls` is `functools.partial`, the call will instead be converted into a
  call to Fiddle's `Partial`. If it is "auto-config eligible" (see
  `_is_auto_config_eligible`), then a `Config` will be create for `fn_or_cls`
  with the provided arguments. Otherwise, `fn_or_cls` is called directly.

  Args:
    fn_or_cls: The function or class being called.
    *args: The positional arguments with which `fn_or_cls` is being called.
    **kwargs: The keyword arguments with which `fn_or_cls` is being called.

  Returns:
    Depending on `fn_or_cls`, either `Partial`, a `Config`, or the result of
    calling `fn_or_cls` with the provided arguments.
  """
  if fn_or_cls is functools.partial:
    return config.Partial(args[0], *args[1:], **kwargs)

  if _is_auto_config_eligible(fn_or_cls):
    return config.Config(fn_or_cls, *args, **kwargs)

  if hasattr(fn_or_cls, 'as_buildable'):
    return fn_or_cls.as_buildable(*args, **kwargs)

  return fn_or_cls(*args, **kwargs)


class UnsupportedLanguageConstructError(SyntaxError):
  pass


class _AutoConfigNodeTransformer(ast.NodeTransformer):
  """A NodeTransformer that adds the auto-config call handler into an AST."""

  def __init__(self,
               source: str,
               filename: str,
               line_offset: int,
               allow_control_flow=False):
    """Initializes the auto config node transformer instance.

    Args:
      source: The source code of the node that will be transformed by this
        instance. This is used for better error reporting.
      filename: The filename `source` is from.
      line_offset: The line offset of `source` within `filename`.
      allow_control_flow: Whether to permit control flow constructs (loops,
        conditionals, comprehensions, etc). By default, this is `False`, and
        control flow constructs will cause an
        `UnsupportedLanguageConstructError` to be raised.
    """
    self._lines = source.splitlines()
    self._filename = filename
    self._line_offset = line_offset
    self._allow_control_flow = allow_control_flow

    self._function_def_depth = 0

  def _location_for(self, node: ast.AST):
    line_number = node.lineno + self._line_offset
    line = self._lines[node.lineno - 1]
    return (self._filename, line_number, node.col_offset, line)

  def _handle_control_flow(self, node: ast.AST, activatable: bool = False):
    if self._allow_control_flow and activatable:
      return self.generic_visit(node)
    msg = f'Control flow ({type(node).__name__}) is unsupported by auto_config.'
    raise UnsupportedLanguageConstructError(msg, self._location_for(node))

  def _generic_visit_inside_function(self, node):
    try:
      self._function_def_depth += 1
      return self.generic_visit(node)
    finally:
      self._function_def_depth -= 1

  # pylint: disable=invalid-name
  def visit_Call(self, node: ast.Call):
    return ast.Call(
        func=ast.Name(id=_CALL_HANDLER_ID, ctx=ast.Load()),
        args=[node.func, *(self.visit(arg) for arg in node.args)],
        keywords=[self.visit(keyword) for keyword in node.keywords],
    )

  def visit_For(self, node: ast.For):
    return self._handle_control_flow(node, activatable=True)

  def visit_While(self, node: ast.While):
    return self._handle_control_flow(node, activatable=True)

  def visit_If(self, node: ast.If):
    return self._handle_control_flow(node, activatable=True)

  def visit_IfExp(self, node: ast.IfExp):
    return self._handle_control_flow(node, activatable=True)

  def visit_ListComp(self, node: ast.ListComp):
    return self._handle_control_flow(node, activatable=True)

  def visit_SetComp(self, node: ast.SetComp):
    return self._handle_control_flow(node, activatable=True)

  def visit_DictComp(self, node: ast.DictComp):
    return self._handle_control_flow(node, activatable=True)

  def visit_GeneratorExp(self, node: ast.GeneratorExp):
    return self._handle_control_flow(node, activatable=True)

  def visit_Try(self, node: ast.Try):
    return self._handle_control_flow(node)

  def visit_Raise(self, node: ast.Try):
    return self._handle_control_flow(node)

  def visit_With(self, node: ast.With):
    return self._handle_control_flow(node)

  def visit_Yield(self, node: ast.Yield):
    return self._handle_control_flow(node)

  def visit_YieldFrom(self, node: ast.YieldFrom):
    return self._handle_control_flow(node)

  def visit_FunctionDef(self, node: ast.FunctionDef):
    if self._function_def_depth > 0:
      msg = 'Nested function definitions are not supported by auto_config.'
      raise UnsupportedLanguageConstructError(msg, self._location_for(node))
    else:
      return self._generic_visit_inside_function(node)

  def visit_Lambda(self, node: ast.Lambda):
    if self._function_def_depth > 0:
      msg = 'Lambda definitions are not supported by auto_config.'
      raise UnsupportedLanguageConstructError(msg, self._location_for(node))
    else:
      return self._generic_visit_inside_function(node)

  def visit_ClassDef(self, node: ast.ClassDef):
    msg = 'Class definitions are not supported by auto_config.'
    raise UnsupportedLanguageConstructError(msg, self._location_for(node))

  def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
    msg = 'Async function definitions are not supported by auto_config.'
    raise UnsupportedLanguageConstructError(msg, self._location_for(node))

  # pylint: enable=invalid-name


def _contains_buildable(structure):
  """Returns `True` if `structure` contains a `fdl.Buildable`."""
  contains_buildable = False

  def traverse(unused_path, value):
    nonlocal contains_buildable
    if isinstance(value, config.Buildable):
      contains_buildable = True
      return  # Stop traversal.
    else:
      yield  # Continue traversal.

  daglish.traverse_with_path(traverse, structure)
  return contains_buildable


def _wrap_ast_for_fn_with_closure_vars(
    module: ast.Module,
    fn: types.FunctionType,
) -> ast.Module:
  """Wraps `module.body` in a function that defines closure variables for `fn`.

  If `fn` has any free variables (i.e., it's `__code__.co_freevars` is not
  empty), we want to make sure that compiling its AST (assumed to be in the body
  of `module`) will create the same set of free variables in the resulting code
  object. However, by default this won't happen, since we would be compiling
  `fn`'s AST in the absence of its original context (e.g., just compiling a
  nested function, and not the containing one).

  To work around this issue, this function wraps `module.body` in another
  `FunctionDef` that defines dummy variables corresponding to `fn`'s free
  variables. This causes the subsequent compile step to create the right set of
  free variables, and allows us to use `fn.__closure__` directly when creating a
  new function object via `types.FunctionType`.

  Effectively, this wrapping looks like the following Python code:

      def __auto_config_closure_wrapper__():
        closure_var_1 = None
        closure_var_2 = None
        ...

        def fn(...):  # Or some expression involving a lambda.
          ...  # Contains references to the closure variables.

  Args:
    module: An `ast.Module` object whose body contains the function definition
      for `fn` (e.g., as an `ast.FunctionDef` or `ast.Lambda`).
    fn: The function to create dummy closure variables for (assumed to
      correspond to the body of `module`).

  Returns:
    A new `ast.Module` containing an additional wrapper `ast.FunctionDef` that
    defines dummy closure variables.
  """
  ast_name = lambda name: ast.Name(id=name, ctx=ast.Store())
  ast_none = ast.Constant(value=None)
  closure_var_definitions = [
      ast.Assign(targets=[ast_name(var_name)], value=ast_none)
      for var_name in fn.__code__.co_freevars
  ]

  wrapper_module = ast.Module(
      body=[
          ast.FunctionDef(
              name=_CLOSURE_WRAPPER_ID,
              args=_EMPTY_ARGUMENTS,
              body=[
                  *closure_var_definitions,
                  *module.body,
              ],
              decorator_list=[])
      ],
      type_ignores=[],
  )
  wrapper_module = ast.fix_missing_locations(wrapper_module)
  return wrapper_module


def _find_function_code(code: types.CodeType, fn_name: str):
  """Finds the code object within `code` corresponding to `fn_name`."""
  code = [
      const for const in code.co_consts
      if inspect.iscode(const) and const.co_name == fn_name
  ]
  assert len(code) == 1, f"Couldn't find function code for {fn_name!r}."
  return code[0]


def _unwrap_code_for_fn(code: types.CodeType, fn: types.FunctionType):
  """Unwraps `code` to find the code object for `fn`.

  This function assumes `code` is the result of compiling an `ast.Module`
  returned by `_wrap_node_for_fn_with_closure_vars`.

  Args:
    code: A code object containing code for `fn`.
    fn: The function to find a code object for within `code`.

  Returns:
    The code object corresponding to `fn`.
  """
  code = _find_function_code(code, _CLOSURE_WRAPPER_ID)
  code = _find_function_code(code, fn.__name__)
  return code


def auto_config(
    fn=None,
    experimental_allow_control_flow=False
) -> Any:  # TODO: More precise return type.
  """Rewrites the given function to make it generate a `Config`.

  This function creates a new function from `fn` by rewriting its AST (abstract
  syntax tree), replacing all `Call` nodes with a custom call handler. When the
  rewritten function is run, the call handler intercepts calls and applies the
  following rules:

    - Calls to builtins, methods, callables without an inferrable signature, or
      other functions that have been `auto_config`ed take place as usual.
    - Calls to `functools.partial` are replaced by calling `fdl.Partial` with
      the same arguments;
    - All other calls are replaced by calling `fdl.Config` with the arguments
      that would have been passed to the called function or class.

  This function may be used standalone or as a decorator. The returned function
  is simply a wrapper around `fn`, but with an additional `as_buildable`
  attribute containing the rewritten function. For example:

      def build_model():
        return Sequential([
            Dense(num_units=128, activation=relu),
            Dense(num_units=128, activation=relu),
            Dense(num_units=1, activation=None),
        ])

      config = auto_config(build_model).as_buildable()

  The resulting `config` is equivalent to the following "manually" constructed
  configuration graph:

      fdl.Config(Sequential, layers=[
          fdl.Config(Dense, num_units=128, activation=relu),
          fdl.Config(Dense, num_units=128, activation=relu),
          fdl.Config(Dense, num_units=1, activation=None),
      ])

  This can then be built with `fdl.build(config)`. Without modification, this
  will result in the same model as just calling `build_model()` directly.
  However, `config` permits changes to the model hyperparameters, for example:

      config.layers[0].num_units = 64
      config.layers[0].activation = 'elu'
      config.layers[1].num_units = 64
      config.layers[1].activation = 'elu'

      modified_model = fdl.build(config)

  Currently, control flow is not supported by default in `auto_config`.
  Experimental support for control flow can be enabled using the
  `experimental_allow_control_flow` argument. If enabled, control flow
  constructs may be used within the function to construct the resulting config
  (for example, a `for` loop could be used to build a list of layers). Control
  flow is never encoded directly as part of the resulting `fdl.Config` (for
  example, there is no `fdl.Config` that will correspond to a conditional or
  loop). While many simple constructs (`for _ in range(10)` etc) work, there
  will also likely be surprising behavior in some circumstances (for example,
  using `itertools` functions in conjunction with a loop will not work, since
  the calls to `itertools` functions will be turned into `fdl.Config` objects).

  Args:
    fn: The function to create a config-generating function from.
    experimental_allow_control_flow: Whether to allow control flow constructs in
      `fn`. By default, control flow constructs will cause an
      `UnsupportedLanguageConstructError` to be thrown.

  Returns:
    A wrapped version of `fn`, but with an additional `as_buildable` attribute
    containing the rewritten function.
  """

  def make_auto_config(fn):
    if isinstance(fn, (staticmethod, classmethod)):
      return type(fn)(make_auto_config(fn.__func__))

    if not inspect.isfunction(fn):
      raise ValueError('`auto_config` is only compatible with functions, '
                       '`@classmethod`s, and `@staticmethod`s.')

    # Get the source code of the function, and remove any indentation which
    # would cause parsing issues when creating the AST (indentation generally is
    # present for nested functions or class methods).
    source = inspect.getsource(fn)
    source = textwrap.dedent(source)

    # Create the NodeTransformer that will transform the AST. The
    # `_AutoConfigNodeTransformer` requires some additional information about
    # the source to provide more informative error messages.
    filename = inspect.getsourcefile(fn)
    line_offset = fn.__code__.co_firstlineno - 1
    node_transformer = _AutoConfigNodeTransformer(
        source=source,
        filename=filename,
        line_offset=line_offset,
        allow_control_flow=experimental_allow_control_flow)

    # Parse the AST, and modify it by intercepting all `Call`s with the
    # `auto_config_call_handler`. Finally, ensure line numbers and code
    # locations match up with the original function, to make errors
    # interpretable.
    node = ast.parse(source)
    node = node_transformer.visit(node)
    node = ast.fix_missing_locations(node)
    node = ast.increment_lineno(node, line_offset)
    assert isinstance(node, ast.Module)

    # In order to allow us to use the original function closure below when
    # constructing a new function object, we have to nest our modified AST
    # within an outer `FunctionDef` that defines variables corresponding to the
    # free variables in `fn`.
    node = _wrap_ast_for_fn_with_closure_vars(node, fn)
    # Compile the modified AST, and then find the function code object within
    # the returned module-level code object.
    code = compile(node, inspect.getsourcefile(fn), 'exec')
    code = _unwrap_code_for_fn(code, fn)

    # Make sure that the proper globals are available to the newly created
    # function, and also add in `auto_config_call_handler`, which is referenced
    # from the modified AST via the `_CALL_HANDLER_ID` (see `ast.Name` node in
    # `_AutoConfigNodeTransformer.visit_Call()`).
    globals_ = {
        **fn.__globals__,
        _CALL_HANDLER_ID: auto_config_call_handler,
    }

    # Then, create a function from the compiled function code object, providing
    # the globals and the original function's closure.
    auto_config_fn = types.FunctionType(code, globals_, closure=fn.__closure__)
    auto_config_fn.__defaults__ = fn.__defaults__
    auto_config_fn.__kwdefaults__ = fn.__kwdefaults__

    # Finally we wrap the rewritten function to perform additional error
    # checking and enforce that the output contains a `fdl.Buildable`.
    @functools.wraps(auto_config_fn)
    def as_buildable(*args, **kwargs):
      output = auto_config_fn(*args, **kwargs)  # pylint: disable=not-callable
      if not _contains_buildable(output):
        raise TypeError(
            f'The `auto_config` rewritten version of `{fn.__qualname__}` '
            f'returned a `{type(output).__name__}`, which is not (or did not '
            'contain) a `fdl.Buildable`. Please ensure this function returns '
            'the result of an `auto_config`-eligible call expression, or a '
            'supported container (list, tuple, dict) containing one.')
      return output

    return AutoConfig(fn, as_buildable)

  # Decorator with empty parenthesis.
  if fn is None:
    return make_auto_config
  else:
    return make_auto_config(fn)


@contextlib.contextmanager
def debug_auto_config():
  """Context manager that enables debugging mode for auto_config.

  Normally, calling an auto_config'ed function `fn(*args, **kwargs)` is
  equivalent to calling `fdl.build(fn.as_buildable(*args, **kwargs))`.  I.e.,
  the auto_config'ed function will first construct a `Buildable` for the
  result, and will then build that result.

  This context manager enables a "debugging mode", where `fn(*args, **kwargs)`
  simply delegates to the underlying function `fn.func(*args, **kwargs)`.  If
  the function has different behavior in normal mode and debugging mode, then
  it's probably using features that are not supported by auto_config (such as
  running code with side effects).

  Note: Fiddle tagging functions, such as `fdl.set_tag` and `MyTag.new`, are
  not supported in debugging mode.

  Yields:
    None.
  """
  global _DEBUG_MODE
  original_value = _DEBUG_MODE
  _DEBUG_MODE = True
  yield
  _DEBUG_MODE = original_value
