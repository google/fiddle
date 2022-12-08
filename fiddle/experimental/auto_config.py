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
import dataclasses
import functools
import inspect
import linecache
import textwrap
import types
from typing import Any, Callable, cast, Optional

from fiddle import building
from fiddle import config
from fiddle._src import mutate_buildable
from fiddle.experimental import auto_config_policy
from fiddle.experimental import daglish_legacy
import libcst as cst

_CALL_HANDLER_ID = '__auto_config_call_handler__'
_CLOSURE_WRAPPER_ID = '__auto_config_closure_wrapper__'
_EMPTY_ARGUMENTS = ast.arguments(
    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
_BUILTINS = frozenset([
    builtin for builtin in builtins.__dict__.values()
    if inspect.isroutine(builtin) or inspect.isclass(builtin)
])


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
  always_inline: bool

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
    return self.func(*args, **kwargs)

  def as_buildable(self, *args, **kwargs) -> config.Buildable:
    return self.buildable_func(*args, **kwargs)

  def __get__(self, obj, objtype=None):
    # pytype: disable=attribute-error
    return AutoConfig(
        func=self.func.__get__(obj, objtype),
        buildable_func=self.buildable_func.__get__(obj, objtype),
        always_inline=self.always_inline)
    # pytype: enable=attribute-error

  @property
  def __wrapped__(self):
    return self.func

  def __getattr__(self, name):
    # Pass through extra things on the thing we wrapped. We use
    # super().__getattribute__('func') here to avoid an infinite recursion.
    return getattr(super().__getattribute__('func'), name)


class UnsupportedLanguageConstructError(SyntaxError):
  pass


class _AutoConfigNodeTransformer(ast.NodeTransformer):
  """A NodeTransformer that adds the auto-config call handler into an AST."""

  def __init__(self,
               source: str,
               filename: str,
               line_number: int,
               allow_control_flow=False):
    """Initializes the auto config node transformer instance.

    Args:
      source: The source code of the node that will be transformed by this
        instance. This is used for better error reporting.
      filename: The filename `source` is from.
      line_number: The line number of `source` within `filename`.
      allow_control_flow: Whether to permit control flow constructs (loops,
        conditionals, comprehensions, etc). By default, this is `False`, and
        control flow constructs will cause an
        `UnsupportedLanguageConstructError` to be raised.
    """
    self._lines = source.splitlines()
    self._filename = filename
    self._line_number = line_number
    self._allow_control_flow = allow_control_flow

    self._function_def_depth = 0

  def _location_for(self, node: ast.AST):
    line_number = self._line_number + node.lineno - 1
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

  def _validate_decorator_ordering(self, node: ast.FunctionDef):
    """Validates that decorators are applied in the right order.

    This is done on a best effort basis to catch cases where @classmethod or
    @staticmethod are applied on top of @auto_config.

    Args:
      node: The `ast.FunctionDef` node to validate decorators for.
    """
    decorator_list = []
    for decorator in node.decorator_list:
      if isinstance(decorator, ast.Attribute):
        decorator = decorator.attr
      if isinstance(decorator, ast.Name):
        decorator = decorator.id
      decorator_list.append(decorator)

    try:
      auto_config_index = decorator_list.index('auto_config')
    except ValueError:
      # Probably auto_config wasn't called as a decorator. Another alternative
      # is that the auto_config function was assigned to a variable with a
      # different name before being applied as a decorator...
      return

    for decorator in decorator_list[:auto_config_index]:
      if decorator in ('classmethod', 'staticmethod'):
        raise AssertionError(
            f'@{decorator} placed above @auto_config on function {node.name} '
            f'at {self._filename}:{self._line_number}. Reorder decorators so '
            f'that @auto_config is placed above @{decorator}.')

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
    return self._handle_control_flow(node, activatable=True)

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
      self._validate_decorator_ordering(node)
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

  daglish_legacy.traverse_with_path(traverse, structure)
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
  free variables, and allows us to use `fn.__closure__` when creating a
  new function object via `types.FunctionType`.

  We also add <_CALL_HANDLER_ID> as a final dummy variable, and append its value
  (the call handler) to `fn.__closure__` when creating the new function object.

  Effectively, this wrapping looks like the following Python code:

      def __auto_config_closure_wrapper__():
        closure_var_1 = None
        closure_var_2 = None
        ...
        <_CALL_HANDLER_ID> = None

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
      for var_name in fn.__code__.co_freevars + (_CALL_HANDLER_ID,)
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


def _make_closure_cell(contents):
  """Returns `types.CellType(contents)`."""
  if hasattr(types, 'CellType'):
    # `types.CellType` added in Python 3.8.
    return types.CellType(contents)  # pytype: disable=wrong-arg-count
  else:
    # For earlier versions of Python, build a dummy function to get CellType.
    dummy_fn = lambda: contents
    cell_type = type(dummy_fn.__closure__[0])
    return cell_type(contents)


def auto_config(
    fn=None,
    *,
    experimental_allow_control_flow: bool = False,
    experimental_always_inline: Optional[bool] = None,
    experimental_exemption_policy: Optional[auto_config_policy.Policy] = None,
    experimental_config_cls=config.Config,
) -> Any:  # TODO(saeta): More precise return type.
  """Rewrites the given function to make it generate a ``Config``.

  This function creates a new function from ``fn`` by rewriting its AST
  (abstract syntax tree), replacing all ``Call`` nodes with a custom call
  handler. When the rewritten function is run, the call handler intercepts calls
  and applies the following rules:

  - Calls to builtins, methods, callables without an inferrable signature, or
    other functions that have been ``auto_config``'ed take place as usual.
  - Calls to ``functools.partial`` are replaced by calling ``fdl.Partial`` with
    the same arguments;
  - All other calls are replaced by calling ``fdl.Config`` with the arguments
    that would have been passed to the called function or class.

  This function may be used standalone or as a decorator. The returned function
  is simply a wrapper around ``fn``, but with an additional ``as_buildable``
  attribute containing the rewritten function. For example::

    def build_model():
      return Sequential([
          Dense(num_units=128, activation=relu),
          Dense(num_units=128, activation=relu),
          Dense(num_units=1, activation=None),
      ])

    config = auto_config(build_model).as_buildable()

  The resulting ``config`` is equivalent to the following "manually" constructed
  configuration graph::

    fdl.Config(Sequential, layers=[
        fdl.Config(Dense, num_units=128, activation=relu),
        fdl.Config(Dense, num_units=128, activation=relu),
        fdl.Config(Dense, num_units=1, activation=None),
    ])

  This can then be built with ``fdl.build(config)``. Without modification, this
  will result in the same model as just calling ``build_model()`` directly.
  However, ``config`` permits changes to the model hyperparameters, for
  example::

      config.layers[0].num_units = 64
      config.layers[0].activation = 'elu'
      config.layers[1].num_units = 64
      config.layers[1].activation = 'elu'

      modified_model = fdl.build(config)

  Currently, control flow is not supported by default in ``auto_config``.
  Experimental support for control flow can be enabled using the
  ``experimental_allow_control_flow`` argument. If enabled, control flow
  constructs may be used within the function to construct the resulting config
  (for example, a ``for`` loop could be used to build a list of layers). Control
  flow is never encoded directly as part of the resulting ``fdl.Config`` (for
  example, there is no ``fdl.Config`` that will correspond to a conditional or
  loop). While many simple constructs (``for _ in range(10)`` etc) work, there
  will also likely be surprising behavior in some circumstances (for example,
  using ``itertools`` functions in conjunction with a loop will not work, since
  the calls to ``itertools`` functions will be turned into ``fdl.Config``
  objects).

  Using ``@auto_config`` is compatible with both ``@staticmethod`` and
  ``@classmethod``, however the ``@auto_config`` decorator must appear above the
  ``@classmethod`` or ``@staticmethod`` in the decorator list.

  Args:
    fn: The function to create a config-generating function from.
    experimental_allow_control_flow: Whether to allow control flow constructs in
      ``fn``. By default, control flow constructs will cause an
      ``UnsupportedLanguageConstructError`` to be thrown.
    experimental_always_inline: If true, this function (when called in an
      ``auto_config`` context) will always be ``inline``'d in-place. See the
      documentation on ``inline`` for an example. The default (if unspecified)
      is currently ``False``, but this may change in the future.
    experimental_exemption_policy: An optional policy to control which function
      calls within the body of ``fn`` should be turned into ``fdl.Config``'s and
      which ones should simply be executed normally during the ``as_buildable``
      interpretation of ``fn``. This predicate should return ``True`` if the
      given callable should be exempted from auto-configuration.
    experimental_config_cls: The class to use to generate configs. By default,
      this is just ``fdl.Config``, but projects with custom ``Config``
      subclasses can use this to override the default. This is experimental and
      may be removed in the future.

  Returns:
    A wrapped version of ``fn``, but with an additional ``as_buildable``
    attribute containing the rewritten function.
  """
  if experimental_always_inline is None:
    experimental_always_inline = True

  if experimental_exemption_policy is None:
    experimental_exemption_policy = auto_config_policy.latest

  def auto_config_call_handler(fn_or_cls, *args, **kwargs):
    """Handles calls in auto_config'ed functions.

    This intercepts calls in an auto-configed function, and determines whether
    the called `fn_or_cls` should be wrapped in a `Config` or `Partial`. If
    `fn_or_cls` is `functools.partial`, the call will instead be converted into
    a call to Fiddle's `Partial`. If it is "auto-config eligible" (see
    `experimental_custom_call_policy`), then a `Config` will be create for
    `fn_or_cls` with the provided arguments. Otherwise, `fn_or_cls` is called
    directly.

    Args:
      fn_or_cls: The function or class being called.
      *args: The positional arguments with which `fn_or_cls` is being called.
      **kwargs: The keyword arguments with which `fn_or_cls` is being called.

    Returns:
      Depending on `fn_or_cls`, either `Partial`, a `Config`, or the result of
      calling `fn_or_cls` with the provided arguments.
    """
    if isinstance(fn_or_cls, AutoConfig) and fn_or_cls.always_inline:
      return fn_or_cls.as_buildable(*args, **kwargs)

    if fn_or_cls is functools.partial:
      return config.Partial(args[0], *args[1:], **kwargs)

    if experimental_exemption_policy(fn_or_cls):
      return fn_or_cls(*args, **kwargs)

    return experimental_config_cls(fn_or_cls, *args, **kwargs)

  def make_auto_config(fn):
    if not isinstance(fn, (types.FunctionType, classmethod, staticmethod)):
      raise ValueError('`auto_config` is only compatible with functions, '
                       f'`@classmethod`s, and `@staticmethod`s.  Got {fn!r} '
                       f'with type {type(fn)!r}.')

    if isinstance(fn, (classmethod, staticmethod)):
      method_type = type(fn)
      fn = fn.__func__
    else:
      method_type = None

    source = _getsource(fn)

    # Create the NodeTransformer that will transform the AST. The
    # `_AutoConfigNodeTransformer` requires some additional information about
    # the source to provide more informative error messages.
    filename = inspect.getsourcefile(fn)
    line_number = fn.__code__.co_firstlineno
    node_transformer = _AutoConfigNodeTransformer(
        source=source,
        filename=filename,
        line_number=line_number,
        allow_control_flow=experimental_allow_control_flow)

    # Parse the AST, and modify it by intercepting all `Call`s with the
    # `auto_config_call_handler`. Finally, ensure line numbers and code
    # locations match up with the original function, to make errors
    # interpretable.
    node = ast.parse(source)
    node = node_transformer.visit(node)
    node = ast.fix_missing_locations(node)
    node = ast.increment_lineno(node, line_number - 1)
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

    # Insert auto_config_call_handler into `fn.__closure__` at the index where
    # _CALL_HANDLER_ID occurs in the freevars.  (_CALL_HANDLER_ID was added to
    # freevars by _wrap_ast_for_fn_with_closure_vars).
    closure = list(fn.__closure__ or ())
    if _CALL_HANDLER_ID in code.co_freevars:
      closure.insert(
          code.co_freevars.index(_CALL_HANDLER_ID),
          _make_closure_cell(auto_config_call_handler))
    closure = tuple(closure)

    # Then, create a function from the compiled function code object, providing
    # the globals and the original function's closure.
    auto_config_fn = types.FunctionType(code, fn.__globals__, closure=closure)
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

    if method_type:
      fn = method_type(fn)
      as_buildable = method_type(as_buildable)
    return AutoConfig(
        fn, as_buildable, always_inline=experimental_always_inline)

  # Decorator with empty parenthesis.
  if fn is None:
    return make_auto_config
  else:
    return make_auto_config(fn)


def auto_unconfig(
    fn=None,
    *,
    experimental_always_inline: Optional[bool] = None
) -> Any:  # TODO(saeta): More precise return type.
  """Converts functions that create buildables directly into auto_config form.

  While most of the time, the benefits of an auto_config representation of
  object configuration and construction are valuable (e.g. static type
  checking and tooling / refactoring support), sometimes it is more convenient
  to manipulate buildable objects directly. ``auto_unconfig`` converts a
  function that directly manipulates ``fdl.Buildable``'s (e.g. ``fdl.Config``'s)
  into one that looks identically to an ``auto_config``'d function, and is fully
  interoperable with the rest of the ``auto_config`` ecosystem.

  Example::

    @auto_unconfig
    def make_experiment_trainer(name: str) -> fdl.Config[MyTrainer]
      model = make_model.as_buildable(name)
      select(model, DropOut).set(rate=0.42)  # Full Fiddle API available!
      dataset = make_dataset.as_buildable()
      # Build fdl.Config's imperatively.
      trainer_config = fdl.Config(MyTrainer)
      trainer_config.model = model
      trainer_config.train_dataset = dataset
      trainer_config.skip_eval = True
      return trainer_config  # Return a `fdl.Buildable`

    # Sample usage within an auto_config'd function.
    @auto_config
    def make_driver():
      return TrainerDriver(
        trainer=make_experiment_trainer('my_experiment'),
        checkpointer=CustomCheckpointer())

    # Sample usage outside of auto_config contexts.
    def main():
      # Use instantiated objects:
      trainer = make_experiment_trainer('my_experiment')
      for example in trainer.train_dataset:
        print_prediction(trainer.model.predict(example))

      # Or manipulate the configuration before calling `fdl.build`:
      trainer_config = make_experiment_trainer.as_buildable('my_experiment')
      trainer_config.skip_eval = False  # Tweak configuration.
      trainer2 = fdl.build(trainer_config)
      run_trainer(trainer2)

  Args:
    fn: The function to convert.
    experimental_always_inline: Whether the output of ``fn`` should always be
      inlined into the caller's config.

  Returns:
    An ``AutoConfig`` that corresponds to ``fn``.
  """

  if experimental_always_inline is None:
    experimental_always_inline = True

  def make_unconfig(fn) -> AutoConfig:

    @functools.wraps(fn)
    def python_implementation(*args, **kwargs):
      previous = building._state.in_build  # pytype: disable=module-attr # pylint: disable=protected-access
      building._state.in_build = False  # pytype: disable=module-attr # pylint: disable=protected-access
      try:
        cfg = fn(*args, **kwargs)
        return building.build(cfg)
      finally:
        building._state.in_build = previous  # pytype: disable=module-attr # pylint: disable=protected-access

    return AutoConfig(
        func=python_implementation,
        buildable_func=fn,
        always_inline=experimental_always_inline)

  # We use this pattern to support using the decorator with and without
  # parenthesis.
  if fn is None:
    return make_unconfig
  return make_unconfig(fn)


def is_auto_config(function_object: Any) -> bool:
  return isinstance(function_object, AutoConfig)


def inline(buildable: config.Config):
  """Converts an ``auto_config``-based ``buildable`` into a DAG of Buildables.

  ``inline`` updates ``buildable`` in place to preserve aliasing within a larger
  Fiddle configuration. If you would like to leave ``buildable`` unmodified,
  make a shallow copy (``copy.copy``) before calling ``inline``.

  Example::

    # shared/input_pipelines.py
    @auto_config(experimental_api_boundary=True)
    def make_input_pipeline(name: str, batch_size: int) -> InputPipeline:
      file_path = '/base_path/'+name
      augmentation = 'my_augmentation_routine'
      # ...
      return InputPipeline(file_path, augmentation, ...)

    # config/main.py
    @auto_config
    def make_experiment():
      data = make_input_pipeline('normal_dataset', batch_size)
      model = ...
      return Experiment(data, model)

    # experiment_configuration.py
    def make_experiment():
      config = make_experiment.as_buildable()
      config.data.name = 'advanced_dataset'
      # config.data.augmentation = 'custom_augmentation'  # Not configurable!!!
      # return fdl.build(config)                          # Works like normal.
      auto_config.inline(config.data)
      print(config.data.file_path)       # Prints: '/base_path/advanced_dataset'
      config.data.augmentation = 'custom_augmentation'    # Now exposed.
      experiment = fdl.build(config)                      # Works like normal.
      return experiment

  Args:
    buildable: The buildable of an ``auto_config``'d function to replace with
      the root of a Fiddle DAG that corresponds to it.

  Raises:
    ValueError: If ``buildable`` is not a ``Config``, or if ``buildable``
      doesn't correspond to an ``auto_config``'d function.
  """
  if not isinstance(buildable, config.Config):
    raise ValueError('Cannot `inline` non-Config buildables; '
                     f'{type(buildable)} is not compatible.')
  if not is_auto_config(buildable.__fn_or_cls__):
    raise ValueError('Cannot `inline` a non-auto_config function; '
                     f'`{buildable.__fn_or_cls__}` is not compatible.')
  # Evaluate the `as_buildable` interpretation.
  auto_config_fn = cast(AutoConfig, buildable.__fn_or_cls__)
  tmp_config = auto_config_fn.as_buildable(**buildable.__arguments__)
  if not isinstance(tmp_config, config.Buildable):
    raise ValueError('You cannot currently inline functions that do not return '
                     '`fdl.Buildable`s.')

  mutate_buildable.move_buildable_internals(
      source=tmp_config, destination=buildable)


def _getsource(fn: Any) -> str:
  """Returns the source code for callable `fn`."""
  if _is_lambda(fn):
    return _getsource_for_lambda(fn)
  else:
    # Remove any indentation which would cause parsing issues when creating the
    # AST (indentation generally is present for nested functions or class
    # methods).
    return textwrap.dedent(inspect.getsource(fn))


def _is_lambda(fn: Any) -> bool:
  """Returns True if `fn` is a lambda function."""
  if not inspect.isfunction(fn):
    return False
  if not (hasattr(fn, '__name__') and hasattr(fn, '__code__')):
    return False
  return ((fn.__name__ == '<lambda>') or (fn.__code__.co_name == '<lambda>'))


class _LambdaFinder(cst.CSTVisitor):
  """CST Visitor that searches for the source code for a given lambda func."""
  METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

  def __init__(self, lambda_fn):
    super().__init__()
    self.lambda_fn = lambda_fn
    self.lineno = lambda_fn.__code__.co_firstlineno
    self.candidates = []

  def visit_Lambda(self, node) -> None:
    loc = self.get_metadata(cst.metadata.PositionProvider, node)
    if loc.start.line == self.lineno:
      self.candidates.append(node)


def _getsource_for_lambda(fn: Callable[..., Any]) -> str:
  """Returns source code for the given lambda function."""
  # Get the source for the module that defines `fn`.
  module = inspect.getmodule(fn)
  filename = inspect.getsourcefile(fn)
  lines = linecache.getlines(filename, module.__dict__)
  source = ''.join(lines)

  # Parse the CST for the module, and search for the lambda.
  module_cst = cst.parse_module(source)
  lambda_finder = _LambdaFinder(fn)
  cst.metadata.MetadataWrapper(module_cst).visit(lambda_finder)

  if len(lambda_finder.candidates) == 1:
    lambda_node = lambda_finder.candidates[0]
    return cst.Module(body=[lambda_node]).code

  elif not lambda_finder.candidates:
    raise ValueError(
        'Fiddle auto_config was unable to find the source code for '
        f'{fn}: could not find lambda on line {lambda_finder.lineno}.')
  else:
    # TODO(b/258671226): If desired, we could narrow down which lambda is
    # used based on the signature (or even fancier things like the checking
    # fn.__code__.co_names).
    raise ValueError(
        'Fiddle auto_config was unable to find the source code for '
        f'{fn}: multiple lambdas found on line {lambda_finder.lineno}; '
        'try moving each lambda to its own line.')
