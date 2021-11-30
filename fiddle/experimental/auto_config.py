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
import functools
import inspect
import textwrap
import types

from fiddle import config

_CALL_HANDLER_ID = '__auto_config_call_handler__'
_BUILTINS = frozenset([
    builtin for builtin in builtins.__dict__.values()
    if inspect.isroutine(builtin) or inspect.isclass(builtin)
])


def _is_auto_config_eligible(fn_or_cls):
  """Helper to determine if `fn_or_cls` is eligible for auto-config."""
  try:
    has_signature = inspect.signature(fn_or_cls) is not None
  except ValueError:
    has_signature = False

  is_buildable = (
      inspect.isclass(fn_or_cls) and issubclass(fn_or_cls, config.Buildable))

  return (
      has_signature and
      fn_or_cls not in _BUILTINS and
      not is_buildable and
      not inspect.ismethod(fn_or_cls) and
      not getattr(fn_or_cls, 'auto_config', False)
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

  return fn_or_cls(*args, **kwargs)


class _AutoConfigNodeTransformer(ast.NodeTransformer):
  """A NodeTransformer that adds the auto-config call handler into an AST."""

  def visit_Call(self, node):  # pylint: disable=invalid-name
    return ast.Call(
        func=ast.Name(id=_CALL_HANDLER_ID, ctx=ast.Load()),
        args=[node.func, *(self.visit(arg) for arg in node.args)],
        keywords=[self.visit(keyword) for keyword in node.keywords],
    )


def auto_config(fn):
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

  This function may be used standalone or as a decorator. For example:

      def build_model():
        layers = []
        for _ in range(2):
          layers.append(Dense(num_units=128, activation=relu))
        layers.append(Dense(num_units=1, activation=None))
        mlp = Sequential(layers=layers)
        return mlp

      build_model_cfg = auto_config(build_model)
      config = build_model_cfg()

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

  While `auto_config` is intended to work well with simple control flow
  constructs and common built-ins, care should be taken with more complex
  constructs (for example certain library calls, like to `itertools` functions,
  may be turned into `fdl.Config` instances).

  Args:
    fn: The function to rewrite into a config-generating function.

  Returns:
    The rewritten function.
  """
  if not inspect.isfunction(fn):
    raise ValueError('Auto-configuration is only compatible with functions.')

  # Get the source code of the function, and remove any indentation which would
  # cause parsing issues when creating the AST (indentation generally is present
  # for nested functions or class methods).
  source = inspect.getsource(fn)
  source = textwrap.dedent(source)

  # Parse the AST, and modify it by intercepting all `Call`s with the
  # `auto_config_call_handler`. Finally, ensure line numbers and code locations
  # match up with the original function, to make errors interpretable.
  node = ast.parse(source)
  node = _AutoConfigNodeTransformer().visit(node)
  node = ast.fix_missing_locations(node)
  node = ast.increment_lineno(node, fn.__code__.co_firstlineno - 1)

  # Compile the modified AST, and then find the function code object within the
  # returned module-level code object. Generally, the function is present at
  # `code.co_consts[0]`, but the comprehension below finds it by name. Assuming
  # compilation was successful, the function should really be there, so an
  # assert is used to verify that it was found.
  code = compile(node, inspect.getsourcefile(fn), 'exec')
  code = [
      const for const in code.co_consts  # pytype: disable=attribute-error
      if inspect.iscode(const) and const.co_name == fn.__name__
  ]
  assert len(code) == 1, "Couldn't find modified function code."
  code = code[0]

  # Make sure that the proper globals and closure variables are available to the
  # newly created function, and also add in `auto_config_call_handler`, which is
  # referenced from the modified AST via the `_CALL_HANDLER_ID` (see `ast.Name`
  # node in `_AutoConfigNodeTransformer.visit_Call()`).
  closure_vars = inspect.getclosurevars(fn)
  scope = {
      **fn.__globals__,
      **closure_vars.globals,
      **closure_vars.nonlocals,
      _CALL_HANDLER_ID: auto_config_call_handler,
  }

  # Finally, create a function from the compiled function code object, providing
  # the scope. Mark it as an auto-config function so that calls to this function
  # from other auto-config functions aren't intercepted (this is checked in
  # `auto_config_call_handler` above.
  auto_config_fn = types.FunctionType(code, scope, argdefs=fn.__defaults__)
  auto_config_fn.auto_config = True
  return auto_config_fn
