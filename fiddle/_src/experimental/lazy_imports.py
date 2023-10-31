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

"""Lazy imports API and utils."""

import builtins
import contextlib
import dataclasses
import functools
import importlib
import inspect
import logging
from typing import Any, Iterator, Optional


@contextlib.contextmanager
def lazy_imports(kw_only=True) -> Iterator[None]:
  """Context manager which replaces import statements with lazy imports.

  A ProxyObject is returned for the import statement as dummy module. The
  actual module is not imported until calling `fdl.build()` on Fiddle
  `Buildable` objects.


  Usage:

  ```python
  with fdl.experimental.lazy_imports():
    import foo.bar.baz as model

  config = fdl.Config(model.MyModel, layer_num=3, dtype='float16')

  my_model = fdl.build(config)  # The module is not actually loaded until now.
  ```

  Warning: Please import modules only when using lazy imports, and do not import
   classes or functions directly. Because it would confuse the API what's the
   correct module path to import.

  ```
  # Errors will be raised when importing classes/functions directly.
  with fdl.experimental.lazy_imports():
    from foo.bar.baz.my_module import MyModel  # Wrong usage!

  # Instead, you should import the module directly only.
  with fdl.experimental.lazy_imports():
    from foo.bar.baz import my_module  # Correct usage!

  config = fdl.Config(my_module.MyModel, ...)
  ```

  Be aware that when using lazy imported modules within `fdl.Partial`, lazy
  imported modules are not loaded until calling the object after `fdl.build()`.
  This is differet from `fdl.Config` where the modules are loaded when calling
  `fdl.build()`.

  ```
  with lazy_imports.lazy_imports(kw_only=True):
    from path.to.the.module import module_1
    from path.to.the.module import module_2

  config = fdl.Config(module_1.MyModel)
  my_model = fld.build(config)  # module_1 is loaded here.

  partial = fdl.Partial(module_2.my_function)
  my_fn = fdl.build(partial)  # module_2 not loaded here.
  assert isinstance(my_fn.func, ProxyObject)
  out = my_fn(x=1, ...)  # module_2 is loaded here during call time.
  ```

  NOTE: When using positional arguments for lazy-imported functions/classes,
  be aware that the config will likely be different from normal-imported
  functions/classes. As Fiddle does not have signature information for
  lazy-imported modules, values for positional arguments, like value 'v' in the
  example below, will be saved as positional-only arguments, and using the index
  0 as the key in `__arguments__`, even if it actually corresponds to a
  positional-or-keyword argument.

  # In foo.py
  def bar(x):
    return x

  # In file_1.py
  with fdl.lazy_import():
    import foo
  cfg_1 = fdl.Config(foo.bar, 'v')
  assert cfg_1.__arguments__ == {0: 'v'}

  # In file_2.py
  import foo
  cfg_2 = fdl.Config(foo.bar, 'v')
  assert cfg_2.__arguments__ == {'x': 'v'}


  Args:
    kw_only: Whether to only allow keyword args usage for lazy imported modules.
      Defaults to True.

  Yields:
    None
  """
  if not kw_only:
    logging.warning(
        'When using positional arguments for lazy-imported functions/classes,'
        ' be aware that the config will likely be different from'
        'normal-imported functions/classes.'
    )

  # Need to mock `__import__` (instead of `sys.meta_path`, as we do not want
  # to modify the `sys.modules` cache in any way)
  origin_import = builtins.__import__
  try:
    builtins.__import__ = functools.partial(
        _fake_import, kw_only=kw_only, proxy_cls=ProxyObject
    )
    yield
  finally:
    builtins.__import__ = origin_import


def _fake_import(
    name: str,
    globals_=None,
    locals_=None,
    fromlist: Any = (),
    level: int = 0,
    *,
    kw_only,
    proxy_cls,
):
  """Mock of `builtins.__import__`."""
  del globals_, locals_  # Unused

  if level:
    raise ValueError(f'Relative import statements not supported ({name}).')

  root_name, *parts = name.split('.')
  root = proxy_cls.from_cache(name=root_name, kw_only=kw_only)
  root.is_import = True
  if not fromlist:
    # import x.y.z
    # import x.y.z as z

    # Register the child modules
    childs = root
    for name in parts:
      childs = childs.child_import(name)

    return root
  else:
    # from x.y.z import a, b

    # Return the inner-most module
    for name in parts:
      root = root.child_import(name)
    # Register the child imports
    for name in fromlist:
      root.child_import(name)
    return root


@dataclasses.dataclass(eq=False)
class ProxyObject:
  """Base class to represent a module, function,..."""

  name: str
  kw_only: bool = False
  parent: Optional['ProxyObject'] = None
  # Whether or not the attribute was from an import or not:
  # `import a.b.c` vs `import a ; a.b.c`
  is_import: bool = False
  loaded_obj: Any = None

  @classmethod
  @functools.lru_cache(maxsize=1024)
  def from_cache(cls, **kwargs):
    """Factory to cache all instances of module.

    Note: The cache is global to all instances of the
    `fake_import` contextmanager.

    Args:
      **kwargs: Init kwargs

    Returns:
      New object
    """
    return cls(**kwargs)

  @property
  def qualname(self) -> str:
    if not self.parent:
      return self.name

    if self.parent.is_import and not self.is_import:
      separator = ':'
    else:
      separator = '.'

    return f'{self.parent.qualname}{separator}{self.name}'

  def __repr__(self) -> str:
    return f'{type(self).__name__}({self.qualname!r})'

  def __getattr__(self, name: str) -> Any:
    if name == '__signature__':
      args = inspect.Parameter('args', inspect.Parameter.VAR_POSITIONAL)
      kwargs = inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD)
      if self.kw_only:
        return inspect.Signature([
            kwargs,
        ])
      else:
        return inspect.Signature([
            args,
            kwargs,
        ])
    if name == '__qualname__':
      return self.qualname
    if name == '__name__':
      return f'ProxyObject_{self.qualname}'
    ret = type(self).from_cache(
        name=name,
        kw_only=self.kw_only,
        parent=self,
        is_import=False,
    )
    return ret

  def __hash__(self):
    return hash((self.qualname, self.parent, self.kw_only))

  def __eq__(self, other):
    if not isinstance(other, ProxyObject):
      return False
    return self.qualname == other.qualname and self.parent == other.parent

  def child_import(self, name: str) -> 'ProxyObject':
    """Returns the child import."""
    obj = getattr(self, name)
    # The cache is shared, so the 2 objects are the same:
    # import a ; a.b.c
    # from a.b import c
    obj.is_import = True
    return obj

  def __call__(self, *args, **kwargs):
    if self.loaded_obj is None:
      self.loaded_obj = import_module(self.qualname)
    if self.kw_only and args:
      return ValueError(
          'This lazily imported function/class is keyword arguments only.'
          f'Got positional arguments: {args}. '
      )
    return self.loaded_obj(*args, **kwargs)


def import_module(qualname_str: str) -> Any:
  """Import module/function/class from a module qualname."""
  parts = qualname_str.split(':')
  if len(parts) == 2:
    [import_str, attributes] = parts
  elif len(parts) == 1:  # Otherwise, assume single attribute
    [qualname_str] = parts
    import_str, attributes = qualname_str.rsplit('.', maxsplit=1)
  else:
    raise ValueError(f'Invalid {qualname_str!r}')

  obj = importlib.import_module(import_str)
  for attr in attributes.split('.'):
    obj = getattr(obj, attr)
  return obj  # pytype: disable=bad-return-type
