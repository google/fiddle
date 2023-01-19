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

"""Internal implementation of autofill functionality.

For additional context, please see the module comment in the API
file.
"""

import dataclasses
import inspect
from typing import Any, Callable, Dict, Optional, Type
import typing_extensions


# TODO(b/265850310): Mark as kw_only when available (Py3.10).
# TODO(b/266030644): Upgrade sphinx and enable section headings.
#                    see also: https://github.com/sphinx-doc/sphinx/pull/10919.
@dataclasses.dataclass(frozen=True)
class Autofill:
  """Annotation of an argument parameter to enable deep configurability.

  Sample usage: ``my_arg: typing.Annotated[UnderlyingType, fdl.Autofill]``. Read
  on for more context.

  **Complete Example**

  While Fiddle Configs support arbitrary nesting of objects, building up the
  full tree of `fdl.Config`'s can be somewhat tedious. The extension in this
  file allows end-users of annotated codebases to construct a single
  ``fdl.Config`` corresponding to the root of the tree, and Fiddle will
  automatically instantiate ``fdl.Config``'s corresponding to every nested
  object based on a type annotation. As an example, given the following types::

    class Root:
      a: A
      b: B

    class A:
      b: B
      x: int

    class B:
      name: str
      c: C

    class C:
      weight: float

  instead of building up the full tree imperatively::

    def without_autofill():
      root = fdl.Config(Root)
      root.a = fdl.Config(A)
      root.a.b = fdl.Config(B)
      root.a.b.c = fdl.Config(C)
      root.b = fdl.Config(B)
      root.b.c = fdl.Config(C)
      return root

  simply instantiate the root config, and all the rest are already
  instantiated::

    def with_autofill():
      root = fdl.Config(Root)
      assert isinstance(root.a.b.c, fdl.Config)  # Same for all other fields.
      assert not hasattr(root.b.name)
      return root

  Because it's incorrect to automatically instantiate `fdl.Config`'s for every
  field (e.g. ``int`` and ``str`` fields), Autofill relies on annotations added
  to the library types as follows::

    class Root:
      a: typing.Annotated[A, fdl.Autofill]
      b: typing.Annotated[B, fdl.Autofill]

    class A:
      b: typing.Annotated[B, fdl.Autofill]
      x: int

    class B:
      name: str
      c: typing.Annotated[C, fdl.Autofill]

  Once the annotations are added, the ``with_autofill`` example will work.

  **Customizing the factory**

  If a library author would like to use a custom function to initialize the
  ``fdl.Buildable`` for a particular argument, they can provide the ``factory``
  argument. For example::

    class MyComplicatedClass:
      x: str = 'wrong default'

    @auto_config
    def default_complicated_class():
      return MyComplicatedClass(x='right default')

    class Root:
      c: typing.Annotated[MyComplicatedClass, fdl.Autofill(
            factory=default_complicated_class.as_buildable)]

    def make_a_config():
      config = fdl.Config(Root)
      assert config.c.x == 'right default'

  > Note: you can use a non-auto_config (aka plain Python) function as well, so
  > long as it returns a ``fdl.Buildable`` instance.

  **Sharing / Aliasing**

  By default, each parameter gets its own ``fdl.Config`` instance; there is no
  aliasing in the constructed tree of ``fdl.Config``'s. If it's important to
  share an instance among two or more parameters (e.g. reuse an expensive object
  such as an embedding layer), use a factory to set up the sharing.

  The following example shares one ``Tower`` instance between the
  ``encoder`` and the ``decoder``::

    @auto_config
    def make_encoder_decoder():
      shared = Tower()
      return EncoderDecoder(encoder=shared, decoder=shared)

    class MyModel(flax.linen.Module):
      encoder_decoder: typing.Annotated[
          EncoderDecoder,
          fdl.Autofill(factory=make_encoder_decoder.as_buildable)]

    config = fdl.Config(MyModel)
    assert config.encoder_decoder.encoder is config.encoder_decoder.decoder
    model = fdl.build(config)
    assert model.encoder_decoder.encoder is model.encoder_decoder.decoder

  **Custom Buildable Type**

  Highly sophisticated Fiddle users might extend ``fdl.Config`` (e.g. to
  facilitate adopting Fiddle in a codebase with non-Fiddle design patterns) with
  a subclass. Specify this custom type as the ``buildable_type`` to cause the
  instantiated autofilled buildable's to be ``buildable_type`` instead of
  ``fdl.Config``. For example::

    class CustomConfig(fdl.Config):
      def instantiate(self):
        "A custom method to facilitate migrating to Fiddle."
        return fdl.build(self)

    class MyClass:
      def __init__(
          self,
          autofill_me: typing.Annotated[
              MyClass,
              fdl.Autofill(buildable_type=CustomConfig),
          ],
      ):
        self.autofill_me = autofill_me

  Attributes:
    factory: A zero-arity (no-parameter) function that returns a subclass of
      ``fdl.Buildable`` suitably configured for this argument. This must be
      ``None`` if ``buildable_type`` is not ``None``.
    buildable_type: A ``fdl.Buildable`` subclass. ``buildable_type`` must be
      ``None`` if ``factory`` is not ``None``.
  """

  factory: Optional[Callable[[], Any]] = None
  buildable_type: Optional[Type[Any]] = None

  def __post_init__(self):
    if self.factory and self.buildable_type is not None:
      raise ValueError('Cannot set both `factory` and `buildable_type`.')


@dataclasses.dataclass(frozen=True)
class ParameterMetadata:
  """Autofill-related metadata for a parameter."""

  autofill_annotation: Optional[Autofill]
  underlying_type: Type[Any]

  @property
  def factory(self) -> Optional[Callable[[], Any]]:
    if self.autofill_annotation:
      return self.autofill_annotation.factory
    return None

  @property
  def buildable_type(self) -> Optional[Type[Any]]:
    if self.autofill_annotation:
      return self.autofill_annotation.buildable_type
    return None


# Set to `True` to see type annotation errors.
# TODO(b/265956870): Make this more user discoverable.
_DEBUG_TYPE_ANNOTATIONS = False


def parameters_to_autofill(
    obj: Any,
    signature: inspect.Signature,
) -> Dict[str, ParameterMetadata]:
  """Inspects `obj` to determine all parameters annotated with autofill.

  This function is used only within the Fiddle library and is not part of the
  Fiddle API.

  Args:
    obj: The arbitrary callable that a ``fdl.Buildable`` is being instantiated
      for.
    signature: The signature of ``obj``.

  Returns:
    A dictionary mapping parameter names annotated with ``Autofill`` to
    autofill-related parameter metadata.
  """
  autofill_params = {}

  if isinstance(obj, type):
    # Use `__init__`'s annotations.
    try:
      annotations = typing_extensions.get_type_hints(
          obj.__init__, include_extras=True
      )
    except (TypeError, NameError):
      # Can occur when the type annotations don't actually refer to types, or
      # there's an unknown name.
      if _DEBUG_TYPE_ANNOTATIONS:
        raise
      return {}
  else:
    try:
      annotations = typing_extensions.get_type_hints(obj, include_extras=True)
    except (TypeError, NameError):
      # Happens when `obj` is not a module, class, method, or function, or when
      # the types (e.g. Any) isn't appropriate imported.
      if _DEBUG_TYPE_ANNOTATIONS:
        raise
      return {}
  for param in signature.parameters:
    annotation = annotations.get(param)
    if (
        typing_extensions.get_origin(annotation)
        is not typing_extensions.Annotated
    ):
      continue
    type_args = typing_extensions.get_args(annotation)
    autofill_args = list(
        filter(
            lambda arg: arg is Autofill or isinstance(arg, Autofill),
            type_args,
        )
    )
    if not autofill_args:
      continue  # No autofill information. Head to the next arg.
    if len(autofill_args) > 1:
      raise TypeError('Cannot specify autofill information more than once.')
    annotation = autofill_args[0]
    if annotation is Autofill:
      annotation = None  # No annotation-specific information; defaults only.
    metadata = ParameterMetadata(
        autofill_annotation=annotation, underlying_type=type_args[0]
    )
    autofill_params[param] = metadata
  return autofill_params
