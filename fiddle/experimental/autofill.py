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

"""Simplifies deep-configurability of nested objects in Fiddle-aware codebases.

While Fiddle Config's support arbitrary nesting of objects, building up the full
tree of `fdl.Config`'s can be somewhat tedious. The extension in this file
allows end-users of annotated codebases to construct a single `fdl.Config`
corresponding to the root of the tree, and Fiddle will automatically instantiate
`fdl.Config`'s corresponding to every nested object based on a type annotation.
As an example, given the following types::

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

simple instantiate the root config, and all the rest are already instantiated::

  def with_autofill():
    root = fdl.Config(Root)
    assert isinstance(root.a.b.c, fdl.Config)  # Same for all the other fields.
    assert not hasattr(root.b.name)
    return root

Because it's incorrect to automatically instantiate `fdl.Config`'s for every
field (e.g. ``int`` and ``str`` fields), Autofill relies on annotations added to
the library types as follows::

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

Note: tags can be added in an analogous fashion (once this feature has been
implemented...).
"""
from fiddle._src import autofill

Autofill = autofill.Autofill
