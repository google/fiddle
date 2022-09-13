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

"""Tests for namespace."""

from absl.testing import absltest
from fiddle.codegen import namespace


class NamespaceTest(absltest.TestCase):

  def test__camel_to_snake(self):
    self.assertEqual(namespace._camel_to_snake("FooBar"), "foo_bar")
    self.assertEqual(namespace._camel_to_snake("fooBar"), "foo_bar")

  def test_namespace_contains_builtins(self):
    self.assertIn("for", namespace.Namespace())
    self.assertNotIn("foo", namespace.Namespace())

  def test_namespace_can_override_builtins(self):
    self.assertNotIn("for", namespace.Namespace(names=set()))

  def test_namespace_add(self):
    ns = namespace.Namespace()
    self.assertEqual(ns.add("foo"), "foo")
    self.assertIn("foo", ns)

  def test_namespace_double_add(self):
    ns = namespace.Namespace()
    self.assertEqual(ns.add("foo"), "foo")
    with self.assertRaisesRegex(ValueError, "Tried to add.*already exists"):
      ns.add("foo")

  def test_namespace_get_new_name(self):
    ns = namespace.Namespace()
    self.assertEqual(ns.get_new_name("foo", ""), "foo")
    self.assertEqual(ns.get_new_name("foo", ""), "foo_2")


if __name__ == "__main__":
  absltest.main()
