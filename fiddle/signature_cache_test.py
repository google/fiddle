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

"""Tests for signature_cache."""

import dataclasses

from absl.testing import absltest
from fiddle import signature_cache


class SignatureCacheTest(absltest.TestCase):

  def test_signature_is_cached(self):

    def some_fn(a, b: int = 1):
      del a, b

    signature_cache.get(some_fn)
    self.assertIn(some_fn, signature_cache._signature_cache)

  def test_can_get_dict_signature(self):
    signature_cache.get(dict)
    self.assertIn(dict, signature_cache._signature_cache)

  def test_can_get_signature_of_unhashable_object(self):

    @dataclasses.dataclass(eq=True)  # Make this unhashable...
    class SomeClass:

      def __call__(self, a, b: int = 1):
        del a, b

    instance = SomeClass()
    self.assertTrue(signature_cache.has(instance))
    signature = signature_cache.get(instance)
    self.assertIn('b', signature.parameters)

  def test_nonexistent_signature(self):

    class SomeClass:
      pass

    self.assertFalse(signature_cache.has(SomeClass()))


if __name__ == '__main__':
  absltest.main()
