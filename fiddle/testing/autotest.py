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

"""Dynamically generated test suites to verify fiddle configs build.

Software engineering can be described as programming integrated over time. (See:
https://www.oreilly.com/library/view/software-engineering-at/9781492082781/ch01.html)
Because Fiddle is engineered for the entire project lifecycle, Fiddle includes
testing utilities to ensure Fiddle configurations stay valid.

This file is a collection of reusable utilities to generate your own customized
tests for advanced users, as well as the business logic to power the
`fiddle_autotest` bazel macro.
"""

import functools
import importlib
from typing import Any, Optional, Type
import unittest

from absl import flags
from absl.testing import absltest
from fiddle import building
from fiddle import module_reflection

Module = Any
# Note: although it's always a subclass of absltest.TestCase, we can't use
# Type[absltest.TestCase] because we have added methods to it, and thus type
# checking fails.
CustomTestClass = Type[Any]

_FLAG_FIDDLE_CONFIG_MODULE = flags.DEFINE_string(
    'fiddle_config_module', None,
    'The name of the Python module containing the configurations and fiddlers '
    'to be tested.')

_FLAG_SKIP_BUILDING = flags.DEFINE_boolean(
    'skip_building', False, 'Skips calling `fdl.build` in the test cases.')


def make_test_class_for_module(module: Module,
                               *,
                               skip_building: bool = False) -> CustomTestClass:
  """Returns a `absltest.TestCase` subclass appropriate for testing `module`."""

  test_functions = {}

  # Tests for all base configurations.
  for name in module_reflection.find_base_config_like_things(module):

    def test_base_config(self, module, fn_name):
      self.assertTrue(hasattr(module, fn_name))
      fn = getattr(module, fn_name)
      cfg = fn()
      if not skip_building:
        _ = building.build(cfg)

    fn = functools.partialmethod(test_base_config, module, name)
    fn.__name__ = f'test_{name}'
    test_functions[fn.__name__] = fn

  # Test all combination of fiddlers and base configurations by default.
  for base_name in module_reflection.find_base_config_like_things(module):
    for fiddler_name in module_reflection.find_fiddler_like_things(module):

      def test_base_and_fiddler(self, module, base_name, fiddler_name):
        self.assertTrue(hasattr(module, base_name))
        self.assertTrue(hasattr(module, fiddler_name))
        base_fn = getattr(module, base_name)
        fiddler_fn = getattr(module, fiddler_name)
        cfg = base_fn()
        fiddler_fn(cfg)
        if not skip_building:
          _ = building.build(cfg)

      fn = functools.partialmethod(test_base_and_fiddler, module, base_name,
                                   fiddler_name)
      fn.__name__ = f'test_{base_name}_and_{fiddler_name}'
      test_functions[fn.__name__] = fn

  if (not module_reflection.find_base_config_like_things(module) and
      module_reflection.find_fiddler_like_things(module)):

    def test_no_base_configuration_sorry(self):
      self.fail(
          'Found a fiddler, but found no base configurations, so auto-fiddle '
          'tests cannot test the fiddler!')

    test_functions[
        'test_no_base_configuration_sorry'] = test_no_base_configuration_sorry

  if hasattr(module, '__name__'):
    test_class_name = module.__name__ + 'FiddleTest'
  else:
    test_class_name = 'FiddleAutoTest'
  return type(test_class_name, (absltest.TestCase,), test_functions)


def load_tests_from_module(loader: unittest.TestLoader, module: Module, *,
                           skip_building: bool) -> unittest.TestSuite:
  test_class = make_test_class_for_module(module, skip_building=skip_building)
  return loader.loadTestsFromTestCase(test_class)


def load_module_from_path(module_name: str) -> Module:
  """Imports and returns the module given a path to the module's file."""
  if '/' in module_name:
    module_name = module_name.replace('/', '.')
  if module_name.endswith('.py'):
    module_name = module_name[:-len('.py')]
  module = importlib.import_module(module_name)
  return module


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite,
               pattern: Optional[str]) -> unittest.TestSuite:
  """An implementation of the load_tests module protocol for dynamic tests.

  For additional context, see the documentation on unittest's load_tests
  protocol, which describes how modules can control test suite discovery.

  Args:
    loader: See unittest's load_tests protocol.
    tests: See unittest's load_tests protocol.
    pattern: See unittest's load_tests protocol.

  Returns:
    A test suite.
  """
  del tests  # Unused.
  del pattern  # Unused.
  module = load_module_from_path(_FLAG_FIDDLE_CONFIG_MODULE.value)
  suite = load_tests_from_module(
      loader, module, skip_building=_FLAG_SKIP_BUILDING.value)
  return suite


# Convenience, for bazel macro.
main = absltest.main
