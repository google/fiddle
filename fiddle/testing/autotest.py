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

# pylint: disable=unused-import
from fiddle._src.testing.autotest import load_tests
from fiddle._src.testing.autotest import main
