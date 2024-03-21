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

"""Small library to ensure a test fixture is imported with a qualified name.

Some test libraries will import the main test file under the main module (module
with name "__main__"), which will generate code differently. In order to ensure
that the test fixture is imported consistently, we just put it in this module.
"""


def foo(dtype):
  return dtype
