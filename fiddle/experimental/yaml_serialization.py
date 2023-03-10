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

"""Serialization helper that writes YAML output.

Note: This API is highly experimental, and primarily intended for dumping
objects in a medium-easy-to-read format, using indentation/spaces instead of
`printing.py`'s dot-separated paths. Please use `serialization.py` whenever
you need to serialize Fiddle objects in a robust manner.
"""

# pylint: disable=unused-import
from fiddle._src.experimental.yaml_serialization import dump_yaml
