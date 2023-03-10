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

"""Functions to help serialize configuration.

This module provides functions that help ensure pickle-compatibility
(`clear_argument_history`), as well as functions and associated helper classes
used to serialize Fiddle structures into a custom JSON-based representation
(`dump_json`, `load_json`).
"""

# pylint: disable=unused-import
from fiddle._src.experimental.serialization import clear_argument_history
from fiddle._src.experimental.serialization import DefaultPyrefPolicy
from fiddle._src.experimental.serialization import DeserializationError
from fiddle._src.experimental.serialization import dump_json
from fiddle._src.experimental.serialization import find_node_traverser
from fiddle._src.experimental.serialization import import_symbol
from fiddle._src.experimental.serialization import load_json
from fiddle._src.experimental.serialization import PyrefPolicy
from fiddle._src.experimental.serialization import PyrefPolicyError
from fiddle._src.experimental.serialization import register_constant
from fiddle._src.experimental.serialization import register_dict_based_object
from fiddle._src.experimental.serialization import register_enum
from fiddle._src.experimental.serialization import register_node_traverser
from fiddle._src.experimental.serialization import UnserializableValueError
