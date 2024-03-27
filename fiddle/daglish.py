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

"""Library for manipulating DAGs."""

# pylint: disable=unused-import
from fiddle._src.daglish import add_path_element
from fiddle._src.daglish import Attr
from fiddle._src.daglish import attr_or_index
from fiddle._src.daglish import BasicTraversal
from fiddle._src.daglish import BuildableAttr
from fiddle._src.daglish import BuildableFnOrCls
from fiddle._src.daglish import collect_paths_by_id
from fiddle._src.daglish import find_node_traverser
from fiddle._src.daglish import follow_path
from fiddle._src.daglish import Index
from fiddle._src.daglish import is_internable
from fiddle._src.daglish import is_memoizable
from fiddle._src.daglish import is_namedtuple_subclass
from fiddle._src.daglish import is_prefix
from fiddle._src.daglish import is_traversable_type
from fiddle._src.daglish import iterate
from fiddle._src.daglish import Key
from fiddle._src.daglish import MemoizedTraversal
from fiddle._src.daglish import NamedTupleType
from fiddle._src.daglish import NodeTraverser
from fiddle._src.daglish import NodeTraverserRegistry
from fiddle._src.daglish import Path
from fiddle._src.daglish import path_str
from fiddle._src.daglish import PathElement
from fiddle._src.daglish import Paths
from fiddle._src.daglish import register_node_traverser
from fiddle._src.daglish import State
from fiddle._src.daglish import SubTraversalResult
from fiddle._src.daglish import Traversal
from fiddle._src.daglish_extensions import is_immutable
from fiddle._src.daglish_extensions import is_unshareable
from fiddle._src.daglish_extensions import parse_path
