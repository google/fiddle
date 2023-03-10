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

"""Library for finding differences between Fiddle configurations."""

# pylint: disable=unused-import
from fiddle._src.diffing import AddTag
from fiddle._src.diffing import align_by_id
from fiddle._src.diffing import align_heuristically
from fiddle._src.diffing import AlignedValueIds
from fiddle._src.diffing import AlignedValues
from fiddle._src.diffing import AlignmentError
from fiddle._src.diffing import AnyCallable
from fiddle._src.diffing import AnyValue
from fiddle._src.diffing import apply_diff
from fiddle._src.diffing import build_diff
from fiddle._src.diffing import build_diff_from_alignment
from fiddle._src.diffing import DeleteValue
from fiddle._src.diffing import Diff
from fiddle._src.diffing import DiffAlignment
from fiddle._src.diffing import DiffOperation
from fiddle._src.diffing import ListPrefix
from fiddle._src.diffing import ModifyValue
from fiddle._src.diffing import Reference
from fiddle._src.diffing import RemoveTag
from fiddle._src.diffing import resolve_diff_references
from fiddle._src.diffing import SetValue
from fiddle._src.diffing import skeleton_from_diff
