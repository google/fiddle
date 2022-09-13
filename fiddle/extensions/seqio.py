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

"""Fiddle extensions to handle SeqIO code more elegantly.

(See https://github.com/google/seqio.)

Currently this just affects codegen, graphviz, and other debugging functions.
"""

from fiddle.codegen import import_manager

_import_aliases = (
    # Seqio  aliases all of these to its top level import.
    ("seqio.dataset_providers", "import seqio"),
    ("seqio.evaluation", "import seqio"),
    ("seqio.feature_converters", "import seqio"),
    ("seqio.loggers", "import seqio"),
    ("seqio.utils", "import seqio"),
    ("seqio.vocabularies", "import seqio"),
)


def enable():
  """Registers SeqIO fiddle extensions.

  This allows for nicer handling of seqio imports.
  """
  for module_str, import_stmt in _import_aliases:
    import_manager.register_import_alias(module_str, import_stmt)
