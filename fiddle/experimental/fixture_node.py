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

"""Declares a fixture node, which allows two-stage configuration.

By delegating implementation of the fixture to the user, we sidestep questions
of what sub-configuation DAGs should be shared, how to replicate them, etc.,
which are hard questions to answer well at the Fiddle library level.
"""

# pylint: disable=unused-import
from fiddle._src.experimental.fixture_node import FixtureNode
from fiddle._src.experimental.fixture_node import materialize
