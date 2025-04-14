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

"""The public API for the absl_flags Fiddle extension."""

# pylint: disable=g-importing-member

from fiddle._src.absl_flags.flags import apply_fiddlers_to
from fiddle._src.absl_flags.flags import apply_overrides_to
from fiddle._src.absl_flags.flags import create_buildable_from_flags
from fiddle._src.absl_flags.flags import DEFINE_fiddle_config
from fiddle._src.absl_flags.flags import fdl_flags_supplied
from fiddle._src.absl_flags.flags import FiddleFlag
from fiddle._src.absl_flags.flags import FiddleFlagSerializer
from fiddle._src.absl_flags.flags import flags_parser
from fiddle._src.absl_flags.sweep_flag import DEFINE_fiddle_sweep
from fiddle._src.absl_flags.sweep_flag import SweepItem
from fiddle._src.absl_flags.utils import set_value_at_path
from fiddle._src.absl_flags.utils import with_overrides
