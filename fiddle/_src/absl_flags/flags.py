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

"""API to use command line flags with Fiddle Buildables."""

from fiddle._src.absl_flags import legacy_flags

# Legacy API
apply_fiddlers_to = legacy_flags.apply_fiddlers_to
apply_overrides_to = legacy_flags.apply_overrides_to
create_buildable_from_flags = legacy_flags.create_buildable_from_flags
flags_parser = legacy_flags.flags_parser
rewrite_fdl_args = legacy_flags.rewrite_fdl_args
fdl_flags_supplied = legacy_flags.fdl_flags_supplied
