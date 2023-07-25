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

"""Checks that custom objects/instances are not present in a configuration.

In general, we recommend building configurations from

(a) primitives like int/float/str/etc.
(b) basic python collections like lists/tuples/dicts
(c) Fiddle configs like fdl.Config, fdl.Partial

Usually, when a custom user object is present in a config, it can be rewritten
to be a fdl.Config that constructs that object. For example, instead of

config.foo.preprocessor = MyPreprocessor(dtype="int32")

write,

config.foo.preprocessor = fdl.Config(MyPreprocessor, dtype="int32")
"""

from fiddle._src.validation.no_custom_objects import check_no_custom_objects  # pylint: disable=unused-import
from fiddle._src.validation.no_custom_objects import get_config_errors  # pylint: disable=unused-import
