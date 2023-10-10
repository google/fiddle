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

"""Init file for the `fiddle` package."""

from fiddle._src.building import build
from fiddle._src.casting import cast
from fiddle._src.config import Buildable
from fiddle._src.config import Config
from fiddle._src.config import get_callable
from fiddle._src.config import NO_VALUE
from fiddle._src.config import ordered_arguments
from fiddle._src.copying import copy_with
from fiddle._src.copying import deepcopy_with
from fiddle._src.materialize import materialize_defaults
from fiddle._src.mutate_buildable import assign
from fiddle._src.mutate_buildable import update_callable
from fiddle._src.partial import ArgFactory
from fiddle._src.partial import Partial
from fiddle._src.signatures import VARARGS
from fiddle._src.tagging import add_tag
from fiddle._src.tagging import clear_tags
from fiddle._src.tagging import get_tags
from fiddle._src.tagging import remove_tag
from fiddle._src.tagging import set_tagged
from fiddle._src.tagging import set_tags
from fiddle._src.tagging import Tag
from fiddle._src.tagging import TaggedValue
from fiddle.version import __version__
