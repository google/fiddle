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

"""Tags attach metadata to arguments & values for ergonomic fiddling.

When defining shared parameters across a project that later could be changed,
for example dtype or activation function, we encourage the following coding
pattern: a tiny shared library file should declare tags for an
entire project, like

  class ActivationDType(fdl.Tag):
    "Data type (e.g. jnp.float32, or jnp.bfloat16) for intermediate values."
  class KernelInitFn(fdl.Tag):
    "A function from RNG & hyperparameters to initial weights."

Then, in library code which configures shared Fiddle fixtures, these tags are
used,

  def layer_norm_fixture() -> fdl.Config[LayerNorm]:
    cfg = fdl.Config(LayerNorm)
    cfg.dtype = ActivationDType.new(default=jnp.float32)

And in experiment code stitching everything together, all of these tagged values
can be set at once,

  def encoder_decoder_fixture() -> fdl.Config[EncoderDecoder]:
    cfg = fdl.Config(EncoderDecoder)
    ...
    cfg.encoder.encoder_norm = layer_norm_fixture()

    # Set all activation dtypes.
    fdl.set_tagged(cfg, ActivationDType, jnp.bfloat16)
    return cfg

  model = fdl.build(encoder_decoder_fixture())

Tags can be defined in a module (file) or inside classes, but cannot be defined
within a function, as there would not be a way to refer to it without
re-invoking the function, defining a new unique Tag.

Tags can inherit from each other. Use this power wisely.

While the normal mechanism for creating a TaggedValue is via $TAG.new(), you can
also create a TaggedValue explicitly, allowing multiple tags to be associated
with the value.
"""

# pylint: disable=unused-import

from fiddle._src.tagging import _NoValue
from fiddle._src.tagging import list_tags
from fiddle._src.tagging import materialize_tags
from fiddle._src.tagging import NO_VALUE
from fiddle._src.tagging import set_tagged
from fiddle._src.tagging import Tag
from fiddle._src.tagging import TaggedValue
from fiddle._src.tagging import TaggedValueCls
from fiddle._src.tagging import TaggedValueNotFilledError
from fiddle._src.tagging import TagType
