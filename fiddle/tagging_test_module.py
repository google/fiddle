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

"""A separate module used by tagging_test."""

import fiddle as fdl


class ParameterDType(fdl.Tag):
  """All trainable parameter's dtype.

  To control mixed precision, or quantization, all trainable parameter's dtypes
  are annotated with the `ParameterDType` tag.
  """


class LinearParamDType(ParameterDType):
  """All linear layer's DTypes."""


class ActivationDType(fdl.Tag):
  """Scalar type for arrays of module outputs."""


class MyModel:
  """Some domain-specific tags nested inside another type."""

  class EncoderDType(fdl.Tag):
    """Parameter dtypes in encoder layers."""

  class DecoderDType(fdl.Tag):
    """Parameter dtypes in decoder layers."""
