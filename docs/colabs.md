# Fiddle Colab Series

<!--#include file="google/colabs_header.md"-->

## Welcome!

This series of colabs introduces Fiddle, a configuration library serving machine
learning use cases.

Please send us feedback if you have any suggestions for improvements! Or if you
built something cool and want to share it out, we'd love to link it up at the
"Where to go from here" section.

[TOC]

## General colab advice

Most entries in this series are colab notebooks (click the blue buttons in each
heading below), allowing you to run our tutorial code interactively. We
encourage you to do that! Play around, change things, see what happens!

<!--#include file="google/colabs_instructions.md"-->

## Codelab 1: Introduction / overview {#intro}

<a href="https://colab.sandbox.google.com/github/google/fiddle/blob/main/fiddle/examples/colabs/fiddle_tutorial_with_flax.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in colab" style="float:left"/></a><br>

This colab walks through how to use Fiddle, using Flax and JAX as a motivating
example. This tutorial includes an overview of the key Fiddle APIs, best
practices, as well as a few tips/tricks.

## Codelab 2: Basic API {#basic-api}

<a href="https://colab.sandbox.google.com/github/google/fiddle/blob/main/fiddle/examples/colabs/basic_api.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in colab" style="float:left"/></a><br>

This colab walks through the core Fiddle abstractions and how they work. If you
prefer learning in a more pedagogical than motivational style, you can start
with this colab and then go through the above intro colab.

## Codelab 3: Visualization / printing / codegen {#visualization}

<a href="https://colab.sandbox.google.com/github/google/fiddle/blob/main/fiddle/examples/colabs/visualization_printing_codegen.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in colab" style="float:left"/></a><br>

This colab introduces a variety of ways to visualize and print Fiddle
configurations. These can be used to communicate experiments, debug errors, and
make configuration more self-contained.

## Codelab 4: `auto_config` usage {#auto_config}

<a href="https://colab.sandbox.google.com/github/google/fiddle/blob/main/fiddle/examples/colabs/auto_config.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in colab" style="float:left"/></a><br>

This colab walks through how to use Fiddle's "auto-config" functionality,
building off of the introduction codelab above. Auto-config allows existing
"glue code" functions that perform object and function wire-up to be transformed
into functions that capture the wire-up as a `fdl.Buildable`, often with no or
very minimal changes to the existing code. This minimizes the boilerplate
necessary to create a Fiddle configuration.

## Codelab 5: CLI flags {#cli-flags}

The [flags code lab](flags_code_lab.md) walks through an
[example application](http://github.com/google/fiddle/tree/main/fiddle/absl_flags/example)
that uses Fiddle and Fiddle's support for overriding configurations with command
line flags.

## Codelab 6: `select()` and Tag APIs {#select-and-tags}

<a href="https://colab.sandbox.google.com/github/google/fiddle/blob/main/fiddle/examples/colabs/select_and_tag_apis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in colab" style="float:left"/></a><br>

The `select()` and `Tag` APIs for Fiddle let users concisely change many values
in a larger configuration structure, facilitating configuration codebases which
are factorized into declaration of a base model, and experimental overrides.

The `select()` API makes it easy to set parameters across all occurrences of
specific functions or classes within a config. For example:

```python
# Set all Dropout classes to have rate 0.1.
select(root_cfg, nn.Dropout).set(rate=0.1)
```

Fiddle also allows values to be tagged with one or more tags, making it easy to
set values that are shared in many places all at once. For example:

```python
# Set all tagged dtypes, which may be on different functions/classes.
select(root_cfg, tag=ActivationDType).set(value=jnp.bfloat16)
```

<!--#include file="google/colabs_internal.md"-->
