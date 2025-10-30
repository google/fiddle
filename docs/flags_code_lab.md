# Fiddle Flag Support

<!--#include file="google/flags_code_lab_header.md"-->

You can easily fiddle with your configuration using command line flags, thanks
to Fiddle's absl_flags integration! This code lab will describe how it all
works. Configuration of fiddle config objects via command line arguments is
supported using 3 APIs:

*   **New API, single config**: Defines custom configurations per binary using
    the API `DEFINE_fiddle_config()` which returns a config object after
    applying all the command line overrides in order. The usage of this API is
    more intuitive than the legacy API, and it provides the ability to define
    custom overrides per binary as well as read serialized configs from a file
    or strings on the command line. Additionally, the overrides are applied in
    order, which is a more intuitive user experience than the current order
    followed by the legacy API.
*   **New API, multiple configs**: Defines a 'sweep' of multiple custom
    configurations per binary using the API `DEFINE_fiddle_sweep()`. This is
    similar to `DEFINE_fiddle_config` but returns a sequence of multiple configs
    each with some metadata. It allows specifying them via a sequence of
    overrides to config attributes, and/or overrides to arguments of the
    function that generates configs. This is intended mainly for use in launch
    binaries of ML experiments which perform hyperparameter sweeps.
*   **Legacy API**: Invoked via `create_buildable_from_flags()` that returns a
    config object. Command line overrides are NOT applied in order; all fiddlers
    are applied first, followed by all tags, followed by all overrides.

> NOTE: New usages of the legacy flags API are discouraged and users should
> migrate their legacy usage to the new API.

See some working examples of the APIs below:

-   [New API](http://github.com/google/fiddle/tree/main/fiddle/_src/absl_flags/sample_test_binary.py)
-   [Legacy API](http://github.com/google/fiddle/tree/main/fiddle/absl_flags/example)


## How to structure your code

Fiddle is designed to require your code to depend on the Fiddle library as
little as possible; this helps projects to work with multiple configuration
systems, increasing reusability, and enables more modular development. A common
Fiddle pattern includes:

1.  **Business logic**: a Python library (or a few!) containing pure Python code
    (no Fiddle dependency).
2.  **Fiddle Configs**: a Python file (module) defining one or more "base
    configurations", and zero or more "fiddlers". It may also define zero or
    more "sweeps", which describe collections of multiple related configs to
    launch together, for example a scan over hyperparameters of an ML model.
3.  **Main**: A module that makes a Fiddle Config from the command line, calls
    `fdl.build`, and then calls into the instantiated business logic (e.g. a
    training loop).

## Definitions

-   **Base Configuration**: A base configuration function is a function without
    required arguments (a nullary function) that returns a `fdl.Buildable`.
-   **Fiddler**: A fiddler is a unary function that takes a `fdl.Buildable` and
    applies some transformations. If the fiddler returns `None`, then it is
    assumed the input `fdl.Buildable` was mutated by the fiddler, and the
    mutated input will be passed on to further flag operations. If the fiddler
    returns a non-`None` value (e.g., a new `fdl.Buildable` obtained from
    applying a `daglish` traversal routine), the return value of the fiddler is
    passed on instead.
-   **Sweep**: A sweep is a function returning a sequence of dicts that describe
    overrides either to arguments of the **base configuration function**, or to
    attributes of the `fdl.Buildable` that it returns, or both.

## Life of a flag-augmented Fiddle program

When our example program is executed, the following steps occur:

<section class="tabs" markdown=1>

### New API (single config)

1.  **Launch**: We run our program on the command line:

    ```sh
    python3 -m fiddle._src.absl_flags.sample_test_binary \
      --sample_config config:base_experiment \
      --sample_config fiddler:'set_dtypes(dtype="float64")' \
      --sample_config set:decoder.mlp.use_bias=True \
      --sample_config set:decoder.mlp.dtype='"float16"'
    ```

2.  **Flag Parsing**: The custom Fiddle flag parser parses Fiddle-related flags
    from the command line, applying any overrides in the order specified in the
    command line, and returns a `fdl.Buildable` object `_SAMPLE_FLAG.value` that
    has all the overrides applied.

3.  **Business logic**: `main` calls `fdl.build` to build the config, then calls
    arbitrary functions on the built objects to carry out whatever task your
    application has been designed to do.

### New API (multiple configs)

1.  **Launch**: We run our launcher program on the command line to launch
    multiple configs:

    ```sh
    python3 -m fiddle._src.absl_flags.sample_launch_binary \
      --sample_config config:base_experiment \
      --sample_config sweep:kernel_init_sweep \
      --sample_config sweep:encoder_bias_sweep \
      --sample_config fiddler:'set_dtypes(dtype="float64")'
    ```

2.  **Flag Parsing**: The custom Fiddle flag parser parses Fiddle-related flags
    from the command line, and applies any overrides specified in the sweeps to
    create a sweep of one or more configs. If multiple sweeps are specified, the
    cartesian product of the sweeps is taken before applying them. Any fiddler:
    or set: commands are then applied to all configs in the sweep, in the order
    specified. `_SAMPLE_FLAG.value` returns a sequence of SweepItem dataclasses,
    each of which has a `.config` property of type `fdl.Buildable`, and an
    `overrides_applied` property which is the dict of overrides and can be
    useful to log as metadata attached to each experiment launched.

3.  **Passing flag to the main binary**: The launch binary will typically then
    serialize each config in the sweep using `FiddleFlagSerializer().serialize`,
    and pass it as a `config_str:` to a main binary's `DEFINE_fiddle_config`
    flag.

4.  **Business logic in the main binary**: Is the same as in the single-config
    case.

### Legacy API

1.  **Launch**: We run our program on the command line:

    ```sh
    python3 -m fiddle.absl_flags.example.example \
      --fdl_config=base \
      --fiddler=swap_weight_and_bias \
      --fdl.model.b=0.73
    ```

2.  **`absl.app.run`**: In `example.py`, `absl.app.run` is called, passing the
    Fiddle flag parser:

    ```py
    if __name__ == '__main__':
      app.run(main, flags_parser=fiddle.absl_flags.flags_parser)
    ```

3.  **Flag Parsing**: The custom Fiddle flag parser parses Fiddle-related flags.

4.  **`main` begins**: `absl.app.run` calls our supplied `main` function.

5.  **Create buildable from flags**: `main` calls
    `fdl.absl_flags.create_buildable_from_flags` passing a Python module that
    contains a set of functions corresponding to base configurations and
    fiddlers (see [the glossary](#definitions)).

6.  **The base configuration is created**: `create_buildable_from_flags`
    identifies the function on the supplied module the user has requested as the
    base configuration based on the `--fdl_config` or `--fdl_config_file`
    command line options.

7.  **Apply Fiddlers**: The `--fiddler` flag can be supplied multiple times; a
    function on the supplied module that corresponds to each `--fiddler` flag
    instance is identified and called, passing in the Fiddle object returned
    from the base configuration function call in the previous step.

8.  **Apply specific overrides**: After fiddlers are run, the `--fdl.$SOMETHING`
    flags are applied. The resulting `fdl.Buildable` object is returned from
    `create_buildable_from_flags`.

9.  **Build**: `main` calls `fdl.build` supplying the `fdl.Buildable` returned
    from `create_buildable_from_flags`.

10. **Business logic**: `main` calls arbitrary functions on the built objects to
    carry out whatever task your application has been designed to do. In the
    case of the example, `main` calls `runner.run()` to (pretend to) train a
    neural network.

</section>

## Flag Syntax

The Fiddle flag integration supports the following flag syntax.

<section class="tabs" markdown=1>

### New API

-   **Base Config**: The base configuration function is specified with the
    `config` command, following the flag name provided to `DEFINE_fiddle_config`
    or `DEFINE_fiddle_sweep`. For example, if the flag object was instantiated
    as `DEFINE_fiddle_config(name="my_flag", ...)`, then the base config is
    specified by using `--my_flag
    config:some_function_returning_fiddle_config_to_be_overridden()`. With
    `DEFINE_fiddle_config` one can also use the command `config_file` to read
    from a JSON serialized config written to a file, or the command `config_str`
    to read from a JSON serialized config in encoded string form (the additional
    encoding involves zlib compression followed by base64 encoding).

-   **Fiddlers**: Fiddlers are specified on the command line with the `fiddler`
    command after the `name` argument for `DEFINE_fiddle_config` or
    `DEFINE_fiddle_sweep`. For example, if the flag object was instantiated as
    `DEFINE_fiddle_config(name="my_flag", ...)` then the fiddlers would be
    invoked like `--my_flag fiddler:name_of_fiddler(value="new_value")`.

-   **Specific Overrides**: Specific overrides allow you to specify specific
    values to arbitrary fields on the command line. The syntax is the `set`
    command after the `name` argument for `DEFINE_fiddle_config` or
    `DEFINE_fiddle_sweep`. For example, if the flag object was instantiated as
    `DEFINE_fiddle_config(name="my_flag", ...)`, then the specific overrides are
    specified using `--my_flag set:some_attr.some_sub_attr=some_value`.

-   **Sweeps**: Sweeps allow you to specify multiple dicts of overrides to
    apply, to generate a sweep of one or more configs. The syntax is the `sweep`
    command after the `name` argument for `DEFINE_fiddle_sweep`. For example, if
    the flag object was instantiated as `DEFINE_fiddle_sweep(name="my_flag",
    ...)`, then one or more sweep functions can be specified using `--my_flag
    sweep:name_of_sweep(arguments=123)` or just `--my_flag sweep:name_of_sweep`
    if no arguments are required to the sweep function.

    A `sweep:` command should specify a function call returning a list of
    dictionaries, where each dictionary represents a single item in the sweep.
    The entries in the dictionary are the overrides to apply, where keys can be
    of the form:

    *   `kwarg:foo` -- to specify or override a keyword argument to the base
        config function specified by the `config:` command.
    *   `arg:0` -- to specify or override a positional argument to the base
        config function specified by the `config:` command.
    *   `path.to.some.attr` -- to specify an override to an attribute in the
        config returned by the base config function. These paths follow the same
        format as is accepted by `set:` commands and can take quite general
        forms like `foo.bar['baz'][0].boz`.

    Multiple `sweep:` commands can be specified, which will result in taking the
    cartesian product of the separate sweeps before applying them.

### Legacy API

-   **Base Config**: The base configuration function is specified with the
    `--fdl_config` flag. Example: `--fdl_config=base`. Alternatively, a
    JSON-serialized configuration can be read from a file with the flag
    `--fdl_config_file`. Example: `--fdl_config_file='/some/path/config.json'`.

-   **Fiddlers**: Fiddlers are specified on the command line with the
    `--fiddler=` flag. This flag can be specified multiple times. Example:
    `--fiddler=swap_weight_and_biases --fiddler=other_fiddler`.

-   **Specific Overrides**: Specific overrides allow you to specify specific
    values to arbitrary fields on the command line. The syntax is
    `--fdl.dotted.path.of.fields=3`, where everything after the `fdl.` prefix
    corresponds to the name of a field in the Fiddle configuration object
    corresponding to exactly the same Python code. For example, if (in Python)
    we wanted to set the value of a nested field to 15, I might write:
    `cfg.model.dense1.parameters = 15`. The corresponding syntax on the command
    line would be: `--fdl.model.dense1.parameters=15`. Due to shell escaping, to
    specify a string value, you often need to use two nested quotes, or escape
    the outer quotes (depending on your shell). For example:
    `--fdl.data.filename='"other.txt"'` (or equivalently:
    `--fdl.data.filename=\"other.txt\"`). Only "literal" values may be specified
    on the command line like this; if you'd like to set a complex value, please
    write a fiddler and invoke it with the previous Fiddlers syntax.

-   **Sweeps**: This is not supported in the legacy API.

</section>

## Name Resolution

Fiddle will attempt to resolve dotted names relative to `default_module`, if
provided. If no module is provided, Fiddle will attempt to resolve the name as
an absolute path.

Concretely, given the flag definition

```py
absl_flags.DEFINE_fiddle_config(
  "config", help_string="Fiddle configuration.", default_module=m
)
```

resolving `--config=config:n.base` will first try to resolve or import
`m.n.base` but will fall back to `n.base`.

## Serializing and forwarding configurations

The new flags API provides a convenient way to serialize and forward a config.
This can be useful if you want to construct a Fiddle config in your XManager
launch script and forward it as an argument to your executable.

For a single-config launch flag via `DEFINE_fiddle_config`, this can be
accomplished by calling `.serialize()` on the absl flag. If using a multi-config
`DEFINE_fiddle_sweep` launch flag, you can apply
`FiddleFlagSerializer().serialize` directly to each config in the sweep.

A basic example on how this can be used in conjunction with XManager is provided
below:

```py
from absl import app
from absl import flags
from fiddle import absl_flags
from xmanager import xm
from xmanager import xm_local

FLAGS = flags.FLAGS

absl_flags.DEFINE_fiddle_config("config", help_string="Fiddle configuration.")

def main(_):
  with xm_local.create_experiment("Fiddle Flag Forwarding") as experiment:
    [executable] = experiment.package([
      xm.bazel_binary(
        label=...,
        executor_spec=xm_local.Local.Spec(),
        args=[FLAGS['config'].serialize()]
      ),
    ])

if __name__ == '__main__':
  app.run(main)
```
