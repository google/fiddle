# Fiddle Flag Support

<!--#include file="google/flags_code_lab_header.md"-->

You can easily fiddle with your configuration using command line flags, thanks
to Fiddle's absl_flags integration! This code lab will describe how it all
works. Configuration of fiddle config objects via command line arguments is
supported using 2 APIs:

| API        | Purpose                                                        |
| ---------- | -------------------------------------------------------------- |
| New API    | Defines custom configurations per binary using the API         |
:            : `DEFINE_fiddle_config()` which returns a built config object   :
:            : handle after applying all the command line overrides in order. :
:            : The usage of this API is more intuitive than the legacy API,   :
:            : and it provides the ability to define custom overrides per     :
:            : binary as well as read serialized configs from a file or       :
:            : strings on the command line. Additionally, the overrides are   :
:            : applied in order, which is a more intuitive user experience    :
:            : than the current order followed by the legacy API.             :
| Legacy API | Invoked via `create_buildable_from_flags()` that returns a     |
:            : built config object. Command line overrides are NOT applied in :
:            : order; all fiddlers are applied first, followed by all tags,   :
:            : followed by all overrides.                                     :

> NOTE: New usages of the legacy flags API are discouraged and users should
> migrate their legacy usage to the new API.

See some working examples of the APIs below.

<section class="tabs" markdown=1>

### New API {.new-tab}

[example](http://github.com/google/fiddle/tree/main/fiddle/_src/absl_flags/sample_test_binary.py)

### Legacy API {.new-tab}

[example](http://github.com/google/fiddle/tree/main/fiddle/absl_flags/example)

</section>

[TOC]

## How to structure your code

Fiddle is designed to require your code to depend on the Fiddle library as
little as possible; this helps projects to work with multiple configuration
systems, increasing reusability, and enables more modular development. A common
Fiddle pattern includes:

1.  **Business logic**: a Python library (or a few!) containing pure Python code
    (no Fiddle dependency).
2.  **Fiddle Configs**: a Python file (module) defining one or more "base
    configurations", and zero or more "fiddlers".
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

## Life of a flag-augmented Fiddle program

When our example program is executed, the following steps occur:

<section class="tabs" markdown=1>

### New API {.new-tab}

1.  **Launch**: We run our program on the command line:

    ```sh
    python3 -m fiddle._src.absl_flags.sample_test_binary \
    --sample_config config:base_experiment \ --sample_config
    fiddler:'set_dtypes(dtype="float64")' \ --sample_config
    set:decoder.mlp.use_bias=True \ --sample_config
    set:decoder.mlp.dtype='"float16"'
    ```

2.  **Flag Parsing**: The custom Fiddle flag parser parses Fiddle-related flags
    from the command line, applying any overrides in the order specified in the
    command line, and returns a built object `_SAMPLE_FLAG.value` that has all
    the overrides applied.

3.  **Business logic**: `main` calls arbitrary functions on the built objects to
    carry out whatever task your application has been designed to do.

### Legacy API {.new-tab}

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

The Fiddle flag integration supports the following flag syntax:

-   **Base Config**: The base configuration function is specified with the flag:

<section class="tabs" markdown=1>

### New API {.new-tab}

that was set as the `name` argument for `DEFINE_fiddle_config` and the command
`config`. For example, if the flag object was instantiated as
`DEFINE_fiddle_config(name="my_flag", ...)`, then the base config is specified
by using `--my_flag
config:some_function_returning_fiddle_config_to_be_overridden()`. One can also
use the command `config_file` to read from a JSON serialized config written to a
file, or the command `config_str` to read from a JSON serialized config in
encoded string form (the additional encoding involves zlib compression followed
by base64 encoding).

### Legacy API {.new-tab}

`--fdl_config`. Example: `--fdl_config=base`. Alternatively, a JSON-serialized
configuration can be read from a file via with the flag `--fdl_config_file`.
Example: `--fdl_config_file='/some/path/config.json'`.

</section>

-   **Fiddlers**: Fiddlers are specified on the command line with the

<section class="tabs" markdown=1>

### New API {.new-tab}

command `fiddler` after the `name` argument for `DEFINE_fiddle_config`. For
example, if the flag object was instantiated as
`DEFINE_fiddle_config(name="my_flag", ...)` then the fiddlers would be invoked
like `--my_flag fiddler:name_of_fiddler(value="new_value")`.

### Legacy API {.new-tab}

`--fiddler=` flag. This flag can be specified multiple times. Example:
`--fiddler=swap_weight_and_biases --fiddler=other_fiddler`.

</section>

-   **Specific Overrides**: Specific overrides allow you to specify specific
    values to arbitrary fields on the command line. The syntax is

<section class="tabs" markdown=1>

### New API {.new-tab}

the command `set` after the `name` argument for `DEFINE_fiddle_config`. For
example, if the flag object was instantiated as
`DEFINE_fiddle_config(name="my_flag", ...)`, then the specific overrides are
specified using `--my_flag set:some_attr.some_sub_attr=some_value`.

### Legacy API {.new-tab}

`--fdl.dotted.path.of.fields=3`, where everything after the `fdl.` prefix
corresponds to the name of a field in the Fiddle configuration object
corresponding to exactly the same Python code. For example, if (in Python) we
wanted to set the value of a nested field to 15, I might write:
`cfg.model.dense1.parameters = 15`. The corresponding syntax on the command line
would be: `--fdl.model.dense1.parameters=15`. Due to shell escaping, to specify
a string value, you often need to use two nested quotes, or escape the outer
quotes (depending on your shell). For example:
`--fdl.data.filename='"other.txt"'` (or equivalently:
`--fdl.data.filename=\"other.txt\"`). Only "literal" values may be specified on
the command line like this; if you'd like to set a complex value, please write a
fiddler and invoke it with the previous Fiddlers syntax.

</section>
