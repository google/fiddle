# Fiddle Flag Support

<!--#include file="google/flags_code_lab_header.md"-->

You can easily fiddle with your configuration using command line flags, thanks
to Fiddle's absl_flags integration! This code lab will describe how it all
works. See also the [complete working example][example].

[example]: http://github.com/google/fiddle/tree/main/fiddle/absl_flags/example

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
    mutates it. (A fiddler returns `None`.)

## Life of a flag-augmented Fiddle program

When our example program is executed, the following steps occur:

1.  **Launch**: We run our program on the command line:

    ```sh
    python3 -m fiddle.absl_flags.example.example \
      --fdl_config=simple \
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

## Flag Syntax

The Fiddle flag integration supports the following flag syntax:

-   **Base Config**: The base configuration function is specified with the flag
    `--fdl_config`. Example: `--fdl_config=simple`. Alternatively, a
    JSON-serialized configuration can be read from a file via with the flag
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
