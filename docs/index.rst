Fiddle documentation
====================

Fiddle is a Python-first configuration library particularly well suited to ML
applications. Fiddle enables deep configurability of parameters in a program,
while allowing configuration to be expressed in readable and maintainable
Python code.

Design Goals
------------

Fiddle attempts to satisfy the following design goals:

.. grid::

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Python first
            :shadow: none
            :class-title: sd-fs-5
            :class-card: sd-border-0

            .. div::

                Configurations are expressed naturally in Python code, and
                represented as Python objects. A Python first approach allows
                configs to leverage Python's extensive existing tooling to
                simplify testing and maintenance.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Readability & Understandability
            :shadow: none
            :class-title: sd-fs-5
            :class-card: sd-border-0

            .. div::

                Fiddle's APIs (powered by Python's flexibility) make the
                structure of a system's configuration easy to read.
                Additionally, Fiddle's printing and visualization tools provide
                multiple lenses to view a configuration to help understand both
                a given model run and how a codebase fits together.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Minimal Boilerplate
            :shadow: none
            :class-title: sd-fs-5
            :class-card: sd-border-0

            .. div::

                Fiddle is designed to reduce boilerplate, to make both writing
                and reading configurations fast. Changing one hyperparameter
                anywhere in the program is just a single line of code.
                Configurations don't require extensive forwarding of parameters
                across multiple files.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Isolation & Modularity
            :shadow: none
            :class-title: sd-fs-5
            :class-card: sd-border-0

            .. div::

                Fiddle allows your library code to remain unaware of what
                configuration system is being used, and doesn't require
                decoration or cooperation on the part of library code. Library
                code should expose parameters as standard constructor and
                function arguments instead of relying on config objects.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Universality & Compatibility
            :shadow: none
            :class-title: sd-fs-5
            :class-card: sd-border-0

            .. div::

                Fiddle configurations can be used to represent all of a Python
                program's configuration, avoiding the need for separate
                configuration systems for different program components. Fiddle
                additionally provides bridges to other configuration systems
                such as Gin and Lingvo Params.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Good errors
            :shadow: none
            :class-title: sd-fs-5
            :class-card: sd-border-0

            .. div::

                Errors that are as close as possible to the problematic line of
                code and whose messages contain all available context help both
                (a) developing code and configuration in the first place, and
                (b) subsequently maintaining it over time as well.


----

Installation
------------

.. code-block:: bash

    pip install fiddle

Just install Fiddle like a standard Python package.

While Fiddle has a minimal set of dependencies by default, Fiddle has support
for command line flags, built on top of ``absl-py``'s command line flags. You
can activate all of this functionality by installing::

    pip install fiddle[flags]



Basic Usage
-----------

TODO

----

Learn More
----------

.. toctree::
   :maxdepth: 3

   colabs.md
   flags_code_lab.md
   api_reference/index
