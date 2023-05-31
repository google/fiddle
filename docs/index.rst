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


----

Quick Start
-----------

This section provides a brief overview of Fiddle's basic mechanics. For more
detailed documentation on Fiddle's core types, please consult the
:ref:`api_reference`.

:class:`fdl.Config<fiddle.Config>` and :class:`fdl.build()<fiddle.build>`
^^^^^^^^^^^^^^^^^^^^^^^^

Fiddle's core type is the :class:`~fiddle.Config` class. A
:class:`~fiddle.Config` instance contains a reference to a Python callable (such
as a function or class), along with parameters to pass to the callable. Valid
parameters are directly determined by the signature of the callable, and
parameter values can be accessed or changed by name using attribute syntax. For
example::

    import fiddle as fdl

    class MomentumOptimizer:
      def __init__(self, learning_rate, momentum=0.9):
          ...

    # Parameters can be set when constructing a `Config` instance ...
    optimizer_config = fdl.Config(MomentumOptimizer, learning_rate=0.1)
    # ... or overridden later via attribute syntax.
    optimizer_config.learning_rate = 0.01
    optimizer_config.momentum = 0.99

A :class:`~fiddle.Config` instance can be "built" using Fiddle's
:func:`~fiddle.build` function, which calls the underlying callable object,
passing in the configured parameters. For example::

    optimizer_instance = fdl.build(optimizer_config)
    assert isinstance(optimizer_instance, MomentumOptimizer)

In other words ``fdl.build(fdl.Config(MomentumOptimizer, learning_rate=0.1))``
is effectively equivalent to ``MomentumOptimizer(learning_rate=0.1)``.

Nesting configuration
^^^^^^^^^^^^^^^^^^^^^

A single :class:`~fiddle.Config` instance may seem a bit like a
``functools.partial`` that you need to call :func:`fdl.build()<fiddle.build>` on
to invoke, with some syntax sugar for accessing/mutating parameters. However,
:class:`~fiddle.Config` instances become much more powerful when they are
nested, i.e., passed as values to parameters in another :class:`~fiddle.Config`
instance. This is because, unlike invoking a ``functools.partial``, Fiddle's
:func:`~fiddle.build` first traverses the parameters of :class:`~fiddle.Config`
instances, recursively building any nested :class:`~fiddle.Config` instances it
finds. For example::

    class Trainer:
      def __init__(self, model, optimizer, num_steps):
        self.model = model
        self.optimizer = optimizer
        ...

    trainer_config = fdl.Config(
        Trainer,
        model=Config(SomeModel, ...),
        optimizer=optimizer_config,  # Defined above.
        num_steps=10000,
    )

    trainer_instance = fdl.build(trainer_config)
    assert isinstance(trainer_instance, Trainer)
    assert isinstance(trainer_instance.optimizer, MomentumOptimizer)

Object sharing
^^^^^^^^^^^^^^

While recursively building nested subconfigurations,
:func:`fdl.build()<fiddle.build>` ensures that each :class:`~fiddle.Config`
instance is only built once --- the value obtained when building a
:class:`~fiddle.Config` is stored in a memoization dictionary, and if the same
instance is encountered again, the result from building it the first time is
reused. In this way, Fiddle maintains a one-to-one correspondence between the
"configuration graph" (graph of :class:`~fiddle.Config` instances) and the
resulting "object graph" returned from :func:`fdl.build()<fiddle.build>`.

This behavior makes object sharing very natural to represent with Fiddle. For
example::

    class DualEncoder:
      def __init__(self, query_encoder, item_encoder):
        self.query_encoder = query_encoder
        self.item_encoder = item_encoder

    # Create an encoder config...
    encoder_config = fdl.Config(SomeEncoder, ...)
    # Share the same encoder config instance by reusing the config.
    dual_encoder_config = fdl.Config(
        DualEncoder, query_encoder=encoder_config, item_encoder=encoder_config)

    dual_encoder_instance = fdl.build(dual_encoder_config)
    assert dual_encoder_instance.query_encoder is dual_encoder_instance.item_encoder

:class:`fdl.Partial<fiddle.Partial>`
^^^^^^^^^^^^^^^

In addition to :class:`fdl.Config<fiddle.Config>`, Fiddle provides one other
core "buildable" type called :class:`fdl.Partial<fiddle.Partial>` (both
:class:`fdl.Config<fiddle.Config>` and :class:`fdl.Partial<fiddle.Partial>`
inherit from a :class:`~fiddle.Buildable` base class). The
:class:`fdl.Partial<fiddle.Partial>` type is in many ways just like
:class:`fdl.Config<fiddle.Config>` --- it maintains a reference to a callable,
and provides mutable access to the callable's parameter values via attribute
syntax. However, when built via :func:`fdl.build()<fiddle.build>`, instead of
invoking the underlying callable with the configured parameters,
:class:`fdl.Partial<fiddle.Partial>` instead creates a corresponding
``functools.partial`` object. In other words,
``fdl.build(fdl.Partial(MomentumOptimizer, learning_rate=0.1))``
is effectively equivalent to
``functools.partial(MomentumOptimizer, learning_rate=0.1)``.

Using :class:`fdl.Partial<fiddle.Partial>` can be a great option especially for
long-running top-level functions. For example::

    def train(model, num_steps):
      ...

    train_partial = fdl.Partial(
        train,
        model=fdl.Config(SomeModel, ...),
        num_steps=10000
    )
    train_fn = fdl.build(train_partial)
    train_fn()

Building ``train_fn`` via a :class:`fdl.Partial<fiddle.Partial>` and then
executing it separately avoids performing what is likely the majority of the
program's workload inside :func:`fdl.build()<fiddle.build>`, and also avoids
Fiddle appearing in any stack traces related to errors during the execution of
``train_fn``.


----

Learn More
----------

.. toctree::
   :maxdepth: 3

   colabs.md
   flags_code_lab.md
   api_reference/index

Developing
----------
.. toctree::

    README.md