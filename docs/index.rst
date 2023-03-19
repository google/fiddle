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

Given a hypothentical (Fiddle-oblivious) codebase as follows::

  # Use dataclasses if you want.
  @dataclasses.dataclass
  class Model:
    predicted_value: int = 42  # Always the answer.

    def __call__(self, x):
      return self.predicted_value

  # Or use regular Python classes
  class Engine:
    def __init__(self, model: Model, batch_size: int = 5):
      self.model = model
      self.batch_size = batch_size
      self._thread = threading.thread(target=self.background_thread)
      self._thread.start()

    ...

  # Inherit from whatever superclasses you'd like.
  class Server(my_grpc.MyApiServicer):
    def __init__(self, engine: Engine, port: int):
      ...

    def run(self):
      server = grpc.server(port=self.port)
      my_grpc.add_MyApiServicer_to_server(self.engine, server)
      server.start()
      server.wait_for_termination()

  # Wire up objects.
  def make_server():
    model = Model(predicted_value=101)  # custom model!
    engine = Engine(model=model)
    server = Server(engine=engine, port=9000)
    return server

  # Main function to run our program.
  def main():
    server = make_server()
    server.run()

You can easily use Fiddle to make configuration changes to arbitrarily deeply
nested objects with just a single line of Python::

  @fdl.auto_config  # Adds a `.as_buildable` function to `make_server`.
  def make_server():
    # ... same as before

  def main():
    server_config = make_server.as_buildable()  # type: fdl.Config[Server]

    server_config.port = 8080  # Can make changes to Server's parameters.
    server_config.engine.model.predicted_value = 3000  # Deep changes in 1 line.
    
    # When you're done making configuration changes, just call `fdl.build`.
    server: Server = fdl.build(config)

    # From here-on, Fiddle is gone, and you just have your plain objects.
    server.run()

To learn more about Fiddle's features, including Fiddle's ability to suggest
typo fixes, Fiddle's time-travelling configuration capabilities, and more, be
sure to check out the Colab series (linked below).

----

Learn More
----------

We recommend new users start with the Colab series, which walks you through
Fiddle features in an interactive Jupyter Notebook. See the API reference for
all of Fiddle's features & how to use them.

.. toctree::
   :maxdepth: 3

   colabs.md
   flags_code_lab.md
   api_reference/index

Developing
----------
.. toctree::

    README.md