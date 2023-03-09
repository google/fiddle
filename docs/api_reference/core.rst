Core Types & Functions
======================

The Fiddle configuration system builds on a core data model and functions.

.. currentModule:: fiddle


``Config``
----------
.. autoclass:: Config

``Partial``
-----------
.. autoclass:: Partial

``build``
---------
.. autofunction:: build

------

Buildable Manipulation Functions
--------------------------------

.. autofunction:: get_callable

.. autofunction:: update_callable

.. autofunction:: assign

.. autofunction:: copy_with

.. autofunction:: deepcopy_with

.. autofunction:: cast

------

``ArgFactory``
--------------
.. autoclass:: ArgFactory

------

Advanced Functionality
----------------------

``Buildable``
~~~~~~~~~~~~~
.. autoclass:: Buildable
    :members: __init__, __build__, __flatten__, __unflatten__, __path_elements__, __setattr__, __delattr__, __dir__, __repr__, __copy__, __deepcopy__, __eq__, __getstate__, __setstate__


ArgFactory Internals
~~~~~~~~~~~~~~~~~~~~
.. currentModule:: fiddle._src.config
.. autoclass:: _BuiltArgFactory

``TiedValue``
~~~~~~~~~~~~~
.. autoclass:: TiedValue

Casting
~~~~~~~
.. autofunction:: register_supported_cast
