AutoConfig: Reinterpret builder functions for deep configurability
==================================================================

The ``@auto_config`` function decorator reinterprets the function to construct a
data structure of ``fdl.Config``'s that correspond to the objects instead of the
objects themselves.

.. currentModule:: fiddle.experimental.auto_config

``auto_config``
---------------

.. autofunction:: auto_config

``is_auto_config``
------------------

.. autofunction:: is_auto_config

``auto_unconfig``
-----------------

.. autofunction:: auto_unconfig

``inline``
----------

.. autofunction:: inline

``AutoConfig``
--------------

.. autoclass:: AutoConfig
    :members: as_buildable
