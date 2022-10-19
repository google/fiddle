Select API: bulk manipulation of configs
========================================

The ``select`` API enables changing multiple parallel values in a single line of
code using a "Beautiful Soup"-inspired fluent API.

.. currentModule:: fiddle.selectors

.. autofunction:: select

.. autoclass:: Selection
    :members: get, set, replace, __iter__

.. autoclass:: NodeSelection

.. autoclass:: TagSelection
