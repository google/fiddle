Tagging: Metadata on fields and values
======================================

Fiddle allows annotating fields of ``Config``'s (and ``Buildable``'s more
generally) and values within a Fiddle configuration DAG with metadata called
``Tag``'s. Tags make bulk update of values easy, and reduces coupling between a
library and its configuration.

.. currentModule:: fiddle

Defining Tags
-------------

.. autoclass:: Tag
    :members: new

Tagging Values
--------------

.. autofunction:: TaggedValue

.. currentModule:: fiddle.tagging

.. autoclass:: TaggedValueCls
    :members: tags


Manipulating tags on `fdl.Buildable`'s
--------------------------------------

Bulk Updates
~~~~~~~~~~~~

.. currentModule:: fiddle

.. autofunction:: set_tagged

.. currentModule:: fiddle.tagging

.. autofunction:: list_tags

.. autofunction:: materialize_tags


Buildable-at-a-time
~~~~~~~~~~~~~~~~~~~

.. currentModule:: fiddle

.. autofunction:: get_tags

.. autofunction:: add_tag

.. autofunction:: set_tags

.. autofunction:: remove_tag

.. autofunction:: clear_tags


Tagging Errors
--------------

.. currentModule:: fiddle.tagging

.. autoclass:: TaggedValueNotFilledError

Tag Type
--------

.. currentModule:: fiddle.tag_type

.. autoclass:: TagType
    :members: __init__, __call__, description, name
