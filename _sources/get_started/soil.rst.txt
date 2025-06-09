Soils
=================

Soils are classes representing different type of soil, which can potentially have temperature and thermal conductivity that vary with the depth and with the simulation time-steps.

Soil
----------

The class :class:`~pydhn.soils.soils.Soil` is the base soil class from which more complex classes can inherit. It has constant properties in time and depth. The following table shows a summary of methods that need to be implemented for soils, as well as the default values used in the base :class:`~pydhn.soils.soils.Soil` class:

.. list-table::
    :widths: 25 10 25 15 10 15
    :header-rows: 1

    * - Name
      - Symbol
      - Method
      - Documentation
      - Default
      - Unit
    * - Temperature
      - :math:`\theta_s`
      - :meth:`~pydhn.soils.soils.Soil.get_temp`
      - :const:`~pydhn.default_values.default_values.T_SOIL`
      - .. autovalue:: pydhn.default_values.default_values.T_SOIL
      - :math:`°C`
    * - Thermal conductivity
      - :math:`k_s`
      - :meth:`~pydhn.soils.soils.Soil.get_k`
      - :const:`~pydhn.default_values.default_values.K_SOIL`
      - .. autovalue:: pydhn.default_values.default_values.K_SOIL
      - :math:`W/(m \cdot K)`

KusudaSoil
------------

The class :class:`~pydhn.soils.soils.KusudaSoil` implements the Kusuda model as described in [RoDa03]_. It has varying temperature in time and depth.
