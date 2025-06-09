Fluids
=================

Fluids are classes representing different type of working fluids, which can potentially have properties that depend on temperature.

Fluid
----------

The class :class:`~pydhn.fluids.fluids.Fluid` provides a blueprint for all fluid classes and defines the main methods that are used during simulations. The following table shows a summary of the fluid properties that need to be implemented and the relevant method:

.. list-table::
    :widths: 40 20 20 20
    :header-rows: 1

    * - Name
      - Symbol
      - Method
      - Unit
    * - Specific heat capacity
      - :math:`c_p`
      - :meth:`~pydhn.fluids.fluids.Fluid.get_cp`
      - :math:`J/(kg \cdot K)`
    * - Density
      - :math:`\rho`
      - :meth:`~pydhn.fluids.fluids.Fluid.get_rho`
      - :math:`kg/m^3`
    * - Dynamic viscosity
      - :math:`\mu`
      - :meth:`~pydhn.fluids.fluids.Fluid.get_mu`
      - :math:`Pa \cdot s`
    * - Thermal conductivity
      - :math:`k`
      - :meth:`~pydhn.fluids.fluids.Fluid.get_k`
      - :math:`W/(m \cdot K)`


ConstantWater
---------------

The class :class:`~pydhn.fluids.fluids.ConstantWater` represents water with constant properties:

.. list-table::
    :widths: 25 25 25 25
    :header-rows: 1

    * - Symbol
      - Documentation
      - Default
      - Unit
    * - :math:`c_p`
      - :const:`~pydhn.default_values.default_values.CP_WATER`
      - .. autovalue:: pydhn.default_values.default_values.CP_WATER
      - :math:`J/(kg \cdot K)`
    * - :math:`\rho`
      - :const:`~pydhn.default_values.default_values.RHO_WATER`
      - .. autovalue:: pydhn.default_values.default_values.RHO_WATER
      - :math:`kg/m^3`
    * - :math:`\mu`
      - :const:`~pydhn.default_values.default_values.MU_WATER`
      - .. autovalue:: pydhn.default_values.default_values.MU_WATER
      - :math:`Pa \cdot s`
    * - :math:`k`
      - :const:`~pydhn.default_values.default_values.K_WATER`
      - .. autovalue:: pydhn.default_values.default_values.K_WATER
      - :math:`W/(m \cdot K)`

Example usage:

.. doctest::

	>>> from pydhn.fluids import ConstantWater
	>>> fluid = ConstantWater()
	>>> # Print the fluid specific heat capacity at 60°C
	>>> print(fluid.get_cp(t=60))
	4182.0
	>>> # Print the fluid density at 40°C
	>>> print(fluid.get_rho(t=40))
	990.0


Water
-------

The class :class:`~pydhn.fluids.fluids.Water` models water with variable properties. The relationships between the fluid properties and its temperature are taken from [CzWo98]_. The range of validity is 0-150°C.

Example usage:

.. doctest::

	>>> from pydhn.fluids import Water
	>>> fluid = Water()
	>>> # Print the fluid specific heat capacity at 60°C
	>>> print(fluid.get_cp(t=60))
	4184.625980441447
	>>> # Print the fluid specific heat capacity at 40°C
	>>> print(fluid.get_cp(t=40))
	4178.8227047312685
