Components
=================


Components are objects representing different elements of the network stored as edges in the network graph. A differentiation is made in PyDHN between **branch** and **edge** components (see :ref:`here <NetPrinciples>`) and between **ideal** and **real** components. These categorizations are used to speed up the hydraulic convergence, currently based on a modified loop method. The difference between *ideal* and *real* components is that the former enforce a setpoint, such as a certain mass flow rate, while the latter define a relationship between pressure losses and mass flow.

Component
----------

The class :class:`~pydhn.components.abstract_component.Component` provides a blueprint for all component classes and defines the main methods that are used during simulations. Each component has its own attributes, stored as a dictionary within the property ``_attrs``. By default, in addition to all the attributes that are enforced as mandatory inputs, ``kwargs`` are also stored there:

.. doctest::

    >>> from pydhn.components import Component
    >>> comp = Component(test_attr=5)
    >>> print(comp._attrs['test_attr'])
    5

Attributes however are not meant to be accessed by directly calling ``_attrs``, but with the built-it :meth:`~pydhn.components.abstract_component.Component.__getitem__` method:

.. doctest::

    >>> from pydhn.components import Component
    >>> comp = Component(test_attr=5)
    >>> print(comp['test_attr'])
    5

and can be changed via the :meth:`~pydhn.components.abstract_component.Component.set` method:

.. doctest::

    >>> from pydhn.components import Component
    >>> comp = Component(test_attr=5)
    >>> comp.set('test_attr', 10)
    >>> print(comp['test_attr'])
    10

In addition, the properties ``_type``, with the unique name of the component type, ``_class`` (either ``"branch_component"`` or ``"leaf_component"``) and ``_is_ideal`` (either ``True`` or ``False``) should be specified when creating a new component class:

.. doctest::

    >>> from pydhn.components import Component
    >>> # Create custom class
    >>> class MyComp(Component):
    ...     def __init__(self, **kwargs):
    ...         super(MyComp, self).__init__()
    ...         # Component class and type
    ...         self._class = "branch_component"
    ...         self._type = "my_custom_component"
    ...         self._is_ideal = True
    ...         # Add new inputs
    ...         self._attrs.update(kwargs)
    ...
    >>> # Instantiate a MyComp object
    >>> my_comp = MyComp()
    >>> # Print component_type
    >>> print(my_comp['component_type'])
    my_custom_component
    >>> # Print component_class
    >>> print(my_comp['component_class'])
    branch_component
    >>> # Print is_ideal
    >>> print(my_comp['is_ideal'])
    True


More complex logic can also be implemented by modifying the method :meth:`~pydhn.components.abstract_component.Component._run_control_logic`, which defines return rules for one or more attributes:

.. doctest::

    >>> from pydhn.components import Component
    >>>
    >>> # Create custom class
    >>> class MyComp(Component):
    ...     def __init__(self, test_value, **kwargs):
    ...         super(MyComp, self).__init__()
    ...         # Component class and type
    ...         self._class = "branch_component"
    ...         self._type = "my_custom_component"
    ...         self._is_ideal = True
    ...         # Add new inputs
    ...         input_dict = {
    ...            "test_value": test_value
    ...         }
    ...         self._attrs.update(input_dict)
    ...         self._attrs.update(kwargs)
    ...
    ...     # Implement the _run_control_logic method
    ...     def _run_control_logic(self, key):
    ...         if key == "test_value":
    ...             return self._attrs["test_value"]**2
    ...         return None
    ...
    >>> # Instantiate a MyComp object
    >>> my_comp = MyComp(test_value=5)
    >>> # Print test_value
    >>> print(my_comp._run_control_logic('test_value'))
    25
    >>> # Check that the original value was not modified
    >>> print(my_comp._attrs['test_value'])
    5


Finally, for each component two private methods defining the functioning during simulations need to be implemented: :meth:`~pydhn.components.abstract_component.Component._compute_delta_p` and :meth:`~pydhn.components.abstract_component.Component._compute_temperatures`.




Pipe
------

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - Component type
     - Component class
     - Is ideal
   * - base_pipe
     - branch_component
     - False

:class:`~pydhn.components.base_pipe.Pipe` is the base implementation for steady-state pipes. It has the following main attributes:

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Input
      - Symbol
      - Documentation
      - Default
      - Unit
    * - diameter
      - :math:`D_p`
      - :const:`~pydhn.default_values.default_values.D_PIPES`
      - .. autovalue:: pydhn.default_values.default_values.D_PIPES
      - :math:`m`
    * - length
      - :math:`L_p`
      - :const:`~pydhn.default_values.default_values.L_PIPES`
      - .. autovalue:: pydhn.default_values.default_values.L_PIPES
      - :math:`m`
    * - roughness
      - :math:`\epsilon`
      - :const:`~pydhn.default_values.default_values.ROUGHNESS`
      - .. autovalue:: pydhn.default_values.default_values.ROUGHNESS
      - :math:`mm`
    * - depth
      - :math:`\delta`
      - :const:`~pydhn.default_values.default_values.DEPTH`
      - .. autovalue:: pydhn.default_values.default_values.DEPTH
      - :math:`m`
    * - k_insulation
      - :math:`k_{ins}`
      - :const:`~pydhn.default_values.default_values.K_INSULATION`
      - .. autovalue:: pydhn.default_values.default_values.K_INSULATION
      - :math:`W/(m·K)`
    * - insulation_thickness
      - :math:`t_{ins}`
      - :const:`~pydhn.default_values.default_values.INSULATION_THICKNESS`
      - .. autovalue:: pydhn.default_values.default_values.INSULATION_THICKNESS
      - :math:`m`
    * - k_internal_pipe
      - :math:`k_{ip}`
      - :const:`~pydhn.default_values.default_values.K_INTERNAL_PIPE`
      - .. autovalue:: pydhn.default_values.default_values.K_INTERNAL_PIPE
      - :math:`W/(m·K)`
    * - internal_pipe_thickness
      - :math:`t_{ip}`
      - :const:`~pydhn.default_values.default_values.INTERNAL_PIPE_THICKNESS`
      - .. autovalue:: pydhn.default_values.default_values.INTERNAL_PIPE_THICKNESS
      - :math:`m`
    * - k_casing
      - :math:`k_{cas}`
      - :const:`~pydhn.default_values.default_values.K_CASING`
      - .. autovalue:: pydhn.default_values.default_values.K_CASING
      - :math:`W/(m·K)`
    * - casing_thickness
      - :math:`t_{cas}`
      - :const:`~pydhn.default_values.default_values.CASING_THICKNESS`
      - .. autovalue:: pydhn.default_values.default_values.CASING_THICKNESS
      - :math:`m`
    * - discretization
      - :math:`-`
      - :const:`~pydhn.default_values.default_values.DISCRETIZATION`
      - .. autovalue:: pydhn.default_values.default_values.DISCRETIZATION
      - :math:`m`

.. _BasePipeHyd:

Hydraulics
""""""""""""

The base Pipe implements the following formula to compute pressure losses:

.. math::

    \begin{equation}\label{pipe_dp}
    \Delta p = \frac{L_p f_D}{4 \pi^2 \rho \left( \frac{D_p}{2} \right) ^5}  \lvert\dot m\lvert\dot m
    \end{equation}

Plus eventually the hydrostatic pressure:

.. math::

    \rho g \Delta z

The friction factor :math:`f_D` is computed as a function of the Reynolds number :math:`Re`. For laminar flow (:math:`Re \leq 2300`) the friction factor is computed as:

.. math::

    \begin{equation}\label{fd_lam1}
    f_D = \frac{64}{Re}
    \end{equation}

For turbulent flow (:math:`Re \geq 4000`) Haaland equation is used:

.. math::

    \begin{equation}\label{fd_lam2}
    f_D = \left\{ 1.8\log_{10} \left[ \left( \frac{\epsilon}{3.7 D_p} \right)^{1.11} + \frac{6.9}{Re}\right]\right\}^{-2}
    \end{equation}

Finally, for the transition regimen (:math:`2320 < Re < 4000`), the relationship given in [HaZa21]_ is used.


.. _BasePipeTher:

Thermal
""""""""""""

The base Pipe implements a simple steady-state model. The outlet temperature, limited by the soil temperature :math:`\theta_{s}`, is computed as:

.. math::

	\begin{equation}\label{dt_pipes}
	\theta_{out} = max\left\{ \theta _{s}, \theta_{in} - \frac{Q}{\dot m c_p} \right\}
	\end{equation}

The thermal losses :math:`Q` (Wh) are computed as:

.. math::

	\begin{equation}\label{pipe_dq}
	Q = \frac{\theta _{in} - \theta _{s}}{R_{0,1} + R_{1,2} + R_{2,3} + R_{conv} + R_{s}} L_p - \frac{\dot m}{\rho} \lvert \Delta p _{fr} \lvert
	\end{equation}

where :math:`\Delta p _{fr}` is the frictional component of the pressure loss and :math:`R_{ip}`, :math:`R_{ins}` and :math:`R_{cas}` are the thermal resistances of the internal pipe, insulation, casing respectively:

.. math::

	\begin{align*}
	R_{ip} = \frac{\ln{\frac{r_{1}}{r_{0}}}}{2 \pi k_{ip}}
	&&
	R_{ins} = \frac{\ln{\frac{r_{2}}{r_{1}}}}{2 \pi k_{ins}}
	&&
	R_{cas} = \frac{\ln{\frac{r_{3}}{r_{2}}}}{2 \pi k_{cas}}
	\end{align*}

And:

.. math::

	\begin{equation}\label{r_conv}
	R_{conv} = \frac{1}{2 \pi r_0 h}
	\end{equation}

.. math::

	\begin{equation}\label{int_tr_coeff}
	h = \frac{ k_{f} Nu}{2 r_0 }
	\end{equation}


Finally, :math:`R_{s}` is the thermal resistance of the soil computed as:

.. math::

	\begin{equation}\label{r_soil_1}
	R_{s} =
		\begin{cases}
			\frac{\ln(\frac{2\delta}{r_3})}{2 \pi k _s},& \text{if } \delta > 3r_0\\
			\frac{\ln(x + \sqrt{x ^2 -1})}{2 \pi k _s},              & \text{otherwise}
		\end{cases}
	\end{equation}

.. math::
	\begin{equation}\label{r_soil_2}
		x = \delta/r_3
	\end{equation}

For the thermal simulation, base Pipes can be discretized in segments of a given length for increasing the accuracy at the cost of computational speed.



Lagrangian Pipe
----------------

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - Component type
     - Component class
     - Is ideal
   * - lagrangian_pipe
     - branch_component
     - False

:class:`~pydhn.components.lagrangian_pipe.LagrangianPipe` is the implementation of a dynamic pipe based on the Lagrangian approach described in [DeAl19]_. It has the following main attributes:

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Input
      - Symbol
      - Documentation
      - Default
      - Unit
    * - diameter
      - :math:`D_p`
      - :const:`~pydhn.default_values.default_values.D_PIPES`
      - .. autovalue:: pydhn.default_values.default_values.D_PIPES
      - :math:`m`
    * - length
      - :math:`L_p`
      - :const:`~pydhn.default_values.default_values.L_PIPES`
      - .. autovalue:: pydhn.default_values.default_values.L_PIPES
      - :math:`m`
    * - roughness
      - :math:`\epsilon`
      - :const:`~pydhn.default_values.default_values.ROUGHNESS`
      - .. autovalue:: pydhn.default_values.default_values.ROUGHNESS
      - :math:`mm`
    * - depth
      - :math:`\delta`
      - :const:`~pydhn.default_values.default_values.DEPTH`
      - .. autovalue:: pydhn.default_values.default_values.DEPTH
      - :math:`m`
    * - k_insulation
      - :math:`k_{ins}`
      - :const:`~pydhn.default_values.default_values.K_INSULATION`
      - .. autovalue:: pydhn.default_values.default_values.K_INSULATION
      - :math:`W/(m·K)`
    * - insulation_thickness
      - :math:`t_{ins}`
      - :const:`~pydhn.default_values.default_values.INSULATION_THICKNESS`
      - .. autovalue:: pydhn.default_values.default_values.INSULATION_THICKNESS
      - :math:`m`
    * - k_internal_pipe
      - :math:`k_{ip}`
      - :const:`~pydhn.default_values.default_values.K_INTERNAL_PIPE`
      - .. autovalue:: pydhn.default_values.default_values.K_INTERNAL_PIPE
      - :math:`W/(m·K)`
    * - internal_pipe_thickness
      - :math:`t_{ip}`
      - :const:`~pydhn.default_values.default_values.INTERNAL_PIPE_THICKNESS`
      - .. autovalue:: pydhn.default_values.default_values.INTERNAL_PIPE_THICKNESS
      - :math:`m`
    * - k_casing
      - :math:`k_{cas}`
      - :const:`~pydhn.default_values.default_values.K_CASING`
      - .. autovalue:: pydhn.default_values.default_values.K_CASING
      - :math:`W/(m·K)`
    * - casing_thickness
      - :math:`t_{cas}`
      - :const:`~pydhn.default_values.default_values.CASING_THICKNESS`
      - .. autovalue:: pydhn.default_values.default_values.CASING_THICKNESS
      - :math:`m`
    * - rho_wall
      - :math:`\rho_p`
      - :const:`~pydhn.default_values.default_values.RHO_INTERNAL_PIPE`
      - .. autovalue:: pydhn.default_values.default_values.RHO_INTERNAL_PIPE
      - :math:`kg/m^3`
    * - cp_wall
      - :math:`cp_p`
      - :const:`~pydhn.default_values.default_values.CP_INTERNAL_PIPE`
      - .. autovalue:: pydhn.default_values.default_values.CP_INTERNAL_PIPE
      - :math:`J/(kg·K)`
    * - h_ext
      - :math:`h_{ext}`
      - :const:`~pydhn.default_values.default_values.H_EXT`
      - .. autovalue:: pydhn.default_values.default_values.H_EXT
      - :math:`W/(m^2·K)`
    * - stepsize
      - :math:`\Delta s`
      - :const:`~pydhn.default_values.default_values.STEPSIZE`
      - .. autovalue:: pydhn.default_values.default_values.STEPSIZE
      - :math:`s`

Each pipe is initialized with a single volume of fluid at a temperature of :autovalue:`pydhn.default_values.default_values.TEMPERATURE` °C. At each time-step, a new volume is inserted in the pipe, pushing all existing volumes.

Hydraulics
""""""""""""

The Lagrangian Pipe implements the same hydraulics of the :class:`~pydhn.components.base_pipe.Pipe` class described :ref:`here <BasePipeHyd>`


Thermal
""""""""""""

The equations used for the thermal simulation in the :class:`~pydhn.components.lagrangian_pipe.LagrangianPipe` class are described in detail in [DeAl19]_. The outlet temperature is computed as the weighted average between the volumes exiting the pipe. If the mass flow is zero, the outlet temperature is the temperature at the outlet section of the pipe. Currently, the function used to compute the temperatures is not vectorized, so the simulation loop will call the method :meth:`~pydhn.components.lagrangian_pipe.LagrangianPipe._compute_temperatures` for each element of this tipe separately using a for loop.


.. note::
	The stepsize is defined for each pipe separately and there is no mechanism in place to check that all dynamic components share the same stepsize.

The component keeps an internal memory of the moving volumes of fluid and their temperatures:

.. doctest::

	>>> from pydhn.components import LagrangianPipe
	>>> from pydhn import ConstantWater
	>>> from pydhn import Soil
	>>> comp = LagrangianPipe(length=100)
	>>> fluid = ConstantWater()
	>>> soil = Soil()
	>>> # Print list of internal volumes
	>>> print(comp._volumes)
	[0.03268513]
	>>> # Print list of internal temperatures
	>>> print(comp._temperatures)
	[50.]
	>>> # Set a mass flow of 1 kg/s
	>>> comp.set("mass_flow", 1.)
	>>> # Simulate one time-step with inlet temperature of 45°C
	>>> _ = comp._compute_temperatures(fluid=fluid, soil=soil, t_in=45)
	>>> # Print list of internal volumes
	>>> print(comp._volumes)
	[0.00505051 0.02763462]
	>>> # Print list of internal temperatures
	>>> print(comp._temperatures)
	[45.        49.9998273]


Producer
-----------------

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - Component type
     - Component class
     - Is ideal
   * - base_producer
     - leaf_component
     - True

:class:`~pydhn.components.base_producer.Producer` models a simple heat source where hydraulics and thermal setpoints are enforced. It has the following main attributes:

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Input
      - Symbol
      - Documentation
      - Default
      - Unit
    * - static_pressure
      - :math:`p_s`
      - :const:`~pydhn.default_values.default_values.STATIC_PRESSURE`
      - .. autovalue:: pydhn.default_values.default_values.STATIC_PRESSURE
      - :math:`Pa`
    * - setpoint_type_hx
      - :math:`-`
      - :const:`~pydhn.default_values.default_values.SETPOINT_TYPE_HX_PROD`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_TYPE_HX_PROD
      - :math:`-`
    * - setpoint_value_hx
      - :math:`\theta_{out}` or :math:`\Delta T` or :math:`Q`
      - :const:`~pydhn.default_values.default_values.SETPOINT_VALUE_HX_PROD`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_VALUE_HX_PROD
      - :math:`°C` or :math:`K` or :math:`Wh`
    * - setpoint_type_hx_rev
      - :math:`-`
      - :const:`~pydhn.default_values.default_values.SETPOINT_TYPE_HX_PROD_REV`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_TYPE_HX_PROD_REV
      - :math:`-`
    * - setpoint_value_hx_rev
      - :math:`\theta_{out}` or :math:`\Delta T` or :math:`Q`
      - :const:`~pydhn.default_values.default_values.SETPOINT_VALUE_HX_PROD_REV`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_VALUE_HX_PROD_REV
      - :math:`°C` or :math:`K` or :math:`Wh`
    * - power_max_hx
      - :math:`Q_{max}`
      - :const:`~pydhn.default_values.default_values.POWER_MAX_HX`
      - .. autovalue:: pydhn.default_values.default_values.POWER_MAX_HX
      - :math:`Wh`
    * - t_out_min_hx
      - :math:`\theta_{min}`
      - :const:`~pydhn.default_values.default_values.T_OUT_MIN`
      - .. autovalue:: pydhn.default_values.default_values.T_OUT_MIN
      - :math:`°C`
    * - setpoint_type_hyd
      - :math:`-`
      - :const:`~pydhn.default_values.default_values.SETPOINT_TYPE_HYD_PROD`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_TYPE_HYD_PROD
      - :math:`-`
    * - setpoint_value_hyd
      - :math:`\Delta p_{set}` or :math:`\dot m_{set}`
      - :const:`~pydhn.default_values.default_values.SETPOINT_VALUE_HYD_PROD`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_VALUE_HYD_PROD
      - :math:`Pa` or :math:`kg/s`


Commonly, one producer is used as the “main” component, enforcing a differential pressure to the network (using a ``'pressure'`` setpoint), while the other producers, if present, have an imposed mass flow.

.. _BaseProdHyd:

Hydraulics
""""""""""""

The base producer has two modes of operations: either the mass flow :math:`\dot m` (kg/s) or pressure difference :math:`\Delta p` (Pa) can be imposed. This is done by setting the relevant :attr:`setpoint_type_hyd`, respectively ``'mass_flow'`` or ``'pressure'``.

.. note::
	A negative pressure difference indicates a pressure lift along the positive direction of the edge.

.. doctest::

	>>> from pydhn.components import Producer
	>>> from pydhn import ConstantWater
	>>> comp = Producer(setpoint_type_hyd='pressure', setpoint_value_hyd=-50000)
	>>> fluid = ConstantWater()
	>>> delta_p, mdot = comp._compute_delta_p(fluid)
	>>> print(delta_p)
	-50000

.. _BaseProdTher:

Thermal
""""""""""""

The base producer can enforce three different types of setpoints for the thermal simulation depending on the value given to :attr:`setpoint_type_hx` the outlet temperature ``'t_out'``, the injected energy ``'delta_q'`` or the temperature difference ``'delta_t'``. The value of the chosen setpoint is then given by :attr:`setpoint_value_hx`. Regardless of the setpoint type, a limitation can be further imposed by limiting the outlet temperature - specifying a value for :attr:`t_out_min_hx` - or the maximum power - specifying a value for :attr:`power_max_hx`.

.. warning::
    :attr:`power_max_hx` comes from an old version of PyDHN where time-steps were assumed to be hourly. What it is actually limiting is the **energy** and not the power of the heat source, according to the formula:

		.. math::

			\begin{equation}\label{dt_prod}
			\theta_{out} = min\left\{ \theta _{set}, \frac{ Q_{max}}{\dot m c_p} + \theta_{in}\right\}
			\end{equation}

    This behaviour will change in a future version.



.. note::
	While the base producer allows different types of setpoints, it is advisable to use at least one ``'t_out'``, as in most cases the solver might not converge in the absence of a fixed nodal temperature in the network.


:attr:`setpoint_type_hx_rev` and :attr:`setpoint_value_hx_rev` are used to specify what should happen in case of reverse flow (IE: negative mass flow). The values and usage are the same as the :attr:`setpoint_type_hx` and :attr:`setpoint_value_hx`.

.. doctest::

	>>> from pydhn.components import Producer
	>>> from pydhn import ConstantWater
	>>> from pydhn import Soil
	>>> comp = Producer(setpoint_type_hx='t_out', setpoint_value_hx=70,
	...                 setpoint_type_hx_rev='delta_t', setpoint_value_hx_rev=0.)
	>>> fluid = ConstantWater()
	>>> soil = Soil()
	>>> # Set a mass flow of 10 kg/s
	>>> comp.set("mass_flow", 10)
	>>> # Simulate one time-step with inlet temperature of 45°C
	>>> t_out = comp._compute_temperatures(fluid=fluid, soil=soil, t_in=45)[1]
	>>> # Print the outlet temperature
	>>> print(t_out)
	70.0
	>>> # Set a mass flow of -1 kg/s
	>>> comp.set("mass_flow", -1)
	>>> # Simulate one time-step with inlet temperature of 45°C
	>>> t_out = comp._compute_temperatures(fluid=fluid, soil=soil, t_in=45)[1]
	>>> # Print the outlet temperature
	>>> print(t_out)
	45.0

Consumer
-----------------

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - Component type
     - Component class
     - Is ideal
   * - base_consumer
     - leaf_component
     - True

:class:`~pydhn.components.base_consumer.Consumer` models a simple consumer where hydraulics and thermal setpoints are enforced. It has the following main attributes:

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Input
      - Symbol
      - Documentation
      - Default
      - Unit
    * - setpoint_type_hx
      - :math:`-`
      - :const:`~pydhn.default_values.default_values.SETPOINT_TYPE_HX_PROD`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_TYPE_HX_PROD
      - :math:`-`
    * - setpoint_value_hx
      - :math:`\theta_{out}` or :math:`\Delta T` or :math:`Q`
      - :const:`~pydhn.default_values.default_values.SETPOINT_VALUE_HX_PROD`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_VALUE_HX_PROD
      - :math:`°C` or :math:`K` or :math:`Wh`
    * - setpoint_type_hx_rev
      - :math:`-`
      - :const:`~pydhn.default_values.default_values.SETPOINT_TYPE_HX_PROD_REV`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_TYPE_HX_PROD_REV
      - :math:`-`
    * - setpoint_value_hx_rev
      - :math:`\theta_{out}` or :math:`\Delta T` or :math:`Q`
      - :const:`~pydhn.default_values.default_values.SETPOINT_VALUE_HX_PROD_REV`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_VALUE_HX_PROD_REV
      - :math:`°C` or :math:`K` or :math:`Wh`
    * - power_max_hx
      - :math:`Q_{max}`
      - :const:`~pydhn.default_values.default_values.POWER_MAX_HX`
      - .. autovalue:: pydhn.default_values.default_values.POWER_MAX_HX
      - :math:`Wh`
    * - t_out_min_hx
      - :math:`\theta_{min}`
      - :const:`~pydhn.default_values.default_values.T_OUT_MIN`
      - .. autovalue:: pydhn.default_values.default_values.T_OUT_MIN
      - :math:`°C`
    * - setpoint_type_hyd
      - :math:`-`
      - :const:`~pydhn.default_values.default_values.SETPOINT_TYPE_HYD_PROD`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_TYPE_HYD_PROD
      - :math:`-`
    * - setpoint_value_hyd
      - :math:`\Delta p_{set}` or :math:`\dot m_{set}`
      - :const:`~pydhn.default_values.default_values.SETPOINT_VALUE_HYD_PROD`
      - .. autovalue:: pydhn.default_values.default_values.SETPOINT_VALUE_HYD_PROD
      - :math:`Pa` or :math:`kg/s`
    * - control_type
      - :math:`-`
      - :const:`~pydhn.default_values.default_values.CONTROL_TYPE_CONS`
      - .. autovalue:: pydhn.default_values.default_values.CONTROL_TYPE_CONS
      - :math:`-`
    * - design_delta_t
      - :math:`\Delta T_{des}`
      - :const:`~pydhn.default_values.default_values.DT_DESIGN`
      - .. autovalue:: pydhn.default_values.default_values.DT_DESIGN
      - :math:`K`
    * - heat_demand
      - :math:`Q_{dem}`
      - :const:`~pydhn.default_values.default_values.HEAT_DEMAND`
      - .. autovalue:: pydhn.default_values.default_values.HEAT_DEMAND
      - :math:`Wh`



Hydraulics
""""""""""""

The base consumer has two possible control types that can be selected by setting the argument :attr:`control_type` as either ``'mass_flow'`` or ``'energy'``. The former behaves exactly like the hydraulic implementation of :class:`~pydhn.components.base_producer.Producer` class described :ref:`here <BaseProdHyd>`.

.. warning::
	Note that selecting ``'mass_flow'`` as :attr:`control_type` also allows to use the differential pressure as setpoint. Furthermore, it requires to se the additional attribute :attr:`setpoint_type_hyd`, that can again be ``'mass_flow'``.	This is indeed confusing and will be changed in a future version.

The control type ``'energy'`` imposes again a mass flow setpoint, with the difference that the value is computed from the :attr:`heat_demand` and :attr:`design_delta_t` values as:

.. math::

    \begin{equation}\label{mdot_from_qdem}
	\dot m_{set} = -\frac{Q _{dem} }{c_{p} \Delta T_{des}}
	\end{equation}

.. note::
	Note that the signs of :math:`Q _{dem}` and :math:`\Delta T_{des}` have to generally be opposite: the demand is normally positive, as it is the energy requested, while the temperature difference is negative, as it represent the (expected) difference between the outlet and inlet temperature of the heat exchanger.

.. doctest::

	>>> from pydhn.components import Consumer
	>>> from pydhn import ConstantWater
	>>> comp = Consumer(control_type='energy', heat_demand=5000.,
	...                 design_delta_t=-30.)
	>>> # Check the hydraulic setpoint value
	>>> comp['setpoint_value_hyd'] # it should be 5000/(30*4182) = 0.03985
	0.03985333970986769


Thermal
""""""""""""

The :class:`~pydhn.components.base_consumer.Consumer` class implements the same method for :meth:`~pydhn.components.base_consumer.Consumer._compute_temperatures`  of the :class:`~pydhn.components.base_producer.Producer` class described :ref:`here <BaseProdTher>`. Commonly, the consumers are given :attr:`setpoint_type_hx` as either ``delta_t`` or ``delta_q``.

.. note::
	:class:`~pydhn.components.base_consumer.Consumer` has the additional argument :attr:`heat_demand`, but this is only used to compute the mass flow setpoint in case :attr:`control_type` is set to ``'energy'``: it has no influence on the thermal simulation.

Branch Valve
-----------------

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - Component type
     - Component class
     - Is ideal
   * - base_branch_valve
     - branch_component
     - False

:class:`~pydhn.components.base_branch_valve.BranchValve` implements a valve controlled by the flow coefficient :math:`K_v`. The model assumes no temperature changes in the fluid across the valve. It has the following main attributes:

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Input
      - Symbol
      - Documentation
      - Default
      - Unit
    * - kv
      - :math:`K_v`
      - :const:`~pydhn.default_values.default_values.KV`
      - .. autovalue:: pydhn.default_values.default_values.KV
      - :math:`m^3/h`

Hydraulics
""""""""""""

The base valve implementation computes the pressure loss according to the following formula:


.. math::

    \begin{equation}\label{kv_valve}
    \Delta p = \frac{1.296\cdot10^9}{\rho K_v^2} |\dot m| \dot m
    \end{equation}

Thermal
""""""""""""

The base valve does not introduce any heat loss, and the outlet temperature is equal to the inlet temperature.


Branch Pump
-----------------

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - Component type
     - Component class
     - Is ideal
   * - branch_pump
     - branch_component
     - True

:class:`~pydhn.components.branch_pump.BranchPump` implements a branch pump imposing a certain :math:`\Delta p`. The model assumes no temperature changes in the fluid across the pump. It has the following main attributes:

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Input
      - Symbol
      - Documentation
      - Default
      - Unit
    * - setpoint_value_hyd
      - :math:`\Delta p`
      - :const:`-`
      - ``0.0``
      - :math:`Pa`

Hydraulics
""""""""""""

The base pump imposes the :math:`\Delta p` value specified by the attribute :attr:`setpoint_type_hyd` regardless of the mass flow.


Thermal
""""""""""""

The branch pump does not introduce any heat loss, and the outlet temperature is equal to the inlet temperature.


Bypass Pipe
-----------------

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - Component type
     - Component class
     - Is ideal
   * - base_bypass_pipe
     - leaf_component
     - False

:class:`~pydhn.components.base_bypass_pipe.BypassPipe` is the implementation of :class:`~pydhn.components.base_pipe.Pipe` as a leaf component. It has the following main attributes:

.. list-table::
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Input
      - Symbol
      - Documentation
      - Default
      - Unit
    * - diameter
      - :math:`D_p`
      - :const:`~pydhn.default_values.default_values.D_BYPASS`
      - .. autovalue:: pydhn.default_values.default_values.D_BYPASS
      - :math:`m`
    * - length
      - :math:`L_p`
      - :const:`~pydhn.default_values.default_values.L_BYPASS_PIPES`
      - .. autovalue:: pydhn.default_values.default_values.L_BYPASS_PIPES
      - :math:`m`
    * - roughness
      - :math:`\epsilon`
      - :const:`~pydhn.default_values.default_values.ROUGHNESS`
      - .. autovalue:: pydhn.default_values.default_values.ROUGHNESS
      - :math:`mm`
    * - depth
      - :math:`\delta`
      - :const:`~pydhn.default_values.default_values.DEPTH`
      - .. autovalue:: pydhn.default_values.default_values.DEPTH
      - :math:`m`
    * - k_insulation
      - :math:`k_{ins}`
      - :const:`~pydhn.default_values.default_values.K_INSULATION`
      - .. autovalue:: pydhn.default_values.default_values.K_INSULATION
      - :math:`W/(m·K)`
    * - insulation_thickness
      - :math:`t_{ins}`
      - :const:`~pydhn.default_values.default_values.BYPASS_INSULATION_THICKNESS`
      - .. autovalue:: pydhn.default_values.default_values.BYPASS_INSULATION_THICKNESS
      - :math:`m`
    * - k_internal_pipe
      - :math:`k_{ip}`
      - :const:`~pydhn.default_values.default_values.K_INTERNAL_PIPE`
      - .. autovalue:: pydhn.default_values.default_values.K_INTERNAL_PIPE
      - :math:`W/(m·K)`
    * - internal_pipe_thickness
      - :math:`t_{ip}`
      - :const:`~pydhn.default_values.default_values.INTERNAL_BYPASS_THICKNESS`
      - .. autovalue:: pydhn.default_values.default_values.INTERNAL_BYPASS_THICKNESS
      - :math:`m`
    * - k_casing
      - :math:`k_{cas}`
      - :const:`~pydhn.default_values.default_values.K_CASING`
      - .. autovalue:: pydhn.default_values.default_values.K_CASING
      - :math:`W/(m·K)`
    * - casing_thickness
      - :math:`t_{cas}`
      - :const:`~pydhn.default_values.default_values.BYPASS_CASING_THICKNESS`
      - .. autovalue:: pydhn.default_values.default_values.BYPASS_CASING_THICKNESS
      - :math:`m`
    * - discretization
      - :math:`-`
      - :const:`~pydhn.default_values.default_values.DISCRETIZATION`
      - .. autovalue:: pydhn.default_values.default_values.DISCRETIZATION
      - :math:`m`

Hydraulics
""""""""""""

See :ref:`here <BasePipeHyd>`.

Thermal
""""""""""""

See :ref:`here <BasePipeTher>`.
