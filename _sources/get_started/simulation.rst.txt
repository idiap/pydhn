Simulation
=================

PyDHN currently implements only a decoupled approach for the network simulation, where the hydraulic state is solved first and the thermal transfer is computed afterwards. Coupling the results could be achieved by iterating over the two simulation phases until the results do not change much between iterations.

Hydraulic simulation
----------------------

While it is relatively easy to create a custom function for the hydraulic simulation, at present the only model implemented is the simplified loop method described in [PYDHN23]_. To reduce the simulation time, some limitations are imposed on the network topology and boundary conditions as described in :ref:`the Network class page <NetPrinciples>`.

The loop method is based on solving the following system of equations using the Newton-Raphson method:

.. math::
	\mathbf{B}\phi(\mathbf{B}^T\mathbf{\tilde{\dot m}}) = \mathbf{0}

where:
	- :math:`\mathbf{B}` is a fundamental cycle matrix of the network graph
	- :math:`\mathbf{\tilde{\dot m}}` is the vector of fundamental cycle mass flows
	- :math:`\phi` is a vector-valued function that maps each component's mass flow to the corresponding pressure difference

The simplified model takes advantage of the constraints in the network topology to find a specific set of fundamental cycles that allows for easily imposing the known setpoints and as such solving some of the equations in the system beforehand.

The model is implemented in the function :func:`~pydhn.solving.hydraulic_simulation.solve_hydraulics`. The following table gives an overview of its main arguments:

.. list-table::
    :widths: 10 70 10 10
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Default
    * - :attr:`net`
      - Network object.
      - :class:`~pydhn.classes.network.Network`
      - ``-``
    * - :attr:`fluid`
      - Fluid object.
      - :class:`~pydhn.fluids.fluids.Fluid`
      - ``-``
    * - :attr:`controller`
      - Controller object.
      - :class:`~pydhn.controllers.controller.Controller`
      - ``None``
    * - :attr:`compute_hydrostatic`
      - Whether to consider hydrostatic pressure.
      - :class:`bool`
      - ``True``
    * - :attr:`compute_singular`
      - Not implemented.
      - :class:`bool`
      - ``False``
    * - :attr:`affine_fd`
      - | Whether to use an affine function for pipe
        | pressure losses in the transitional flow regime.
      - :class:`bool`
      - ``False``
    * - :attr:`max_iters`
      - Maximum number of iterations for the solver.
      - :class:`int`
      - ``100``
    * - :attr:`error_threshold`
      - Error threshold for the solver in Pa.
      - :class:`float`
      - ``100``
    * - :attr:`damping_factor`
      - Damping factor for the Newton iterations.
      - :class:`float`
      - ``1``
    * - :attr:`decreasing`
      - | Whether to reduce the damping factor at each
        | Newton iteration.
      - :class:`bool`
      - ``False``
    * - :attr:`adaptive`
      - Whether to reduce the damping factor on plateau.
      - :class:`bool`
      - ``False``
    * - :attr:`verbose`
      - Controls the verbosity of the simulation.
      - :class:`int`
      - ``1``
    * - :attr:`ts_id`
      - Specifies the ID of the current time-step.
      - :class:`int`
      - ``None``


The following snippet shows the usage of :func:`~pydhn.solving.hydraulic_simulation.solve_hydraulics`:

.. doctest::

	>>> from pydhn import solve_hydraulics
	>>> from pydhn import ConstantWater
	>>> from pydhn.networks import star_network
	>>> net = star_network()
	>>> fluid = ConstantWater()
	>>> # Assign mass flow setpoints to consumers
	>>> net.set_edge_attribute(
	...     value="mass_flow", name="control_type", mask=net.consumers_mask
	...     )
	>>> net.set_edge_attribute(
	...     value="mass_flow", name="setpoint_type_hyd", mask=net.consumers_mask
	... )
	>>> setpoints = [0.01, 0.03, 0.2]
	>>> net.set_edge_attributes(
	...     values=setpoints, name="setpoint_value_hyd", mask=net.consumers_mask
	... )
	>>> # Assign pressure lift and starting pressure to the producer
	>>> net.set_edge_attribute(
	...     value="pressure", name="setpoint_type_hyd", mask=net.producers_mask
	... )
	>>> net.set_edge_attributes(
	...     values=[-50000], name="setpoint_value_hyd", mask=net.producers_mask
	... )
	>>> net.set_edge_attributes(
	...     values=[100000], name="static_pressure", mask=net.producers_mask
	... )
	>>> # Check mass flow, differential pressure and nodal pressure
	>>> _, mass_flow_init, delta_p_init = net.edges(["mass_flow", "delta_p"])
	>>> print(mass_flow_init)
	[1.e-05 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05
	 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05 1.e-05]
	>>> print(delta_p_init)
	[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
	>>> _, pressure_init = net.nodes("pressure")
	>>> print(pressure_init)
	[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
	>>> # Solve
	>>> results = solve_hydraulics(net, fluid, max_iters=25, verbose=0)
	>>> # Check convergence
	>>> print(results['history']) # doctest: +SKIP
	{'hydraulics converged': True, 'hydraulics iterations': 2,
	'hydraulics errors': [3075.485197359615, 385.627545640828, 38.28381758111959]}
	>>> # Check mass flow, differential pressure and nodal pressure
	>>> _, mass_flow, delta_p = net.edges(["mass_flow", "delta_p"])
	>>> print(mass_flow) # doctest: +SKIP
	[0.24        0.14018399  0.09981601 -0.05981601  0.2         0.06981601
	0.03        0.01        0.01        0.03        0.2         0.24
	0.24        0.14018248  0.09981752 -0.05981752  0.06981752  0.01
	0.03        0.2       ]
	>>> print((delta_p).astype(int)) # Pa # doctest: +SKIP
	[ 40607   1480    791   -313   2877    413     92     10 -33570 -32984
	-39931 -50000  40607   1480    791   -313    413     10     92   2877]
	>>> _, pressure = net.nodes("pressure")
	>>> print(pressure) # Pa # doctest: +SKIP
	[100000.          59392.25343201  57911.79761464  58638.85286523
	58225.20256416  58253.11545484  58545.9487053   55072.6298674
	50038.20169509  90645.94826308  92126.37435409  91437.57252758
	91812.95519263  91823.24399703  91530.47668751  95003.74379642]


Thermal simulation
----------------------

The thermal simulation is carried out, assuming the mass flow has been computed, using the method described in [PYDHN23]_. Considering a modified network graph :math:`\mathcal G' = (\mathcal V, \mathcal E')`, where edges with negative mass flow are reversed in order to have only positive mass flows, the method is based on solving the heat balance at each node :math:`i \in \mathcal{V}` given by:

.. math::

	\mathbf{A^+} \cdot (|\mathbf{\dot{m}}| \odot \mathbf{c_p} \odot \mathbf{\psi}(\boldsymbol{\theta}_{\text{in}})) = \mathbf{A^-} \cdot (|\mathbf{\dot{m}}| \odot \mathbf{c_p} \odot \boldsymbol{\theta}_{\text{in}})

.. math::
	\boldsymbol{\theta}_{\text{in}} = (\mathbf{A^-})^\top \boldsymbol{\theta}

where:
	- :math:`\mathbf{\dot{m}}` is the vector of mass flows along each edge,
	- :math:`\mathbf{c_p}` is the vector of specific heat capacities for each edge,
	- :math:`\boldsymbol{\theta}_{\text{in}}` and :math:`\boldsymbol{\theta}_{\text{out}}` are vectors of inlet and outlet temperatures, respectively,
	- :math:`\boldsymbol{\theta}` is the vector of nodal temperatures,
	- :math:`\odot` denotes element-wise multiplication,
	- :math:`\mathbf{\psi}` is a vector-valued function that maps each inlet temperature :math:`\theta_{\text{in}, i}` to its corresponding outlet temperature :math:`\theta_{\text{out}, i}` based on component-specific behavior,
	- :math:`\mathbf{A^+} \in \{0, 1\}^{|\mathcal V| \times |\mathcal E'|}` and :math:`\mathbf{A^-} \in \{0, 1\}^{|\mathcal V| \times |\mathcal E'|}` are respectively the positive and negative part of the incidence matrix of :math:`\mathcal G'`.


The system is solved using the Newton-Raphson method.

The model is implemented in the function :func:`~pydhn.solving.thermal_simulation.solve_thermal`. The following table gives an overview of its main arguments:


.. list-table::
    :widths: 10 70 10 10
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Default
    * - :attr:`net`
      - Network object.
      - :class:`~pydhn.classes.network.Network`
      - ``-``
    * - :attr:`fluid`
      - Fluid object.
      - :class:`~pydhn.fluids.fluids.Fluid`
      - ``-``
    * - :attr:`soil`
      - Soil object.
      - :class:`~pydhn.soils.soils.Soil`
      - ``-``
    * - :attr:`max_iters`
      - Maximum number of iterations for the solver.
      - :class:`int`
      - ``100``
    * - :attr:`error_threshold`
      - Error threshold for the solver in Wh.
      - :class:`float`
      - ``1e-6``
    * - :attr:`mass_flow_min`
      - Mass flow (kg/s) used to approximate 0.
      - :class:`float`
      - ``1e-16``
    * - :attr:`damping_factor`
      - Damping factor for the Newton iterations.
      - :class:`float`
      - ``1``
    * - :attr:`decreasing`
      - | Whether to reduce the damping factor at each
        | Newton iteration.
      - :class:`bool`
      - ``False``
    * - :attr:`adaptive`
      - Whether to reduce the damping factor on plateau.
      - :class:`bool`
      - ``False``
    * - :attr:`verbose`
      - Controls the verbosity of the simulation.
      - :class:`int`
      - ``1``
    * - :attr:`ts_id`
      - Specifies the ID of the current time-step.
      - :class:`int`
      - ``None``

The following snippet shows the usage of :func:`~pydhn.solving.thermal_simulation.solve_thermal`:

.. doctest::

	>>> from pydhn import solve_hydraulics
	>>> from pydhn import solve_thermal
	>>> from pydhn import ConstantWater
	>>> from pydhn import Soil
	>>> from pydhn.networks import star_network
	>>> net = star_network()
	>>> fluid = ConstantWater()
	>>> soil = Soil()
	>>> # Solve hydraulics
	>>> setpoints = [0.01, 0.03, 0.2]
	>>> net.set_edge_attributes(
	...    values=setpoints, name="setpoint_value_hyd", mask=net.consumers_mask
	... )
	>>> _ = solve_hydraulics(net, fluid, max_iters=25, verbose=0)
	>>> # Check init values
	>>> _, init_temp = net.nodes("temperature")
	>>> print(init_temp)
	[50. 50. 50. 50. 50. 50. 50. 50. 50. 50. 50. 50. 50. 50. 50. 50.]
	>>> _, init_delta_q = net.edges("delta_q")
	>>> print(init_delta_q)
	[nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
	 nan nan]
	>>> # Set thermal setpoints
	>>> net.set_edge_attribute(
	...     value="delta_q", name="setpoint_type_hx", mask=net.consumers_mask
	... )
	>>> setpoints = [4000., 8000., 3500.] # Wh
	>>> net.set_edge_attribute(
	...     value="t_out", name="setpoint_type_hx", mask=net.producers_mask
	... )
	>>> net.set_edge_attributes(
	...    values=setpoints, name="setpoint_value_hx", mask=net.consumers_mask
	... )
	>>> net.set_edge_attribute(
	...    value=80, name="setpoint_value_hx", mask=net.producers_mask # °C
	... )
	>>> # Solve thermal
	>>> results = solve_thermal(net, fluid, soil, max_iters=25, verbose=0)
	>>> # Check convergence
	>>> print(results['history']) # doctest: +SKIP
	{'thermal converged': True, 'thermal iterations': 1,
	'thermal errors': [3.586800573888093, 1.7763568394002505e-15]}
	>>> # Check new values
	>>> _, temperature = net.nodes("temperature")
	>>> print(temperature) # °C # doctest: +SKIP
	[80.         77.39387785 76.90555315 76.85854098 75.38150756 74.63815243
	76.09889036 76.14538386 42.36033088 43.64937803 35.68422391 52.92974914
	50.16778038 50.63815243 55.09889036 28.14538386]
	>>> _, delta_q = net.edges("delta_q")
	>>> print(delta_q) # Wh # doctest: +SKIP
	[-1303.06107746  -127.69486047  -127.68008654  -126.48168132
	-126.69488196  -124.14867849  -126.6084377   -123.89252241
	-4000.         -3500.         -8000.         18819.83456219
	-644.5235786    -50.93011554   -82.66195626   -77.40107693
	-76.02622531   -78.3953416    -86.59758414   -37.03645745]
