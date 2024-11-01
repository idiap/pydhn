The Network class
=================

.. _NetPrinciples:

Principles
----------

Most functions in PyDHN work with a network object, which holds information about the network graph and its components. This object includes methods to read, compute, and modify these information. To accommodate various conventions for constructing this class, PyDHN provides a blueprint with essential functionalities through the :class:`~pydhn.classes.abstract_network.AbstractNetwork` class. This class is intended to serve as a parent class for specific network implementations, rather than being used directly.

In practice, all functions in PyDHN currently expect a specific structure for the child class, as defined by the :class:`~pydhn.classes.network.Network` class. In this structure, the network graph is directed, with edge directions establishing the reference frame. Components are stored in the edges, representing the various elements of the network, while nodes are only connection points between these elements.

The architecture of a Network object's graph is subject to some constraints, mostly due to the custom loop method implemented to simulate the network's hydraulics. It must be a connected graph with nodes of total degree 2 or 3 and must follow a specific layout. This layout is characterized by two parallel lines, one for supply and one for return, interconnected only by out-of-line elements. Components that can be used in one of the two lines are called **branch components**, while components that are designed to be used between the two lines are called **leaf components**. At present, connecting two leaf components in sequence is not supported. Furthermore, one of the leaf components must be the "main" component, where a setpoint pressure difference is enforced. A simple example of this structure is shown below:

.. code-block:: none

    1---->2---->3---->4
    ^     |     |     |
    |     |     |     |
    M    S1    S2    S3
    |     |     |     |
    |     v     v     v
    8<----7<----6<----5

In this example, nodes *1*-*4* form the supply line, and nodes *5*-*8* form the return line. The horizontal edges are branch components, while the vertical edges are leaf components. Among these, *M* represents the main node, such as a heat plant where an array of pumps enforces a predefined pressure lift.

Creating a network
------------------

We can easily build this toy example using the methods provided by the class. Let's start with the supply line. First, we add nodes *1*-*4* using the method :func:`~pydhn.classes.abstract_network.AbstractNetwork.add_node`. We have to specify a ``name`` that can be any hashable type, and optionally we can add the ``x`` and ``y`` coordinates for plotting. If we specify a ``z`` value, this can then be used to automatically compute the altitude difference between the two ends of a component, which can be useful if we want to take into account the hydrostatic pressure in the simulation.

.. doctest::

    >>> from pydhn import Network
    >>> net = Network()
    >>> net.add_node(name=1, x=0, y=1, z=1)
    >>> net.add_node(name=2, x=1, y=1, z=1)
    >>> net.add_node(name=3, x=2, y=1, z=1)

We can also add any other named attribute that we like:

.. doctest::

    >>> net.add_node(name=4, x=3, y=1, z=1, note='test')

Now, we need to add the pipes. One way would be to create a :class:`~pydhn.components.base_pipe.Pipe` object and add it to the network using the method :func:`~pydhn.classes.network.Network.add_component`. For the most common components, however, specific methods are implemented that makes this process easier. In the case of the base pipe, we can use :func:`~pydhn.classes.network.Network.add_pipe`:

.. doctest::

    >>> net.add_pipe(name='P_1-2', start_node=1, end_node=2)
    >>> net.add_pipe('P_2-3', 2, 3)
    >>> net.add_pipe('P_3-4', 3, 4)

We don't need to add the nodes beforehand: if an edge is added between two nodes that don't exist yet, these will be added automatically:

.. doctest::

    >>> net.add_pipe('P_5-6', 5, 6)
    >>> net.add_pipe('P_6-7', 6, 7)
    >>> net.add_pipe('P_7-8', 7, 8)

We can now go on and add the remaing edges, one producer and three consumers:

.. doctest::

    >>> net.add_producer('M', 8, 1)
    >>> net.add_consumer('S1', 2, 7)
    >>> net.add_consumer('S2', 3, 6)
    >>> net.add_consumer('S3', 4, 5)



Accessing data
-----------------

The network object has an internal ordering for nodes and edges. By calling the methods :func:`~pydhn.classes.abstract_network.AbstractNetwork.nodes` and :func:`~pydhn.classes.abstract_network.AbstractNetwork.edges`, an array of node and edge names can be retrieved that follows this order:

.. warning::
    These methods currently have different behaviours. In order to fix this, :func:`~pydhn.classes.abstract_network.AbstractNetwork.nodes` will be modified in a future version.

.. doctest::

    >>> nodes, _ = net.nodes()
    >>> edges = net.edges()
    >>> nodes
    array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> edges
    array([[1, 2],
           [2, 3],
           [2, 7],
           [3, 4],
           [3, 6],
           [4, 5],
           [5, 6],
           [6, 7],
           [7, 8],
           [8, 1]])

We can also use the same methods to access one attribute, or multiple attributes at the same time:

.. doctest::

    >>> nodes, altitude = net.nodes('z')
    >>> altitude
    array([1., 1., 1., 1., 0., 0., 0., 0.])
    >>> edges, delta_z, th_ins = net.edges(['dz', 'insulation_thickness'])
    >>> delta_z
    array([ 0.,  0., nan,  0., nan, nan,  0.,  0.,  0., nan])
    >>> th_ins
    array([0.034, 0.034,   nan, 0.034,   nan,   nan, 0.034, 0.034, 0.034,
             nan])

Masks can be used to find the indices of components meeting a certain criterion. The most general method is :func:`~pydhn.classes.network.Network.mask`, which allows to specify different conditions:

.. doctest::

    >>> net.mask(attr='insulation_thickness', value=0.034, condition='equality')
    array([0, 1, 3, 6, 7, 8], dtype=int64)
    >>> net.mask(attr='component_type', value='base_consumer', condition='equality')
    array([2, 4, 5], dtype=int64)
    >>> net.mask(attr='component_type', value=['base_producer', 'base_consumer'], condition='membership')
    array([2, 4, 5, 9], dtype=int64)

However, some specific masks also have their own method:

.. doctest::

    >>> net.producers_mask
    array([9], dtype=int64)
    >>> net.consumers_mask
    array([2, 4, 5], dtype=int64)
    >>> net.pipes_mask
    array([0, 1, 3, 6, 7, 8], dtype=int64)

A mask can be used to access the attributes of specific components:

.. doctest::

    >>> _, diameter = net.edges('insulation_thickness', mask=net.pipes_mask)
    >>> diameter
    array([0.034, 0.034, 0.034, 0.034, 0.034, 0.034])


Another way to access attrubutes is using the methods :func:`~pydhn.classes.abstract_network.AbstractNetwork.get_nodes_attribute_array` and :func:`~pydhn.classes.abstract_network.AbstractNetwork.get_edges_attribute_array`, which however do not support masks:

.. doctest::

    >>> altitude = net.get_nodes_attribute_array('z')
    >>> altitude
    array([1., 1., 1., 1., 0., 0., 0., 0.])
    >>> th_ins = net.get_edges_attribute_array('insulation_thickness')
    >>> th_ins
    array([0.034, 0.034,   nan, 0.034,   nan,   nan, 0.034, 0.034, 0.034,
             nan])

In order to access data from single attributes, we can use the ``[]`` syntax:

.. doctest::

    >>> net[1]
    {'pos': (0, 1), 'z': 1, 'temperature': 50.0}
    >>> net[1]['pos']
    (0, 1)
    >>> net[(1, 2)] # doctest: +SKIP
    <pydhn.components.base_pipe.Pipe at 0x256b58847a0>
    >>> net[(1, 2)]['name']
    'P_1-2'
    >>> net[(1, 2)]._get_type()
    'base_pipe'


Updating attributes
--------------------

.. warning::
    These methods currently use a different convention for the order and name of key and value inputs. This behavious will be changed in a future version.

Node and edge attributes can be changed in bulk using the methods :func:`~pydhn.classes.abstract_network.AbstractNetwork.set_node_attribute`  and :func:`~pydhn.classes.abstract_network.AbstractNetwork.set_edge_attribute` if the value to be set is the same for all elements:

.. doctest::

    >>> net.set_node_attribute(value=35., name='temperature')
    >>> net.get_nodes_attribute_array('temperature')
    array([35., 35., 35., 35., 35., 35., 35., 35.])
    >>> net.set_edge_attribute(10., 'length', mask=net.pipes_mask)
    >>> net.get_edges_attribute_array('length')
    array([10., 10., nan, 10., nan, nan, 10., 10., 10., nan])


Or :func:`~pydhn.classes.abstract_network.AbstractNetwork.set_node_attributes`  and :func:`~pydhn.classes.abstract_network.AbstractNetwork.set_edge_attributes` if the value to be set is different for all elements:

.. doctest::

    >>> import numpy as np
    >>> ran = np.arange(net.n_nodes)
    >>> net.set_node_attributes(values=ran, name='new_attr')
    >>> net.get_nodes_attribute_array('new_attr')
    array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> ran_edge = np.arange(len(net.pipes_mask))
    >>> net.set_edge_attributes(ran_edge, 'new_attr', mask=net.pipes_mask)
    >>> net.get_edges_attribute_array('new_attr')
    array([ 0.,  1., nan,  2., nan, nan,  3.,  4.,  5., nan])

Single component attributes can finally be modified using the :func:`~pydhn.components.abstract_component.Component.set` method:

.. ::

    >>> net[(1, 2)].set(key='name', value='P_1-2_supply')
    >>> net[(1, 2)]['name']
    'P_1-2_supply'
