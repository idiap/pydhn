#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions to load data from different sources"""

from warnings import warn

import networkx as nx

from .utilities import isiter

# From Pandas #################################################################


def add_catalogue_from_dataframe(net, catalogue, key_col="DN", key_attr="net"):
    """
    Add properties to pipes in a Network object based on a manufacturer's
    catalogue from a Pandas DataFrame.
    """
    # TODO: avoid for loop
    import pandas as pd

    # Prepare the catalogue
    catalogue = catalogue.set_index(key_col)

    # Assign values to edges
    G = net._graph
    for u, v in G.edges():
        if G[u][v]["edge_type"] == "pipe":
            val = G[u][v][key_attr]
            assert type(catalogue.loc[val]) == pd.Series
            d = catalogue.loc[val].to_dict()
            nx.set_edge_attributes(G, {(u, v): d})


def add_nodes_from_dataframe(net, df, name_col, x_col, y_col, z_col=None):
    """Add nodes to a Network object from a Pandas DataFrame."""
    # TODO: avoid for loop
    for i in df.index:
        name = df.loc[i, name_col]
        x = df.loc[i, x_col]
        y = df.loc[i, y_col]
        if z_col is not None:
            z = df.loc[i, z_col]
            kwargs = df.drop([x_col, y_col, name_col, z_col], axis=1).loc[i]
            net.add_node(name=name, x=x, y=y, z=z, **kwargs)
        else:
            kwargs = df.loc[i].drop([x_col, y_col, name_col]).to_dict()
            net.add_node(name=name, x=x, y=y, **kwargs)


def add_edges_from_dataframe(
    net, df, name_col, start_node_col, end_node_col, edge_type_col=None, edge_type="all"
):
    """Add edges to a Network object from a Pandas DataFrame."""
    # TODO: avoid for loop
    raise NotImplementedError()


# From Postgres ###############################################################


def _df_from_postgres(table_name, schema, conn):
    """
    Imports the specified table from a Postgres database as a Pandas DataFrame.
    """

    # Import necessary libraries
    import pandas as pd

    # Fetch data
    print(f"""Downloading table "{table_name}" """)
    query = "SELECT * FROM {}.{}".format(schema, table_name)

    try:
        return pd.read_sql(query, conn)
    except:
        warn(f"Unable to fetch table {table_name}")
        return None


def _df_from_postgis(table_name, schema, conn, geometry_col):
    """
    Imports the specified table from a Postgis database as either a Pandas
    DataFrame or a GeoPandas GeoDataFrame.
    """

    # Import necessary libraries
    import geopandas as gpd
    import pandas as pd

    # Fetch data
    print(f"""Downloading table "{table_name}" """)
    query = f"SELECT * FROM {schema}.{table_name}"

    try:
        # Check if the table has geometry
        ifquery = f"SELECT EXISTS (SELECT 1 FROM information_schema.columns \
            WHERE table_schema='{schema}' AND table_name='{table_name}' \
            AND column_name='{geometry_col}')"
        if conn.execute(ifquery).fetchone()[0] == True:
            df = gpd.read_postgis(sql=query, con=conn, geom_col=geometry_col)
        else:
            df = pd.read_sql(query, conn)
        return df
    except:
        warn(f"Unable to fetch table {table_name}")
        return None


def _load_tables_from_postgres(table_names, engine, schema, geometry_col=None):
    """
    Downloads one or more tables from a Postgres database. If geometry_col
    is specified, the database is supposed to have Postgis installed, and
    GeoPandas is used.supporting Postgis.

    Usage example:

        import sqlalchemy as sqla

        server = "servername.com"
        port = 0000
        user = "Username"
        pwd = "Password"
        database = "database_name"
        address = f"postgresql://{user}:{pwd}@{server}:{port}/{database}"
        engine = sqla.create_engine(address)

        schema = "schema"

        table_names = ["table_a_name, table_b_name"]

        [table_a, table_b] = _load_tables_from_postgres(table_names=table_names,
                                                        engine=engine,
                                                        schema=schema,
                                                        geometry_col='geom')
    """
    # If the name of a single table is given as str for table_names create a
    # list
    if not isiter(table_names) and type(table_names) == str:
        table_names = [table_names]

    # Connect and load dfs
    dfs = []
    with engine.connect() as conn:
        print("Connection success!")
        for table_name in table_names:
            if geometry_col is not None:
                df = _df_from_postgis(table_name, schema, conn, geometry_col)
            else:
                df = _df_from_postgres(table_name, schema, conn)
            dfs.append(df)
    return dfs


def add_nodes_from_postgres(
    net,
    nodes_table,
    name_col,
    x_col,
    y_col,
    engine,
    schema,
    z_col=None,
    geometry_col=None,
):
    """
    Adds nodes to a Network object from a Postgres database. Supports Postgis.
    """
    [df] = _load_tables_from_postgres(nodes_table, engine, schema, geometry_col)
    add_nodes_from_dataframe(net, df, name_col, x_col, y_col, z_col)


def add_catalogue_from_postgres(
    net, catalogue_table, key_col, key_attr, engine, schema
):
    """
    Add properties to pipes in a Network object based on a manufacturer's
    catalogue stored in a Postgres database.
    """
    [df] = _load_tables_from_postgres(catalogue_table, engine, schema)
    add_catalogue_from_dataframe(net, df, key_col=key_col, key_attr=key_attr)
