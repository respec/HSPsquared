''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''


from pandas import DataFrame, read_hdf
from networkx import DiGraph, Graph, topological_sort, connected_components


def graph_from_HDF5(hdfname):
    '''
    Builds a Directed Acyclic Graph (DAG) from the links data in the HDF5 file.

    Parameters
    ----------
    hdfname : string
        Name of the HSP2 HDF5 file.

    Returns
    -------
    dg : Networkx DiGraph
        DAG for HDF5 file.
    '''

    links = read_hdf(hdfname, 'CONTROL/LINKS')
    links['source']      = links.apply(lambda row: f'{row.SVOL}_{row.SVOLNO}', axis='columns')
    links['destination'] = links.apply(lambda row: f'{row.TVOL}_{row.TVOLNO}', axis='columns')
    links = links[['source','destination']].copy()

    dg = DiGraph()
    for i, (source, destination) in links.iterrows():
        dg.add_edge(source, destination)
    return dg


def make_opsequence(hdfname, delt=60):
    '''
    Puts topologically valid /CONTROL/OP_SEQUENCE table into the HSP2 HDF file.

    Parameters
    ----------
    hdfname : string
        Name of HSP2 HDF file.
    delt : TYPE, integer
        Value for OP_SEQUENCE delt value.
        The default is 60.

    Returns
    -------
    None.
    '''

    data = []
    for o in topological_sort(graph_from_HDF5(hdfname)):
        op, seg = o.split('_')
        data.append((op, seg, delt))

    df = DataFrame(data, columns=['OPERATION', 'SEGMENT', 'INDELT_minutes'])
    df.to_hdf(hdfname, 'CONTROL/OP_SEQUENCE', )
    return


def HDF5_isconnected(hdfname):
    '''
    Boolean value if the DAG for the HDF5 file has only one connected subset

    Parameters
    ----------
    hdfname : string
        Name of HSP2 HDF5 file.

    Returns
    -------
    Boolean
        DAG has only one connected subset.
    '''

    dg = graph_from_HDF5(hdfname)
    a = list(connected_components(Graph(dg)))
    return True if len(a) == 1 else False


def component_list(hdfname):
    '''
    List of connected components in HDF5 file DAG

    Parameters
    ----------
    hdfname : string
        Name of HSP2 HDF5 file.

    Returns
    -------
    List of string lists
        Each element of list is a list of connected subset nodes.
    '''

    dg = graph_from_HDF5(hdfname)
    a  = connected_components(Graph(dg))
    return list(a)


def color_graph(hdfname):
    '''
    Demonstrates how DAG can be colored to provide additional info to plot.

    Parameters
    ----------
    hdfname : string
        Name of HSP2 HDF file.

    Returns
    -------
    Tuple(Networkx DiGraph, List of string color names)
        The Networkx draw_network(DAG, colorlist) displays Graph in color.
    '''

    dg = graph_from_HDF5(hdfname)
    cm = []
    for x in dg.nodes:
        npred = len(list(dg.predecessors(x)))
        nsucc = len(list(dg.successors(x)))

        if x.startswith('RCHRES'):
            if   npred == 0 and nsucc == 0: cm.append('red')
            elif npred == 0:                cm.append('orange')
            elif nsucc == 0:                cm.append('yellow')
            else:                           cm.append('blue')
        elif x.startswith('PERLND'):        cm.append('green')
        elif x.startswith('IMPLND'):        cm.append('gray')
        else:                               cm.append('black')

    return dg, cm


# smart_opseq(master, slave) will return soom
