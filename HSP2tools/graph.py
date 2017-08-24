''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D. '''


import networkx as nx
import pandas   as pd
import numpy as np


def build_links(hdfname):
    ''' Reads HDF5 NETWORK, SCHEMTIC, MASS_LINK tables to build linkage DataFrame
           CALL build_links(hdfname) '''

    return pd.read_hdf(hdfname, 'CONTROL/LINKS')[['SVOL', 'SVOLNO', 'TVOL', 'TVOLNO']]


def graph_fromHDF(hdfname, sep='_'):
    ''' Converts HDF5 file links to a Directed Acyclic Graph
        CALL: Graph_fromHDF(name, sep)
            name is the DataFrame name
            sep is the optional character between VOL and VOLNO for display'''

    return build_graph(build_links(hdfname), sep)


def build_graph(links, sep):
    ''' Converts a Link DataFrame into a Directed Acyclic Graph
    CALL: build_graph(linkDataframe, seperator)'''

    dg = nx.DiGraph()
    for i,r in links.iterrows():
        sourceID       =  r['SVOL'] + sep + r['SVOLNO']
        destinationID  =  r['TVOL'] + sep + r['TVOLNO']
        dg.add_edge(sourceID, destinationID)   #, label=edgelabel)

    for x in dg.nodes():
        npred = len(dg.predecessors(x))
        nsucc = len(dg.successors(x))

        dg.node[x]['optype']  = x.split(sep)[0]
        dg.node[x]['segment'] = x.split(sep)[1]
        dg.node[x]['fillcolor'] = ''
        if 'RCHRES' in x:
            if  npred == 0 and nsucc == 0:  dg.node[x]['fillcolor'] = 'firebrick'
            elif npred == 0:                dg.node[x]['fillcolor'] = 'darkgreen'
            elif nsucc == 0:                dg.node[x]['fillcolor'] = 'gold'
    return dg


def make_opseq(hdfname):
    order = nx.topological_sort(graph_fromHDF(hdfname))
    order = [o.split('_') for o in order]
    operation, segment = zip(*order)

    opseq = pd.DataFrame(pd.Series(operation), columns=['TARGET'])
    opseq['ID'] = pd.Series(segment)
    opseq['DELT'] = 60

    opseq.TARGET = opseq.TARGET.astype(str)
    opseq.ID     = opseq.ID.astype(str)
    opseq.DELT   = opseq.DELT.astype(str)

    opseq.to_hdf(hdfname, '/CONTROL/OP_SEQUENCE', data_columns=True, format='table')


def smart_opseq(master, slave):
    ''' Creates the simplist OP_SEQUENCE table from changes in the slave relative to the master
    CALL: smart_opseq(master, slave)
      master and slave are names of HDF5 files'''

    with pd.get_store(master, mode='r') as store:
        keys = set(store['HSP2/KEYS'])

    dnodes = []
    k = [key for key in keys if key.startswith('/PERLND')]
    for kk in k:
        dfm = pd.read_hdf(master, kk).fillna('')
        dfs = pd.read_hdf(slave,  kk).fillna('')
        c = np.any(dfm != dfs, axis=1)
        nodes = ['PERLND_' + pls for pls in dfm.index[c].values]
        if nodes:
            dnodes.extend(nodes)

    k = [key for key in keys if key.startswith('/IMPLND')]
    for kk in k:
        dfm = pd.read_hdf(master, kk).fillna('')
        dfs = pd.read_hdf(slave,  kk).fillna('')
        c = np.any(dfm != dfs, axis=1)
        nodes = ['IMPLND_' + ils for ils in dfm.index[c].values]
        if nodes:
            dnodes.extend(nodes)

    k = [key for key in keys if key.startswith('/RCHRES')]
    for kk in k:
        dfm = pd.read_hdf(master, kk).fillna('')
        dfs = pd.read_hdf(slave,  kk).fillna('')
        c = np.any(dfm != dfs, axis=1)
        nodes = ['RCHRES_' + ils for ils in dfm.index[c].values]
        if nodes:
            dnodes.extend(['RCHRES_' + ils for ils in dfm.index[c].values])

    # check Ext_Sources for changes
    if '/CONTROL/EXT_SOURCES' in keys:
        dfm = pd.read_hdf(master, '/CONTROL/EXT_SOURCES').fillna('')
        dfs = pd.read_hdf(slave,  '/CONTROL/EXT_SOURCES').fillna('')
        c = np.any(dfm != dfs, axis=1)
        if np.any(c):
            for indx in dfm[c]:
                dnodes.append[dfm.at[indx, 'TVOL'] + '_' + dfm.at[indx, 'TOPFST']]

    # check LINKS for changes
    if '/CONTROL/SCHEMATIC' in keys:
        dfm = pd.read_hdf(master, '/CONTROL/LINKS').fillna('')
        dfs = pd.read_hdf(slave,  '/CONTROL/LINKS').fillna('')
        c = np.any(dfm != dfs, axis=1)
        if np.any(c):
            for indx in dfm[c]:
                dnodes.append[dfm.at[indx, 'TVOL'] + '_' + dfm.at[indx, 'TVOLNO']]

    # wrap up
    if not dnodes:
        print('No differences found between master and sim files. OP_SEQUENCE table NOT changed.')
        return
    else:
        print('The following DAG nodes showed differences:')
        print(' '.join(sorted(dnodes)))


    dg = graph_fromHDF(master)
    for x in nx.topological_sort(dg):
        if x not in dnodes:
            dg.remove_node(x)
        else:
            dnodes.extend(dg.successors(x))

    order = nx.topological_sort(dg)
    order = [o.split('_') for o in order]
    operation, segment = zip(*order)

    opseq = pd.DataFrame(pd.Series(operation), columns=['TARGET'])
    opseq['ID']   = pd.Series(segment)
    opseq['DELT'] = 60

    opseq.TARGET = opseq.TARGET.astype(str)
    opseq.ID     = opseq.ID.astype(str)
    opseq.DELT   = opseq.DELT.astype(str)

    opseq.to_hdf(slave, '/CONTROL/OP_SEQUENCE', data_columns=True, format='table')


def check_network(HDFname):
    ''' Simple check for disconnected parts of the HDF watershed
    CALL: check_network(HDFname)'''

    g = nx.Graph(graph_fromHDF(HDFname))
    a = list(nx.connected_components(g))

    if (len(a) == 1):
        print('No disconnected nodes found')
    else:
        print(' '.join([len(a) - 1, 'disconnected nodes found']))


def graphtutoral_test10(hdfname):
    ''' TUTORIAL routine to color a watershed DAG'''

    df = pd.read_hdf(hdfname, 'PERLND/GENERAL_INFO')
    df['GISx'] = 7.0
    df['GISy'] = 10.0
    df['GIScolor'] = 'lightblue'
    df['GISarea'] = 6000.0
    df.to_hdf(hdfname, '/PERLND/GENERAL_INFO', data_columns=True, format='t')

    df = pd.read_hdf(hdfname, 'IMPLND/GENERAL_INFO')
    df['GISx'] = 3.0
    df['GISy'] = 1.0
    df['GIScolor'] = 'yellow'
    df['GISarea'] = 3000.0
    df.to_hdf(hdfname, '/IMPLND/GENERAL_INFO', data_columns=True, format='t')

    df = pd.read_hdf(hdfname, 'RCHRES/GENERAL_INFO')
    df['GISx'] = [5.0, 3.0, 7.0, 5.0, 5.0]
    df['GISy'] = [9.0, 6.0, 6.0, 3.5, 0.0]
    df['GIScolor'] = 'lawngreen'
    df.to_hdf(hdfname, '/RCHRES/GENERAL_INFO', data_columns=True, format='t')
