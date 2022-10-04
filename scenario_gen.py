import json
from scipy.stats import norm
import configargparse
import os
from rasterio import mask
import rasterio as rio
from scipy.constants import c as speed_of_light
import numpy as np
import shapely.wkt as wkt
import csv
from diskcache import Cache
from pprint import pprint
import glob
import networkx as nx
import matplotlib.pyplot as plt
import enum
import networkit as nk
import math as m
import osmnx as ox
from distutils.util import strtobool
ox.settings.use_cache = True
ox.settings.log_console = False
cache = Cache(".cache")


raster_dir = 'dsm/'
truenets_dir = 'results/'

tx_power = 30
tx_gain = 10
rx_gain_backhaul = 10  # from Polese et Al
rx_gain_fronthaul = 3  # need to find a reference
sigma_los = 4
sigma_nlos = 7.8
nrv_los = norm(0, sigma_los)
nrv_nlos = norm(0, sigma_nlos)
h_bs = 10
h_ut = 1.5


def transform(shape):
    import pyproj
    from shapely.ops import transform
    wgs84 = pyproj.CRS('EPSG:4326')
    montemario = pyproj.CRS('EPSG:3003')
    project = pyproj.Transformer.from_crs(montemario, wgs84, always_xy=True).transform
    return transform(project, shape)


@cache.memoize()
def get_road_graph(boundbox):
    osmg = ox.graph_from_polygon(boundbox)
    posmg = ox.project_graph(osmg, 'EPSG:3003')
    osm_road = ox.get_undirected(posmg)
    return osm_road


@cache.memoize()
def get_buildings(boundbox):
    tags = {"building": True}
    gdf = ox.geometries_from_polygon(transform(boundbox), tags)
    pgdf = ox.project_gdf(gdf, 'EPSG:3003')
    return pgdf


@cache.memoize()
def get_area(area,  sub_area_id):
    with open(f'{area.lower()}.csv') as sacsv:
        subareas_csv = list(csv.reader(sacsv, delimiter=','))
    for row in subareas_csv:
        if row[1] == sub_area_id:
            # Read the WKT of this subarea
            sub_area = wkt.loads(row[0])
            # Create a buffer of max_d / 2 around it
            return sub_area, sub_area.buffer(300/2)


@cache.memoize()
def read_dsm_transform(area, boundbox):
    big_dsm = rio.open(
        "%s/%s.tif" % (raster_dir, area.lower()), crs=3003)
    raster, transform1 = mask.mask(
        big_dsm, [boundbox], crop=True, indexes=1)
    return transform1


def set_donors(vg, donors_ratio, coverage_graph):
    # Set donors by applying group centrality to each component. Isolated nodes are all donors
    cg = [vg.subgraph(cc) for cc in nx.connected_components(vg)]
    donors = []
    if donors_ratio >= 1:
        return vg.nodes()
    for g in cg:
        if len(g) == 1:
            donors.append(list(g.nodes())[0])
        else:
            n_donors = m.ceil(len(g)*donors_ratio)
            orig_nodes = list(g.nodes())
            vg_nk = nk.nxadapter.nx2nk(g, weightAttr='distance')
            close = nk.centrality.GroupDegree(vg_nk, n_donors)
            close.run()
            d = close.groupMaxDegree()
            mapped_donors = [orig_nodes[di] for di in d]
            donors += mapped_donors   # Map from nk id to nx ids
            assert(len(set(mapped_donors)) == n_donors)

    for n in vg.nodes():
        if n in donors:
            coverage_graph.nodes[n]['type'] = 'donor'
            vg.nodes[n]['type'] = 'donor'
        else:
            coverage_graph.nodes[n]['type'] = 'relay'
            vg.nodes[n]['type'] = 'relay'


def pathloss(d, f, los=True):
    # ETSI TR38.901 Channel Model
    if d < 10:
        d = 10  # Workaround for antennas in the same location as the BS
        # TODO: use 3d distance
    breakpoint_distance = 2*m.pi*h_bs*h_ut*f*1e9/speed_of_light
    if d < breakpoint_distance:
        pl_los = 32.4 + 21*m.log10(d)+20*m.log10(f) + nrv_los.rvs(1)[0]
    else:
        pl_los = 32.4 + 40*m.log10(d)+20*m.log10(f) - 9.5*m.log10((breakpoint_distance)**2 + (h_bs-h_ut)**2) + nrv_los.rvs(1)[0]

    pl_nlos = 22.4 + 35.3*m.log10(d)+21.3*m.log10(f) - 0.3*(h_ut - 1.5) + nrv_nlos.rvs(1)[0]

    if los:
        return pl_los
    else:
        return max(pl_los, pl_nlos)


def pathgain(d, f, los, fronthaul):
    pl = pathloss(d, f, los)
    if fronthaul:
        return tx_power + tx_gain - pl + rx_gain_fronthaul
    else:
        return tx_power + tx_gain - pl + rx_gain_backhaul


def delay(distance):
    return distance / speed_of_light


def make_coverage_graph(n_subs, visgraph, inverse_transmat, viewsheds, f):
    rng = np.random.default_rng()
    subs_loc = rng.choice(a=inverse_transmat, size=n_subs, axis=0)
    coverage_graph = nx.Graph()
    coverage_graph.add_nodes_from(visgraph.nodes(data=True))
    for src in coverage_graph.nodes():
        for tgt in coverage_graph.nodes():
            if src == tgt:
                continue
            if (src, tgt) in visgraph.edges():
                dist = visgraph[src][tgt]['distance']
                coverage_graph.add_edge(src, tgt, distance=dist, pathloss=pathgain(dist, f, los=True, fronthaul=False), delay=delay(dist))
            else:
                p1 = np.array([visgraph.nodes[src]['x'], visgraph.nodes[src]['y'], visgraph.nodes[src]['z']])
                p2 = np.array([visgraph.nodes[tgt]['x'], visgraph.nodes[tgt]['y'], visgraph.nodes[tgt]['z']])
                dist = np.linalg.norm(p2-p1)
                coverage_graph.add_edge(src, tgt, distance=dist, pathloss=pathgain(dist, f, los=False, fronthaul=False), delay=delay(dist))
    if n_subs:
        # For each viewshed (each BS)
        for ndx, viewshed in enumerate(viewsheds):
            pos = np.array([visgraph.nodes[ndx]['x'], visgraph.nodes[ndx]['y']])
            for p in subs_loc:
                coverage_graph.add_node(f'{p[0]}_{p[1]}', type='ue', x=p[0], y=p[1])
                dist = np.linalg.norm(pos-p)
                if viewshed[p[0], p[1]]:
                    # if the element in position pos is in LOS, then add to the graph together with th
                    coverage_graph.add_edge(ndx, f'{p[0]}_{p[1]}', distance=dist, pathloss=pathgain(dist, f, los=True, fronthaul=True), delay=delay(dist))
                else:
                    coverage_graph.add_edge(ndx, f'{p[0]}_{p[1]}', distance=dist, pathloss=pathgain(dist, f, los=False, fronthaul=True), delay=delay(dist))

    return coverage_graph


def remove_isolated_gnb(graph):
    unisolated_graph = graph.copy()
    for n in graph.nodes():
        if graph.degree[n] == 0:
            print(f'removing {n}')
            unisolated_graph.remove_node(n)
    return unisolated_graph


def double_iab_nodes(coverage_graph, pl, delay):
    doubled_graph = nx.Graph()
    for n in coverage_graph.nodes():
        if coverage_graph.nodes[n]['type'] != 'ue':
            doubled_graph.add_node(f'{n}_mt', **coverage_graph.nodes[n])
            doubled_graph.add_node(f'{n}_relay', **coverage_graph.nodes[n])
            doubled_graph.add_edge(f'{n}_mt', f'{n}_relay', distance=0, pathloss=pl, delay=delay)
            for e in coverage_graph[n].items():
                if coverage_graph.nodes[e[0]]['type'] != 'ue':
                    doubled_graph.add_edge(f'{n}_mt', f'{e[0]}_mt', **e[1])
                    doubled_graph.add_edge(f'{n}_relay', f'{e[0]}_relay', **e[1])
                else:
                    doubled_graph.add_edge(f'{n}_mt', e[0], **e[1])
                    doubled_graph.add_edge(f'{n}_relay', e[0], **e[1])

        else:
            doubled_graph.add_node(f'{n}', **coverage_graph.nodes[n])
    return doubled_graph


def order_nodes(coverage_graph):
    ordered_nodes = []
    for n, d in coverage_graph.nodes(data=True):
        if d['type'] == 'donor':
            ordered_nodes.append(n)
    for n, d in coverage_graph.nodes(data=True):
        if d['type'] == 'relay':
            ordered_nodes.append(n)
    for n, d in coverage_graph.nodes(data=True):
        if d['type'] == 'ue':
            ordered_nodes.append(n)
    return ordered_nodes


def generate_colosseum(graph, scenario_name):
    # Generate nodes positions file
    name_map = {}
    nodesPositions = []
    nodes = order_nodes(graph)
    for ndx, n in enumerate(nodes):
        d = graph.nodes[n]
        name_map[ndx] = {'type': d['type'], 'id': n}
        nodesPositions.append([d['x'], d['y'], 1, n])  # TODO: fix z height of UE
    nodesPositions = np.array(nodesPositions)
    np.savetxt(f'scenarios/{scenario_name}/colosseum/nodesPosition.csv', nodesPositions, fmt='%s')
    # Generate pathloss matrix file
    pathlossMatrix = np.zeros(shape=(len(nodes), len(nodes)))
    delayMatrix = np.zeros(shape=(len(nodes), len(nodes)))
    for src in name_map.keys():
        for tgt in name_map.keys():
            if src != tgt:
                try:
                    pathlossMatrix[src][tgt] = graph[name_map[src]['id']][name_map[tgt]['id']]['pathloss']
                    delayMatrix[src][tgt] = graph[name_map[src]['id']][name_map[tgt]['id']]['delay']
                except KeyError:
                    pathlossMatrix[src][tgt] = 0
                    delayMatrix[src][tgt] = 0

    np.savetxt(f'scenarios/{scenario_name}/colosseum/pathlossMatrix.csv', pathlossMatrix, fmt='%.2f')
    np.savetxt(f'scenarios/{scenario_name}/colosseum/delayMatrix.csv', delayMatrix, fmt='%.9f')


def print_map(g, coverage_graph, boundbox, scenario_name):
    def get_node_color(type):
        if type == 'relay':
            return 'orange'
        elif type == 'donor':
            return 'firebrick'
        return 'yellow'

    # Save image of the network
    pos = {n: (d['y'], d['x']) for n, d in g.nodes(data=True) if d['type'] != 'ue'}
    labels = {}
    for ndx, n in enumerate(order_nodes(coverage_graph)):
        n_old = int(n.split('_')[0])
        if n_old in pos:
            labels[n_old] = ndx
    # labels = {n.split('_')[0]:ndx for ndx, n in enumerate(order_nodes(coverage_graph)) if n.split('_')[0] in pos.keys()}

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    buildings = get_buildings(boundbox)
    transform = read_dsm_transform(args.area, boundbox)
    transf_geom = buildings.geometry.affine_transform((~transform).to_shapely())
    transf_geom.plot(ax=ax, color='silver')
    nx.draw_networkx(g,
                     pos,
                     ax=ax,
                     node_size=15,
                     edge_color='mediumseagreen',
                     width=0.5,
                     labels=labels,
                     node_color=[get_node_color(d['type']) for n, d in g.nodes(data=True)],
                     font_size=6,
                     horizontalalignment='left',
                     verticalalignment='bottom',
                     font_color='black')
    ues = np.array([[d['x'], d['y']] for r, d in coverage_graph.nodes(data=True) if d['type'] == 'ue'])
    if ues.size > 0:
        ax.scatter(ues[:, 1], ues[:, 0], s=5)
    # plt.title(f'{scenario_name}')
    plt.grid('on')
    plt.axis('on')
    plt.tight_layout()
    #ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.savefig(f'scenarios/{scenario_name}/map.png')


def print_nodes(g, coverage_graph, boundbox, scenario_name):
    def get_node_color(type):
        return 'orange'

    # Save image of the network
    pos = {n: (d['y'], d['x']) for n, d in g.nodes(data=True) if d['type'] != 'ue'}
    labels = {}
    for ndx, n in enumerate(order_nodes(coverage_graph)):
        n_old = int(n.split('_')[0])
        if n_old in pos:
            labels[n_old] = ndx
    # labels = {n.split('_')[0]:ndx for ndx, n in enumerate(order_nodes(coverage_graph)) if n.split('_')[0] in pos.keys()}

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    buildings = get_buildings(boundbox)
    transform = read_dsm_transform(args.area, boundbox)
    transf_geom = buildings.geometry.affine_transform((~transform).to_shapely())
    transf_geom.plot(ax=ax, color='silver')
    nx.draw_networkx(g,
                     pos,
                     ax=ax,
                     node_size=15,
                     edge_color='mediumseagreen',
                     width=0,
                     labels=labels,
                     node_color=[get_node_color(d['type']) for n, d in g.nodes(data=True)],
                     font_size=6,
                     horizontalalignment='left',
                     verticalalignment='bottom',
                     font_color='black')
    ues = np.array([[d['x'], d['y']] for r, d in coverage_graph.nodes(data=True) if d['type'] == 'ue'])
    if ues.size > 0:
        ax.scatter(ues[:, 1], ues[:, 0], s=5)
    # plt.title(f'{scenario_name}')
    plt.grid('on')
    # plt.axis('on')
    plt.tight_layout()
    #ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.savefig(f'scenarios/{scenario_name}/nodes.png')


def print_nodemap(graph, scenario_name):
    nodemap = {}
    for ndx, n in enumerate(order_nodes(graph)):
        nodemap[f'Node {ndx+1}'] = "None"
    with open(f"scenarios/{scenario_name}/nodemap.json", 'w') as fw:
        json.dump(nodemap, fw, indent=4)


def main(args):
    path = f'{truenets_dir}/{args.area}/twostep/{args.sub_area}/r1/1/100.0/{args.lambda_gnb}/visibility.graphml.gz'
    vgs = glob.glob(path)
    if not len(vgs):
        print(f'{path} not found')
        exit(0)
    g = nx.read_graphml(vgs[0], node_type=int)
    subscriber_area, boundbox = get_area(args.area, args.sub_area)
    area = subscriber_area.area*1e-6  # km2
    n_subs = m.ceil(area*args.lambda_ue) if args.lambda_ue else 0
    viewsheds = np.load(vgs[0].replace('visibility.graphml.gz', 'viewsheds.npy'))
    invtransmat = np.load(f'{truenets_dir}{args.area}/twostep/{args.sub_area}/inverse_translation_matrix.npy')
    if args.remove_isolated:
        g = remove_isolated_gnb(g)
        g = nx.convert_node_labels_to_integers(g)
    coverage_graph = make_coverage_graph(n_subs,
                                         g,
                                         invtransmat,
                                         viewsheds,
                                         args.frequency)

    set_donors(g, args.p_donor, coverage_graph)
    scenario_name = f'{args.area}{args.sub_area}_{args.lambda_gnb}_{args.p_donor}_{args.frequency}'
    os.makedirs(f'scenarios/{scenario_name}/colosseum/', exist_ok=True)
    if args.double_nodes:
        doubled_graph = double_iab_nodes(coverage_graph,
                                         args.doubled_nodes_pl,
                                         args.doubled_nodes_delay)
    else:
        doubled_graph = coverage_graph
    generate_colosseum(doubled_graph, scenario_name)
    print_nodes(g, doubled_graph, boundbox, scenario_name)
    print_map(g, doubled_graph, boundbox, scenario_name)
    print_nodemap(doubled_graph, scenario_name)
    nx.write_graphml(doubled_graph, f"scenarios/{scenario_name}/graph.graphml")
    os.system(f'matlab -batch "convertColosseum "{scenario_name}" "')


if __name__ == '__main__':
    p = configargparse.ArgParser(default_config_files=['sim.yaml'])
    p.add('--frequency', required=True, type=float)
    p.add('--lambda_ue', required=True, type=int)
    p.add('--lambda_gnb', required=True, type=int)
    p.add('--area', required=True, type=str)
    p.add('--sub_area', required=True, type=str)
    p.add('--p_donor', required=True, type=float)
    p.add('--double_nodes', required=True, type=lambda x: bool(strtobool(x)))
    p.add('--remove_isolated', required=True, type=lambda x: bool(strtobool(x)))
    p.add('--doubled_nodes_pl', required=True, type=float)
    p.add('--doubled_nodes_delay', required=True, type=float)
    args = p.parse_args()
    main(args)
