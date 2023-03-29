import configargparse
import os
import rasterio as rio
from scipy.constants import c as speed_of_light
import numpy as np
import shapely.wkt as wkt
import csv
from diskcache import Cache
import glob
import networkx as nx
import matplotlib.pyplot as plt
import networkit as nk
import math as m
import osmnx as ox
from distutils.util import strtobool
ox.settings.use_cache = True
ox.settings.log_console = False
cache = Cache(".cache")
from rasterio.io import MemoryFile
import pyproj
from shapely.ops import transform
from shapely.geometry import Point
import rasterio.mask
import math as m



def mytransform(shape, epsg):
    wgs84 = pyproj.CRS('EPSG:4326')
    source_epsg = pyproj.CRS(f'EPSG:{epsg}')
    project = pyproj.Transformer.from_crs(source_epsg, wgs84, always_xy=True).transform
    return transform(project, shape)


# @cache.memoize()
# def get_road_graph(boundbox, epsg):
#     osmg = ox.graph_from_polygon(boundbox)
#     posmg = ox.project_graph(osmg, f'EPSG:{epsg}')
#     osm_road = ox.get_undirected(posmg)
#     return osm_road


@cache.memoize()
def get_buildings(boundbox, epsg):
    tags = {"building": True}
    gdf = ox.geometries_from_polygon(mytransform(boundbox, epsg), tags)
    pgdf = ox.project_gdf(gdf, f'EPSG:{epsg}')
    return pgdf

class ScenarioGenerator():
    def __init__(self, frequency, lambda_gnb, area, sub_area, p_donor, double_nodes, remove_isolated, only_los, double_nodes_pl, double_nodes_delay, epsg, strategy, ratio, colosseum_base_loss, subset, directed, truenets_dir, raster_dir):
        self.frequency = frequency
        self.lambda_gnb = lambda_gnb
        self.area = area
        self.sub_area = sub_area
        self.p_donor = p_donor
        self.double_nodes = double_nodes
        self.remove_isolated = remove_isolated
        self.only_los = only_los
        self.double_nodes_pl = double_nodes_pl
        self.double_nodes_delay = double_nodes_delay
        self.epsg = epsg
        self.strategy = strategy
        self.ratio = ratio
        self.colosseum_base_loss = colosseum_base_loss
        self.subset = subset
        self.directed =  directed
        self.area= area
        self.truenets_dir = truenets_dir
        self.raster_dir = raster_dir 
        self.tx_power = 30
        self.tx_gain = 10
        self.rx_gain_backhaul = 10  # from Polese et Al
        self.rx_gain_fronthaul = 3  # need to find a reference
        self.sigma_los = 4
        self.sigma_nlos = 7.8
        #nrv_los = norm(0, sigma_los)
        #nrv_nlos = norm(0, sigma_nlos)
        self.h_bs = 10
        self.h_ut = 1.5
        self.epsg = 3003
        self.kb = 1.380649*(10**-23)  # J/K
        self.t = 300  # K
        self.nf = 5
        self.subscriber_area, self.boundbox = self.get_area(self.sub_area)
        self.raster = self.read_dsm_transform(self.boundbox)
        self.buildings = get_buildings(self.boundbox, self.epsg)
        


    
    def set_lambda_ue(self, lambda_ue):
        self.lambda_ue = lambda_ue

    def get_area(self, sub_area_id):
        with open(f'{self.area.lower()}.csv') as sacsv:
            subareas_csv = list(csv.reader(sacsv, delimiter=','))
        for row in subareas_csv:
            if int(row[1]) == int(sub_area_id):
                # Read the WKT of this subarea
                sub_area = wkt.loads(row[0])
                # Create a buffer of max_d / 2 around it
                return sub_area, sub_area.buffer(50)

    def set_donors(self, vg, donors_ratio, coverage_graph):
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
                vg.nodes[n]['role'] = 'donor'
            else:
                coverage_graph.nodes[n]['type'] = 'relay'
                vg.nodes[n]['type'] = 'relay'


    def pathloss(self, d, f, los=True):
        # ETSI TR38.901 Channel Model
        if d < 10:
            d = 10  # Workaround for antennas in the same location as the BS
            # TODO: use 3d distance
        breakpoint_distance = 2*m.pi*self.h_bs*self.h_ut*f*1e9/speed_of_light
        if d < breakpoint_distance:
            pl_los = 32.4 + 21*m.log10(d)+20*m.log10(f)  # + nrv_los.rvs(1)[0]
        else:
            pl_los = 32.4 + 40*m.log10(d)+20*m.log10(f) - 9.5*m.log10((breakpoint_distance)**2 + (self.h_bs-self.h_ut)**2)  # + nrv_los.rvs(1)[0]

        pl_nlos = 22.4 + 35.3*m.log10(d)+21.3*m.log10(f) - 0.3*(self.h_ut - 1.5)  # + nrv_nlos.rvs(1)[0]

        if los:
            return pl_los
        else:
            return max(pl_los, pl_nlos)

    def get_snr(self, loss, bandwidth):
        noise = 10*m.log10(1000*self.kb*self.t*bandwidth)
        return loss - noise  - self.nf

    def get_shannon_capacity(self, snr, bandwidth):
        if snr<1:
            return 0
        else:
            return bandwidth*m.log2(1+snr) / 10**6 #to get Mbps
        
    def get_indoor_pl(self, f):
        #TR 38.901 model O2I high loss
        l_glass = 2+0.2*f
        l_concrete = 5+4*f
        loss = 5 - 10*m.log10(0.7*10**(-l_glass/10) + 0.3*10**(-l_concrete/10))
        return loss


    def pathgain(self, d, f, los, fronthaul, indoor):
        pl = self.pathloss(d, f, los)
        if fronthaul:
            if indoor:
                return self.tx_power + self.tx_gain - pl + self.rx_gain_fronthaul + self.get_indoor_pl(f)
            else:
                return self.tx_power + self.tx_gain - pl + self.rx_gain_fronthaul
        else:
            return self.tx_power + self.tx_gain - pl + self.rx_gain_backhaul


    def delay(self, distance):
        return distance / speed_of_light

    def get_indoor_ues(self, outdoor_ues, n):
        import pdb;
        
        max_x = np.max(outdoor_ues[:,0])
        max_y = np.max(outdoor_ues[:,1])
        outdoor_ues = outdoor_ues.tolist()
        i=0
        indoor_ues = []
        while i<n:
            x = np.random.randint(0, max_x, 1)
            y = np.random.randint(0, max_y, 1)
            x_3003, y_3003 = self.raster.xy(x[0],y[0])

            #import pdb; pdb.set_trace()
            if self.buildings.intersects(Point(x_3003, y_3003)).any():
                indoor_ues.append((x[0],y[0]))
                i+=1                
        return indoor_ues
        
            
        

    def make_coverage_graph(self, n_subs, visgraph, inverse_transmat, viewsheds, f):
        #print(f"{n_subs} subscribers")
        rng = np.random.default_rng()
        outdoor_ues = rng.choice(a=inverse_transmat, size=int(n_subs*0.2), axis=0)
        indoor_ues = self.get_indoor_ues(inverse_transmat, int(n_subs*0.8))
        coverage_graph = nx.Graph()
        coverage_graph.add_nodes_from(visgraph.nodes(data=True))
        for src in coverage_graph.nodes():
            for tgt in coverage_graph.nodes():
                if src == tgt:
                    continue
                if (src, tgt) in visgraph.edges():
                    dist = visgraph[src][tgt]['distance']
                    pl = self.pathgain(dist, f, los=True, fronthaul=False, indoor=False)
                    snr = self.get_snr(pl, 400e6)
                    capacity = self.get_shannon_capacity(snr, 400e6)
                    coverage_graph.add_edge(src, tgt, distance=dist, pathloss=pl, snr=snr, capacity=capacity, delay=self.delay(dist), los=True)
                else:
                    p1 = np.array([visgraph.nodes[src]['x'], visgraph.nodes[src]['y'], visgraph.nodes[src]['z']])
                    p2 = np.array([visgraph.nodes[tgt]['x'], visgraph.nodes[tgt]['y'], visgraph.nodes[tgt]['z']])
                    dist = np.linalg.norm(p2-p1)
                    pl = self.pathgain(dist, f, los=False, fronthaul=False, indoor=False)
                    snr = self.get_snr(pl, 400e6)
                    capacity = self.get_shannon_capacity(snr, 400e6)
                    coverage_graph.add_edge(src, tgt, distance=dist, pathloss=pl, snr=snr, capacity=capacity, delay=self.delay(dist), los=False)
        if n_subs:
            # For each viewshed (each BS)
            for ndx, viewshed in enumerate(viewsheds):
                if ndx in visgraph.nodes():
                    pos = np.array([visgraph.nodes[ndx]['x'], visgraph.nodes[ndx]['y']])
                    for p in outdoor_ues:
                        #This is bugged, must fix the mapping
                        pos_3003 = self.raster.xy(p[0], p[1])
                        coverage_graph.add_node(f'{p[0]}_{p[1]}', type='ue', x=p[0], y=p[1], x_3003 = pos_3003[0], y_3003 = pos_3003[1])
                        dist = np.linalg.norm(pos-p)
                        pl = self.pathgain(dist, f, los=bool(viewshed[p[0], p[1]]), fronthaul=True, indoor=False)
                        snr = self.get_snr(pl, 40e6)
                        capacity = self.get_shannon_capacity(snr, 40e6)
                        coverage_graph.add_edge(f'{p[0]}_{p[1]}', ndx, distance=dist,pathloss=pl, snr=snr, capacity=capacity, delay=self.delay(dist), los=bool(viewshed[p[0], p[1]]), indoor=False, fronthaul=True)
                    for p in indoor_ues:
                        pos_3003 = self.raster.xy(p[0], p[1])
                        coverage_graph.add_node(f'{p[0]}_{p[1]}', type='ue', x=p[0], y=p[1], x_3003 = pos_3003[0], y_3003 = pos_3003[1])
                        dist = np.linalg.norm(pos-p)
                        pl = self.pathgain(dist, f, los=False, fronthaul=True, indoor=True)
                        snr = self.get_snr(pl, 40e6)
                        capacity = self.get_shannon_capacity(snr, 40e6)
                        coverage_graph.add_edge(f'{p[0]}_{p[1]}', ndx, distance=dist,pathloss=pl, snr=snr, capacity=capacity, delay=self.delay(dist), los=False, indoor=True, fronthaul=True)
        #print(f"{coverage_graph}")
        return coverage_graph


    def remove_isolated_gnb(self, graph):
        unisolated_graph = graph.copy()
        for n in graph.nodes():
            if graph.degree[n] == 0:
                print(f'removing {n}')
                unisolated_graph.remove_node(n)
        return unisolated_graph


    def double_iab_nodes(self, coverage_graph):
        doubled_graph = nx.Graph()
        for n in coverage_graph.nodes():
            if coverage_graph.nodes[n]['type'] != 'ue':
                doubled_graph.add_node(f'{n}_mt', **coverage_graph.nodes[n], iab_type='mt', role='mt')
                doubled_graph.add_node(f'{n}_relay', **coverage_graph.nodes[n], iab_type='gnb', role='du')
                doubled_graph.add_edge(f'{n}_mt', f'{n}_relay', distance=0, pathloss=self.pl, delay=self.delay, type="wired")
                for e in coverage_graph[n].items():
                    if coverage_graph.nodes[e[0]]['type'] != 'ue':
                        doubled_graph.add_edge(f'{n}_mt', f'{e[0]}_mt', **e[1], type="wireless")
                        doubled_graph.add_edge(f'{n}_relay', f'{e[0]}_relay', **e[1], type="wireless")
                        doubled_graph.add_edge(f'{n}_mt', f'{e[0]}_relay', **e[1], type="wireless")
                        doubled_graph.add_edge(f'{n}_relay', f'{e[0]}_mt', **e[1], type="wireless")
                    else:
                        doubled_graph.add_edge(f'{n}_mt', e[0], **e[1], type="wireless")
                        doubled_graph.add_edge(f'{n}_relay', e[0], **e[1], type="wireless")
            else:
                doubled_graph.add_node(f'{n}', **coverage_graph.nodes[n])
        return doubled_graph


    def order_nodes(self, coverage_graph):
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


    def reindex_nodes(self, graph):
        for ndx, n in enumerate(self.order_nodes(graph)):
            graph.nodes[n]['index'] = ndx+1


    def generate_colosseum(self, graph, scenario_name):
        # Generate nodes positions file
        name_map = {}
        nodesPositions = []
        nodes = self.order_nodes(graph)
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
                try:
                    pathlossMatrix[src][tgt] = -graph[name_map[src]['id']][name_map[tgt]['id']]['pathloss']-self.colosseum_base_loss
                    delayMatrix[src][tgt] = graph[name_map[src]['id']][name_map[tgt]['id']]['delay']
                except KeyError:
                    pathlossMatrix[src][tgt] = np.inf
                    delayMatrix[src][tgt] = 0

        np.savetxt(f'scenarios/{scenario_name}/colosseum/pathlossMatrix.csv', pathlossMatrix, fmt='%.2f')
        np.savetxt(f'scenarios/{scenario_name}/colosseum/delayMatrix.csv', delayMatrix, fmt='%.9f')


    def remove_nlos(self, graph: nx.Graph):
        to_del = []
        for e in graph.edges(data=True):
            if 'los' in e[2]:
                if e[2]['los'] == False:
                    to_del.append(e)
        graph.remove_edges_from(to_del)

    def read_dsm_transform(self, boundbox):
        big_dsm = rio.open(f"{self.raster_dir}/{self.area}.tif", crs=f'EPSG:{self.epsg}')
        raster, transform1 = rio.mask.mask(big_dsm, [boundbox], crop=True, indexes=1)
        with MemoryFile() as memfile:
            new_dataset = memfile.open(driver='GTiff',
                                        height=raster.shape[0],
                                        width=raster.shape[1],
                                        count=1, dtype=str(raster.dtype),
                                        crs=f'EPSG:{self.epsg}',
                                        transform=transform1,
                                        nodata=-9999
                                        )
            new_dataset.write(raster, 1)
            new_dataset.close()
            dataset = memfile.open(crs=f'EPSG:{self.epsg}')
            return dataset


    def print_map(self, coverage_graph, scenario_name):
        def get_node_color(type):
            if type == 'relay':
                return 'orange'
            elif type == 'donor':
                return 'firebrick'
            return 'yellow'

        def get_edge_width(d):
            if d.get('fronthaul', False):
                return 0
            if d.get('los', False):
                return 0.5
            else:
                return 0

        # Save image of the network
        gnbs = [n for n, d in coverage_graph.nodes(data=True)]
        pos = {n: (d['x_3003'], d['y_3003']) for n, d in coverage_graph.nodes(data=True)}
        labels = {}
        for ndx, n in enumerate(self.order_nodes(coverage_graph)):
            if n in pos and ('_' not in str(n) or 'mt' in str(n)):
                labels[n] = f'{ndx+1}'  #f'{n} ({ndx+1})'
        # labels = {n.split('_')[0]:ndx for ndx, n in enumerate(order_nodes(coverage_graph)) if n.split('_')[0] in pos.keys()}

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
        
        #transf_geom = buildings.geometry.affine_transform((~transform).to_shapely())
        self.buildings.plot(ax=ax, color='silver')
        nx.draw_networkx_nodes(coverage_graph,
                            pos,
                            nodelist=gnbs,
                            ax=ax,
                            node_color=[get_node_color(d['type']) for n, d in coverage_graph.nodes(data=True) if n in gnbs],
                            node_size=15)
        nx.draw_networkx_edges(coverage_graph,
                            pos,
                            nodelist=gnbs,
                            ax=ax,
                            edge_color='mediumseagreen',
                            edgelist=[(s,r) for s,r,d, in coverage_graph.edges(data=True) if not d.get('fronthaul', False)])
                            #width=[get_edge_width(d) for s, t, d in coverage_graph.edges(data=True)])
        # nx.draw_networkx_labels(coverage_graph,
        #                         pos,
        #                         ax=ax,
        #                         font_size=6,
        #                         horizontalalignment='left',
        #                         verticalalignment='bottom',
        #                         font_color='black',
        #                         labels=labels)
        ues = np.array([[d['x_3003'], d['y_3003']] for r, d in coverage_graph.nodes(data=True) if d['type'] == 'ue'])
        # if ues.size > 0:
        #     ax.scatter(ues[:, 1], ues[:, 0], s=5)
        # plt.title(f'{scenario_name}')
        plt.grid('on')
        plt.axis('off')
        plt.tight_layout()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.tight_layout()
        plt.savefig(f'scenarios/{scenario_name}/map.png')

    # def plot_map_cartopy(self, g, scenario_name, epsg):
    #     def get_edge_width(d):
    #         if d.get('los', 'False'):
    #             return 1
    #         else:
    #             return 0
    #     def get_node_color(type):
    #         if type == 'relay':
    #             return 'black'
    #         elif type == 'donor':
    #             return 'red'
    #         return 'yellow'
        
    #     pos = {n:(k['x_3003'], k['y_3003']) for n,k in g.nodes(data=True)}

    #     max_x = max(pos.values(), key=lambda x: x[0])[0]+100
    #     max_y = max(pos.values(), key=lambda x: x[1])[1]+100
    #     min_x = min(pos.values(), key=lambda x: x[0])[0]-100
    #     min_y = min(pos.values(), key=lambda x: x[1])[1]-100
    #     extent = [min_x, max_x, min_y, max_y]
    #     proj = ccrs.epsg(f'{epsg}')
    #     fig = plt.figure(figsize=(5, 5))
    #     ax = fig.add_subplot(1, 1, 1, projection=proj)
    #     #ax = plt.axes(projection=proj)
    #     ax.set_extent(extent, proj)
    #     request = cimgt.OSM(cache=True)
    #     ax.add_image(request, 15)    # 5 = zoom level
        
    #     ## graph
    #     nx.draw_networkx_nodes(g, 
    #                         pos=pos,
    #                         node_color=[get_node_color(d['type']) for n, d in g.nodes(data=True)],
    #                         node_size=25)
    #     colors = []
    #     nx.draw_networkx_edges(g, 
    #                         pos=pos,
    #                         edge_color='grey',
    #                         width=[get_edge_width(d) for s, t, d in g.edges(data=True)])
        

    #     plt.savefig(f'scenarios/{scenario_name}/nodes.png', dpi=600,  bbox_inches='tight')
    #     plt.savefig(f'scenarios/{scenario_name}/nodes.pdf', dpi=600,  bbox_inches='tight')
    #     plt.clf()

    def make_directed(self, graph):
        digraph = nx.DiGraph()
        digraph.add_nodes_from(graph.nodes(data=True))
        for e in graph.edges(data=True):
            s = digraph.nodes[e[0]]
            t = digraph.nodes[e[1]]
            d = e[2]
            if s['type'] == 'relay' and t['type'] == 'relay':
                #relay to relay
                digraph.add_edge(e[0],e[1],**d)
                digraph.add_edge(e[1],e[0],**d)
            elif s['type'] == 'ue' or t['type'] == 'donor':
                #print(s,"\n",t,"\n", d)
                #if ue is source or donor is destination, only uplink edge
                digraph.add_edge(e[0],e[1],**d)
            elif s['type'] == 'donor' or t['type'] == 'ue':
                #print(s,"\n",t,"\n", d)
                #if ue is source or donor is destination, only uplink edge
                digraph.add_edge(e[1],e[0],**d)
        return digraph

    def generate_scenario(self,vg_path, frequency, lambda_gnb, only_los, lambda_ue):
        g = nx.read_graphml(vg_path, node_type=int)
        scenario_name = self.get_scenario_name(only_los, lambda_gnb, lambda_ue, frequency)
        viewsheds = np.load(vg_path.replace('visibility.graphml.gz', 'viewsheds.npy'))
        invtransmat = np.load(f'{self.truenets_dir}{self.area}/{self.strategy}/{self.sub_area}/inverse_translation_matrix.npy')
        if self.subset:
            to_remove = [n for n in g.nodes() if n not in self.subset]
            for n in to_remove:
                g.remove_node(n)
        if self.remove_isolated:
            g = self.remove_isolated_gnb(g)
            g = nx.convert_node_labels_to_integers(g)
        coverage_graph = self.make_coverage_graph(int(lambda_ue*self.subscriber_area.area/1000000),
                                                  g,
                                                  invtransmat,
                                                  viewsheds,
                                                  frequency)

        self.set_donors(g, self.p_donor, coverage_graph)
        
        os.makedirs(f'scenarios/{scenario_name}/colosseum/', exist_ok=True)
        if self.double_nodes:
            doubled_graph = self.double_iab_nodes(coverage_graph)
        else:
            doubled_graph = coverage_graph

        if only_los:
            self.remove_nlos(doubled_graph)
            
        self.reindex_nodes(doubled_graph)
        if self.directed:
            doubled_graph = self.make_directed(doubled_graph)
        else:
            self.generate_colosseum(doubled_graph, scenario_name)
        
        return doubled_graph
        
    def read_vg(self, lambda_gnb):
        path = f'{self.truenets_dir}/{self.area}/{self.strategy}/{self.sub_area}/r1/1/{self.ratio}/{lambda_gnb}/visibility.graphml.gz'
        vgs = glob.glob(path)
        if not len(vgs):
            print(f'{path} not found')
            return None
        return vgs[0]

    def get_scenario_name(self, only_los, lambda_gnb, lambda_ue, frequency):
        if only_los:
            scenario_name = f'{self.area}{self.sub_area}_{lambda_gnb}_{lambda_ue}_{self.p_donor}_{frequency}_onlylos'
        else:
            scenario_name = f'{self.area}{self.sub_area}_{lambda_gnb}_{lambda_ue}_{self.p_donor}_{frequency}_nlos'
        return scenario_name

    def main(self):
        for frequency in self.frequency:
            for l in self.lambda_gnb:
                for ol in self.only_los:
                    for lue in self.lambda_ue:
                        vg_path = self.read_vg(lue)
                        scenario_name = self.get_scenario_name(ol, l, lue, frequency)
                        doubled_graph = self.generate_scenario(vg_path, frequency, l, ol, lue)
                        self.print_map(doubled_graph, scenario_name)
                        #self.plot_map_cartopy(doubled_graph, scenario_name, self.epsg)
                        nx.write_graphml(doubled_graph, f"scenarios/{scenario_name}/graph.graphml")
                        print("Now run:")
                        print(f'matlab -batch "convertColosseum "{scenario_name}""')     


if __name__ == '__main__':
    p = configargparse.ArgParser(default_config_files=['sim.yaml'])
    p.add('--frequency', required=True, type=float, action='append')
    p.add('--lambda_ue', required=True, type=int, action='append')
    p.add('--lambda_gnb', required=True, type=int, action='append')
    p.add('--area', required=True, type=str)
    p.add('--sub_area', required=True, type=str)
    p.add('--p_donor', required=True, type=float)
    p.add('--double_nodes', required=True, type=lambda x: bool(strtobool(x)))
    p.add('--remove_isolated', required=True, type=lambda x: bool(strtobool(x)))
    p.add('--only_los', required=True, type=lambda x: bool(strtobool(x)), action='append')
    p.add('--doubled_nodes_pl', required=True, type=float)
    p.add('--doubled_nodes_delay', required=True, type=float)
    p.add('--epsg', required=True, type=int)
    p.add('--strategy', required=True, type=str)
    p.add('--ratio', required=True, type=float)
    p.add('--colosseum_base_loss', required=True, type=float)
    p.add('--subset', required=False, type=int, action='append')
    p.add('--directed', required=False, default=False, type=lambda x: bool(strtobool(x)))


    args = p.parse_args()
    epsg = args.epsg
    truenets_dir = '/home/gabriele.gemmi/nfs/gabriel/results/MedComNet/results/'
    raster_dir = '/home/gabriele.gemmi/nfs/gabriel/data/dtm_fusion/'
    sc = ScenarioGenerator(args.frequency, args.lambda_gnb, args.area, args.sub_area, args.p_donor, args.double_nodes, args.remove_isolated, args.only_los, args.double_nodes_pl, args.double_nodes_delay, args.epsg, args.strategy, args.ratio, args.colosseum_base_loss, args.subset, args.directed, truenets_dir, raster_dir)
    sc.set_lambda_ue(args.lambda_ue)
    sc.main()
