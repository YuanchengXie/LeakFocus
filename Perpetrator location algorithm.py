import torch
import networkx as nx
from torch_geometric.utils import from_networkx


class Create_Links():
    '''
    Used to extract graph data from csv
    Output in txt format
    The default output name can be set to update_graph.txt
    '''

    def __init__(self, csv_file, output_file):
        self.csv_file = csv_file  # csv file path
        self.output_file = output_file  # Output file path

    def create_links(self):

        link_set = set()

        with open(self.csv_file, 'r+') as f:
            for line in f:
                if 'fields' in line:  # Skip header
                    continue
                line = line.split(',')
                if line[2] == "A" or line[2] == "R":
                    aspath_col = line[7]
                    aspath = aspath_col.split(':')[1].replace('"', '')  # Data cleaning
                    aspath = aspath.split(' ')
                    print(aspath)
                for i in range(1, len(aspath) - 1):
                    if aspath[i] == aspath[i + 1]:
                        continue
                    AS1 = aspath[i].replace('{', '').replace('}', '')
                    AS2 = aspath[i + 1].replace('{', '').replace('}', '')
                    link = AS1 + '|' + AS2
                    print(link)
                    link_set.add(link)

        link_set = list(link_set)
        with open(self.output_file, 'w+') as f:  # Output
            for i in link_set:
                f.write(i + '\n')

class Get_Bad_AS():

    def __init__(self, graph_file, sus_ASN, sus_ASN_weight):
        # Initialization parameters
        self.graph_file = graph_file  # Path to graph data
        self.sus_ASN = sus_ASN  # Suspect AS List
        self.sus_ASN_weight = sus_ASN_weight  # Suspect AS weight list

    def create_general_graph(self):
        '''
        Convert the input txt graph data into a networkx graph
        :return: Overall Internet topology G
        '''
        edges = []
        with open(self.graph_file, 'r+', encoding='utf-8') as f:
            for line in f:
                line = line.replace('\n', '')
                src_ASN = int(line.split('|')[0])
                dst_ASN = int(line.split('|')[1])
                edge = (src_ASN, dst_ASN)
                edges.append(edge)
        G = nx.Graph()
        G.add_edges_from(edges)
        return G

    def get_subgraph(self, G, center_node, distance):
        '''
        Extract the subgraph consisting of all nodes within the specified hop distance centered on center_node from the overall topology G

        :param G: overall Internet topology
        :param center_node: center node
        :param distance: specified hop distance
        :return:
        '''
        # Get all nodes within a specified number of hops from the central node
        nodes = nx.single_source_shortest_path_length(G, center_node, cutoff=distance)
        # Extract subgraph
        subgraph = G.subgraph(nodes).copy()
        return subgraph

    def get_bad_AS(self, sus_ASN, sus_ASN_weight, G):

        '''

        :param sus_ASN:           Suspect ASN          list
        :param sus_ASN_weight:    Suspect ASN corresponding weight    list
        :param G:                 Internet overall topology diagram    nx.Graph
        :return:
        '''
        ASN_2_graph = {}
        updated_weight = []

        for ASN in sus_ASN:  # Traversing suspect nodes
            # Get the subgraph (centered on the suspect node, with a distance of 3 hops)
            ASN = int(ASN)
            center_node = ASN
            distance = 3
            subgraph = self.get_subgraph(G, center_node, distance)

            # Convert NetworkX graph to PyTorch Geometric graph
            data = from_networkx(subgraph)

            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)  # Initialize the weights of all nodes to 0

            for node in subgraph.nodes:  # Enter the weight of the suspect node
                if node in sus_ASN:
                    node_index = list(subgraph.nodes).index(node)
                    data.x[node_index] = sus_ASN_weight[sus_ASN.index(node)]

            print("Node weight before update:")
            for i, weight in enumerate(data.x):
                print(f"node {list(subgraph.nodes)[i]}: {weight.item()}")

            # Update the weight value of the 2-hop node
            for node in subgraph.nodes:
                if nx.shortest_path_length(subgraph, center_node, node) == 2:
                    neighbors_3_hop = []
                    neighbors_3_hop.append(node)
                    for neighbor in subgraph.neighbors(node):
                        if nx.shortest_path_length(subgraph, center_node, neighbor) == 3:
                            neighbors_3_hop.append(neighbor)
                    neighbor_weights = torch.stack([data.x[list(subgraph.nodes).index(n)] for n in neighbors_3_hop])
                    new_weight = torch.mean(neighbor_weights)
                    data.x[list(subgraph.nodes).index(node)] = new_weight

            # Update the weight value of the 1-hop node
            for node in subgraph.nodes:
                if nx.shortest_path_length(subgraph, center_node, node) == 1:
                    neighbors_2_hop = []
                    neighbors_2_hop.append(node)
                    for neighbor in subgraph.neighbors(node):
                        if nx.shortest_path_length(subgraph, center_node, neighbor) == 2:
                            neighbors_2_hop.append(neighbor)
                    neighbor_weights = torch.stack([data.x[list(subgraph.nodes).index(n)] for n in neighbors_2_hop])
                    new_weight = torch.mean(neighbor_weights)
                    data.x[list(subgraph.nodes).index(node)] = new_weight

            # Update the weight value of the central node
            center_neighbors = [n for n in subgraph.neighbors(center_node)]
            center_neighbors.append(center_node)
            center_neighbor_weights = torch.stack([data.x[list(subgraph.nodes).index(n)] for n in center_neighbors])
            center_node_weight = torch.mean(center_neighbor_weights)
            data.x[list(subgraph.nodes).index(center_node)] = center_node_weight

            print("\n Node weight after update:")
            for i, weight in enumerate(data.x):
                print(f"node {list(subgraph.nodes)[i]}: {weight.item()}")

            # 获取中心节点的权重
            center_node_weight = data.x[list(subgraph.nodes).index(center_node)].item()
            print(f"the weight of central Node {center_node}: {center_node_weight}")
            ASN_2_graph[center_node] = subgraph  # Save the subgraph of suspect nodes
            updated_weight.append(center_node_weight)  # Save the updated weight of the suspect node

        output_updated_weight = []

        for i in updated_weight:  # Get the updated weight of each suspect AS
            output_updated_weight.append(i)

        bad_ASN = []

        while sus_ASN:
            # Find the maximum weight and its corresponding ASN
            max_weight_index = updated_weight.index(max(updated_weight))
            max_weight_ASN = sus_ASN[max_weight_index]
            bad_ASN.append(max_weight_ASN)
            # Get the subgraph of the ASN
            subgraph = ASN_2_graph[max_weight_ASN]

            # Check if the other nodes are in the subgraph. If so, remove the node and its weight from sus_ASN and updated_weight.
            nodes_in_subgraph = set(subgraph.nodes)
            indices_to_remove = [i for i, asn in enumerate(sus_ASN) if
                                 asn in nodes_in_subgraph and asn != max_weight_ASN]
            indices_to_remove.append(max_weight_index)

            for index in sorted(indices_to_remove, reverse=True):
                del sus_ASN[index]
                del updated_weight[index]

        print(bad_ASN)
        return bad_ASN, output_updated_weight

    def main(self):
        G = self.create_general_graph()
        bad_ASN, output_updated_weight = self.get_bad_AS(self.sus_ASN, self.sus_ASN_weight, G)
        return bad_ASN, output_updated_weight

class Find_Bad_AS():
    def __init__(self,csv_path,sus_AS,sus_weight):

        self.csv_path = csv_path
        self.sus_ASN = sus_AS # Suspect AS List
        self.sus_ASN_weight = sus_weight  # Suspect AS weight list


    def find_bad_AS(self):
        cl = Create_Links(self.csv_path,'update_graph.txt')
        cl.create_links()
        gba = Get_Bad_AS('update_graph.txt',self.sus_ASN,self.sus_ASN_weight)
        bad_ASN, output_updated_weight = gba.main()
        return bad_ASN, output_updated_weight

if __name__ == '__main__':
    csv_file_path = ''
    sus_AS = []
    sus_weight = []
    fba = Find_Bad_AS(csv_file_path,sus_AS,sus_weight)
    bad_ASN, output_updated_weight = fba.find_bad_AS()
