from utils.functions import pairwise_iteration

from datanetAPI.datanetAPI import DatanetAPI

import networkx as nx
import numpy as np
import copy
import os
import gin.tf


DEFAULT_EDGE_ATTRIBUTES = {
    'increments': 1,
    'reductions': 1,
    'weight': 0.0,
    'traffic': 0.0
}


@gin.configurable
class Environment(object):

    def __init__(self,
                 env_type='NSFNet',
                 traffic_profile='uniform',
                 routing='ecmp',
                 init_sample=0,
                 seed_init_weights=1,
                 min_weight=1.0,
                 max_weight=4.0,
                 weight_change=1.0,
                 weight_update='sum',
                 weigths_to_states=True,
                 link_traffic_to_states=True,
                 probs_to_states=False,
                 reward_magnitude='link_traffic',
                 base_reward='min_max',
                 reward_computation='change',
                 base_dir='datasets'):
        
        env_type = [env for env in env_type.split('+')]#env_type if type(env_type) == list else [env_type]
        self.env_type = env_type 
        self.traffic_profile = traffic_profile
        self.routing = routing

        self.num_sample = init_sample-1
        self.seed_init_weights = seed_init_weights
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_change = weight_change
        self.weight_update = weight_update

        num_features = 0
        self.weigths_to_states = weigths_to_states
        if self.weigths_to_states: num_features += 1
        self.link_traffic_to_states = link_traffic_to_states
        if self.link_traffic_to_states: num_features += 1
        self.probs_to_states = probs_to_states
        if self.probs_to_states: num_features += 2
        self.num_features = num_features
        self.reward_magnitude = reward_magnitude
        self.base_reward = base_reward
        self.reward_computation = reward_computation

        self.base_dir = base_dir
        self.dataset_dirs = []
        for env in env_type:
            self.dataset_dirs.append(os.path.join(base_dir, env, traffic_profile))
        
        self.initialize_environment()
        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()
        self.reward_measure = self.compute_reward_measure()
        self.set_target_measure()


    def load_topology_object(self):
        try:
            nx_file = os.path.join(self.base_dir, self.network, 'graph_attr.txt')
            self.topology_object = nx.DiGraph(nx.read_gml(nx_file, destringizer=int))
        except: 
            self.topology_object = nx.DiGraph()
            capacity_file = os.path.join(self.dataset_dir, 'capacities', 'graph.txt')
            with open(capacity_file) as fd:
                for line in fd:
                    if 'Link_' in line:
                        camps = line.split(" ")
                        self.topology_object.add_edge(int(camps[1]),int(camps[2]))
                        self.topology_object[int(camps[1])][int(camps[2])]['bandwidth'] = int(camps[4])

    def load_capacities(self):
        if self.traffic_profile == 'gravity_full':
            capacity_file = os.path.join(self.dataset_dir, 'capacities', 'graph-TM-'+str(self.num_sample)+'.txt')
        else:
            capacity_file = os.path.join(self.dataset_dir, 'capacities', 'graph.txt')
        with open(capacity_file) as fd:
            for line in fd:
                if 'Link_' in line:
                    camps = line.split(" ")
                    self.G[int(camps[1])][int(camps[2])]['capacity'] = int(camps[4])

    def load_traffic_matrix(self):
        tm_file = os.path.join(self.dataset_dir, 'TM', 'TM-'+str(self.num_sample))
        self.traffic_demand = np.zeros((self.n_nodes,self.n_nodes))
        with open(tm_file) as fd:
            fd.readline()
            fd.readline()
            for line in fd:
                camps = line.split(" ")
                self.traffic_demand[int(camps[1]),int(camps[2])] = float(camps[3])
        self.get_link_probs()

    def initialize_environment(self, num_sample=None, random_env=True):
        if num_sample is not None:
            self.num_sample = num_sample
        else:
            self.num_sample += 1
        if random_env:
            num_env = np.random.randint(0,len(self.env_type))
        else:
            num_env = self.num_sample % len(self.env_type)
        self.network = self.env_type[num_env]
        self.dataset_dir = self.dataset_dirs[num_env]

        self.load_topology_object()
        self.generate_graph()
        self.load_capacities()
        self.load_traffic_matrix()

    def next_sample(self):
        if len(self.env_type) > 1:
            self.initialize_environment()
        else:
            self.num_sample += 1
            self._reset_edge_attributes()
            self.load_capacities()
            self.load_traffic_matrix()

    def define_num_sample(self, num_sample):
        self.num_sample = num_sample - 1

    def reset(self, change_sample=False):
        if change_sample:
            self.next_sample()
        else:
            if self.seed_init_weights is None: self._define_init_weights()
            self._reset_edge_attributes()
        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()
        self.reward_measure = self.compute_reward_measure()    
        self.set_target_measure()


    def generate_graph(self):
        G  = copy.deepcopy(self.topology_object)
        self.n_nodes = G.number_of_nodes()
        self.n_links = G.number_of_edges()
        self._define_init_weights()
        idx = 0
        link_ids_dict = {}
        for (i,j) in G.edges():
            G[i][j]['id'] = idx
            G[i][j]['increments'] = 1
            G[i][j]['reductions'] = 1
            G[i][j]['weight'] = copy.deepcopy(self.init_weights[idx])
            link_ids_dict[idx] = (i,j)
            G[i][j]['capacity'] = G[i][j]['bandwidth']
            G[i][j]['traffic'] = 0.0
            idx += 1
        self.G = G
        incoming_links, outcoming_links = self._generate_link_indices_and_adjacencies()
        self.G.add_node('graph_data', link_ids_dict=link_ids_dict, incoming_links=incoming_links, outcoming_links=outcoming_links)


    def set_target_measure(self):
        self.target_sp_routing = copy.deepcopy(self.sp_routing)
        self.target_reward_measure = copy.deepcopy(self.reward_measure)
        self.target_link_traffic = copy.deepcopy(self.link_traffic)
        self.get_weights()
        self.target_weights = copy.deepcopy(self.raw_weights)


    def get_weights(self, normalize=True):
        weights = [0.0]*self.n_links
        for i,j in self.G.edges():
            weights[self.G[i][j]['id']] = copy.deepcopy(self.G[i][j]['weight'])
        self.raw_weights = weights
        max_weight = self.max_weight*3
        self.weights = [weight/max_weight for weight in weights]

    def get_state(self):
        state = []
        link_traffic = copy.deepcopy(self.link_traffic)
        weights = copy.deepcopy(self.weights)
        if self.link_traffic: 
            state += link_traffic
        if self.weigths_to_states: 
            state += weights
        if self.probs_to_states:
            state += self.p_in
            state += self.p_out
        return np.array(state, dtype=np.float32)

    def define_weight(self, link, weight):
        i, j = link
        self.G[i][j]['weight'] = weight
        self._generate_routing()
        self._get_link_traffic()
        
    def update_weights(self, link, action_value, step_back=False):
        i, j = link
        if self.weight_update == 'min_max':
            if action_value == 0:
                self.G[i][j]['weight'] = max(self.G[i][j]['weight']-self.weight_change, self.min_weight)
            elif action_value == 1:
                self.G[i][j]['weight'] = min(self.G[i][j]['weight']+self.weight_change, self.max_weight)
        else: 
            if self.weight_update == 'increment_reduction':
                if action_value == 0:
                    self.G[i][j]['reductions'] += 1
                elif action_value == 1:
                    self.G[i][j]['increments'] += 1
                self.G[i][j]['weight'] = self.G[i][j]['increments'] / self.G[i][j]['reductions']
            elif self.weight_update == 'sum':
                if step_back:
                    self.G[i][j]['weight'] -= self.weight_change
                else:    
                    self.G[i][j]['weight'] += self.weight_change
            
    def reinitialize_weights(self, seed_init_weights=-1, min_weight=None, max_weight=None):
        if seed_init_weights != -1: 
            self.seed_init_weights = seed_init_weights
        if min_weight: self.min_weight = min_weight
        if max_weight: self.max_weight = max_weight

        self.generate_graph()
        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()

    def reinitialize_routing(self, routing):
        self.routing = routing
        self._get_link_traffic()

    def step(self, action, step_back=False):
        #link_id, action_value = action
        link = self.G.nodes()['graph_data']['link_ids_dict'][action]
        #self.update_weights(link, action_value, step_back)
        self.update_weights(link, 0, step_back)
        self.get_weights()
        self._generate_routing()
        self._get_link_traffic()
        state = self.get_state()
        reward = self._compute_reward()
        return state, reward

    def step_back(self, action):
        state, reward = self.step(action, step_back=True)
        return state, reward


    # in the q_function we want to use info on the complete path (src_node, next_hop, n3, n4, ..., dst_node)
    # this function returns the indices of links in the path
    def get_complete_link_path(self, node_path):
        link_path = []
        for i, j in pairwise_iteration(node_path):
            link_path.append(self.G[i][j]['id'])
        # pad the path until "max_length" (implementation is easier if all paths have same size)
        link_path = link_path + ([-1] * (self.n_links-len(link_path)))
        return link_path



    """
    ****************************************************************************
                 PRIVATE FUNCTIONS OF THE ENVIRONMENT CLASS
    ****************************************************************************
    """
    
    def _define_init_weights(self):
        np.random.seed(seed=self.seed_init_weights)
        self.init_weights = np.random.randint(self.min_weight,self.max_weight+1,self.n_links)
        np.random.seed(seed=None)
        

    # generates indices for links in the network
    def _generate_link_indices_and_adjacencies(self):
        # for the q_function, we want to have info on link-link connection points
        # there is a link-link connection between link A and link B if link A
        # is an incoming link of node C and link B is an outcoming node of node C.
        # For connection "i", the incoming link is incoming_links[i] and the
        # outcoming link is outcoming_links[i]
        incoming_links = []
        outcoming_links = []
        # iterate through all links
        for i in self.G.nodes():
            for j in self.G.neighbors(i):
                incoming_link_id = self.G[i][j]['id']
                # for each link, search its outcoming links
                for k in self.G.neighbors(j):
                    outcoming_link_id = self.G[j][k]['id']
                    incoming_links.append(incoming_link_id)
                    outcoming_links.append(outcoming_link_id)

        return incoming_links, outcoming_links

    def _reset_edge_attributes(self, attributes=None):
        if attributes is None:
            attributes = list(DEFAULT_EDGE_ATTRIBUTES.keys())
        if type(attributes) != list: attributes = [attributes]
        for (i,j) in self.G.edges():
            for attribute in attributes:
                if attribute == 'weight':
                    self.G[i][j][attribute] = copy.deepcopy(self.init_weights[self.G[i][j]['id']])
                else:
                    self.G[i][j][attribute] = copy.deepcopy(DEFAULT_EDGE_ATTRIBUTES[attribute])

    def _normalize_traffic(self):
        for (i,j) in self.G.edges():
            self.G[i][j]['traffic'] /= self.G[i][j]['capacity']

    def _generate_routing(self, next_hop=None):
        self.sp_routing = dict(nx.all_pairs_dijkstra_path(self.G))
        #self.path_lengths = dict(nx.all_pairs_dijkstra_path_length(self.G))


    def successive_equal_cost_multipaths(self, src, dst, traffic):
        new_srcs = self.next_hop_dict[src][dst]
        traffic /= len(new_srcs)
        for new_src in new_srcs:
            self.G[src][new_src]['traffic'] += traffic
            if new_src != dst:
                self.successive_equal_cost_multipaths(new_src, dst, traffic)


    # returns a list of traffic volumes of each link
    def _distribute_link_traffic(self, routing=None):
        self._reset_edge_attributes('traffic')
        if self.routing == 'sp':
            if routing is None: routing = self.sp_routing
            for i in self.G.nodes():
                if i=='graph_data': continue
                for j in self.G.nodes():
                    if j=='graph_data' or i == j: continue
                    traffic = self.traffic_demand[i][j]
                    for u,v in pairwise_iteration(routing[i][j]):
                        self.G[u][v]['traffic'] += traffic
        elif self.routing == 'ecmp':
            visited_pairs = set()
            self.next_hop_dict = {i : {j : set() for j in range(self.G.number_of_nodes()-1) if j != i} for i in range(self.G.number_of_nodes()-1)}
            for src in range(self.G.number_of_nodes()-1):
                for dst in range(self.G.number_of_nodes()-1):
                    if src == dst: continue
                    if (src,dst) not in visited_pairs:
                        routings = set([item for sublist in [[(routing[i],routing[i+1]) for i in range(len(routing)-1)] for routing in nx.all_shortest_paths(self.G, src, dst, 'weight')] for item in sublist])
                        for (new_src,next_hop) in routings:
                            self.next_hop_dict[new_src][dst].add(next_hop)
                            visited_pairs.add((new_src,dst))
                    traffic = self.traffic_demand[src][dst]
                    self.successive_equal_cost_multipaths(src, dst, traffic)
        
        self._normalize_traffic()

    def _get_link_traffic(self, routing=None):
        self._distribute_link_traffic(routing)
        link_traffic = [0]*self.n_links
        for i,j in self.G.edges():
            link_traffic[self.G[i][j]['id']] = self.G[i][j]['traffic']
        self.link_traffic = link_traffic
        self.mean_traffic = np.mean(link_traffic)
        self.get_weights()

    def get_link_traffic(self):
        link_traffic = [0]*self.n_links
        for i,j in self.G.edges():
            link_traffic[self.G[i][j]['id']] = self.G[i][j]['traffic']
        return link_traffic

    def get_link_probs(self):
        traffic_in = np.sum(self.traffic_demand, axis=0)
        traffic_out = np.sum(self.traffic_demand, axis=1)
        node_p_in = list(traffic_in / np.sum(traffic_in))
        node_p_out = list(traffic_out / np.sum(traffic_out))
        self.p_in = [0]*self.n_links
        self.p_out = [0]*self.n_links
        for i,j in self.G.edges():
            self.p_in[self.G[i][j]['id']] = node_p_out[i]
            self.p_out[self.G[i][j]['id']] = node_p_in[j]

    # reward function is currently quite simple
    def compute_reward_measure(self, measure=None):
        if measure is None:
            if self.reward_magnitude == 'link_traffic':
                measure = self.link_traffic
            elif self.reward_magnitude == 'weights':
                measure = self.raw_weights
        
        if self.base_reward == 'mean_times_std':
            return np.mean(measure) * np.std(measure)
        elif self.base_reward == 'mean':
            return np.mean(measure)
        elif self.base_reward == 'std':
            return np.std(measure)
        elif self.base_reward == 'diff_min_max':
            return np.max(measure) - np.min(measure)
        elif self.base_reward == 'min_max':
            return np.max(measure)

    def _compute_reward(self, current_reward_measure=None):
        if current_reward_measure is None:
            current_reward_measure = self.compute_reward_measure()
        
        if self.reward_computation == 'value':
            reward = - current_reward_measure
        elif self.reward_computation == 'change':
            reward = self.reward_measure - current_reward_measure

        self.reward_measure = current_reward_measure
        
        return reward
