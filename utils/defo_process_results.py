#!/usr/bin/python3

import numpy as np
import re
import sys
sys.path.append('../')

node_to_index_dic = {}
index_to_node_lst = []

def index_to_node(n):
    return(index_to_node_lst[n])

def node_to_index(node):
    return(node_to_index_dic[node])


class Defo_results:
    
    net_size = 0
    MP_matrix = None
    ecmp_routing_matrix = None
    routing_matrix = None
    links_bw = None
    
    def __init__(self, graph_file, results_file):
        self.graph_file = graph_file
        self.results_file = results_file
        
        self.process_graph_file()
        
        self.process()
    
    def process_graph_file(self):
        with open(self.graph_file) as fd:
            line = fd.readline()
            camps = line.split(" ")
            self.net_size = int(camps[1])
            # Remove : label x y
            line = fd.readline()
            
            for i in range (self.net_size):
                line = fd.readline()
                node = line[0:line.find(" ")]
                node_to_index_dic[node] = i
                index_to_node_lst.append(node)
                
            self.links_bw = []
            for i in range(self.net_size):
                self.links_bw.append({})
            for line in fd:
                if (not line.startswith("Link_") and not line.startswith("edge_")):
                    continue
                camps = line.split(" ")
                src = int(camps[1])
                dst = int(camps[2])
                bw = float(camps[4])
                self.links_bw[src][dst] = bw
                
 
            
    
    def process (self):
        with open(self.results_file) as fd:
            while (True):
                line = fd.readline()
                if (line == ""):
                    break
                if (line.startswith("*")):
                    if (line == "***Next hops priority 2 (sr paths)***\n"):
                        self._read_middle_points(fd)
                    if (line == "***Next hops priority 3 (ecmp paths)***\n"):
                        self._read_ecmp_routing(fd)
                        break
        self._gen_routing_matrix()

    def _read_middle_points(self,fd):
        self.MP_matrix = np.zeros((self.net_size,self.net_size),dtype="object")
        while (True):
            pos = fd.tell()
            line = fd.readline()
            #print (line)
            if (line.startswith("*")):
                fd.seek(pos)
                return
            if (not line.startswith("seq")):
                continue
            line = line[line.find(": ")+2:]
            if (line[-1]=='\n'):
                line = line[:-1]
            
            ptr = 0
            mp_path = []
            while (True):
                prev_ptr = ptr
                ptr = line.find(" -> ",ptr)
                if (ptr == -1):
                    mp_path.append(line[prev_ptr:])
                    break
                else:
                    mp_path.append(line[prev_ptr:ptr])
                    ptr += 4
            src = node_to_index(mp_path[0])
            dst = node_to_index(mp_path[-1])
            self.MP_matrix[src,dst] = mp_path
        
    
    def _read_ecmp_routing(self,fd):
        self.ecmp_routing_matrix = np.zeros((self.net_size,self.net_size),dtype="object")
        next_node_matrix = np.zeros((self.net_size,self.net_size),dtype="object")
        dst_node = None
        while (True):
            line = fd.readline()
            if (line == ""):
                break
            if (line.startswith("Destination")):
                dst_node_str = line[line.find(" ")+1:-1]
                dst_node = node_to_index(dst_node_str)
            if (line.startswith("node")):
                src_node_str = line[6:line.find(", ")]
                src_node = node_to_index(src_node_str)
                sub_line = line[line.find("[")+1:line.find("]")]
                ptr = 0
                next_node_lst = []
                while (True):
                    prev_ptr = ptr
                    ptr = sub_line.find(", ",ptr)
                    if (ptr == -1):
                        next_node_lst.append(sub_line[prev_ptr:])
                        break
                    else:
                        next_node_lst.append(sub_line[prev_ptr:ptr])
                        ptr += 2

                next_node_matrix[src_node,dst_node] = next_node_lst

        for i in range (self.net_size):
            for j in range (self.net_size):
                end_paths = []
                paths_info = [{"path":[index_to_node(i)],"proportion":1.0}]
                while (len(paths_info) != 0):
                    for path_info in paths_info:
                        path = path_info["path"]
                        if (node_to_index(path[-1]) == j):
                            paths_info.remove(path_info)
                            end_paths.append(path_info)
                            continue
                        next_lst = next_node_matrix[node_to_index(path[-1]),j]
                        num_next_hops = len(next_lst)
                        if (num_next_hops > 1):
                            for next_node in next_lst:
                                new_path = list(path)
                                new_path.append(next_node)
                                paths_info.append({"path":new_path,"proportion":path_info["proportion"]/num_next_hops})
                            paths_info.remove(path_info)
                        else:
                            path.append(next_lst[0])
                self.ecmp_routing_matrix[i,j] = end_paths
        
    def _gen_routing_matrix(self):
        self.routing_matrix = np.zeros((self.net_size,self.net_size),dtype="object")
        for i in range(self.net_size):
            for j in range(self.net_size):
                if (i == j):
                    continue
                #print(self.MP_matrix)
                end_path_info_list = []
                mp_path = self.MP_matrix[i,j]
                #print (i,j,mp_path)
                if type(mp_path) is not list:
                    continue
                src_mp = mp_path[0]
                for mp in mp_path:
                    dst_mp = mp
                    sub_path_info_lst =  self.ecmp_routing_matrix[node_to_index(src_mp),node_to_index(dst_mp)]
                    if (len(end_path_info_list) == 0):
                        for sub_path_info in sub_path_info_lst:
                            end_path_info_list.append({"path":sub_path_info["path"][:-1],"proportion":sub_path_info["proportion"]})
                    elif (len(sub_path_info_lst) > 1):
                        aux_end_path_list = []
                        for path_info in end_path_info_list:
                            for sub_path_info in sub_path_info_lst:
                                new_path = list(path_info["path"])
                                new_path.extend(sub_path_info["path"][:-1])
                                aux_end_path_list.append({"path":new_path,"proportion":path_info["proportion"]*sub_path_info["proportion"]})
                        end_path_info_list = aux_end_path_list
                    else:
                        for path_info in end_path_info_list:
                            path_info["path"].extend(sub_path_info_lst[0]["path"][:-1])
                    src_mp = dst_mp
                for path_info in end_path_info_list:
                    path_info["path"].append(dst_mp)
                self.routing_matrix[i,j] = end_path_info_list
    
    def _get_traffic_matrix (self,traffic_file):
        tm = np.zeros((self.net_size,self.net_size))
        with open(traffic_file) as fd:
            fd.readline()
            fd.readline()
            for line in fd:
                camps = line.split(" ")
                tm[int(camps[1]),int(camps[2])] = float(camps[3])
        return (tm)
    
    def _link_utilization(self, routing_matrix, traffic_file):
        link_utilization = []
        traffic_matrix = self._get_traffic_matrix(traffic_file)
        for i in range(self.net_size):
            link_utilization.append({})
        for i in range(self.net_size):
            for j in range (self.net_size):
                if (i==j):
                    continue
                traffic_all_path = traffic_matrix[i,j]
                routings_lst = routing_matrix[i,j]
                if routings_lst == 0:
                    continue
                for path_info in routings_lst:
                    path = path_info["path"]
                    traffic = traffic_all_path*path_info["proportion"]
                    n0 = path[0]
                    for n1 in path[1:]:
                        N0 = node_to_index(n0)
                        N1 = node_to_index(n1)
                        if N1 in link_utilization[N0]:
                            link_utilization[N0][N1] += traffic
                        else:
                            link_utilization[N0][N1] = traffic
                        n0 = n1
        max_lu = 0
        
        for i in range(self.net_size):
            for j in link_utilization[i].keys():
                link_traffic = link_utilization[i][j]
                link_capacity = self.links_bw[i][j]
                link_utilization[i][j] = link_traffic / link_capacity
                if (link_utilization[i][j] > max_lu):
                    max_lu = link_utilization[i][j]
        #print ("max link utilization :",max_lu)
        return (link_utilization)
    
    def get_opt_link_utilization(self,traffic_file):
        return (self._link_utilization(self.routing_matrix,traffic_file))
    
    def get_direct_link_utilization(self,traffic_file):
        return (self._link_utilization(self.ecmp_routing_matrix,traffic_file))


def get_traffic_matrix(tm_file, net_size=14):
    tm = np.zeros((net_size,net_size))
    with open(tm_file) as fd:
        fd.readline()
        fd.readline()
        for line in fd:
            camps = line.split(" ")
            tm[int(camps[1]),int(camps[2])] = float(camps[3])
    return (tm)

def get_link_utilization(results):
    return [item for sublist in [[value for value in results[i].values()] for i in range(len(results))] for item in sublist]


if (__name__ == "__main__"):
    
    network = 'NSFNet'
    defo_dir = "./datasets/defo_" + network + '_dataset/'
    init_seed = 100
    final_seed = 199
    
    ecmp, defo = [], []
    graph_file = defo_dir + network + '_defo.txt'
    for i in range(init_seed, final_seed+1):
        results_file =  defo_dir +'Results-' + str(i)
        tm_file =  defo_dir + 'TM-' + str(i)
        results = Defo_results(graph_file,results_file)
        results_ecmp = results.get_direct_link_utilization(tm_file)
        lu_ecmp = get_link_utilization(results_ecmp)
        results_defo = results.get_opt_link_utilization(tm_file)
        lu_defo = get_link_utilization(results_defo)
        ecmp.append(np.max(lu_ecmp))
        defo.append(np.max(lu_defo))
    
    #print ("============== Direct =====================")
    #print (results.get_direct_link_utilization(tm_file))
    #print ("============== Optim =====================")
    #print (results.get_opt_link_utilization(tm_file))
