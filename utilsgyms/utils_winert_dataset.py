import h5py
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import torch

def get_h5py_dataset(h5py_path):
    f = h5py.File(h5py_path, 'r')
    return f

def destroy_h5py_dataset(h5py_dataset):
    h5py_dataset.close()

def get_unique_floor_idx(dir_path = '../raytracingdata/wi3rooms/' ):
    h5_data_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(".h5")]
    all_unique_floor_idx = set()
    for path in h5_data_paths:
        f = h5py.File(path, 'r')
        # get all unique value of np.unique(f['floor_idx'][:])
        floor_idx = f['floor_idx'][:]
        all_unique_floor_idx.update(np.unique(floor_idx))
        f.close()
    return all_unique_floor_idx

def get_surronding_coord(coord, radius=3):
    x, y, z = coord
    surrounding_coord = []

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            for k in range(-radius, radius + 1):
                surrounding_coord.append([x + i * 0.1, y + j * 0.1, z + k * 0.1])

    # Use list comprehension to remove the center coord
    surrounding_coord = [c for c in surrounding_coord if c != list(coord)]

    return surrounding_coord
def get_surronding_coord_tensor(interType_points, radius=3):
    # Input: interType_points: (L,N, 4)
    # L: number of paths
    # N: number of points in each path
    # 4: 0: interType, 1: x, 2: y, 3: z
    # Output: surronding_coord_tensor: (L,N, 4, radius^3)
    offsets = np.arange(-radius, radius + 1) * 0.1
    offset_grid = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1, 3)
    center_mask = ~(np.isclose(offset_grid, 0)).all(axis=1)
    offset_grid = offset_grid[center_mask]

    # Extract interTypes and coordinates
    interTypes = interType_points[:, :, 0]
    coords = interType_points[:, :, 1:]  # Extract just the x,y,z coordinates

    # Use broadcasting to add offsets to coords
    # New shape will be L x N x (radius^3 - 1) x 3
    surr_coords = coords[:, :, np.newaxis, :] + offset_grid

    # Repeat the interTypes across the new axis and match the shape
    repeated_interTypes = np.repeat(interTypes[:, :, np.newaxis], surr_coords.shape[2], axis=2)

    # Combine interTypes with coords
    # New shape will be L x N x (radius^3 - 1) x 4
    surrounding_coord_tensor = np.concatenate((repeated_interTypes[..., np.newaxis], surr_coords), axis=-1)

    return surrounding_coord_tensor
    

def get_surronding_coord_new(coord, radius=3):
    x, y, z = coord
    surrounding_coord = []

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            for k in range(-radius, radius + 1):
                surrounding_coord.append([x + i * 0.1, y + j * 0.1, z + k * 0.1])

    # Use list comprehension to remove the center coord
    surrounding_coord = [c for c in surrounding_coord if c != list(coord)]
    return surrounding_coord

def get_graph_from_sample(x, intereaction_tensor, debug=False, sampling_mode = False):
    if debug:
        intereaction_tensor =  np.expand_dims(intereaction_tensor, axis = 0)
        for key in x.keys():
            x[key] = np.expand_dims(x[key], axis = 0)
    _,T,_,R,K,L,_ = intereaction_tensor.shape
    graph_dict = {}
    graph_dict_full_edges = {}
    graph_dict_addtional_nodes = {}
    if sampling_mode:
        T = 1
        R = 5
        print ("sampling mode")
        
    for Tx in range(T):
        for Rx in tqdm(range(R)):
            # digraph, graph with only ground truth edges
            digraph = nx.DiGraph()
            # digraph_full_edges, graph with all possible edges, on ground truth nodes
            digraph_full_edges = nx.Graph()
            # digraph_addtional_nodes, graph with all possible edges, on ground truth nodes and surronding nodes
            digraph_addtional_nodes = nx.DiGraph()
    
            # Use list so that we are faster
            # addtional_nodes_dict = {}
            additional_nodes_list = []
            

            
            
            for path in range(K):
                
                previous_point = None  # To store the previous point
                previous_surronding_point_list = [] # To store the previous surronding point
                
                points = intereaction_tensor[0, Tx, 0, Rx, path, :]
                # ‘channels’ (F, T, 1, R, D=8, K)
                assert len(points.shape) == 2, print("points shape is: ", points.shape)
                validaty = x['channels'][0, Tx, 0, Rx, 7 , path]
                if validaty == 1:
                    for point in points:
                        interaction_type = int(point[0].item())
                        coordinates = point[1:].tolist()
                        current_point = [interaction_type,coordinates]
                        digraph.add_node(current_point, coordinates = coordinates,  interaction=interaction_type, path=path)
                        # If there's a previous point, add an edge from the previous point to the current one
                        if previous_point and not debug:
                            digraph.add_edge(previous_point,current_point, path=path)
                        # Update the previous_point
                        previous_point = current_point
                        
                        # Add all possible edges to the graph
                        surronding_point_list = [ [interaction_type, x] for x in get_surronding_coord(coordinates)]
                        # addtional_nodes_dict[current_point] = surronding_point_list
                        additional_nodes_list.append(surronding_point_list)
                        for surronding_point in surronding_point_list :
                            digraph_addtional_nodes.add_node(surronding_point, coordinates = surronding_point[1],  interaction=interaction_type, path=path)
                        if len(previous_surronding_point_list) > 0:
                            for prev_node in previous_surronding_point_list:
                                for node in surronding_point_list:
                                    digraph_addtional_nodes.add_edge(prev_node,node, path=path)
                                    # print ("digraph_addtional_nodes add edge success")
                        previous_surronding_point_list = surronding_point_list
                                    
                # Store the graph in the dictionary
                else:
                    continue
            if digraph.number_of_nodes() > 0 and not debug:
                graph_dict[(Tx, Rx)] = digraph
                nodes_with_attributes = digraph.nodes(data=True)
                digraph_full_edges = nx.complete_graph(digraph.nodes())
                attrs_dict = {node: attrs for node, attrs in nodes_with_attributes}
                nx.set_node_attributes(digraph_full_edges, attrs_dict)
                graph_dict_full_edges[(Tx, Rx)] = digraph_full_edges
                
                
            if digraph_addtional_nodes.number_of_nodes() > 0:
                # for point in addtional_nodes_dict.keys():
                for point_idx in range(len(additional_nodes_list)):
                    # for another_point in addtional_nodes_dict.keys():
                    for another_point_idx in range(len(additional_nodes_list)):
                        # if point != another_point:
                        if point_idx != another_point_idx:
                            # for node1 in addtional_nodes_dict[point]:
                                # for node2 in addtional_nodes_dict[another_point]:
                            for node1 in additional_nodes_list[point_idx]:
                                for node2 in additional_nodes_list[another_point_idx]:
                                    digraph_addtional_nodes.add_edge(node1,node2)   
                                    
                graph_dict_addtional_nodes[(Tx, Rx)] = digraph_addtional_nodes
                if debug:
                    print ("good")
                    break
                
    return graph_dict, graph_dict_full_edges, graph_dict_addtional_nodes
        
def get_degree_centrality_distribution_average_clustering(graph_dict, debug=False):
    degree_dict = {}
    centrality_dict = {}
    # average_clustering_dict = {}
    keys = list(graph_dict.keys())
    
    for key in keys: 
        TxRx = [x for x,y in graph_dict[key].nodes(data=True) if y['interaction']==0]
        for tranceiver in TxRx: 
            degree_dict[key,tranceiver] = graph_dict[key].degree(tranceiver)
            centrality_dict[key,tranceiver] = list(nx.degree_centrality(graph_dict[key]).values())
        # average_clustering_dict[key] = nx.average_clustering(graph_dict[key])
        if debug:
            if len(centrality_dict.keys()) > 2:
                break
        else:
            if len(centrality_dict.keys()) > 5:
                break
        
    return degree_dict, centrality_dict, # average_clustering_dict


def update_fig_w_path(figure, plot_points, Tx_coord, Rx_coord, visualize=True, return_fig = False):
    # Assuming plot_points is a list of tuples containing (x, y, z) coordinates
    x_vals = [point[0] for point in plot_points]
    y_vals = [point[1] for point in plot_points]
    z_vals = [point[2] for point in plot_points]

    fig = go.Figure(figure)
    
    # Scatter plot for Tx and Rx, Tx is red, Rx is blue
    fig.add_trace(go.Scatter3d(x=[Tx_coord[0], Rx_coord[0]], 
                            y=[Tx_coord[1], Rx_coord[1]], 
                            z=[Tx_coord[2], Rx_coord[2]],
                            mode='markers',
                            marker=dict(size=5, color=['red', 'blue']),
                            text=['Tx', 'Rx'], # this is for hover text
                            hoverinfo='text'))
    

    # Scatter plot for nodes
    fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, 
                            mode='markers',
                            marker=dict(size=5, color='red',symbol=['triangle-up', 'triangle-down']),
                            text=ploting_text, # this is for hover text
                            hoverinfo='text'))

    # Line plot for paths
    # Here I am assuming consecutive points in plot_points should be connected
    for i in range(len(plot_points) - 1):
        fig.add_trace(go.Scatter3d(x=[plot_points[i][0], plot_points[i+1][0]],
                                y=[plot_points[i][1], plot_points[i+1][1]],
                                z=[plot_points[i][2], plot_points[i+1][2]],
                                mode='lines',
                                line=dict(color='blue', width=2)))

    # Adjusting layout if needed
    fig.update_layout(scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'))
    if visualize:
        fig.show()
    if return_fig:
        return fig
    


def get_edgelist_from_sample(x, interaction_tensor, debug=False, sampling_mode=False):
    pass