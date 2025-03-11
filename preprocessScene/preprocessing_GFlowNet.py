import numpy as np
import matplotlib.pyplot as plt
# from utils.utils import *
# from utils import *
from WINeRT_Dataset import WINeRT_Dataset
import multiprocessing

import torch
from get_scene import *
from tqdm import tqdm
import pandas as pd
import os
# from torch_geometric.data import Data, Batch
import os.path as osp
import argparse
from datetime import datetime

### Prepare surface index ###
def parse_SegMap_PointMap(segments_map, point_map):
    vertices = []
    vertex_to_index = {}

    for seg in segments_map.values():
        if seg['type'] == 'WALL':
            for z in seg['z']:
                for v_key in seg['connect']:
                    coord = point_map[v_key] + [z]
                    point = tuple(coord)
                    if point not in vertex_to_index:
                        vertex_to_index[point] = len(vertices)
                        vertices.append(point)
        else:
            for v_key in seg['connect']:
                coord = point_map[v_key] + [seg['z'][0]]
                
                point = tuple(coord)
                if point not in vertex_to_index:
                    vertex_to_index[point] = len(vertices)
                    vertices.append(point)

    # 2. Map segment endpoints to unique indices
    faces_rect = []
    for id in range(len(segments_map)):
        seg_id = id + 1 # segment ids start from 1
        if segments_map[seg_id]['type'] == 'WALL':
            start, end = segments_map[seg_id]['connect']
            idx_start_top = vertex_to_index[tuple(point_map[start] + [segments_map[seg_id]['z'][1]])]
            idx_start_bottom = vertex_to_index[tuple(point_map[start] + [segments_map[seg_id]['z'][0]])]
            idx_end_top = vertex_to_index[tuple(point_map[end] + [segments_map[seg_id]['z'][1]])]
            idx_end_bottom = vertex_to_index[tuple(point_map[end] + [segments_map[seg_id]['z'][0]])]
            # Representing line segments with degenerate triangles
            faces_rect.append((idx_start_top, idx_start_bottom, idx_end_top, idx_end_bottom))
        else:
            start1, start2, end1, end2 = segments_map[seg_id]['connect']
            idx_start1_top = vertex_to_index[tuple(point_map[start1] + [segments_map[seg_id]['z'][0]])]
            idx_start2_top = vertex_to_index[tuple(point_map[start2] + [segments_map[seg_id]['z'][0]])]
            idx_end1_top = vertex_to_index[tuple(point_map[end1] + [segments_map[seg_id]['z'][0]])]
            idx_end2_top = vertex_to_index[tuple(point_map[end2] + [segments_map[seg_id]['z'][0]])]
            faces_rect.append((idx_start1_top, idx_start2_top, idx_end1_top, idx_end2_top))
    return torch.tensor(vertices), torch.tensor(faces_rect)

def edge_vector(v1, v2):
    """ Compute the vector from v1 to v2 """
    return v2 - v1

def compute_plane_normal(v1, v2, v3):
    """ Compute the normal vector of the plane defined by v1, v2, and v3 """
    a = v2 - v1
    b = v3 - v1
    normal = torch.cross(a, b)
    return normal / torch.norm(normal)

def is_point_in_projection(point, rect_vertices):
    """ Check if the point is within the projected bounds of the rectangle """
    ab = edge_vector(rect_vertices[0], rect_vertices[1])
    ad = edge_vector(rect_vertices[0], rect_vertices[2])
    # ipdb.set_trace()
    ap = point - rect_vertices[0]
    ap = ap.type(torch.FloatTensor)
    is_in_ab = 0 <= torch.dot(ap, ab) <= torch.dot(ab, ab)
    is_in_ad = 0 <= torch.dot(ap, ad) <= torch.dot(ad, ad)
    return is_in_ab and is_in_ad

def point_to_line_distance(p, a, b):
    """ Compute the distance from point p to line segment ab """
    ap = (p - a).type(torch.FloatTensor)
    ab = b - a
    t = torch.dot(ap, ab) / torch.dot(ab, ab)
    t = torch.clamp(t, 0, 1)
    closest = a + t * ab
    return torch.norm(p - closest)

def orthogonal_projection_to_plane(point, plane_point, normal):
    """ Calculate the orthogonal projection of a point onto a plane """
    v = point - plane_point
    v = v.type(torch.FloatTensor)
    distance = torch.dot(v, normal)
    projection = point - distance * normal
    return projection, torch.abs(distance)

def distance_to_rectangle(point, rect_vertices):
    normal = compute_plane_normal(rect_vertices[0], rect_vertices[1], rect_vertices[2])
    if is_point_in_projection(point, rect_vertices):
        _, distance = orthogonal_projection_to_plane(point, rect_vertices[0], normal)
        return distance
    else:
        """ Calculate the distance from the point to the nearest edge or corner of the rectangle """
        distances = [
            point_to_line_distance(point, rect_vertices[0], rect_vertices[1]),
            point_to_line_distance(point, rect_vertices[0], rect_vertices[2]),
            point_to_line_distance(point, rect_vertices[1], rect_vertices[3]),
            point_to_line_distance(point, rect_vertices[2], rect_vertices[3])
        ]
        return torch.min(torch.stack(distances))

def classify_points(path_points, surfaces, vertices, debug=False):
    """ Classify each point to the nearest rectangle surface """
    labels = []
    assert len(path_points.shape) == 3, "len(points.shape) != 3"
    assert path_points.shape[2] == 3, "points.shape[2] != 3"
    path_num = path_points.shape[0]
    hop_num = path_points.shape[1]
    labels = -1 * torch.ones(path_num, hop_num, 1)
    min_distance_tensor = -1 * torch.ones(path_num, hop_num, 1)
    for i in range(path_num):
        for j in range(hop_num):
            point = path_points[i,j,:]
            if j == 0:
                labels[i,j,0] = -1 # the first point is always the Tx
            elif torch.equal(point, torch.tensor([-1., -1., -1.])):
                labels[i,j,0] = -3 # the point is not valid
                labels[i,j-1,0] = -2 # the previous point is the Rx
                if min_distance_tensor[i,j-1,0] > 0.1 and labels[i,j-1,0] != -1 and labels[i,j-1,0] != -2:
                    print("WARNIGN: point {} is not close to any surface, min_distance {}, closest_surface {}".format(path_points[i,j-1,:],  min_distance_tensor[i,j-1,0], labels[i,j-1,0]))
            else:
                min_distance = float('inf')
                closest_surface = -1
                closest_surface_vertices = None
                for k, surface_indices in enumerate(surfaces):
                    rect_vertices = vertices[surface_indices]
                    distance = distance_to_rectangle(point, rect_vertices)
                    if distance < min_distance:
                        min_distance = distance
                        closest_surface = k
                        closest_surface_vertices = rect_vertices
                labels[i,j,0] = closest_surface
                min_distance_tensor[i,j,0] = min_distance

                if debug:
                    print("Point: {}, Closest surface: {}, closest_surface_vertices {}, min_distance {}".format(point, closest_surface, closest_surface_vertices, min_distance))
    return labels

def prepare_suface_index(lay_data_path = "/proj/gaia/RayDT/dataset/processed_data/train/objs/1.obj", surface_index_path = "/proj/gaia/RayDT/dataset/processed_data/train/objs/"):
    # Phase 1: get surface index
    lay_data = None        
    # get obj id
    obj_id = lay_data_path.split('/')[-1].split('.')[0]
    
    surface_index_path = surface_index_path  + "/"+ obj_id+"_surface_index.pt"
    faces_rect_path = surface_index_path.replace("surface_index.pt", "faces_rect_path.pt")
    vertices_path = surface_index_path.replace("surface_index.pt", "vertices_path.pt")
    
    assert surface_index_path.split('/')[-1] == obj_id+"_surface_index.pt", "surface_index_path is not correct {}".format(surface_index_path)
    assert faces_rect_path.split('/')[-1] == obj_id+"_faces_rect_path.pt", "faces_rect_path is not correct {}".format(faces_rect_path)
    
    if os.path.exists(surface_index_path):
        surface_index = torch.load(surface_index_path)
        faces_rect = torch.load(faces_rect_path)
        vertices = torch.load(vertices_path)
    else:
        with open(lay_data_path, 'r') as file:
            lay_data = file.read()
            point_map = extract_points_from_lay(lay_data)
            segments_map = extract_segments_from_lay(lay_data,point_map)
            # print ("point_map", point_map)
            vertices, faces_rect = parse_SegMap_PointMap(segments_map, point_map)
            surface_index = torch.tensor(np.arange(len(faces_rect)))
            torch.save(surface_index, surface_index_path)
            torch.save(faces_rect, faces_rect_path)
            torch.save(vertices, vertices_path)
    return surface_index, faces_rect, vertices

    
def count_graph_stats(intereaction_tensor, radius):
    num_paths = intereaction_tensor.shape[4] 
    num_node_per_path = intereaction_tensor.shape[5]
    num_total_nodes = num_node_per_path* (2 * radius + 1)**3
    num_total_edges = num_node_per_path - 1
    return num_paths, num_node_per_path, num_total_nodes, num_total_edges

def get_obj_lay_path_list(data_dir):
    # get all lay file path
    lay_path_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".lay"):
                lay_path_list.append(os.path.join(root, file))
    # sort by name. 1,2,3,4, NOT 1,10,11,12
    lay_path_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return lay_path_list

def per_Tx_Rx_process_cuda_optim(args):
    # target.shape[1], target.shape[3], input, target, radius, i, storage_dir,  obj_list[i], get_graph_data
    T, R, x, intereaction_tensor, _, i, storage_dir, obj_dir, _ = args
    obj_id_bug_list = [21,34,55,59,72,78,92,121,124] # these obj_id has some problem, skip
    # obj_dir = /proj/raygnn/workspace/raytracingdata/testenv/objs/1.obj, obj_id = 1
    obj_id = obj_dir.split('/')[-1].split('.')[0]
    if int(obj_id) in obj_id_bug_list:
        print ("obj_id {} is in obj_id_bug_list, skip".format(obj_id))
        return
    else:
        pass
    ### Bootstrapping, since all graph are the same stucture, lets take it outside the loop ###
    ### 1. prepare node_attr_tensor, edge_attr_tensor ### 
    # intereaction_tensor = torch.tensor(intereaction_tensor).to('cpu')
    num_paths, num_node_per_path, num_total_nodes, _ = count_graph_stats(intereaction_tensor, 0)
    ### 2,  prepare origin edge_index ###
    _, faces_rect, vertices = prepare_suface_index(obj_dir)
    faces_rect = faces_rect.to('cpu')
    vertices = vertices.to('cpu')
    ### 3,  prepare surrounding edge_index ###
    node_attr_tensor = torch.zeros(int(T),int(R), int(num_paths), int(num_node_per_path), 7) # 6 dim for each node + 1 dim for surface index
    
    gt = x['channels'][0, :, 0, :, :, :].type(torch.float16).to('cpu')
    os.makedirs(storage_dir, exist_ok=True)
    torch.save(gt, os.path.join(storage_dir, f'gt_obj_{obj_id}.pt'))
    Tx_Rx_validaty = gt[:,:,7,:]
    del gt
    
    for Tx in range(T): # rahge(1)
        # 4 dim for each edge: validaty, distance, radians, azimuth
        for Rx in range(R): # range(10)
            #### Phase1: Node handling ####
            interType_points = intereaction_tensor[0, Tx, 0, Rx, :, :]
            # 1.1 acquire texture label
            interType_points_coord = interType_points[:, :, 1:4]
            # get surface label
            surface_label_tensor = classify_points (interType_points_coord, faces_rect, vertices)
            surface_label_tensor = surface_label_tensor.type(torch.int).to('cpu')
            # TODO: some function to check if such point is on some 2d plane in 3d space
            mask = (interType_points[:, :, 3] == 0) | (interType_points[:, :, 3] == 3)
            # All texture labels are 0: Bricks
            texture_label = torch.zeros_like(mask, dtype=torch.bool)
            # Where the mask is True, set to 1: Concrete
            texture_label[mask] = 1
            texture_label = texture_label.unsqueeze(-1)
            texture_label = texture_label.type(torch.float16)
            # Concatenate along the last dimension to append the values
            node_w_texture = torch.cat((interType_points, texture_label), dim=2)
            # 1.2 acquire path validaty
            D = Tx_Rx_validaty[Tx, Rx, :].unsqueeze(1)
            D = D.type(torch.float16).to('cpu')
            # Extract the validity of each path
            path_validaty = D.unsqueeze(1).repeat(1, 6, 1)
            
            # interaction_type, x, y, z, texture_label, path_validaty surface_label
            node_w_texture_w_D = torch.cat([node_w_texture, path_validaty], dim = 2)
            node_w_texture_w_D_w_surface = torch.cat([node_w_texture_w_D, surface_label_tensor], dim = 2)
            node_attr_tensor[Tx , Rx, :, :, :] = node_w_texture_w_D_w_surface

    # Save each Data object
    save_path = os.path.join(storage_dir, f'obj_{obj_id}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert node_attr_tensor.shape == (int(T), int(R), int(num_paths), int(num_total_nodes), 7), "Shape of node_attr_tensor is not correct {}".format(node_attr_tensor.shape)
    torch.save(node_attr_tensor, os.path.join(storage_dir, f'node_attr_tensor_{obj_id}_{i}.pt'))


# Define the function that creates argument tuples for each task
def prepare_args(train_loader, radius, storage_dir,data_dir, debug = False, get_graph_data = False):
    obj_dir = data_dir + "objs"
    obj_list = get_obj_lay_path_list(obj_dir)
    if not debug:
        return [(target.shape[1], target.shape[3], input, target, radius, i, storage_dir,  obj_list[i], get_graph_data) for i, (input, target) in enumerate(train_loader)]
    else:
        results = []
        for i, (input, target) in enumerate(train_loader):
            results.append((target.shape[1], target.shape[3], input, target, radius, i, storage_dir, obj_list[i], get_graph_data))
            if i == 99:
                break
        return results
    
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--radius', type=int, default=1)
    argparser.add_argument('--num_workers', type=int, default=0)
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--data_dir', type=str, default='/proj/gaia/RayDT/dataset/raw_data/wi3rooms/')
    argparser.add_argument('--storage_dir', type=str, default='/proj/gaia/RayDT/dataset/processed_data/train')
    argparser.add_argument('--get_graph_data',  action='store_true')
    argparser.add_argument('--testset',  type=str, default='None') # 'None' or 'test', 'genz', 'gendiag'
    radius = argparser.parse_args().radius
    debug = argparser.parse_args().debug
    num_workers = argparser.parse_args().num_workers
    data_dir = argparser.parse_args().data_dir
    testset_type = argparser.parse_args().testset
    if testset_type == "None":
        num_proc = 8
    else:
        num_proc = 4
    if debug:
        storage_dir = "/proj/gaia/RayDT/dataset/processed_data/train"
        type_winertloader = "train"
    elif testset_type != "None":
        train_storage_dir = argparser.parse_args().storage_dir
        storage_dir = train_storage_dir.replace("train", testset_type)
        type_winertloader = testset_type
    else:
        storage_dir = argparser.parse_args().storage_dir
        type_winertloader = "train"
    
    
    # Load your train_loader or whatever setup you have
    train_dataset = WINeRT_Dataset(data_dir, type = type_winertloader)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=1
    )
    
    # Prepare arguments for multiprocessing
    args_train = prepare_args(train_loader,
                              radius, 
                              storage_dir,
                              data_dir,
                              debug = debug, 
                              get_graph_data=False)
    
    if debug:
        time_now = datetime.now()
    with multiprocessing.Pool(processes=num_proc) as pool:
        for _ in tqdm(pool.imap_unordered(per_Tx_Rx_process_cuda_optim, args_train), total=len(args_train)):
            pass
    # for arg in tqdm(args_train):
    #     per_Tx_Rx_process_cuda_optim(arg)
    if debug:
        time_end = datetime.now()
        time_spent = (time_end - time_now).total_seconds()  
        print ("Time spent: ", time_spent)
if __name__ == '__main__':
    main()