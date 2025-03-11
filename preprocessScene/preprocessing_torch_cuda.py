import numpy as np
import matplotlib.pyplot as plt

from WINeRT_Dataset import WINeRT_Dataset
import torch
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import multiprocessing
from get_scene import *


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
    ap = point - rect_vertices[0]
    ap = ap.type(torch.cuda.FloatTensor)
    is_in_ab = 0 <= torch.dot(ap, ab) <= torch.dot(ab, ab)
    is_in_ad = 0 <= torch.dot(ap, ad) <= torch.dot(ad, ad)
    return is_in_ab and is_in_ad

def point_to_line_distance(p, a, b):
    """ Compute the distance from point p to line segment ab """
    ap = (p - a).type(torch.cuda.FloatTensor)
    ab = b - a
    t = torch.dot(ap, ab) / torch.dot(ab, ab)
    t = torch.clamp(t, 0, 1)
    closest = a + t * ab
    return torch.norm(p - closest)

def orthogonal_projection_to_plane(point, plane_point, normal):
    """ Calculate the orthogonal projection of a point onto a plane """
    v = point - plane_point
    v = v.type(torch.cuda.FloatTensor)
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
                found_rx = False
                labels[i,j,0] = -1 # the first point is always the Tx
            elif torch.equal(point, torch.tensor([-1., -1., -1.]).to('cuda')):
                labels[i,j,0] = -3 # the point is not valid
                if not found_rx:      
                    labels[i,j-1,0] = -2 # the previous point is the Rx
                    found_rx = True
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
                    if debug:
                        print("Point: {}, surface: {}, distance {}, min_distance {}".format(point, k, distance, min_distance))
                labels[i,j,0] = closest_surface
                min_distance_tensor[i,j,0] = min_distance

                if debug:
                    print("Point: {}, Closest surface: {}, closest_surface_vertices {}, min_distance {}".format(point, closest_surface, closest_surface_vertices, min_distance))
    return labels.to('cuda')



def classify_points_vec(path_points, surfaces, vertices, debug=False):
    path_num, hop_num, _ = path_points.shape
    labels = -1 * torch.ones(path_num, hop_num, 1, dtype=torch.float32).to('cuda')
    min_distance_tensor = torch.full((path_num, hop_num, 1), float('inf'), dtype=torch.float32)

    points_expanded = path_points[:, :, None, :].expand(-1, -1, len(surfaces), -1)  # Shape: (path_num, hop_num, num_surfaces, 3)
    surfaces_vertices = vertices[surfaces]  # Shape: (num_surfaces, 4, 3)

    def compute_normals(surfaces_vertices):
        v1 = surfaces_vertices[:, 0, :]
        v2 = surfaces_vertices[:, 1, :]
        v3 = surfaces_vertices[:, 2, :]
        a = v2 - v1
        b = v3 - v1
        normals = torch.cross(a, b)
        return normals / torch.norm(normals, dim=1, keepdim=True)

    def orthogonal_projection_to_plane(points_expanded, plane_point, normals):
        v = points_expanded - plane_point
        distance = torch.sum(v * normals, dim=-1, keepdim=True)
        projection = points_expanded - distance * normals
        return projection, torch.abs(distance)

    normals = compute_normals(surfaces_vertices)  # Shape: (num_surfaces, 3)
    plane_point = surfaces_vertices[:, 0, :]  # Shape: (num_surfaces, 3)

    projections, distances_to_planes = orthogonal_projection_to_plane(points_expanded, plane_point, normals)
    
    ab = surfaces_vertices[:, 1, :] - surfaces_vertices[:, 0, :]
    ad = surfaces_vertices[:, 2, :] - surfaces_vertices[:, 0, :]
    
    cd = surfaces_vertices[:, 2, :] - surfaces_vertices[:, 3, :]
    bd = surfaces_vertices[:, 1, :] - surfaces_vertices[:, 3, :]
    
    ap = points_expanded.type(torch.float32) - surfaces_vertices[:, 0, :]
    dp = points_expanded.type(torch.float32) - surfaces_vertices[:, 3, :]

    ab_dot = torch.sum(ab * ab, dim=-1, keepdim=True)
    ad_dot = torch.sum(ad * ad, dim=-1, keepdim=True)
    ap_ab_dot = torch.sum(ap * ab, dim=-1, keepdim=True)
    ap_ad_dot = torch.sum(ap * ad, dim=-1, keepdim=True)

    is_in_ab = (0 <= ap_ab_dot) & (ap_ab_dot <= ab_dot)
    is_in_ad = (0 <= ap_ad_dot) & (ap_ad_dot <= ad_dot)
    is_in_projection = is_in_ab & is_in_ad  # Shape: (path_num, hop_num, num_surfaces)

    # This line is buggy, it should be torch.where(is_in_projection, distances_to_planes, distances_to_edges)
    edge1_distances = torch.norm(torch.cross(ab.expand_as(ap), ap, dim=-1), dim=-1) / torch.norm(ab, dim=-1)
    edge2_distances = torch.norm(torch.cross(ad.expand_as(ap), ap, dim=-1), dim=-1) / torch.norm(ad, dim=-1)
    edge3_distances = torch.norm(torch.cross(cd.expand_as(dp), dp, dim=-1), dim=-1) / torch.norm(cd, dim=-1)
    edge4_distances = torch.norm(torch.cross(bd.expand_as(dp), dp, dim=-1), dim=-1) / torch.norm(bd, dim=-1)
    
    distances_to_edges = torch.min(torch.stack([edge1_distances, edge2_distances, edge3_distances, edge4_distances], dim=-1), dim=-1)[0].unsqueeze(-1)
    distances = torch.where(is_in_projection, distances_to_planes, distances_to_edges).squeeze(-1)

    min_distances, min_indices = torch.min(distances, dim=-1)

    labels[:, :, 0] = min_indices
    min_distance_tensor[:, :, 0] = min_distances

    # Handle specific labeling rules
    labels[:, 0, 0] = -1  # The first point is always the Tx

    padding_mask = torch.all(path_points == torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64).to('cuda'), dim=-1)
    labels[padding_mask] = -3  # Padding location
    max_indices =  torch.argmax(padding_mask.int(), dim=1) #torch.argmax(non_padding_mask.int(), dim=1)
    rx_indices = torch.max(max_indices-1, torch.zeros_like(max_indices))
    rx_indices_mask = rx_indices > 0
    labels[rx_indices_mask, rx_indices[rx_indices_mask], 0] = -2  # The last valid point in each row is labeled as Rx
    # Set Rx labels
        #torch.roll(padding_mask, shifts=-1, dims=1)


    # # Ensure correct labeling: only the last valid point in each row is labeled as Rx (-2)
    # last_valid_indices = torch.max(torch.where(~padding_mask, torch.arange(hop_num).unsqueeze(0), torch.tensor(-1)), dim=1).values
    # for i in range(path_num):
    #     if last_valid_indices[i] >= 0:
    #         labels[i, last_valid_indices[i], 0] = -2

    return labels

def prepare_suface_index(lay_data_path,
                         surface_index_path):
    # Phase 1: get surface index
    lay_data = None        
    # get obj id
    obj_id = lay_data_path.split('/')[-1].split('.')[0]
    lay_data_path = lay_data_path.replace(".obj", ".lay")
    if not os.path.exists(surface_index_path):
        os.makedirs(surface_index_path, exist_ok=True)
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
    obj_path_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".lay"):
                lay_path_list.append(os.path.join(root, file))
            elif file.endswith(".obj"):
                obj_path_list.append(os.path.join(root, file))
    # sort by name. 1,2,3,4, NOT 1,10,11,12
    lay_path_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    obj_path_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return lay_path_list, obj_path_list


def per_Tx_Rx_process_cuda_optim_vec(args):
    T, R, x, intereaction_tensor, i, storage_dir, obj_dir, debug_flag = args
    obj_id_bug_list = [21, 34, 55, 59, 72, 78, 92, 121, 124]  # these obj_id has some problem, skip
    
    obj_id = obj_dir.split('/')[-1].split('.')[0]
    if int(obj_id) in obj_id_bug_list:
        print("obj_id {} is in obj_id_bug_list, skip".format(obj_id))
        return
    
    num_paths = intereaction_tensor.shape[4]
    num_node_per_path = intereaction_tensor.shape[5]
    
    surface_index_path = storage_dir.replace("node_attr_tensor", "surface_index")
    _, faces_rect, vertices = prepare_suface_index(obj_dir, surface_index_path)
    faces_rect = faces_rect.to('cuda')
    vertices = vertices.to('cuda')
    
    node_attr_tensor = torch.zeros(int(T), int(R), int(num_paths), int(num_node_per_path), 7)  # 6 dim for each node + 1 dim for surface index
    
    gt = x['channels'][0, :, 0, :, :, :].type(torch.float16)
    os.makedirs(storage_dir, exist_ok=True)
    torch.save(gt, os.path.join(storage_dir, f'gt_{i}.pt'))
    Tx_Rx_validaty = gt[:, :, 7, :].to('cuda')
    del gt
    
    if debug_flag:
        tx_iter = 1
        rx_iter = 18
    else:
        tx_iter = T
        rx_iter = R
    
    tx_indices = torch.arange(tx_iter)
    rx_indices = torch.arange(rx_iter)
    
    tx_indices_broadcasted = tx_indices.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, rx_iter, num_paths, num_node_per_path)
    rx_indices_broadcasted = rx_indices.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(tx_iter, -1, num_paths, num_node_per_path)
    
    interType_points = intereaction_tensor[0, tx_indices_broadcasted, 0, rx_indices_broadcasted, :, :].to('cuda')
    interType_points_coord = interType_points[:, :, :, :, 1:4]
    
    surface_label_tensor = classify_points_vec(interType_points_coord.reshape(-1, num_node_per_path, 3), faces_rect, vertices)
    surface_label_tensor = surface_label_tensor.view(tx_iter, rx_iter, num_paths, num_node_per_path, 1)
    
    mask = (interType_points[:, :, :, :, 3] == 0) | (interType_points[:, :, :, :, 3] == 3)
    
    texture_label = torch.zeros_like(mask, dtype=torch.bool)
    texture_label[mask] = 1
    texture_label = texture_label.unsqueeze(-1).type(torch.float16)
    
    node_w_texture = torch.cat((interType_points, texture_label), dim=-1)
    
    D = Tx_Rx_validaty[tx_indices_broadcasted, rx_indices_broadcasted, :].unsqueeze(-1).type(torch.float16)
    path_validaty = D.unsqueeze(-1).expand(-1, -1, -1, num_node_per_path, 6)
    
    node_w_texture_w_D = torch.cat([node_w_texture, path_validaty], dim=-1)
    node_w_texture_w_D_w_surface = torch.cat([node_w_texture_w_D, surface_label_tensor], dim=-1)
    
    node_attr_tensor[:tx_iter, :rx_iter, :, :, :] = node_w_texture_w_D_w_surface
    
    save_path = os.path.join(storage_dir, f'obj_{obj_id}')
    
    if debug_flag:
        return node_attr_tensor.cpu()
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(node_attr_tensor.cpu(), os.path.join(storage_dir, f'node_attr_tensor_{obj_id}_{i}.pt'))
        return


def per_Tx_Rx_process_cuda_optim(args):
    T, R, x, intereaction_tensor, i, storage_dir, obj_dir, debug_flag = args
    intereaction_tensor = intereaction_tensor.to('cuda')
    obj_id_bug_list = [21,34,55,59,72,78,92,121,124] # these obj_id has some problem, skip
    # obj_dir = /proj/raygnn/workspace/raytracingdata/testenv/objs/1.obj, obj_id = 1
    # import ipdb; ipdb.set_trace()
    obj_id = obj_dir.split('/')[-1].split('.')[0]
    if int(obj_id) in obj_id_bug_list:
        print ("obj_id {} is in obj_id_bug_list, skip".format(obj_id))
        return
    else:
        pass
    ### Bootstrapping, since all graph are the same stucture, lets take it outside the loop ###
    ### 1. prepare node_attr_tensor, edge_attr_tensor ### 
    # intereaction_tensor = torch.tensor(intereaction_tensor)
    num_paths = intereaction_tensor.shape[4]
    num_node_per_path = intereaction_tensor.shape[5]
    num_total_nodes = num_node_per_path
    ### 2,  prepare origin edge_index ###
    surface_index_path = storage_dir.replace("node_attr_tensor", "surface_index")
    _, faces_rect, vertices = prepare_suface_index(obj_dir, surface_index_path)
    faces_rect = faces_rect.to('cuda')
    vertices = vertices.to('cuda')
    ### 3,  prepare surrounding edge_index ###
    node_attr_tensor = torch.zeros(int(T),int(R), int(num_paths), int(num_node_per_path), 7) # 6 dim for each node + 1 dim for surface index        
    # node_attr_tensor = torch.zeros(1,10, int(num_paths), int(num_total_nodes), 7) # 6 dim for each node + 1 dim for surface index
    gt = x['channels'][0, :, 0, :, :, :].type(torch.float16)
    os.makedirs(storage_dir, exist_ok=True)
    torch.save(gt, os.path.join(storage_dir, f'gt_{i}.pt'))
    Tx_Rx_validaty = gt[:,:,7,:].to('cuda')
    del gt
    
    if debug_flag:
        tx_iter = 1
        rx_iter = 18
    else:
        tx_iter = T
        rx_iter = R
    for Tx in range(tx_iter): #tqdm(range(T)): 
        # 4 dim for each edge: validaty, distance, radians, azimuth
        for Rx in range(rx_iter): #range(R):
            #### Phase1: Node handling ####
            interType_points = intereaction_tensor[0, Tx, 0, Rx, :, :]
            # 1.1 acquire texture label
            interType_points_coord = interType_points[:, :, 1:4]
            # get surface label
            surface_label_tensor_vec = classify_points_vec (interType_points_coord, faces_rect, vertices)
            # surface_label_tensor = classify_points (interType_points_coord, faces_rect, vertices)
            # assert torch.equal(surface_label_tensor, surface_label_tensor_vec), "Two functions are not equal"
            # TODO: some function to check if such point is on some 2d plane in 3d space
            mask = (interType_points[:, :, 3] == 0) | (interType_points[:, :, 3] == 3)
            # All texture labels are 0: Bricks
            texture_label = torch.zeros_like(mask, dtype=torch.bool).to('cuda')
            # Where the mask is True, set to 1: Concrete
            texture_label[mask] = 1
            texture_label = texture_label.unsqueeze(-1)
            texture_label = texture_label.type(torch.float16)
            # Concatenate along the last dimension to append the values
            node_w_texture = torch.cat((interType_points, texture_label), dim=2)
            # 1.2 acquire path validaty
            D = Tx_Rx_validaty[Tx, Rx, :].unsqueeze(1)
            D = D.type(torch.float16)
            # Extract the validity of each path
            path_validaty = D.unsqueeze(1).repeat(1, 6, 1)
            
            # interaction_type, x, y, z, texture_label, path_validaty surface_label
            node_w_texture_w_D = torch.cat([node_w_texture, path_validaty], dim = 2)
            node_w_texture_w_D_w_surface = torch.cat([node_w_texture_w_D, surface_label_tensor_vec], dim = 2)
            node_attr_tensor[Tx, Rx, :, :, :] = node_w_texture_w_D_w_surface
    # Save each Data object
    save_path = os.path.join(storage_dir, f'obj_{obj_id}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert node_attr_tensor.shape == (int(T), int(R), int(num_paths), int(num_total_nodes), 7), "Shape of node_attr_tensor is not correct {}".format(node_attr_tensor.shape)
    if debug_flag:
        return node_attr_tensor
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(node_attr_tensor.cpu(), os.path.join(storage_dir, f'node_attr_tensor_{obj_id}_{i}.pt'))

# Define the function that creates argument tuples for each task
def prepare_args(train_loader, storage_dir, data_dir, debug = False):
    obj_dir = data_dir + "objs"
    lay_list, obj_list = get_obj_lay_path_list(obj_dir)
    if not debug:
        return [(target.shape[1], target.shape[3], input, target, i, storage_dir,  obj_list[i], debug) for i, (input, target) in enumerate(train_loader)]
    else:
        results = []
        for i, (input, target) in enumerate(train_loader):
            #  T, R, x, intereaction_tensor, _, i, storage_dir, obj_dir = args
            results.append((target.shape[1], target.shape[3], input, target, i, storage_dir,  obj_list[i], debug))
            if i == 19:
                break
        return results
    
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='wi3rooms') # 'wi3rooms' or 'wiIndoor'
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--data_dir', type=str, default='/proj/raygnn/workspace/raytracingdata/wi3rooms/')
    argparser.add_argument('--storage_dir', type=str, default='/proj/raygnn_storage/HFdata/raw_data/train/node_attr_tensor/')
    argparser.add_argument('--testset',  type=str, default='None') # 'None' or 'test', 'genz', 'gendiag'

    dataset = argparser.parse_args().dataset    
    debug = argparser.parse_args().debug
    data_dir = argparser.parse_args().data_dir
    testset_type = argparser.parse_args().testset
    storage_dir = argparser.parse_args().storage_dir
    assert dataset in ['wi3rooms', 'wiIndoor'], "dataset is not correct"
    if dataset == 'wi3rooms':
        print ("dataset is wi3rooms")
    else:
        print ("dataset is wiIndoor")
        data_dir = data_dir.replace("wi3rooms", "wiIndoor")
        storage_dir = storage_dir.replace("raw_data", "wiIndoor_raw_data")

    if debug:
        print (f"debug: {debug}, data_dir: {data_dir}, storage_dir: {storage_dir}, testset_type: {testset_type}")
        type_winertloader = "train"
    elif testset_type != "None": # test set
        train_storage_dir = storage_dir
        storage_dir = train_storage_dir.replace("train", testset_type)
        type_winertloader = testset_type
    else: # training case
        type_winertloader = "train"
    
    # Load your train_loader or whatever setup you have
    train_dataset = WINeRT_Dataset(data_dir, type = type_winertloader)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
    )
    
    # Prepare arguments for multiprocessing
    args_train = prepare_args(train_loader,
                              storage_dir,
                              data_dir,
                              debug = debug)
    

    ### Only for debug
    if debug:
        # time_diff_1 =[]
        # time_diff_2 =[]
        for arg in tqdm(args_train):
            # time_now = datetime.now()
            node_attr_tensor_vec = per_Tx_Rx_process_cuda_optim(arg)
            # time_end1 = datetime.now()
            # node_attr_tensor = per_Tx_Rx_process_cuda_optim_vec(arg)
            # time_end2 = datetime.now()
            # assert torch.equal(node_attr_tensor, node_attr_tensor_vec), "Two functions are not equal"
            # time_diff_1.append((time_end1 - time_now).total_seconds())
            # time_diff_2.append((time_end2 - time_end1).total_seconds())
        # time_diff_1 = np.array(time_diff_1)
        # time_diff_2 = np.array(time_diff_2)
        # print ("time_diff_1", time_diff_1.mean(), time_diff_1.std(), time_diff_1.max(), time_diff_1.min())
        # print ("time_diff_2", time_diff_2.mean(), time_diff_2.std(), time_diff_2.max(), time_diff_2.min())
    else:
        for arg in tqdm(args_train):
            per_Tx_Rx_process_cuda_optim(arg)
    print ("Done")
            
if __name__ == '__main__':
    main()