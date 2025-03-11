import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import plotly.graph_objects as go
import random
import os
import re

def get_random_color_node(seed=None, path_id=None):
    r_green, g_green, b_green = 0, 255, 0
    r_blue, g_blue, b_blue = 0, 0, 255
    
    # Get a random ratio between 0 and 1
    if seed:
        np.random.seed(seed)
        ratio = np.random.random()
    elif path_id:
        ratio = 0.01 * path_id
    else:
        ratio = random.random()
    # Interpolate between green and blue
    r = int(r_green + ratio * (r_blue - r_green))
    g = int(g_green + ratio * (g_blue - g_green))
    b = int(b_green + ratio * (b_blue - b_green))
    
    return 'rgb({}, {}, {})'.format(r, g, b)

def parse_obj(obj_content):
    vertices = []
    faces = []
    lines = obj_content.split('\n')
    vertex_count = 0
    faces_count = 0
    for line in lines:
        if line.startswith('v '):
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))
            vertex_count += 1
        elif line.startswith('f '):
            _, *f = line.split()
            f = [int(i) for i in f]
            faces.append(f)
            faces_count += 1
    print(f"Found {vertex_count} vertices and {faces_count} faces")
    return vertices, faces

def extract_segments_from_lay(lay_content,points_map, dataset = "wi3rooms"):
    segments_map = {}
    lines = lay_content.split('\n')
    inside_segments = False
    for line in lines: 
        # print (line)
        if line.startswith("[segments]"):
            inside_segments = True
            continue
        if inside_segments and line.startswith('['):
            break
        if inside_segments:
            # print (line)
            # line = "1 = {'name': 'WALL', 'connect': [-6, -10], 'z': (0.0, 3)}"
            match_pattern = re.search(r"(\d+) = \{'name': '(\w+)', 'connect': \[(-?\d+), (-?\d+)\], 'z': \(([\d\.]+), ([\d\.]+)\)\}", line)
            if match_pattern:
                segment_name = int(match_pattern.group(1))
                segment_type = str(match_pattern.group(2))
                connect_1 = float(match_pattern.group(3))
                connect_2 = float(match_pattern.group(4))
                z_1 = float(match_pattern.group(5))
                z_2 = float(match_pattern.group(6))
                segments_map[segment_name] = {'type': segment_type, 'connect': [connect_1, connect_2], 'z': [z_1, z_2]}
    if dataset == "wi3rooms":
        point_left_lower_corner_coord = [10.0, 0.0]
        point_right_lower_corner_coord = [10.0, 5.0] # not tuple
        point_left_upper_corner_coord = [0.0, 0.0]
        point_right_upper_corner_coord = [0.0, 5.0]
        point_left_lower_corner, point_right_lower_corner, point_left_upper_corner, point_right_upper_corner = None, None, None, None
        for key, value in points_map.items():
            if value == point_left_lower_corner_coord:
                point_left_lower_corner = key
            elif value == point_right_lower_corner_coord:
                point_right_lower_corner = key
            elif value == point_left_upper_corner_coord:
                point_left_upper_corner = key
            elif value == point_right_upper_corner_coord:
                point_right_upper_corner = key
        assert point_left_lower_corner and point_right_lower_corner and point_left_upper_corner and point_right_upper_corner, "cannot find 4 corners"
        len_segment_map = len(segments_map)
        segments_map[len_segment_map+1] = {'type': 'CEIL', 'connect': [point_left_lower_corner, point_right_lower_corner, point_left_upper_corner, point_right_upper_corner], 'z': [3]}
        segments_map[len_segment_map+2] = {'type': 'FLOOR', 'connect': [point_left_lower_corner, point_right_lower_corner, point_left_upper_corner, point_right_upper_corner], 'z': [0]}
    return segments_map

def decompose_segments(segments_map, points_map, color_map, materials_map):
    # 1. Extract all unique vertices
    vertices = []
    vertex_to_index = {}

    for seg in segments_map.values():
        if seg['type'] == 'WALL':
            for z in seg['z']:
                for v_key in seg['connect']:
                    coord = points_map[v_key] + [z]
                    point = tuple(coord)
                    if point not in vertex_to_index:
                        vertex_to_index[point] = len(vertices)
                        vertices.append(point)
        else:
            for v_key in seg['connect']:
                coord = points_map[v_key] + [seg['z'][0]]
                point = tuple(coord)
                if point not in vertex_to_index:
                    vertex_to_index[point] = len(vertices)
                    vertices.append(point)

    # 2. Map segment endpoints to unique indices
    faces = []
    colors = []
    materials = []
    for id in range(len(segments_map)):
        seg_id = id + 1 # segment ids start from 1
        if segments_map[seg_id]['type'] == 'WALL':
            start, end = segments_map[seg_id]['connect']
            idx_start_top = vertex_to_index[tuple(points_map[start] + [segments_map[seg_id]['z'][1]])]
            idx_start_bottom = vertex_to_index[tuple(points_map[start] + [segments_map[seg_id]['z'][0]])]
            idx_end_top = vertex_to_index[tuple(points_map[end] + [segments_map[seg_id]['z'][1]])]
            idx_end_bottom = vertex_to_index[tuple(points_map[end] + [segments_map[seg_id]['z'][0]])]
            
            # Representing line segments with degenerate triangles
            faces.append((idx_start_top, idx_start_bottom, idx_end_top))
            faces.append((idx_end_top, idx_start_bottom, idx_end_bottom))
            colors.append(color_map[id])
            colors.append(color_map[id])
        else:
            start1, start2, end1, end2 = segments_map[seg_id]['connect']
            idx_start1_top = vertex_to_index[tuple(points_map[start1] + [segments_map[seg_id]['z'][0]])]
            idx_start2_top = vertex_to_index[tuple(points_map[start2] + [segments_map[seg_id]['z'][0]])]
            idx_end1_top = vertex_to_index[tuple(points_map[end1] + [segments_map[seg_id]['z'][0]])]
            idx_end2_top = vertex_to_index[tuple(points_map[end2] + [segments_map[seg_id]['z'][0]])]
            faces.append((idx_start1_top, idx_start2_top, idx_end1_top))
            faces.append((idx_end1_top, idx_start2_top, idx_end2_top))
            colors.append(color_map[id])
            colors.append(color_map[id])
            
        material_info = materials_map[segments_map[seg_id]['type']]
        # del material_info['raw_color']
        materials.append(material_info)
    return vertices, faces, colors, materials

def extract_points_from_lay(lay_content):
    points_map = {}
    lines = lay_content.split('\n')
    inside_points = False
    for line in lines:
        if line.startswith("[points]"):
            inside_points = True
            continue
        if inside_points and line.startswith('['):
            break
        if inside_points:
            #-1 = (4.3, 1.8)
            match_pattern = re.search(r"(-?\w+) = \(([\d\.]+), ([\d\.]+)\)", line)
            if match_pattern:
                point_name = int(match_pattern.group(1))
                x = float(match_pattern.group(2))
                y = float(match_pattern.group(3))
                points_map[point_name] = [x, y]
    return points_map
            
            

def extract_materials_from_lay(lay_content):
    materials_map = {}
    lines = lay_content.split('\n')
    inside_materials = False
    for line in lines:
        if line.startswith("[materials]"):
            inside_materials = True
            continue
        if inside_materials and line.startswith('['):
            break
        if inside_materials:
            # Check for both material patterns
            match_pattern_1 = re.search(r"(\w+) = \{.*?'mur': \(([\d\.+-]+[\d+-j]*)\), 'epr': \(([\d\.+-]+[\d+-j]*)\), 'roughness': ([\d\.]+), 'sigma': ([\d\.]+)", line)
            match_pattern_2 = re.search(r"(\w+) = \{'sigma': ([\d\.]+), 'roughness': ([\d\.]+), 'epr': \(([\d\.+-]+[\d+-j]*)\), 'mur': \(([\d\.+-]+[\d+-j]*)\)\}", line)
            
            if match_pattern_1:
                material_name = match_pattern_1.group(1)
                mur = complex(match_pattern_1.group(2))
                epr = complex(match_pattern_1.group(3))
                roughness = float(match_pattern_1.group(4))
                sigma = float(match_pattern_1.group(5))
                materials_map[material_name] = {'mur': mur, 'epr': epr, 'roughness': roughness, 'sigma': sigma}
            elif match_pattern_2:
                material_name = match_pattern_2.group(1)
                sigma = float(match_pattern_2.group(2))
                roughness = float(match_pattern_2.group(3))
                epr = complex(match_pattern_2.group(4))
                mur = complex(match_pattern_2.group(5))
                materials_map[material_name] = {'mur': mur, 'epr': epr, 'roughness': roughness, 'sigma': sigma}
    return materials_map


def extract_lay_colors(lay_content):
    color_map = {}
    lines = lay_content.split('\n')
    inside_slabs = False
    for line in lines:
        if line.startswith("[slabs]"):
            inside_slabs = True
            continue
        if inside_slabs and line.startswith('['):
            break
        if inside_slabs:
            # {'color': 'grey20', 'linewidth': 3, 'lthick': [0.05], 'lmatname': ['BRICK']}
            match = re.search(r"(\w+) = \{'color': '(\w+)', 'linewidth': (\d+), 'lthick': \[(\d+\.\d+)\], 'lmatname': \['(\w+)'\]", line)
            if match:
                color = match.group(2)
                linewidth = int(match.group(3))
                lthick = float(match.group(4))
                lmatname = match.group(5)
                color_map[match.group(1)] = {'raw_color': color, 'linewidth': linewidth, 'lthick': lthick, 'lmatname': lmatname}
    return color_map

def visualize(vertices, faces, seg_colors, materials, material_labels = False, return_fig = False, show = True):
    x, y, z = zip(*vertices)
    i, j, k = zip(*faces)

    # Create a list to store hover text for each data point
    if material_labels:
        hover_text = materials
    else:
        hover_text = []
    no_material_info_faces = []

    # Create a list of face colors based on the colors list
    face_colors = [seg_colors[i] for i in range(len(seg_colors))]
    # print (f"face_colors: {face_colors[:4]}")
    # Create a Mesh3d trace with facecolor
    info = [len(x),len(y),len(z),len(i),len(j),len(k),len(seg_colors)]
    assert len(x) == len(y) == len(z), "length of x,y,z is: " + str(info[:3])
    assert len(i) == len(j) == len(k) == len(seg_colors), "length of i,j,k,seg_colors is: " + str(info[3:])
    trace = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        facecolor=face_colors,  # Specify the face colors directly
        hoverinfo="text",
        text=hover_text
    )

    # Create the figure and add the trace
    fig = go.Figure(data=[trace])

    # Configure layout settings if needed
    fig.update_layout(
        scene=dict(
            # Adjust other scene properties here if needed
        )
    )
    # Show the figure
    if show:
        fig.show()
    if return_fig:
        return fig
    else:
        return
    
def visualize_update(fig, vertices, faces, seg_colors=None, materials=None, material_labels=False, return_fig=False):
    x, y, z = zip(*vertices)
    i, j, k = zip(*faces)
    i_new = [x-1 for x in i]
    j_new = [x-1 for x in j]
    k_new = [x-1 for x in k]
    if material_labels:
        hover_text = list(range(len(faces)))
    else:
        hover_text = []

    if seg_colors:
        face_colors = [seg_colors[i] for i in range(len(faces))]
    print(face_colors)
    adding_trace = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i_new,
        j=j_new,
        k=k_new,
        facecolor=face_colors,
        hoverinfo="text",
        text=hover_text,
        name="objs"
    )

    fig.add_trace(adding_trace)
    fig.show()

    if return_fig:
        return fig
    
def get_scene_plotly_lay_objs(dir_path, env_id, show=False, return_3D_list = True):
    lay_data_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(".lay")]
    print("get ", len(lay_data_paths), " lay files")
    lay_data_paths = [lay_data_paths[env_id]]
    all_vertices = []
    all_faces = []
    all_colors = []
    all_materials = []
    # non_air_count = 0  # to keep track of non-air objects
    color_map = None
    point_map = None
    segments_map = None
    materials_map = None
    if len(all_vertices) == 0:
        assert len(all_vertices) == len(all_faces) == len(all_colors) == len(all_materials) == 0, "all_vertices, all_faces, all_colors, all_materials should be empty"
        # Extract materials and colors from the .lay file, only once per dataset
        for idx, lay_data_path in enumerate(lay_data_paths):
            with open(lay_data_path, 'r') as file:
                lay_data = file.read()
                
            # Extract materials and colors from the .lay file
            color_map = extract_lay_colors(lay_data)
            print(f"lay {idx} has {len(color_map)} colors")
            materials_map = extract_materials_from_lay(lay_data)
            print (f"lay {idx} has {len(materials_map)} materials")
            point_map = extract_points_from_lay(lay_data)
            print (f"lay {idx} has {len(point_map)} points")
            segments_map = extract_segments_from_lay(lay_data, point_map)
            print (f"lay {idx} has {len(segments_map)} segments ")
            
            if True: # used as debug
                unique_point_index_in_point_map = set(point_map.keys())
                unique_point_index_in_segment_map = set([int(segments_map[x]['connect'][0]) for x in segments_map.keys()] + [int(segments_map[x]['connect'][1]) for x in segments_map.keys()])
                assert unique_point_index_in_point_map == unique_point_index_in_segment_map, "point index in point map and segment map are not the same"
            
            # Convert color names to uppercase
            color_map = {k.upper(): v for k, v in color_map.items()}
            
            colors = []
            for obj in segments_map.keys():
                material_name = segments_map[obj]['type']
                
                # non_air_count += 1
                # Use RGBA color values for each material
                if material_name == 'WALL':
                    material_color = 'rgba(156, 102, 31, 0.35)'  # brick color with transparency
                elif material_name == 'CEIL':
                    material_color = 'rgba(105, 105, 105, 0.2)'  # dim grey with 20% opacity
                elif material_name == 'FLOOR':
                    material_color = 'rgba(128, 128, 128, 0.2)'  # grey with 20% opacity
                else:
                    material_color = 'rgba(173, 216, 230, 0.2)'  # Default color with 20% opacity
                    # non_air_count -= 1
                colors.append(material_color)  # Use len(faces) to associate all faces with the current material
            
            # Add additional processing here if needed.
            all_colors.extend(colors)
        # Decompose segments into vertices and faces    
        all_vertices,all_faces,all_colors,all_materials = decompose_segments(segments_map, point_map, all_colors, color_map)
    else:
        print (".lay files have been processed before")
    fig = None
    if show:
        fig = visualize(all_vertices, all_faces, all_colors, all_materials, return_fig=True, material_labels=True)
    else:
        fig = visualize(all_vertices, all_faces, all_colors, all_materials, return_fig=True, material_labels=True, show=False)
    if return_3D_list:
        return all_vertices, all_faces, all_colors, all_materials, fig
    else:
        return color_map, point_map, segments_map, materials_map, fig

def update_scene_from_obj(dir_path, figure, material_labels = []):
    fig = figure
    obj_data_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(".obj")]
    print("get ", len(obj_data_paths), " obj files")
    # print("get ", len(lay_data_paths), " lay files")
    # assert len(obj_data_paths) == len(lay_data_paths), "number of obj files and lay files should be the same"

    all_vertices = []
    all_faces = []
    all_colors = []

    for idx, obj_data_path in enumerate(obj_data_paths):
        with open(obj_data_path, 'r') as file:
            obj_data = file.read()
            
        vertices, faces = parse_obj(obj_data)

        colors = []
    
        material_color = 'rgba(103, 242, 209, 0.35)'  # cyan color with  20% opacity
        colors.extend([material_color] * len(faces))  # Use len(faces) to associate all faces with the current material
        
        all_vertices.extend(vertices)
        all_faces.extend(faces)
        all_colors.extend(colors)

    # Use materials_map to map colors to material names for visualization
    fig_update = visualize_update(fig, all_vertices, all_faces, all_colors, material_labels, material_labels = False, return_fig = True)
    return fig_update
