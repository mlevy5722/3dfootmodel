import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import pythreejs as three
from ipywidgets import HTML, VBox

def load_and_process_model(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    return mesh

def identify_sole(mesh):
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    downward_points = vertices[normals[:, 1] < 0]
    if len(downward_points) == 0:
        sorted_indices = np.argsort(vertices[:, 1])
        downward_points = vertices[sorted_indices[:len(vertices)//10]]
    clustering = DBSCAN(eps=0.02, min_samples=5).fit(downward_points)
    if np.max(clustering.labels_) < 0:
        return downward_points
    sole_points = downward_points[clustering.labels_ == np.argmax(np.bincount(clustering.labels_[clustering.labels_ >= 0]))]
    return sole_points

def identify_toes(mesh):
    vertices = np.asarray(mesh.vertices)
    max_y = np.max(vertices[:, 1])
    y_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    toe_threshold = max_y - 0.2 * y_range
    toe_points = vertices[vertices[:, 1] > toe_threshold]
    return toe_points

def align_model(mesh, sole_points):
    sole_pcd = o3d.geometry.PointCloud()
    sole_pcd.points = o3d.utility.Vector3dVector(sole_points)
    sole_pcd.estimate_normals()
    sole_normal = np.mean(np.asarray(sole_pcd.normals), axis=0)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(
        (np.arctan2(sole_normal[2], -sole_normal[1]), np.arctan2(sole_normal[0], -sole_normal[1]), 0))
    mesh.rotate(rotation_matrix, center=(0, 0, 0))
    return mesh

def maximize_view(mesh):
    vertices = np.asarray(mesh.vertices)
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    mesh.translate(-center)
    mesh.scale(1 / np.max(size), center=(0, 0, 0))
    # Rotate 180 degrees around X-axis to flip toes up
    R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    return mesh

def crop_to_sole(mesh):
    vertices = np.asarray(mesh.vertices)
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    threshold = min_z + 0.1 * (max_z - min_z)  # Keep bottom 10%
    sole_indices = vertices[:, 2] < threshold
    mesh = mesh.select_by_index(np.where(sole_indices)[0])
    return mesh

def save_mesh(mesh, file_path):
    o3d.io.write_triangle_mesh(file_path, mesh)
    print(f"Processed mesh saved to {file_path}")

def visualize_mesh_threejs(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    geometry = three.BufferGeometry(
        attributes={
            'position': three.BufferAttribute(vertices.astype('float32'), normalized=False),
            'index': three.BufferAttribute(faces.ravel().astype('uint32'), normalized=False),
        }
    )
    
    material = three.MeshPhongMaterial(color='tan', side='DoubleSide')
    mesh_3js = three.Mesh(geometry, material)
    
    scene = three.Scene()
    scene.add(mesh_3js)
    
    camera = three.PerspectiveCamera(position=[0, -2, 0], up=[0, 0, 1])
    camera.lookAt([0, 0, 0])
    
    light = three.DirectionalLight(position=[0, -1, 1])
    scene.add(light)
    
    renderer = three.Renderer(camera=camera, scene=scene, controls=[three.OrbitControls(controlling=camera)])
    
    return renderer

def process_and_visualize_foot_model(input_file_path, output_file_path):
    try:
        mesh = load_and_process_model(input_file_path)
        sole_points = identify_sole(mesh)
        toe_points = identify_toes(mesh)
        mesh = align_model(mesh, sole_points)
        mesh = maximize_view(mesh)
        mesh = crop_to_sole(mesh)
        save_mesh(mesh, output_file_path)
        print("Processing completed successfully.")
        renderer = visualize_mesh_threejs(mesh)
        display(VBox([renderer, HTML("Use mouse to rotate, zoom, and pan")]))
        print(f"The processed model has been saved as {output_file_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check if the input file exists and is in the correct format.")

# Usage
process_and_visualize_foot_model("model-mobile001.obj", "processed_foot_model.obj")

