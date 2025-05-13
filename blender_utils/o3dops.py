import open3d as o3d
import numpy as np
DEAFAULT_CAM_INFO = {
    'cam_intri': np.eye(3),
}
def meshface2obj(mesh, face, save_path = 'temp/test.obj'):
    mesh = mesh.reshape(-1, 3)
    face = face.reshape(-1, 3)
    mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.points), o3d.utility.Vector3iVector(face))
    o3d.io.write_triangle_mesh(save_path, mesh)
    print('mesh saved to', save_path)
    
def meshface2o3dmesh(mesh, face):
    mesh = mesh.reshape(-1, 3)
    face = face.reshape(-1, 3)
    mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.points), o3d.utility.Vector3iVector(face))
    return mesh
    
def load_model(model_path):
    if isinstance(model_path, str):
        if model_path.endswith('.obj'):
            mesh = o3d.io.read_triangle_mesh(model_path)
    elif isinstance(model_path, np.ndarray):
        mesh = meshface2o3dmesh(model_path[:, :3], model_path[:, 3:])
    elif isinstance(model_path, list):
        model_mesh_info = model_path[0]
        if isinstance(model_mesh_info, str):
            if model_mesh_info.endswith('.npy'):
                mesh = np.load(model_mesh_info)
        elif isinstance(model_mesh_info, np.ndarray):
            mesh = model_mesh_info
        mesh = mesh.reshape(-1, 3)
        model_face_info = model_path[1]
        if isinstance(model_face_info, str):
            if model_face_info.endswith('.npy'):
                face = np.load(model_face_info)
        elif isinstance(model_face_info, np.ndarray):
            face = model_face_info
        face = face.reshape(-1, 3)
        mesh = meshface2o3dmesh(mesh, face)
    return mesh
    
class Open3dTool():
    def __init__(self, cam_info = None):
        if cam_info is None:
            self.cam_intri = np.eye(3)
            self.cam_extri = np.eye(4)
            self.img_size = np.array([1920, 1080])
            self.distort_coffe = None
        else:
            self.load_cam_info_from_dict(cam_info)
            
    def load_cam_info_from_dict(self, cam_info):
        self.cam_intri = cam_info['cam_intri']
        self.cam_extri = cam_info['cam_extri']
        self.img_size = cam_info['img_size']
        if 'distort_coffe' in cam_info:
            self.distort_coffe = cam_info['distort_coffe']
        else:
            self.distort_coffe = None
    
    def visual_models(self, models, save_path = 'temp/test.png'):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for model_info in models:
            model = load_model(model_info)
            vis.add_geometry(model)
        vis.run()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        print('image saved to', save_path)