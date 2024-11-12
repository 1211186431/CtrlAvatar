"""
Visualize 3D meshes in aitviewer.
"""
import os.path as osp
import glob
import argparse
import trimesh
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from aitviewer.viewer import Viewer
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import VariableTopologyMeshes


def main(data_root,out_path,model_type):
    cols, rows = 800, 1200
    # if want to save video, use the following line and line 34
    # v = HeadlessRenderer(size=(cols, rows))
    # if want to visualize, use the following line and line 35
    v = Viewer(size=(cols, rows))
    v.playback_fps = 30.0
    v.scene.camera.fov = 80.0
    v.scene.camera.position[0] = -1.3
    v.scene.camera.position[1] = 2.0
    v.scene.camera.position[2] = 1.8
    
    # Example 2: Load 3D scans (.ply format)
    if model_type == 'ply':
        smplx_meshes_names = sorted(glob.glob(osp.join(data_root, '*.ply')))
        smplx_meshes = VariableTopologyMeshes.from_plys(smplx_meshes_names, name='Scan Mesh')
    elif model_type == 'ply_no_color':
        # no color
        meshes_list = []
        for mesh_name in smplx_meshes_names:
            mesh = trimesh.load(mesh_name)
            meshes_list.append(mesh)
        smplx_meshes = VariableTopologyMeshes.from_trimeshes(meshes_list, name='Scan Mesh')
    elif model_type == 'obj':   
        # obj
        smplx_meshes = VariableTopologyMeshes.from_directory(data_root, name='Scan Mesh')
    v.scene.add(smplx_meshes)

    #v.save_video(video_dir=out_path)
    v.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/ps/dy/save_mesh/00016')
    parser.add_argument('--out_root', type=str, default='/home/ps/dy/')
    parser.add_argument('--subject', type=str, default='00016')
    parser.add_argument('--method', type=str, default='Ours')
    parser.add_argument('--mode_type', type=str, default='ply')
    args = parser.parse_args()
    out_path = osp.join(args.out_root,args.method+"_"+args.subject+".mp4")
    main(data_root=args.data_root,out_path=out_path,model_type=args.mode_type)
