import glob
import os
from dataset.render_util import render_data
import numpy as np
import tqdm
import torch
import argparse
os.environ['IO_DISABLE_TELEMETRY'] = '1'
def eval_render(obj_files,config):
    out_list = []
    for obj_file in tqdm.tqdm(obj_files):
        with torch.no_grad():
            out_data_list = render_data(obj_file,None,None,image_size=config['imge_size'],is_obj=config['is_obj'],renderer_type=config['renderer_type'])
        ## (views, height, width, channels)
        out_data = np.stack(out_data_list, axis=0)
        out_list.append(out_data)
    out_list = np.stack(out_list, axis=0)
    return out_list
def eval(base_path,config,is_gt=False,subject=None):
    if is_gt:
        mesh_list_path = os.path.join(base_path,subject,"test")
        obj_files = sorted(glob.glob(os.path.join(mesh_list_path, '*', "meshes_obj", '*.obj')))
    elif config["is_obj"]:
        mesh_list_path = base_path
        obj_files = sorted(glob.glob(os.path.join(base_path,'*.obj')))
    elif not config["is_obj"]:
        mesh_list_path = base_path
        obj_files = sorted(glob.glob(os.path.join(base_path,'*.ply')))
    out_list = eval_render(obj_files,config)
    return out_list

def main(args):
    data_path = args.data_path
    is_gt = args.is_gt
    method = args.method
    subject = args.subject
    out_dir = args.out_dir
    renderer_type = args.renderer_type
    config = {
        "imge_size":1024,
        "is_obj":True,
        "renderer_type":renderer_type
    }
    if is_gt:
        eval_data = eval(data_path,config,True,subject)
    else:
        config["is_obj"]=False
        eval_data = eval(data_path,config,False)
    np.save(os.path.join(out_dir,method+"_"+subject+".npy"),eval_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--subject', type=str, default='00016')
    parser.add_argument('--data_path', type=str, default='/home/ps/dy/CtrlAvatar/outputs/test/00016/mesh_test')
    parser.add_argument('--is_gt', type=bool, default=False)
    parser.add_argument('--method', type=str, default='Ctrl')
    parser.add_argument('--out_dir', type=str, default='/home/ps/dy/')
    parser.add_argument('--renderer_type', type=str, default='pytorch3d')
    args = parser.parse_args()
    main(args)
