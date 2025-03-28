from model.renderer.camera import CameraManager
from model.renderer.nvdiff_renderer import Nviff_renderer
from model.mesh import load_mesh
from PIL import Image
import os
import torch
import argparse
def render_images(mesh_path, output_path, iter_res, elev_list):
    mesh = load_mesh(mesh_path)
    mesh.transform_size("normalize", 1.0)
    renderer = Nviff_renderer(None, None, None)
    camera_manager = CameraManager(
        iter_res=iter_res
    )
    cameras = camera_manager.sample_camera(
        "rotating", 
        elev_list=elev_list
    )

    render_imgs = renderer.render_gt(
            mesh, cameras, iter_res,
            return_types=["rgb_from_v_color","normals"], need_bg=True
    )
    type_name = {
        "rgb_from_v_color": "",
        "normals": "_normal"
    }

    view_name = {
        0: "front",
        1: "back",
        2: "left",
        3: "right"
    }

    for type in render_imgs:
        images = render_imgs[type] * 255
        images = images.type(torch.uint8).cpu()
        for i, _ in enumerate(images):
            image = images[i].numpy()
            image = Image.fromarray(image, "RGB")
            image.save(os.path.join(output_path, f"base_{view_name[i]}{type_name[type]}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default='00122_Inner')
    parser.add_argument('--mesh_path', type=str, default='/home/ps/data/dy/aaaiplus/data/00122_Inner/model0319.obj')
    parser.add_argument('--output_path', type=str, default='/home/ps/data/dy/aaaiplus/data/00122_Inner/base_images')
    parser.add_argument('--iter_res', type=int, default=512)
    args = parser.parse_args()
    iter_res = [args.iter_res, args.iter_res]
    subject = args.subject
    output_path = args.output_path
    mesh_path = args.mesh_path
    elev_list=[90, 270, 180, 0]
    os.makedirs(output_path, exist_ok=True)
    render_images(mesh_path, output_path, iter_res, elev_list)