import torch
import os
from dataset.render_util import load_mesh
from model.color_net import MyColorNet
import nvdiffrast.torch as dr
import numpy as np
from PIL import Image
import trimesh
import cv2
from trimesh.visual.texture import TextureVisuals
def diffuse_color_with_mask(img_m, img_c, num_iter=1, ksize=3):
    """
    cv.findContours: http://t.zoukankan.com/wojianxin-p-12602490.html
    """
    img_m[img_m != 0] = 255

    hksize = ksize // 2
    k_range = range(-hksize, hksize + 1)

    #* expand
    img_m = cv2.copyMakeBorder(img_m, hksize, hksize, hksize, hksize, cv2.BORDER_CONSTANT, value=(0))
    img_c = cv2.copyMakeBorder(img_c, hksize, hksize, hksize, hksize, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    for _ in range(num_iter):
        uu, vv = np.where(img_m == 0)

        #* remove border
        m = True
        m &= (uu >= hksize)
        m &= (uu < img_m.shape[0] - hksize)
        m &= (vv >= hksize)
        m &= (vv < img_m.shape[1] - hksize)
        uu = uu[m]
        vv = vv[m]

        #* select silhouette. only 3x3 patch
        m = False
        for tu in [-1, 0, 1]:
            for tv in [-1, 0, 1]:
                m |= (img_m[uu + tu, vv + tv] == 255)

        uu = uu[m]
        vv = vv[m]
        img_m[uu, vv] = 127  #! set silhouette value

        #* calc weights: 0/1 | sum and mean
        c = 0
        w = 0
        for tu in k_range:
            for tv in k_range:
                tw = (img_m[uu + tu, vv + tv] == 255).astype(np.float32).reshape(-1, 1)
                tc = (img_c[uu + tu, vv + tv]).astype(np.float32)
                w += tw
                c += tw * tc
        img_c[uu, vv] = (c / w).astype(np.float32)
        img_m[img_m == 127] = 255  #!

    img_m = img_m[hksize:-hksize, hksize:-hksize]
    img_c = img_c[hksize:-hksize, hksize:-hksize].astype(np.uint8)

    return img_m, img_c
def texture_padding(img_c0, img_m0, fac=1.25):
    """
    * question: https://blender.stackexchange.com/a/265246/82691
        Here are some related keywords/links: 
            [Texture Padding](https://www.youtube.com/watch?v=MVsIIkJNkjM&ab_channel=malcolm341), 
            `Solidify` in [Free Plug-ins](http://www.flamingpear.com/free-trials.html) 
            and [Seam Fixing](https://www.youtube.com/watch?v=r9l8RfTvqyI&ab_channel=NamiNaeko); 
            [TexTools](https://github.com/SavMartin/TexTools-Blender) for Blender.
    * reference:
        [inpainting for atlas/texture map](https://blender.stackexchange.com/questions/264966/inpainting-for-atlas-texture-map)
        [mipmap](https://substance3d.adobe.com/documentation/spdoc/padding-134643719.html)
        [distance transform](https://stackoverflow.com/questions/26421566/pixel-indexing-in-opencvs-distance-transform)
        [seamlessClone](https://learnopencv.com/seamless-cloning-using-opencv-python-cpp/)
        [torch-interpol](https://github.com/balbasty/torch-interpol/issues/1)
    """

    assert 1 < fac < 1.5

    if np.all(img_m0 > 0):
        return img_c0

    img_m0[img_m0 != 0] = 255

    img_m0, img_c0 = diffuse_color_with_mask(img_m0, img_c0, 2)  #* diffuse 2 pixels (2x2 downsampling)

    img_m1 = img_m0.copy()
    img_c1 = img_c0.copy()
    while np.any(img_m1 == 0):
        img_m1 = cv2.resize(img_m1, (int(img_m1.shape[0] / fac), int(img_m1.shape[1] / fac)), interpolation=cv2.INTER_LINEAR)
        img_c1 = cv2.resize(img_c1, (int(img_c1.shape[0] / fac), int(img_c1.shape[1] / fac)), interpolation=cv2.INTER_LINEAR)
        img_m1[img_m1 != 255] = 0
        img_c1[img_m1 == 0] = 0
        img_m1, img_c1 = diffuse_color_with_mask(img_m1, img_c1, 2)

        img_m2 = img_m1.copy()
        img_c2 = img_c1.copy()
        while img_m2.shape[0] != img_m0.shape[0]:
            if (img_m0.shape[0] < img_m2.shape[0] * fac < img_m0.shape[0] * fac):
                img_shape = (img_m0.shape[0], img_m0.shape[1])
            else:
                img_shape = (int(img_m2.shape[0] * fac), int(img_m2.shape[1] * fac))
            img_m2 = cv2.resize(img_m2, img_shape, interpolation=cv2.INTER_LINEAR)
            img_c2 = cv2.resize(img_c2, img_shape, interpolation=cv2.INTER_LINEAR)

            img_m2[img_m2 != 255] = 0
            img_c2[img_m2 == 0] = 0

        nnz = np.nonzero(~img_m0 & img_m2)
        img_c0[nnz] = img_c2[nnz]
        img_m0 = img_m2

    return img_c0

def export_color(model,mesh_data,res=[4096,4096]):
    model.eval()
    with torch.no_grad():
        h, w = res[0],res[1]
        uv = mesh_data["vt"] *2.0 - 1.0
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)
        glctx = dr.RasterizeCudaContext()
        rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), mesh_data["faces"][0].int(), (h, w)) # [1, h, w, 4]
        xyzs, _ = dr.interpolate(mesh_data["verts"], rast, mesh_data["faces"][0].int()) # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(mesh_data["verts"][0][:, :1]).unsqueeze(0), rast, mesh_data["faces"][0].int()) # [1, h, w, 1]
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)
        feats = torch.zeros(h * w, 3, device='cuda', dtype=torch.float32)
        xyzs = xyzs[mask]
        batch_size = 300000
        pred_colors = []
        for start_idx in range(0, xyzs.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, xyzs.shape[0])
            batch_xyzs = xyzs[start_idx:end_idx]  
            batch_xyzs = batch_xyzs.unsqueeze(0)
            batch_pred_color = model.pred_color(batch_xyzs)
            batch_pred_color = batch_pred_color.squeeze(0)
            pred_colors.append(batch_pred_color)
    pred_color = torch.cat(pred_colors, dim=0)
    feats[mask] = pred_color
    feats = feats.reshape(h, w, 3)
    feats = texture_padding((feats.reshape(h, w, 3).cpu().numpy()*255).astype(np.uint8), (mask.reshape(h, w).cpu().numpy()*255).astype(np.uint8))
    PIL_image = Image.fromarray(feats)
    return PIL_image

def main(config):
    base_path = config['base_path']
    subject = config['subject']
    t_mesh_name = config['t_mesh_name']
    model_path = config['model_path']
    save_path = os.path.join(base_path, 'data',subject,'t_mesh')
    mesh_path = os.path.join(base_path, 'data',subject,'t_mesh',t_mesh_name)
    model_path = os.path.join(base_path, model_path)

    verts,faces,normals,texture = load_mesh(mesh_path,return_texture=True)
    model = MyColorNet(None,None,None,d_in_color=3).cuda()
    model.load_state_dict(torch.load(model_path))
    mesh_data={'verts':verts,'faces':faces,'normals':normals,'texture':texture}
    
    mesh = trimesh.load(mesh_path)    
    uv = mesh.visual.uv
    uv = torch.tensor(uv, device='cuda', dtype=torch.float32)
    mesh_data['vt'] = uv
    ## !!!
    mesh_data['vt'][...,1] = 1 - mesh_data['vt'][...,1]
    PIL_image = export_color(model,mesh_data)

    texture_visuals = TextureVisuals(mesh.visual.uv, image=PIL_image)
    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=texture_visuals)
    out_mesh.export(os.path.join(save_path,"t_mesh_with_uv.obj"))
