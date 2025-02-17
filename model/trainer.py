import torch.optim as optim
from torch.utils.data import DataLoader
from model.mesh import load_mesh_by_dicts
from model.mesh import load_mesh
from PIL import Image
from model.networks.loss import img_loss
import torch
import tqdm
import copy
import os

class Trainer:
    def __init__(self, config, dataset, renderer, camera_manager):
        self.config = config
        self.device = torch.device(config.device if hasattr(config, "device") else "cuda")
        self.dataset = dataset
        self.renderer = renderer.to(self.device)
        self.camera_manager = camera_manager
        self.optimizer = self._configure_optimizer()
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=False
        )
        self.base_t_mesh = self._load_base_mesh(config.t_mesh_path)
        self.current_epoch = 0
        
    def set_dataset(self, dataset):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False
        )
    def _configure_optimizer(self):
        return optim.Adam(
            self.renderer.parameters(),
            lr=self.config.train.lr,
            betas=self.config.train.betas
        )

    def _load_base_mesh(self, path):
        mesh = load_mesh(path)
        mesh.transform_size("normalize", 1.0)
        return mesh

    def _compute_loss(self, pred, gt, mask):
        return img_loss(pred*mask, gt*mask)

    def train_epoch(self):
        self.renderer.train()
        total_loss = 0.0
        
        for data in self.dataloader:
            self.optimizer.zero_grad()
            
            # get camera parameters
            cameras_1 = self.camera_manager.sample_camera(
                "rotating", 
                elev_list=self.config.camera.rotating_elevations
            )
            cameras_2 = self.camera_manager.sample_camera(
                "random",
                batch_size=self.config.camera.random_batch_size
            )
            cameras = self.camera_manager.merged_camera([cameras_1, cameras_2])
            
            # get mesh data
            t_mesh = copy.deepcopy(self.base_t_mesh)
            gt_mesh = load_mesh_by_dicts(data['gt_mesh'])[0]
            
            # render
            render_img = self.renderer.render_def(t_mesh, data, cameras, self.config.render.iter_res)
            render_gt_img = self.renderer.render_gt(
                gt_mesh, cameras, self.config.render.iter_res,
                return_types=["rgb_from_texture"], need_bg=True
            )
            
            pred_rgb = render_img['rgb_from_texture']
            gt_rgb = render_gt_img['rgb_from_texture']
            mask = render_img['mask']
            
            loss = self._compute_loss(pred_rgb, gt_rgb, mask)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.dataloader)

    def train(self):
        for epoch in range(self.current_epoch, self.config.train.num_epochs):
            epoch_loss = self.train_epoch()
            self._log_metrics(epoch, epoch_loss)
            if self.config.logging.save_interval > 0 and epoch % self.config.logging.save_interval == 0:
                self.validate()
            self.current_epoch += 1
            
    def validate(self):
        cameras_1 = self.camera_manager.sample_camera(
                "rotating", 
                elev_list=[0]
            )
        cameras = cameras_1
        data = next(iter(self.dataloader))
        t_mesh = copy.deepcopy(self.base_t_mesh)
        render_img = self.renderer.render_def(t_mesh, data, cameras, self.config.render.iter_res)
        pred_rgb = render_img['rgb_from_texture']
        mask = render_img['mask']
        self._save_image(self.current_epoch, pred_rgb, mask)
        self.save_checkpoint(self.current_epoch)
    
    def test(self):
        self.renderer.eval()   
        t_mesh = copy.deepcopy(self.base_t_mesh)
        if self.config.test.texture_format == "texture":
            texture_mesh = self.renderer.export_texture(t_mesh)
        elif self.config.test.texture_format == "v_color":
            texture_mesh = self.renderer.export_v_color(t_mesh)
        else:
            raise ValueError("Unsupported texture format")
        for i,data in tqdm.tqdm(enumerate(self.dataloader)):    
            def_mesh = self.renderer.deform_model.forward(copy.deepcopy(texture_mesh), data['smplx_cond'][0], data['smplx_tfs'][0], inverse=False)
            def_mesh.export(os.path.join(self.config.test.test_out,"def_mesh_{}.obj".format(i)))

        
    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'model_state': self.renderer.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }
        
        filename = f"checkpoint_{epoch}.pth"
        torch.save(state, f"{self.config.logging.checkpoint_dir}/{filename}")
        
    def _save_image(self, epoch, pred_rgb, mask):
        images = (pred_rgb*mask*255).type(torch.uint8).cpu()
        img = images[0].numpy()
        Image.fromarray(img.squeeze(), 'RGB').save(
            f"{self.config.logging.output_dir}/rgb_img_{epoch}.png"
        )
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.renderer.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.current_epoch = checkpoint['epoch'] + 1

    def _log_metrics(self, epoch, loss):
        tqdm.tqdm.write(f"Epoch {epoch+1} | Loss: {loss:.4f}")

