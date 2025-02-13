from .lbs import ForwardDeformer, skinning
from .IDN import IDN
import torch
import torch.nn as nn

class DeformMesh(nn.Module):
    def __init__(self, pretrained_path, max_batch_size=300000,config=None):
        super().__init__()
        self.lbs = ForwardDeformer(d_out=59,model_type='smplx').to("cuda:0")
        self.IDN = IDN(cond_dim=73,multires=4).to("cuda:0")
        self.max_batch_size = max_batch_size
        self.w = None
        if pretrained_path is not None:
            print('Loading pretrained model from {}'.format(pretrained_path))
            self.load_lbs_weights(pretrained_path=pretrained_path)
            self.IDN.load_IDN_weights(pretrained_path=pretrained_path)
        else:
            assert False, 'No deformer model found'
        self.freeze_model()
         
    def freeze_model(self):
        for param in self.lbs.parameters():
            param.requires_grad = False
        for param in self.IDN.parameters():
            param.requires_grad = False
        print('Deform Model frozen')
    
    def load_lbs_weights(self, pretrained_path):
        """
        Load deformer network weights from a checkpoint into the specified model.
        """
        pretrained_model = torch.load(pretrained_path, map_location=torch.device('cuda:0'))
        deformer_state_dict = {key: value for key, value in pretrained_model['state_dict'].items() if key.startswith('deformer.lbs_network')}
        adjusted_state_dict = {'lbs_network.' + key.replace('deformer.lbs_network.', ''): value for key, value in deformer_state_dict.items()}
        self.lbs.load_state_dict(adjusted_state_dict) 
    
    def forward_deform(self, xc, cond, tfs):
        """
        Perform forward deformation, transforming from the canonical space xc to the deformed space xd.
        
        Args:
            xc (Tensor): Input vertices in the canonical space, shape (batch_size, num_vertices, 3).
            cond (Tensor): SMPL-X conditions, shape (batch_size, 73).
            tfs (Tensor): Joint rotation matrices, shape (batch_size, 55, 4, 4).
            
        Returns:
            Tensor: Deformed vertices in the deformed space, shape (batch_size, num_vertices, 3).
        """
        xo = self.IDN(xc, cond, inverse=False)
        xd = skinning(xo, self.w, tfs, inverse=False)
        return xd
    
    def inverse_deform(self, xd, cond, tfs):
        """
        Perform inverse deformation, transforming from the deformed space xd to the canonical space xc.
        
        Args:
            xd (Tensor): Input vertices in the deformed space, shape (batch_size, num_vertices, 3).
            cond (Tensor): SMPL-X conditions, shape (batch_size, 73).
            tfs (Tensor): Joint rotation matrices, shape (batch_size, 55, 4, 4).
            
        Returns:
            Tensor: Vertices in the canonical space, shape (batch_size, num_vertices, 3).
        """
        xc = skinning(xd, self.w, tfs, inverse=True)
        xo = self.IDN(xc, cond, inverse=True)
        return xo
    
    def get_lbs_weights(self,v_pos):
        weights = self.lbs.query_weights(v_pos,None).clamp(0, 1)
        return weights
        
    def forward(self, mesh, cond, smpl_tfs, inverse=False):
        """
        Perform a forward pass through the model, either forward deformation or inverse deformation based on the `inverse` flag.
        
        Args:
            mesh (Mesh): Mesh
            cond (Tensor): SMPL-X conditions, shape (batch_size, 73).
            smpl_tfs (Tensor): Joint transformation matrices, shape (batch_size, 55, 4, 4).
            inverse (bool): If True, performs inverse deformation, otherwise forward deformation.
            
        Returns:
            Mesh: Deformed Mesh.
        """
        mesh.transform_size("restore")
        v_pos = mesh.v_pos.unsqueeze(0)
        if self.w is None:
            self.w = self.get_lbs_weights(v_pos)
        if not inverse:
            v_pos = self.forward_deform(v_pos, cond, smpl_tfs)
        else:
            v_pos = self.inverse_deform(v_pos, cond, smpl_tfs)
        mesh.v_pos = v_pos[0]
        mesh.transform_size("normalize")
        return mesh
    