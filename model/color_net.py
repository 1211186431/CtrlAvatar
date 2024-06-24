import torch
import torch.nn as nn
from .deformer import ForwardDeformer, skinning
from .network import ImplicitNetwork
from .delta_mlp_0611 import condNet
from .smpl import SMPLXServer

import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
class MyColorNet(nn.Module):
    def __init__(self,meta_info,pretrained_path=None,smpl_model_path=None):
        super(MyColorNet, self).__init__()
        if meta_info is not None:
            gender = str(meta_info['gender']) if 'gender' in meta_info else None
            betas = meta_info['betas'] if 'betas' in meta_info else None
            use_pca = meta_info['use_pca'] if 'use_pca' in meta_info else None
            num_pca_comps = meta_info['num_pca_comps'] if 'num_pca_comps' in meta_info else None
            flat_hand_mean = meta_info['flat_hand_mean'] if 'flat_hand_mean' in meta_info else None
        if smpl_model_path is not None:
            self.smpl_server = SMPLXServer(gender=gender,
                                    betas=betas,
                                    use_pca=use_pca,
                                    num_pca_comps=num_pca_comps,
                                    flat_hand_mean=flat_hand_mean,model_path=smpl_model_path)
            for param in self.smpl_server.parameters():
                param.requires_grad = False
        self.deformer = ForwardDeformer(d_out=59,model_type='smplx')
        self.delta_net = ImplicitNetwork(d_in=54,d_out=3,skip_layer=[3],depth=4,width=256,multires=0,geometric_init=False)
        self.color_net = ImplicitNetwork(d_in=3,d_out=3,skip_layer=[3],depth=6,width=256,multires=10,geometric_init=False)
        self.cond_net = condNet(cond_dim=73)
        self.sigmoid = torch.nn.Sigmoid()
        if pretrained_path is not None:
            self.load_deformer_weights(pretrained_path=pretrained_path)
            self.load_cond_weights(pretrained_path=pretrained_path)
            self.load_delta_weights(pretrained_path=pretrained_path)
        self.freeze_model()
    def forward(self, x,cond=None):
        color = self.pred_color(x)
        if cond is None:
            return color
        pts_c = self.pred_point(x,cond)
        return color,pts_c
    def pred_color(self,x):
        color = self.color_net(x)
        color = self.sigmoid(color)
        return color
    
    def pred_point(self,x,cond):
        pts_c_emb = self.cond_net(x,cond)
        pts_c_delta = self.delta_net(pts_c_emb,return_feature=False)
        pts_c = x - pts_c_delta*0.1
        return pts_c
    
    def deform(self,verts,smpl_tfs):
        if verts.dim() == 3:
            verts = verts.squeeze(0)
        weights = self.deformer.query_weights(verts[None],
                                                None).clamp(0, 1)[0]

        verts_mesh_deformed = skinning(verts.unsqueeze(0),
                                        weights.unsqueeze(0),
                                        smpl_tfs).data.cpu().numpy()[0]
        return torch.from_numpy(verts_mesh_deformed).float().unsqueeze(0).to('cuda:0')
    


    def load_deformer_weights(self, pretrained_path):
        """
        Load deformer network weights from a checkpoint into the specified model.

        Args:
        model (torch.nn.Module): The model that contains a 'deformer' attribute.
        pretrained_path (str): Path to the pretrained model checkpoint.
        """
        # 加载预训练模型
        pretrained_model = torch.load(pretrained_path, map_location=torch.device('cuda:0'))

        # 获取deformer网络的状态字典
        deformer_state_dict = {key: value for key, value in pretrained_model['state_dict'].items() if key.startswith('deformer.lbs_network')}

        # 根据错误信息调整，添加 'lbs_network.' 前缀来匹配模型中的键
        adjusted_state_dict = {'lbs_network.' + key.replace('deformer.lbs_network.', ''): value for key, value in deformer_state_dict.items()}

        # 加载权重
        self.deformer.load_state_dict(adjusted_state_dict) 
    def load_cond_weights(self, pretrained_path):
        """
        Load deformer network weights from a checkpoint into the specified model.

        Args:
        model (torch.nn.Module): The model that contains a 'deformer' attribute.
        pretrained_path (str): Path to the pretrained model checkpoint.
        """
        # 加载预训练模型
        pretrained_model = torch.load(pretrained_path, map_location=torch.device('cuda:0'))
        cond_state_dict = {key: value for key, value in pretrained_model['state_dict'].items() if key.startswith('cond_net')}
        adjusted_state_dict = {key.replace('cond_net.', ''): value for key, value in cond_state_dict.items()}

        # 加载权重
        self.cond_net.load_state_dict(adjusted_state_dict)
        
    def load_delta_weights(self, pretrained_path):
        """
        Load deformer network weights from a checkpoint into the specified model.

        Args:
        model (torch.nn.Module): The model that contains a 'deformer' attribute.
        pretrained_path (str): Path to the pretrained model checkpoint.
        """
        pretrained_model = torch.load(pretrained_path, map_location=torch.device('cuda:0'))
        delta_state_dict = {key: value for key, value in pretrained_model['state_dict'].items() if key.startswith('delta_net')}
        adjusted_state_dict = {key.replace('delta_net.', ''): value for key, value in delta_state_dict.items()}

        # 加载权重
        self.delta_net.load_state_dict(adjusted_state_dict)
        
    def freeze_model(self):
        for param in self.deformer.parameters():
            param.requires_grad = False
            
        for param in self.cond_net.parameters():
            param.requires_grad = False
        for param in self.delta_net.parameters():
            param.requires_grad = False
        print('Model frozen')
    
