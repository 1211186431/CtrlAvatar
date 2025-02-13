import torch
import numpy as np
from geometry.code.lib.smplx import SMPLX
import pickle as pkl

class SMPLXServer(torch.nn.Module):
    def __init__(
            self,
            gender='male',
            betas=None,
            v_template=None,
            use_pca=False,
            num_pca_comps=12,
            flat_hand_mean=False,model_path=None):
        super().__init__()

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps if use_pca else 45
        self.flat_hand_mean = flat_hand_mean
        self.model = SMPLX(
            model_path=model_path,
            gender=gender,
            batch_size=1,
            num_pca_comps=num_pca_comps,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            dtype=torch.float32).cuda()

        self.bone_parents = self.model.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        for i in range(len(self.bone_parents)):
            self.bone_ids.append([self.bone_parents[i], i])

        if v_template != None:
            self.v_template = torch.tensor(v_template).float().cuda()
        else:
            self.v_template = None

        if betas is not None:
            self.betas = torch.tensor(betas).float().cuda()
        else:
            self.betas = None

        # define the canonical pose
        param_canonical = torch.zeros((1, 99 + 2 * self.num_pca_comps),
                                          dtype=torch.float32).cuda()
        
        param_canonical[0, 0] = 1
        param_canonical[0, 9] = np.pi / 6
        param_canonical[0, 12] = -np.pi / 6
        param_canonical[0, 76 + 2 * self.num_pca_comps] = 0.2
        
        if flat_hand_mean == False:
            param_canonical[
                0, 70:115] = -self.model.left_hand_mean.unsqueeze(0)
            param_canonical[
                0, 115:160] = -self.model.right_hand_mean.unsqueeze(0)
        if self.betas is not None and self.v_template is None:
            param_canonical[0, -20:-10] = self.betas

        output = self.forward(param_canonical, absolute=True)

        self.verts_c = output['smpl_verts']
        self.joints_c = output['smpl_jnts']
        self.tfs_c_inv = output['smpl_tfs'].squeeze(0).inverse()
        self.pose_feature_c = output['smpl_pose_feature']
        self.expr_c = output['smpl_expr']


    def forward(self, model_params, absolute=False):
        """return SMPLX output from params

        Args:
            model_params : smplx parameters. shape: [B, 99+2*num_pca_comps].
            absolute (bool): if true return smpl_tfs wrt thetas=0. else wrt thetas=thetas_canonical. 

        Returns:
            smpl_verts: vertices. shape: [B, 10475. 3]
            smpl_tfs: bone transformations. shape: [B, 55, 4, 4]
            smpl_jnts: joint positions. shape: [B, 56, 3]
        """

        output = {}
        scale, transl, global_orient, body_pose, left_hand_pose, right_hand_pose, leye_pose, reye_pose,\
            jaw_pose, betas, expression = torch.split(
            model_params, [1, 3, 3, 63, self.num_pca_comps, self.num_pca_comps, 3, 3, 3, 10, 10], dim=1)
        
        # ignore betas if v_template is provided
        if self.v_template is not None:
            betas = torch.zeros_like(betas)
            expression = torch.zeros_like(expression)

        smpl_output = self.model.forward(betas=betas,
                                         expression=expression,
                                         transl=torch.zeros_like(transl),
                                         global_orient=global_orient,
                                         body_pose=body_pose,
                                         left_hand_pose=left_hand_pose,
                                         right_hand_pose=right_hand_pose,
                                         leye_pose=leye_pose,
                                         reye_pose=reye_pose,
                                         jaw_pose=jaw_pose,
                                         return_verts=True,
                                         return_full_pose=True,
                                         v_template=self.v_template)

        verts = smpl_output.vertices.clone()
        output['smpl_verts'] = verts * scale.unsqueeze(1) + transl.unsqueeze(1)

        joints = smpl_output.joints.clone()
        output['smpl_jnts'] = joints * scale.unsqueeze(1) + transl.unsqueeze(1)

        tf_mats = smpl_output.T.clone()
        tf_mats[:, :, :3, :] *= scale.unsqueeze(1).unsqueeze(1)
        tf_mats[:, :, :3, 3] += transl.unsqueeze(1)

        if not absolute:
            tf_mats = torch.einsum('bnij,njk->bnik', tf_mats, self.tfs_c_inv)

        output['smpl_tfs'] = tf_mats
        output['smpl_pose_feature'] = smpl_output.pose_feature.clone()
        output['smpl_expr'] = smpl_output.expression.clone()

        return output
    
class SMPLXModel(torch.nn.Module):
    def __init__(self,smpl_model_path,meta_info_path):
        super().__init__()
        meta_info = self.load_meta_info(meta_info_path)
        self.meta_info = meta_info
        gender = str(meta_info['gender']) if 'gender' in meta_info else None
        betas = meta_info['betas'] if 'betas' in meta_info else None
        use_pca = meta_info['use_pca'] if 'use_pca' in meta_info else None
        num_pca_comps = meta_info['num_pca_comps'] if 'num_pca_comps' in meta_info else None
        flat_hand_mean = meta_info['flat_hand_mean'] if 'flat_hand_mean' in meta_info else None
        self.smpl_server = SMPLXServer(gender=gender,
                                    betas=betas,
                                    use_pca=use_pca,
                                    num_pca_comps=num_pca_comps,
                                    flat_hand_mean=flat_hand_mean,model_path=smpl_model_path)
        for param in self.smpl_server.parameters():
            param.requires_grad = False
            
    def load_meta_info(self,meta_info_path):
        meta_info = np.load(meta_info_path, allow_pickle=True)
        return meta_info
    
    def load_smplx_data(self,smplx_path):
        betas = self.meta_info['betas']
        num_hand_pose = self.meta_info['num_pca_comps'].item() if self.meta_info['use_pca'].item() else 45
        f = pkl.load(open(smplx_path, 'rb'), encoding='latin1')
        smplx_params = np.zeros(99+2*num_hand_pose)
        smplx_params[0] = 1
        smplx_params[1:4] = f['transl']
        smplx_params[4:7] = f['global_orient']
        smplx_params[7:70] = f['body_pose']
        smplx_params[70:70+num_hand_pose] = f['left_hand_pose']
        smplx_params[70+num_hand_pose:70+2*num_hand_pose] = f['right_hand_pose']
        smplx_params[70+2*num_hand_pose:73+2*num_hand_pose] = np.zeros(3)
        smplx_params[73+2*num_hand_pose:76+2*num_hand_pose] = np.zeros(3)
        smplx_params[76+2*num_hand_pose:79+2*num_hand_pose] = f['jaw_pose']
        smplx_params[79+2*num_hand_pose:89+2*num_hand_pose] = betas
        smplx_params[89+2*num_hand_pose:99+2*num_hand_pose] = f['expression']
        smplx_params= torch.tensor(smplx_params).unsqueeze(0).float().cuda()
        return smplx_params
    
    def forward(self,smplx_params):
        smplx_data = self.smpl_server.forward(smplx_params, absolute=False)
        smplx_tfs = smplx_data['smpl_tfs']
        
        smplx_thetas = smplx_params[:, 7:70]
        smplx_exps = smplx_params[:, -10:]
        cond = torch.cat([smplx_thetas / np.pi, smplx_exps], dim=-1)
        return smplx_tfs, cond

