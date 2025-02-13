import torch
import trimesh
from typing import Tuple, Union
import sys
from torchvision import transforms
# TODO: does this support borders?
def calc_edges(
        faces: torch.Tensor,  # F,3 long - first face may be dummy with all zeros
        with_edge_to_face: bool = False,
        with_dummies=True
):
    """
    returns tuple of
    - edges E,2 long, 0 for unused, lower vertex index first
    - face_to_edge F,3 long
    - (optional) edge_to_face shape=E,[left,right],[face,side]

    o-<-----e1     e0,e1...edge, e0<e1
    |      /A      L,R....left and right face
    |  L /  |      both triangles ordered counter clockwise
    |  / R  |      normals pointing out of screen
    V/      |
    e0---->-o
    """

    F = faces.shape[0]

    # make full edges, lower vertex index first
    face_edges = torch.stack((faces, faces.roll(-1, 1)), dim=-1)  # F*3,3,2
    full_edges = face_edges.reshape(F * 3, 2)
    sorted_edges, _ = full_edges.sort(dim=-1)  # F*3,2 TODO min/max faster?

    # make unique edges
    edges, full_to_unique = torch.unique(input=sorted_edges, sorted=True, return_inverse=True, dim=0)  # (E,2),(F*3)
    E = edges.shape[0]
    face_to_edge = full_to_unique.reshape(F, 3)  # F,3

    if not with_edge_to_face:
        return edges, face_to_edge

    is_right = full_edges[:, 0] != sorted_edges[:, 0]  # F*3
    edge_to_face = torch.zeros((E, 2, 2), dtype=torch.long, device=faces.device)  # E,LR=2,S=2
    scatter_src = torch.cartesian_prod(torch.arange(0, F, device=faces.device),
                                       torch.arange(0, 3, device=faces.device))  # F*3,2
    edge_to_face.reshape(2 * E, 2).scatter_(dim=0, index=(2 * full_to_unique + is_right)[:, None].expand(F * 3, 2),
                                            src=scatter_src)  # E,LR=2,S=2
    if with_dummies:
        edge_to_face[0] = 0
    return edges, face_to_edge, edge_to_face  # =EF