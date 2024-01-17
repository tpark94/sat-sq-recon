'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha Park (tpark94@stanford.edu)
'''

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from matplotlib import pyplot as plt

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
CAMERA = dict(x=0.75, y=0.75, z=0.5)

def denormalize(image):
    std   = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1)
    mean  = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1)
    image = image.cpu() * std + mean

    return image


def imshow(image, is_tensor=False, is_mask=False, savefn=None):
    # image: torch.tensor (C x H x W)

    # image to numpy.array
    if is_tensor:
        if not is_mask:
            image = denormalize(image)
            image = image.mul(255).clamp(0,255).permute(1,2,0).byte().numpy()
        else:
            image  = image.mul(255).clamp(0,255).byte().cpu().numpy()

    # plot
    plt.imshow(image)
    plt.axis('off')
    if savefn is not None:
        plt.savefig(savefn, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def plot_3dmesh(meshes, markers_for_vertices=True, add_line=False, groundtruth=False, savefn=None):
    if not isinstance(meshes, list):
        meshes = [meshes]

    data = []

    # palette = px.colors.qualitative.Pastel
    palette = px.colors.qualitative.Safe

    # Mesh data
    for i in range(len(meshes)):
        v, f = meshes[i].get_mesh_verts_faces(0) # [V x 3], [F x 3]

        data.append(
            go.Mesh3d(
                x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2],
                opacity=1.0,
                color='gray' if groundtruth else palette[i]
            )
        )

        if markers_for_vertices:
            tri_points = np.array([v[i] for i in f.reshape(-1)])
            Xe, Ye, Ze = tri_points.T

            if add_line:
                mode = 'lines+markers'
                line_dict = dict(color='black', width=2)
            else:
                mode = 'markers'
                line_dict = None

            data.append(
                go.Scatter3d(
                    x=Xe, y=Ye, z=Ze, mode=mode,
                    line=line_dict,
                    marker=dict(
                        size=5.0, opacity=1.0, color='black'
                    )
                )
            )

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            scene=dict(aspectmode='cube'),
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            margin=go.layout.Margin(
                l=0, r=0, b=0, t=0
            ),

        )
    )

    ax_lim = 0.6
    tick_size = 14
    font_size = 28
    axis_visible = False
    fig.update_scenes(
        xaxis_visible=axis_visible,
        yaxis_visible=axis_visible,
        zaxis_visible=axis_visible,
        xaxis_range=[-ax_lim, ax_lim],
        yaxis_range=[-ax_lim, ax_lim],
        zaxis_range=[-ax_lim, ax_lim],
        xaxis_tickfont_size=tick_size,
        yaxis_tickfont_size=tick_size,
        zaxis_tickfont_size=tick_size,
        xaxis_titlefont_size=font_size,
        yaxis_titlefont_size=font_size,
        zaxis_titlefont_size=font_size,
        camera=dict(eye=CAMERA)
    )

    if savefn:
        fig.write_image(savefn)
    else:
        fig.show()


def plot_3dpoints(pcls, groundtruth=False, savefn=None):
    if not isinstance(pcls, list):
        pcls = [pcls]

    data = []

    palette = px.colors.qualitative.Safe

    for i, p in enumerate(pcls):
        points = p.points_padded()[0]

        data.append(
            go.Scatter3d(
                x=points[..., 0], y=points[..., 1], z=points[..., 2], mode='markers',
                marker=dict(
                    size=5.0, opacity=1.0,
                    color='black' if groundtruth else palette[i]
                )
            )
        )

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            scene=dict(aspectmode='cube'),
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            margin=go.layout.Margin(
                l=0, r=0, b=0, t=0
            ),
            showlegend=False
        )
    )

    ax_lim = 0.6
    tick_size = 14
    font_size = 28
    axis_visible = False
    fig.update_scenes(
        xaxis_visible=axis_visible,
        yaxis_visible=axis_visible,
        zaxis_visible=axis_visible,
        xaxis_range=[-ax_lim, ax_lim],
        yaxis_range=[-ax_lim, ax_lim],
        zaxis_range=[-ax_lim, ax_lim],
        xaxis_tickfont_size=tick_size,
        yaxis_tickfont_size=tick_size,
        zaxis_tickfont_size=tick_size,
        xaxis_titlefont_size=font_size,
        yaxis_titlefont_size=font_size,
        zaxis_titlefont_size=font_size,
        camera=dict(eye=CAMERA)
    )

    if savefn:
        fig.write_image(savefn)
    else:
        fig.show()


def plot_occupancy_labels(points, occupancy, savefn=None):

    # Inside
    inside  = points[occupancy == 1]
    outside = points[occupancy == 0]
    data = [
        go.Scatter3d(
            x=inside[..., 0], y=inside[..., 1], z=inside[..., 2], mode='markers',
            marker=dict(
                size=2.0, opacity=1.0, color='black'
            )
        ),
        go.Scatter3d(
            x=outside[..., 0], y=outside[..., 1], z=outside[..., 2], mode='markers',
            marker=dict(
                size=2.0, opacity=0.1, color='red'
            )
        )
    ]

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            scene=dict(aspectmode='cube'),
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            margin=go.layout.Margin(
                l=0, r=0, b=0, t=0
            ),
            showlegend=False
        )
    )

    ax_lim = 0.6
    tick_size = 14
    font_size = 28
    axis_visible = False
    fig.update_scenes(
        xaxis_visible=axis_visible,
        yaxis_visible=axis_visible,
        zaxis_visible=axis_visible,
        xaxis_range=[-ax_lim, ax_lim],
        yaxis_range=[-ax_lim, ax_lim],
        zaxis_range=[-ax_lim, ax_lim],
        xaxis_tickfont_size=tick_size,
        yaxis_tickfont_size=tick_size,
        zaxis_tickfont_size=tick_size,
        xaxis_titlefont_size=font_size,
        yaxis_titlefont_size=font_size,
        zaxis_titlefont_size=font_size,
        camera=dict(eye=CAMERA)
    )

    if savefn:
        fig.write_image(savefn)
    else:
        fig.show()



def plot_3dpoints_on_mesh(points, mesh):

    data = []

    data.append(
        go.Scatter3d(
            x=points[..., 0], y=points[..., 1], z=points[..., 2], mode='markers',
            marker=dict(
                size=2.0, opacity=1.0, color='black'
            )
        )
    )

    v, f = mesh.get_mesh_verts_faces(0) # [V x 3], [F x 3]

    data.append(
        go.Mesh3d(
            x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2],
            opacity=0.25
        )
    )

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            scene=dict(aspectmode='data')
        )
    )

    fig.show()