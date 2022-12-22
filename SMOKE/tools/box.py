import numpy as np
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class detectionInfo(object):
    def __init__(self, line):
        self.name = line[0]


        # self.truncation = float(line[1])
        # self.occlusion = int(line[2])

        # # local orientation = alpha + pi/2
        # self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[2])
        self.ymin = float(line[3])
        self.xmax = float(line[4])
        self.ymax = float(line[5])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[6])
        self.w = float(line[7])
        self.l = float(line[8])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[9])
        self.ty = float(line[10])
        self.tz = float(line[11])

        # global orientation [-pi, pi]
        self.rot_global = float(line[12])







def compute_3Dbox(P2, line):
    obj = detectionInfo(line)
    # Draw 2D Bounding Box
    xmin = int(obj.xmin)
    xmax = int(obj.xmax)
    ymin = int(obj.ymin)
    ymax = int(obj.ymax)
    # width = xmax - xmin
    # height = ymax - ymin
    # box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
    # ax.add_patch(box_2d)

    # Draw 3D Bounding Box

    R = np.array([[np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                  [0, 1, 0],
                  [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)]])

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
    z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    print("Corners 3D_1 : ",corners_3D_1)
    print("Corners 3D_1 Shape: ",corners_3D_1.shape)
    print("P2: ",P2)
    print("P2 Shape: ",P2.shape)
    corners_2D = P2.dot(corners_3D)
    print("Dot Product Output: ",corners_2D)
    print("Dot Product Output Shape: ",corners_2D.shape)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]
    print("corners 2D Final Version: ",corners_2D)

    return corners_2D

def draw_3Dbox( P2, line, color,ax):

    for i in range(len(line)):

        corners_2D = compute_3Dbox(P2, line[i])

        # draw all lines through path
        # https://matplotlib.org/users/path_tutorial.html
        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
        bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
        verts = bb3d_on_2d_lines_verts.T
        codes = [Path.LINETO] * verts.shape[0]
        codes[0] = Path.MOVETO
        # codes[-1] = Path.CLOSEPOLYq
        pth = Path(verts, codes)
        p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

        width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
        height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
        # put a mask on the front
        front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
        ax.add_patch(p)
        ax.add_patch(front_fill)


def compute_birdviewbox(line, shape, scale):
    
    npline = [np.float64(line[i]) for i in range(len(line))]
    #print('npline: ',npline)
    #obj = detectionInfo(line)
    #print('Number of items in npline: ',len(npline))
    h = npline[6] * scale
    w = npline[7] * scale
    l = npline[8] * scale
    x = npline[9] * scale
    y = npline[10] * scale
    z = npline[11] * scale
    rot_y = npline[12]
    #print('(X,Z): ',(x,z))

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2


    x_corners += -w / 2
    z_corners += -l / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    #corners_2D[0] += int(shape[0]/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0,:]))

def draw_birdeyes(ax2, birdimage, line_p, shape):
    # shape = 900
    scale = 1

    for i in range(len(line_p)):

        pred_corners_2d = compute_birdviewbox(line_p[i], shape, scale)
        #gt_corners_2d = compute_birdviewbox(line_gt, shape, scale)

        # codes = [Path.LINETO] * gt_corners_2d.shape[0]
        # codes[0] = Path.MOVETO
        # codes[-1] = Path.CLOSEPOLY
        # pth = Path(gt_corners_2d, codes)
        # p = patches.PathPatch(pth, fill=False, color='orange', label='ground truth')
        # ax2.add_patch(p)

        codes = [Path.LINETO] * pred_corners_2d.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        pth = Path(pred_corners_2d, codes)
        p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
        #print('Patch: ',p)
        ax2.add_patch(p)
        ax2.imshow(birdimage)


def visualize(shape,predictions_list,P2,image):
    fig = plt.figure(figsize=(20.00, 5.12), dpi=100)
    # fig.tight_layout()
    gs = GridSpec(1, 4)
    gs.update(wspace=0.25)  # set the spacing between axes.

    ax = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 3:])

    # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):

    birdimage = np.zeros((shape[0] ,shape[1], 3), np.uint8)

    # plot camera view range
    x1 = np.linspace(-shape[0]/2, 0)
    x2 = np.linspace(0, shape[0]/2)
    ax2.plot(x1, - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
    ax2.plot(x2, x2, ls='--', color='grey', linewidth=1, alpha=0.5)
    ax2.plot(0, 0.2, marker='+', markersize=16, markeredgecolor='red')


    draw_3Dbox(P2,predictions_list,'green',ax)
    draw_birdeyes(ax2,birdimage,predictions_list,shape)

    # visualize 3D bounding box
    ax.imshow(image,animated=True)
    # for track in tracker.tracks:
    #     bbox = track.to_tlbr()
    #     ax.text(int(bbox[0]), int(bbox[1]),'ID: '+str(track.track_id),color='green')
    ax.set_xticks([]) #remove axis value
    ax.set_yticks([])

    # visualize bird eye view
    ax2.imshow(birdimage, extent=[-shape[0]/2, shape[0]/2, 0, shape[1]],animated=True)
    # print('type of image',type(birdimage))
    # print('type of image 2',type(image))
    return fig,image,birdimage
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    
