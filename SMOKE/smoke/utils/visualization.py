import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image
def test_module(arg1):
    print("Test Complete")

def compute_birdviewbox(predictions_list, shape, scale):
    #npline = [np.float64(line[i]) for i in range(1, len(line))]
    h = predictions_list[7] * scale
    w = predictions_list[8] * scale
    l = predictions_list[9] * scale
    x = predictions_list[10] * scale
    y = predictions_list[11] * scale
    z = predictions_list[12] * scale
    rot_y = predictions_list[13]

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
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0,:]))

def draw_birdeyes(ax2, line_p, shape):
    # shape = 900
    scale = 15
    pred_corners_2d = compute_birdviewbox(line_p, shape, scale)
    
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
    ax2.add_patch(p)