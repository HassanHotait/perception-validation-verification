
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.patches as patches
from matplotlib.path import Path
import cv2
import matplotlib

class GtInfo():
    def __init__(self, line):
        self.name = line[0]


        self.truncation = float(line[1])
        self.occlusion = float(line[2])

        # local orientation = alpha + pi/2
        self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

        self.labels_list=[self.name,self.truncation,self.occlusion,
                self.alpha,
                self.xmin,self.ymin,self.xmax,self.ymax,
                self.h,self.w,self.l,
                self.tx,self.ty,self.tz,
                self.rot_global]

    def get_labels_list(self):

        self.labels_list=[self.name,self.truncation,self.occlusion,
                        self.alpha,
                        self.xmin,self.ymin,self.xmax,self.ymax,
                        self.h,self.w,self.l,
                        self.tx,self.ty,self.tz,
                        self.rot_global]





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
class SMOKE_Viz(object):

    def __init__(self, lat_range_m,long_range_m,scale):
        self.shape=(lat_range_m*scale,long_range_m*scale)
        self.scale=scale
        self.fig = plt.figure(figsize=(20.00, 5.12), dpi=100)
        # fig.tight_layout()
        gs = GridSpec(1, 4)
        gs.update(wspace=0.25)  # set the spacing between axes.

        self.ax = self.fig.add_subplot(gs[0, :3])
        self.ax2 = self.fig.add_subplot(gs[0, 3:])

        # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):

        self.birdimage = np.zeros((self.shape[0] ,self.shape[1], 3), np.uint8)

        # plot camera view range
        x1 = np.linspace(-self.shape[0]/2, 0)
        x2 = np.linspace(0, self.shape[0]/2)
        self.ax2.plot(x1, - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
        self.ax2.plot(x2, x2, ls='--', color='grey', linewidth=1, alpha=0.5)
        self.ax2.plot(0, 0.2, marker='+', markersize=16, markeredgecolor='red')

    def compute_3Dbox(self,P2, obj):
        #obj=gt_obj
        # if gt_list==None:
        #obj = GtInfo(line)
        # else:
        #     obj=detectionInfo(gt_list)
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

    def draw_3Dbox( self,image,P2, predictions_list=None,gt_list=None):
        b,g,r=cv2.split(image)
        rgb_frame=cv2.merge([r,g,b])

        if predictions_list==None and gt_list==None:
            raise print("No Input Object Provided")

        if predictions_list!=None:
            obj_list=predictions_list
            color='red'
        else:
            obj_list=gt_list
            color='green'

        for obj in obj_list:

            #obj=GtInfo(line[i])
            corners_2D = self.compute_3Dbox(P2, obj)

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
            self.ax.add_patch(p)
            self.ax.add_patch(front_fill)
            self.ax.imshow(rgb_frame)


    def compute_birdviewbox(self,obj):

        scale=10
        
        #npline = [np.float64(line[i]) for i in range(len(line))]
        #obj = GtInfo(line)
        #print('npline: ',npline)
        #obj = detectionInfo(line)
        #print('Number of items in npline: ',len(npline))
        h = obj.h * scale
        w = obj.w * scale
        l = obj.l * scale
        x = obj.tx * scale
        y = obj.ty * scale
        z = obj.tz * scale
        rot_y = obj.rot_global

        print('Project Angle [Deg]: ',np.degrees(rot_y))

        #print('(X,Z): ',(x,z))

        R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                    [np.sin(rot_y), np.cos(rot_y)]])
        t = np.array([x, z]).reshape(1, 2).T


        # x_corners = [-(w/2), l-(w/2), l-(w/2), -(w/2)]  # -l/2
        # z_corners = [w-(l/2), w-(l/2), -(l/2), -(l/2)]  # -w/2

        x_corners = [-l/2, l/2, l/2, -l/2]  # -l/2
        z_corners = [w/2, w/2, -w/2, -w/2]  # -w/2

        # x_corners = [0, l, l, 0]  # -l/2
        # z_corners = [w, w, 0, 0]  # -w/2

        # x_corners = [i - obj.l / 2 for i in x_corners]
        # z_corners = [i - obj.w / 2 for i in z_corners]


        # x_corners += -w / 2
        # z_corners += -l / 2

        # bounding box in object coordinate
        corners_2D = np.array([x_corners, z_corners])
        # rotate
        corners_2D = R.dot(corners_2D)
        # translation
        corners_2D = t - corners_2D
        # in camera coordinate
        #corners_2D[0] += int(self.shape[0]/2)
        corners_2D = (corners_2D).astype(np.int16)
        corners_2D = corners_2D.T

        print('Corners 2D: ',corners_2D)

        return np.vstack((corners_2D, corners_2D[0,:])),rot_y

    def draw_birdeyes(self,  obj_list):
        shape = self.shape

        #for i in range(len(gt_list)):
        for obj in obj_list:

            if obj.name=='DontCare':
                continue
            # print('line_p[i]: ',gt_list[i])

            #obj=GtInfo(gt_list[i])
            pred_corners_2d,rot = self.compute_birdviewbox(obj)
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
            print('Patch: ',p)


            # if i==0:
            #     t2 = matplotlib.transforms.Affine2D().rotate_deg(np.degrees(rot)) + self.ax2.transData
            # else:
            #     t2 = matplotlib.transforms.Affine2D().rotate_deg(np.degrees(rot)) - self.ax2.transData
            # p.set_transform(t2)


            self.ax2.add_patch(p)

            self.ax2.imshow(self.birdimage,extent=[-shape[0]/2, shape[0]/2, 0, shape[1]])
            self.ax2.set_yticks(range(0,shape[1],self.scale))
            self.ax2.set_xticks(range(-int(shape[0]/2),int(shape[0]/2),self.scale))

    def show(self):
        plt.show()


