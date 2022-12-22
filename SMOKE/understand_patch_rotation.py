import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

fig = plt.figure()
ax = fig.add_subplot(111)




for i in range(4):
    blue_patch= patches.Rectangle((15*i,0), 10, 20, color="blue", alpha=0.50)
    red_patch = patches.Rectangle((15*i,0), 10, 20, color="red",  alpha=0.50)

    t2 = mpl.transforms.Affine2D().rotate_deg(45) + ax.transData
    red_patch.set_transform(t2)

    #ax.add_patch(blue_patch)
    ax.add_patch(red_patch)




plt.xlim(-20, 80)
plt.ylim(-20, 60)

plt.grid(True)

plt.show()