import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy import stats

x = np.linspace(0, 1, 200)
pdfx = stats.beta(2, 5).pdf(x)

path = Path(np.array([x, pdfx]).transpose())
patch = PathPatch(path, facecolor='none', alpha=0)
plt.gca().add_patch(patch)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ["#00bfbf", "#00bfbf", "#126a8a"])

im = plt.imshow(np.array([[1, 0, 0], [1, 1, 0]]),
                cmap=cmap, interpolation='bicubic',
                origin='lower', extent=[0, 1, 0.0, 5],
                aspect="auto", clip_path=patch, clip_on=True)

plt.axis('off')
plt.ylim(0, 5.5)
plt.xlim(0, 0.9)

bbox = Bbox([[0.75, 0.5], [5.4, 2.2]])

#plt.savefig('logo_00.png', dpi=300,  bbox_inches=bbox, transparent=True)
plt.text(x=0.04, y=-0.01, s='ArviZ',
         fontdict={'name': 'ubuntu mono', 'fontsize': 62}, color='w')

plt.savefig('ArviZ.png', dpi=300, bbox_inches=bbox, transparent=True)
plt.savefig('ArviZ.pdf', dpi=300, bbox_inches=bbox, transparent=True)
plt.savefig('ArviZ.svg', dpi=300, bbox_inches=bbox, transparent=True)
plt.savefig('ArviZ.eps', dpi=70, bbox_inches=bbox, transparent=True)
plt.savefig('ArviZ.jpg', dpi=300, bbox_inches=bbox, transparent=True)
