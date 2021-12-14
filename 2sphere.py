from mpl_toolkits import mplot3d
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import *
#from numba import njit

class fibonacci_sphere:
    def __init__(self, N):
        self.N = N
        # golden ratio
        Phi = (1+5**0.5)/2
        i = np.arange(N)
        self.phi = 2*np.pi*i/Phi
        self.theta = np.arccos(1-2*(i+0.5)/N)

    def spherical_to_cartesian(self):
        x = np.cos(self.phi)*np.sin(self.theta)
        y = np.sin(self.phi)*np.sin(self.theta)
        z = np.cos(self.theta)
        return x, y, z

    def wiring(self, d_0):
        edges = np.zeros((self.N*(self.N-1), 2))
        counter = 0
        for i in range(self.N):
            distances = np.sqrt(
                2 - 2 * (np.sin(self.theta[i]) * np.sin(self.theta) * np.cos(self.phi[i] - self.phi) + np.cos(self.theta[i]) * np.cos(self.theta)))
            indices = (np.where((distances<= d_0) & (distances>0))[0]).astype(int)
            edges[counter:counter+len(indices)]=np.array([np.ones(len(indices))*i, indices]).T
            counter += len(indices)
        edges = edges[:counter].astype(int)
        # now build regular sparse matrix
        L_0 = csr_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])))
        L_0 += csr_matrix((-np.ones(len(edges)), (edges[:, 0], edges[:, 0])))
        self.L_0 = L_0
        return L_0


# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

if __name__=='__main__':
    al = fibonacci_sphere(100)
    a=al.wiring(1.9)
    print(a.toarray())
    plt.imshow(a.toarray())
    print(np.sum(a.toarray()[0]))
    plt.show()
    # x, y, z = al.spherical_to_cartesian()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(x, y, z, s=2)
    # #ax.set_xlim3d(-1, 1)
    # #ax.set_ylim3d(-1,1)
    # #ax.set_zlim3d(-1,1)
    # ax.set_box_aspect([1,1,1])
    # set_axes_equal(ax)
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.savefig('pictures/2sphere.svg', format='svg', dpi=1000)
    # plt.show()


