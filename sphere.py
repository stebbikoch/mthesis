from mpl_toolkits import mplot3d
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import *
#from numba import njit

class eq_partition_alg:
    def __init__(self, N):
        self.N = N
        self.V = 4 * np.pi

    @property
    def theta_c(self):
        """
        colatitude of polar caps
        :return:
        """
        return np.arccos(1-self.V/(2*np.pi*self.N))

    @property
    def theta_l(self):
        """
        ideal collar angle
        :return:
        """
        return np.sqrt(self.V/self.N)

    @property
    def ideal_num_collars(self):
        return (np.pi-2*self.theta_c)/self.theta_l

    @property
    def real_num_collars(self):
        return int(max(1,np.round(self.ideal_num_collars)))

    @property
    def ideal_numb_regions_per_colar(self):
        real_colar_angle=(np.pi-2*self.theta_c)/self.real_num_collars
        output = []
        theta_A = self.theta_c
        for i in range(self.real_num_collars):
            theta_B = theta_A+real_colar_angle
            V_collar = 2*np.pi*np.abs((np.cos(theta_A)-np.cos(theta_B)))
            output.append(V_collar/(self.V/self.N))
            theta_A = theta_B
        return output

    @property
    def actual_numb_regions_per_colar(self):
        m_in = self.ideal_numb_regions_per_colar
        m_out = []
        alpha=0
        for i in range(self.real_num_collars):
            m_out.append(round(m_in[i]+alpha))
            alpha+=m_in[i]-m_out[i]
            #print(alpha)
        #print('ideal {}, \n actual {}'.format(m_in, m_out))
        assert sum(m_out)==self.N-2, 'problem!!! sum of m_out is {}'.format(sum(m_out))
        return m_out

    @property
    def actual_colatitudes(self):
        m = self.actual_numb_regions_per_colar
        thetaborders=[]
        thetaborders.append(self.theta_c)
        for i in range(1,self.real_num_collars):
            next_border = np.arccos(np.cos(thetaborders[i - 1]) - (self.V / self.N * m[i-1]) / (2 * np.pi))
            thetaborders.append(next_border)
        return thetaborders

    def grid(self):
        m = self.actual_numb_regions_per_colar
        thetas = self.actual_colatitudes
        thetas.append(np.pi-self.theta_c)
        phis_out = np.zeros(self.N)
        thetas_out = np.zeros(self.N)
        phis_out[0]=0
        phis_out[-1]=0
        thetas_out[0]=0
        thetas_out[-1]=np.pi
        count=0
        for i in range(len(m)):
            for j in range(m[i]):
                phis_out[1+count]=(j)*2*np.pi/m[i]
                thetas_out[1+count]=(thetas[i]+thetas[1+i])/2
                count+=1
        #print(thetas_out, phis_out)
        return phis_out, thetas_out

class fibonacci_sphere:
    def __init__(self, N, random=False, eq_partition=False):
        np.random.seed(1)
        self.N = N
        if random:
            self.phi = np.zeros(N)
            self.theta = np.zeros(N)
            for i in range(N):
                again=True
                while again:
                    self.phi[i] = np.random.random()*2*np.pi
                    sign = np.random.randint(0,2)
                    #print(sign)
                    theta = np.arccos(np.random.random())
                    self.theta[i] = theta
                    self.theta[i] = np.abs(sign*np.pi-theta)
                    if i>0:
                        arg = 2 - 2 * (np.sin(self.theta[i]) * np.sin(self.theta[:i]) * np.cos(self.phi[i] - self.phi[:i]) \
                                   + np.cos(self.theta[i]) * np.cos(self.theta[:i]))
                        #print(arg)
                        if np.min(np.sqrt(arg))<np.sqrt(4*np.pi/N)*0.75:
                            again=True
                        else:
                            again=False
                    else:
                        again=False
            print(self.theta[-1], self.phi[-1])

        elif eq_partition:
            instance = eq_partition_alg(self.N)
            self.phi, self.theta = instance.grid()
        else:
            # golden ratio
            Phi = (1 + 5 ** 0.5) / 2
            i = np.arange(N)
            self.phi = 2 * np.pi * i / Phi
            self.theta = np.arccos(1 - 2 * (i + 0.5) / N)
            #print(self.theta, self.phi)


    def spherical_to_cartesian(self):
        x = np.cos(self.phi)*np.sin(self.theta)
        y = np.sin(self.phi)*np.sin(self.theta)
        z = np.cos(self.theta)
        return x, y, z

    def wiring(self, d_0):
        edges = np.zeros((self.N*(self.N-1), 2))
        counter = 0
        for i in range(self.N):
            arg = 2 - 2 * (np.sin(self.theta[i]) * np.sin(self.theta) * np.cos(self.phi[i] - self.phi)  \
                           + np.cos(self.theta[i]) * np.cos(self.theta))
            assert np.all(arg>-1e-8), "negative distances occuring!!! For example {}".format(arg[arg<-1e-8])
            arg[arg<0]=0
            distances = np.sqrt(arg)
            indices = (np.where((distances<= d_0) & (distances>0))[0]).astype(int)
            edges[counter:counter+len(indices)]=np.array([np.ones(len(indices))*i, indices]).T
            counter += len(indices)
        edges = edges[:counter].astype(int)
        # now build regular sparse matrix
        L_0 = csr_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])))
        L_0 += csr_matrix((-np.ones(len(edges)), (edges[:, 0], edges[:, 0])))
        self.L_0 = L_0
        #print(L_0.diagonal())
        self.k = abs(np.mean(L_0.diagonal()))
        print('average k {} for r_0 {}'.format(self.k, d_0))
        self.N_tot = self.N
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
    al = fibonacci_sphere(1000, random=True, eq_partition=False)
    #for r_0 in [0.2, 0.3, 0.5, 1, 1.3, 1.7]:
     #   a=al.wiring(r_0)
      #  print('k :', abs(np.mean(al.L_0.diagonal())))
    x, y, z = al.spherical_to_cartesian()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, s=2)
    #ax.set_xlim3d(-1, 1)
    #ax.set_ylim3d(-1,1)
    #ax.set_zlim3d(-1,1)
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    ax.set_axis_off()
    #ax.view_init(elev=13, azim=-66)
    plt.tight_layout()
    #plt.savefig('figures/2sphere_fibo.svg', format='svg', dpi=1000)
    plt.show()


