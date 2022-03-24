from scipy.special import legendre as lg
import numpy as np
from matplotlib import pyplot as plt

def f(l,x):
    output = (lg(l-1)(x)-lg(l+1)(x))/(2*l+1)
    return output

def l_approx(x):
    theta_0 = np.arccos(-x)
    l = np.round((2*theta_0/np.pi + 5)/(4*(1-theta_0/np.pi)))
    return l

def tangent1(x):
    #output = -lg(2)(-1/2)*(x+1/2)+1/5*(lg(1)(-1/2)-lg(3)(-1/2))
    output = 1/8*(x-1)
    return output

def tangent2(x):
    u = -0.505943
    v = 0.0202611475201925
    output = - lg(2)(u)*(x-u)+(1/5*(lg(1)(u)-lg(3)(u)))
    return output

if __name__=="__main__":
    x = np.arange(1000)/500-1
    #fig = plt.figure()
    fig,(ax1,ax2, ax3) = plt.subplots(nrows=3, sharex=True)#, constrained_layout=True)
    i=0
    n=11
    line1=[0 for i in range(n-1)]
    for l in range(1,n):
        line1[i], =ax1.plot(x, f(l,x), label=r'$l = {}$'.format(l))
        line2, =ax2.plot(x, lg(l)(x), color=line1[i].get_color())
        i+=1
    #ax1.plot(x, tangent2(x), '--')
    line3, =ax1.plot(x, tangent1(x), '--')
    # draw the stepfunction, for which l to choose
    ax3.step(x,l_approx(x), label=r'$\left \lfloor{\frac{2 \frac{\arccos{x}}{\pi}+5}{4(1-\frac{\arccos{x}}{\pi})}-0.5} \right \rfloor$')
    ax3.set_ylim(0,10.5)
    #legend0 = ax1.legend()
    legend1 = ax1.legend([line1[-1], line3], [r"$\frac{1}{2l+1}(P_{l-1}(x)-P_{l+1}(x))$", r"$\frac {1}{8}(x-1)$"], loc='upper left')
    #ax1.legend(loc=7)
    ax1.legend(line1, [r'$l={}$'.format(l) for l in range(1,n)], loc='upper right', bbox_to_anchor=(1.25,1))
    plt.subplots_adjust(right=0.8)
    #legend0= ax1.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    legend2=ax2.legend([line2], [r'$P_l(x)$'], loc='upper left')
    legend3=ax3.legend(loc='upper left')
    ax1.add_artist(legend1)
    ax3.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax2.set_ylabel(r'$y$')
    ax3.set_ylabel(r'$l$')
    #fig.tight_layout()
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.savefig('figures/legendrepolynomials.svg', format='svg', dpi=1000)
    plt.show()
