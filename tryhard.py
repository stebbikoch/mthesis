import numpy as np

def func(p, d):
    output = 0
    degree = 0
    for i in range(-d, d+1):
        for j in range(-d, d+1):
            for k in range(-d, d+1):
                if not (i==0 and j==0 and k==0):
                    degree += 1
                    output += np.exp(1j*np.pi*2*np.dot(np.array([i,j,k]), p))
    print(degree)
    print(output)

if __name__=='__main__':
    #print(np.sin(np.pi/2))
    #d = 6
    #func(np.array([1, 0, 0])/(2*d+1), d)
    print('Hello World!')