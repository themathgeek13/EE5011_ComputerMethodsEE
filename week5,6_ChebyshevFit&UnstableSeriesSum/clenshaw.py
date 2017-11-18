#Clenshaw Algorithm:
#S(x)=sum(0 to n, a_k*phi_k(x))
#phi_(k+1)(x)=alpha_k(x)*phi_k(x)+beta_k(x)*phi_(k-1)(x)
import numpy as np
import scipy.special

def alpha(i,x):
    return 2*x

def beta(i,x):
    return -1

def phi(i,x):
    if(i==0):
        return 1
    if(i==1):
        return x

def a(i):
    return coeff[i]

def S(phi,alpha,beta,a,x,n):
    b=np.zeros(n+2)
    b[-1]=b[-2]=0
    for i in range(n-1,-1,-1):
        b[i]=a(i)+alpha(i,x)*b[i+1]+beta(i+1,x)*b[i+2]
    return phi(0,x)*a(0)+phi(1,x)*b[0]+beta(1,x)*phi(0,x)*b[1]
