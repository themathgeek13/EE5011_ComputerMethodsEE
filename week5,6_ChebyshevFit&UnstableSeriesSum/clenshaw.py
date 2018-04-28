#Clenshaw Algorithm:
#S(x)=sum(0 to n, c_k*F_k(x))
#F_(n+1)(x)=alpha(n,x)*F_n(x)+beta(n,x)*F_(n-1)(x)
import numpy as np
import scipy.special

def alpha(i,x):
    #return x/(2*(i+1))
    return 2*i/x

def beta(i,x):
    #return x**2/(2*(i+1)*i)
    return -1

def F(i,x):
    if(i==0):
        return scipy.special.jn(0,x)
    if(i==1):
        return scipy.special.jn(1,x)

def a(i):
    #return coeff[i]
    return 1/(i+1)

def S(F,alpha,beta,a,x,n):
    b=np.zeros(n+3)
    b[-1]=b[-2]=0
    for i in range(n,0,-1):
        b[i]=a(i)+alpha(i,x)*b[i+1]+beta(i+1,x)*b[i+2]
    print(F(0,x)*a(0), F(1,x)*b[1], beta(1,x)*F(0,x)*b[2])
    return F(0,x)*a(0)+F(1,x)*b[1]+beta(1,x)*F(0,x)*b[2]
