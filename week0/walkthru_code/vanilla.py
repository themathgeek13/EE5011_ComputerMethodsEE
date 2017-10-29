from pylab import *
N=50
M=100
c=1.0/(1.0+arange(N+1)**2)
x=linspace(0,10,M)
def fourier(N,c,x):
	z=zeros(x.shape) # create and initialize z
	for i in range(M):
		zz=0.0;xx=x[i]
		for k in range(N+1):
			zz += c[k]*cos(k*xx)
		z[i]=zz
	return(z)
