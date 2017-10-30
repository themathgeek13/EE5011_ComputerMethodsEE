from pylab import *
N=50
M=100
c=1.0/(1.0+arange(N+1)**2)
x=linspace(0,10,M)
def fourierv(N,c,x):
	z=zeros(x.shape) # create and initialize z
	for k in range(N+1):
		z += c[k]*cos(k*x)
	return(z)
