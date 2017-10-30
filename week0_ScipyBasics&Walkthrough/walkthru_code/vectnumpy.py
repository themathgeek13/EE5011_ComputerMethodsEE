from pylab import *
N=50
M=100
c=1.0/(1.0+arange(N+1)**2)
x=linspace(0,10,M)
def fouriervm(N,c,x):
	A=cos(outer(x,arange(N+1)))
	return (dot(A,c))
