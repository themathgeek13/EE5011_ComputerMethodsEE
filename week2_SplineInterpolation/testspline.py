from scipy import *
from matplotlib.pyplot import *
import weave

#define support code
with open("spline.c","r") as f:
	scode=f.read()
	
h=logspace(-3,0,16)
N=(2.0*pi)/h
err=zeros(h.shape)

for i in range(len(h)):
	x=linspace(0,2*pi,N[i])
	y=sin(x)
	n=int(N[i])
	xx=linspace(0,2*pi,10*n+1)
	y2=zeros(x.size)
	#y2=cos(x)
	u=zeros(x.size)
	yy=zeros(xx.size)
	code="""
	#include <math.h>
	int i;
	double xp;
	spline(x,y,n,cos(x[0]),cos(x[n-1]),y2,u);
	for(i=0; i<=10*n; i++){
		xp=xx[i];
		splint(x,y,y2,n,xp,yy+i);
	}
	"""
	weave.inline(code,["x","y","n","y2","u","xx","yy"],support_code=scode,extra_compile_args=["-g"],compiler="gcc")
	
	if i==0:
		figure(0)
		plot(x,y,'ro')
		plot(xx,yy,'g')
		title("Interpolated values and data points for n=%d" % N[i])
	z=abs(yy-sin(xx))
	err[i]=z.max()

figure(1)
loglog(h,err)
title("Error vs. spacing")
show()
