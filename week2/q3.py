from scipy import *
from scipy import special
from matplotlib.pyplot import *
import weave
​
def func(x):
	num=pow(x,1+special.jn(0,x))
	densqr=(1+100*x*x)*(1-x)
	den=sqrt(densqr)
	return num/den
​
#define support code
with open("spline.c","r") as f:
	scode=f.read()
	
h=logspace(-4,-2,20)
N=(0.8)/h
print N
err=zeros(h.shape)
figure(0)
for i in range(len(h)):
	x=linspace(0.1,0.9,N[i])
	y=func(x)
	n=int(N[i])
	xx=linspace(0.1,0.9,10*n+1)
	y2=zeros(x.size)
	#y2=cos(x)
	u=zeros(x.size)
	yy=zeros(xx.size)
	code="""
	#include <math.h>
	int i;
	double xp;
	spline(x,y,n,1e40,1e40,y2,u);
	for(i=0; i<=10*n; i++){
		xp=xx[i];
		splint(x,y,y2,n,xp,yy+i);
	}
	"""
	weave.inline(code,["x","y","n","y2","u","xx","yy"],support_code=scode,extra_compile_args=["-g"],compiler="gcc")
	if i==0:
		figure(2)
		plot(x,y)
		plot(xx,yy)
		title("Interpolated values and data points for n=%d" % N[i])
		show()
	figure(0)
	z=abs(yy-func(xx))
	plot(xx,z,label="N=%d"%N[i])
	err[i]=z.max()
​
xlabel("Location (x)")
ylabel("Error profile")
legend(loc="upper left")
figure(1)
loglog(h,err)
xlabel("Spacing")
ylabel("Error")
title("Error vs. spacing")
show()