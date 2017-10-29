# coding: utf-8
get_ipython().magic(u'paste')
S(4)
S(1.5)
S(15)
get_ipython().magic(u'paste')
vals1p5
get_ipython().magic(u'paste')
vals1p5
get_ipython().magic(u'paste')
vals15
plot(vals15)
from pylab import *
plot(vals15)
show()
semilogy(vals15)
show()
np.cumsum(vals15)
len(np.cumsum(vals15))
len(vals15)
vals15[0]
vals15[1]
vals15[2]
fwd15=abs(np.cumsum(vals15))
fwd1p5=abs(np.cumsum(vals1p5))
semilogy(fwd15)
show()
semilogy(fwd1p5)
show()
n=range(41)
n
S(1.5,n)
get_ipython().magic(u'paste')
get_ipython().magic(u'paste')
S(1.5,n)
np.vectorize(S)
s1=np.vectorize(S)
s1(1.5,n)
s1(15,n)
err1p5=s1(1.5,n)-fwd1p5
err1p5=s1(1.5,n)-fwd1p5[:41]
semilogy(abs(err1p5))
show()
semilogy(abs(err1p5),'ro')
semilogy(abs(err1p5))
show()
semilogy(abs(err1p5),'ro')
semilogy(abs(err1p5))
title("Error for x=1.5 for different values of n")
xlabel("Value of n")
ylabel("Error (logscale)")
show()
err15=s1(15,n)-fwd15[:41]
semilogy(abs(err15))
semilogy(abs(err15),'ro')
title("Error for x=15 for different values of n")
xlabel("Value of n")
ylabel("Error (logscale)")
show()
get_ipython().magic(u'paste')
get_ipython().magic(u'paste')
vals1p5
len(vals1p5)
range(62)[::-1]
x=range(62)[::-1]
x.index(0)
vals1p5/=vals1p5[-1]
vals1p5=np.array(vals1p5)
vals1p5
vals1p5/=vals1p5[-1]
vals1p5
vals1p5[::-1]
vals1p5[::-1][:41]
np.cumsum(vals1p5[::-1][:41])
s
s2
get_ipython().magic(u'paste')
vals15
vals15=np.array(vals15)
vals15/=vals15[-1]
vals15
get_ipython().magic(u'paste')
vals15
get_ipython().magic(u'paste')
vals15
vals15=np.array(vals15)
vals15/=vals15[-1]
vals15
vals15=vals15[::-1]
np.cumsum(vals15)
jn(0,1.5)
jn(0,15)
vals15
jn(0,15)
vals15*=jn(0,15)
np.cumsum(vals15[:41])
vals1p5
vals1p5=vals1p5[::-1]
vals1p5*=jn(0,1.5)
np.cumsum(vals1p5[:41])
get_ipython().magic(u'paste')
n=range(41)
len(vals15)
vals15=vals15[:41]
vals15/(n+1)
vals15=np.array(vals15)
vals15/(n+1)
n
n+1
n=np.array(n)
vals15/(n+1)
vals15=vals15/(n+1)
vals1p5=np.array(vals1p5)
len(vals1p5)
vals1p5=np.array(vals1p5[:41])
len(vals1p5)
vals1p5=vals1p5/(n+1)
abs(np.cumsum(vals1p5))
get_ipython().magic(u'paste')
semilogy(err1p5)
show()
err1p5
semilogy(abs(err1p5))
show()
get_ipython().magic(u'paste')
n=np.array(range(41))
n
vals15=np.array(vals15)
vals1p5=np.array(vals1p5)
vals15/=vals15[-1]
vals1p5/=vals1p5[-1]
vals15=vals15[::-1]
vals1p5=vals1p5[::-1]
vals15
vals1p5
vals15=vals15[:41]
vals1p5=vals1p5[:41]
get_ipython().magic(u'paste')
err15
err1p5
vals15
vals1p5
vals15*=jn(0,15)
vals15
vals1p5*=jn(0,1.5)
vals1p5
get_ipython().magic(u'paste')
err15
err1p5
plot(abs(err15))
show()
semilogy(abs(err15))
show()
semilogy(abs(err15))
semilogy(abs(err15),'ro')
title("Error for x=15 for different values of n")
xlabel("Value of n")
ylabel("Error (logscale)")
show()
semilogy(abs(err1p5),'ro')
semilogy(abs(err1p5))
title("Error for x=1.5 for different values of n")
xlabel("Value of n")
ylabel("Error (logscale)")
show()
err1p5
semilogy(abs(err1p5))
semilogy(abs(err1p5),'ro')
xlabel("Value of n")
ylabel("Error (logscale)")
title("Error for x=1.5 for different values of n")
show()
def recfn(n,x,f1,f2,b1,b2):
    if(n==-1):
        return b2
    if(n==0):
        return b1
    else:
        return f1(x)*recfn(n-1,x,f1,f2,b1,b2)+f2(x)*recfn(n-2,x,f1,f2,b1,b2)
    
def f1(x):
    return 2*x
def f2(x):
    return -1
recfn(1,0.5,f1,f2,1,0.5)
recfn(1,0.5,f1,f2,0.5,1)
recfn(2,0.5,f1,f2,0.5,1)
recfn(3,0.5,f1,f2,0.5,1)
recfn(4,0.5,f1,f2,0.5,1)
recfn(10,0.5,f1,f2,0.5,1)
import scipy.quad
from scipy import quad.integrate
from scipy import quad
from scipy.integrate import quad
get_ipython().magic(u'pinfo quad')
def chebpoly(n,x):
    if(n==0):
        return 1
    if(n==1):
        return x
    return 2*x*chebpoly(n-1,x)-chebpoly(n-2,x)
chebpoly(0,1)
chebpoly(1,1)
chebpoly(1,0.5)
chebpoly(2,0.5)
quad(chebpoly(5,x),-1,1)
def f(x):
    return x*jn(1,x)
import numpy.polynomial.chebyshev.Chebyshev
import numpy.polynomial.chebyshev.Chebyshev.fit as fitcheb
import numpy.polynomial.chebyshev.chebfit as chebfit
import numpy.polynomial.chebyshev.chebfit
import numpy.polynomial.chebyshev.chebfit
import numpy.polynomial.chebyshev.chebfit
import numpy.polynomial.chebyshev.chebfromroots
x=linspace(0,5,50)
y=cos(5*acos(x))
y=cos(5*arccos(x))
x
x=linspace(-1,1,50)
y=cos(5*arccos(x))
plot(x,y)
show()
def chebfit(xdata,ydata,x):
    n=length(xdata)
    xmax=max(xdata)
    xmin=min(xdata)
    xdata=(2*xdata-xmax-xmin)/(xmax-xmin)
    T=np.zeros(n,n)
    T[:,0]=np.ones(n,1)
    T[:,1]=xdata
    for j in range(2,n):
        T[:,j]=2*xdata*T[:,j-1]-T[:,j-2]
    b=T\ydata
def chebfit(xdata,ydata,x):
    n=length(xdata)
    xmax=max(xdata)
    xmin=min(xdata)
    xdata=(2*xdata-xmax-xmin)/(xmax-xmin)
    T=np.zeros(n,n)
    T[:,0]=np.ones(n,1)
    T[:,1]=xdata
    for j in range(2,n):
        T[:,j]=2*xdata*T[:,j-1]-T[:,j-2]
    b=T/ydata
    x=(2*x-xmax-xmin)/(xmax-xmin)
    y=np.zeros(len(x))
    for j in range(n):
        y=y+b[j]*cos((j-1)*arccos(x));
        
xdata=linspace(0,5,50)
ydata=f(xdata)
ydaya
ydata
x=linspace(0,5,10)
chebfit(xdata,ydata,x)
def chebfit(xdata,ydata,x):
    n=len(xdata)
    xmax=max(xdata)
    xmin=min(xdata)
    xdata=(2*xdata-xmax-xmin)/(xmax-xmin)
    T=np.zeros(n,n)
    T[:,0]=np.ones(n,1)
    T[:,1]=xdata
    for j in range(2,n):
        T[:,j]=2*xdata*T[:,j-1]-T[:,j-2]
    b=T/ydata
    x=(2*x-xmax-xmin)/(xmax-xmin)
    y=np.zeros(len(x))
    for j in range(n):
        y=y+b[j]*cos((j-1)*arccos(x));
        
chebfit(xdata,ydata,x)
np.zeros(n,n)
np.zeros(3,3)
import numpy as np
np.zeros(3,3)
np.zeros((3,3))
def chebfit(xdata,ydata,x):
    n=len(xdata)
    xmax=max(xdata)
    xmin=min(xdata)
    xdata=(2*xdata-xmax-xmin)/(xmax-xmin)
    T=np.zeros((n,n))
    T[:,0]=np.ones((n,1))
    T[:,1]=xdata
    for j in range(2,n):
        T[:,j]=2*xdata*T[:,j-1]-T[:,j-2]
    b=T/ydata
    x=(2*x-xmax-xmin)/(xmax-xmin)
    y=np.zeros(len(x))
    for j in range(n):
        y=y+b[j]*cos((j-1)*arccos(x));
        
chebfit(xdata,ydata,x)
def chebfit(xdata,ydata,x):
    n=len(xdata)
    xmax=max(xdata)
    xmin=min(xdata)
    xdata=(2*xdata-xmax-xmin)/(xmax-xmin)
    T=np.zeros((n,n))
    T[:,0]=1
    T[:,1]=xdata
    for j in range(2,n):
        T[:,j]=2*xdata*T[:,j-1]-T[:,j-2]
    b=T/ydata
    x=(2*x-xmax-xmin)/(xmax-xmin)
    y=np.zeros(len(x))
    for j in range(n):
        y=y+b[j]*cos((j-1)*arccos(x));
        
chebfit(xdata,ydata,x)
n
len(xdata)
def chebfit(xdata,ydata,x):
    n=len(xdata)
    xmax=max(xdata)
    xmin=min(xdata)
    xdata=(2*xdata-xmax-xmin)/(xmax-xmin)
    T=np.zeros((n,n))
    T[:,0]=1
    T[:,1]=xdata
    for j in range(2,n):
        T[:,j]=2*xdata*T[:,j-1]-T[:,j-2]
    b=T/ydata
    x=(2*x-xmax-xmin)/(xmax-xmin)
    y=np.zeros(len(x))
    return T
    for j in range(n):
        y=y+b[j]*cos((j-1)*arccos(x));
        
chebfit(xdata,ydata,x)
T=chebfit(xdata,ydata,x)
ydata
np.linalg.inv(T)
Tinv=np.linalg.inv(T)
Tinv*ydata
b=Tinv*ydata
def chebfit(xdata,ydata,x):
    n=len(xdata)
    xmax=max(xdata)
    xmin=min(xdata)
    xdata=(2*xdata-xmax-xmin)/(xmax-xmin)
    T=np.zeros((n,n))
    T[:,0]=1
    T[:,1]=xdata
    for j in range(2,n):
        T[:,j]=2*xdata*T[:,j-1]-T[:,j-2]
    b=np.dot(np.linalg.inv(T),ydata)
    x=(2*x-xmax-xmin)/(xmax-xmin)
    y=np.zeros(len(x))
    for j in range(n):
        y=y+b[j]*cos((j-1)*arccos(x));
        
chebfit(xdata,ydata,x)
def chebfit(xdata,ydata,x):
    n=len(xdata)
    xmax=max(xdata)
    xmin=min(xdata)
    xdata=(2*xdata-xmax-xmin)/(xmax-xmin)
    T=np.zeros((n,n))
    T[:,0]=1
    T[:,1]=xdata
    for j in range(2,n):
        T[:,j]=2*xdata*T[:,j-1]-T[:,j-2]
    b=np.dot(np.linalg.inv(T),ydata)
    x=(2*x-xmax-xmin)/(xmax-xmin)
    y=np.zeros(len(x))
    for j in range(n):
        y=y+b[j]*cos((j-1)*arccos(x));
    return y
chebfit(xdata,ydata,x)
chebfit(xdata,ydata,x)-f(x)
x=xdata
chebfit(xdata,ydata,x)-f(x)
def chebfit(xdata,ydata,x):
    n=len(xdata)
    xmax=max(xdata)
    xmin=min(xdata)
    xdata=(2*xdata-xmax-xmin)/(xmax-xmin)
    T=np.zeros((n,n))
    T[:,0]=1
    T[:,1]=xdata
    for j in range(2,n):
        T[:,j]=2*xdata*T[:,j-1]-T[:,j-2]
    b=np.dot(np.linalg.inv(T),ydata)
    x=(2*x-xmax-xmin)/(xmax-xmin)
    y=np.zeros(len(x))
    for j in range(n):
        y=y+b[j]*cos((j-1)*arccos(x));
    return y,b
    
chebfit(xdata,ydata,x)
y,b=chebfit(xdata,ydata,x)
plot(abs(b))
show()
semilogy(abs(b))
semilogy(abs(b),'ro')
show()
plot(abs(b))
show()
semilogy(abs(b),'ro')
semilogy(abs(b))
show()
semilogy(abs(b),'ro')
show()
semilogy(abs(b))
semilogy(abs(b),'ro')
title("Magnitude of coefficients for Chebyshev fitting")
xlabel("Coefficient index")
ylabel("Magnitude")
show()
y,b=chebfit(xdata,ydata,x)
y-f(x)
plot(abs(y-f(x))
)
show(
)
plot(abs(y-f(x)))/y
plot(abs(y-f(x)/y))
show()
plot(abs(y-f(x)/y))
show()
max(abs(y-f(x)))
xdata=linspace(0,5,500)
ydata=f(xdata)
y,b=chebfit(xdata,ydata,xdata)
plot(abs(y-f(x)))
plot(abs(y-f(xdata)))
show()
f(xdata)
y
xdata=linspace(0,5,100)
ydata=f(xdata)
y,b=chebfit(xdata,ydata,xdata)
y
xdata=linspace(0,5,50)
ydata=f(xdata)
y,b=chebfit(xdata,ydata,xdata)
y
ydata
plot(abs(y-f(xdata)))
show()
len(xdata)
xdata=linspace(0,5,100)
xdata
xdata=linspace(0,5,101)
xdata
ydata=f(xdata)
y,b=chebfit(xdata,ydata,xdata)
y
np.polynomial.chebyshev.chebfit
np.polynomial.chebyshev.chebfit()
np.polynomial.chebyshev.chebfit(xdata,ydata,xdata)
np.polynomial.chebyshev.chebfit(xdata,ydata)
np.polynomial.chebyshev.chebfit(xdata,ydata,10)
np.polynomial.chebyshev.chebfit(xdata,ydata,50)
np.polynomial.chebyshev.chebfit(xdata,ydata,20)
np.polynomial.chebyshev.chebfit(xdata,ydata,15)
len(xdata)
xdata=linspace(0,5,50)
ydata=f(xdata)
np.polynomial.chebyshev.chebfit(xdata,ydata,15)
coeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
semilogy(coeff)
semilogy(coeff,'ro')
title("Magnitude of coefficients for Chebyshev fitting")
xlabel("Coefficient index")
ylabel("Magnitude")
show()
semilogy(abs(coeff))
semilogy(abs(coeff),'ro')
title("Magnitude of coefficients for Chebyshev fitting")
xlabel("Coefficient index")
ylabel("Magnitude")
show()
def chebpoly(n,x):
    if(n==0):
        return 1
    if(n==1):
        return x
    return 2*x*chebpoly(n-1,x)-chebpoly(n-2,x)
jn(2,0.5)
jn(1,0.5)
jn(1,0.5j)
def chebapproxfn(coeff,x):
    n=len(coeff)
    s=0
    for i in range(n):
        s+=coeff[i]*chebpoly(i,x)
    return s
chebapproxfn(coeff,xdata)
err=chebapproxfn(coeff,xdata)
max(abs(err))
coeff=np.polynomial.chebyshev.chebfit(xdata,ydata,20)
chebapproxfn(coeff,xdata)
err=chebapproxfn(coeff,xdata)
max(abs(err))
xdata=linspace(0,5,100)
chebapproxfn(coeff,xdata)
err=chebapproxfn(coeff,xdata)
max(abs(err))
np.polynomial.chebyshev.chebder(xdata,f(xdata),15)
np.polynomial.chebyshev.chebder(xdata,f(xdata))
xdata
coeff
np.polynomial.chebyshev.chebder(coeff)
dercoeff=np.polynomial.chebyshev.chebder(coeff)
chebapproxfn(dercoeff,xdata)
xdata*jn(0,xdata)
errd=chebapproxfn(dercoeff,xdata)-xdata*jn(0,xdata)
plot(abs(errd))
show()
semilogy(abs(errd))
show()
chebapproxfn(coeff,xdata)
chebapproxfn(coeff,xdata)-f(xdata)
xdata
len(xdata)
xdata=linspace(0,5,50)
coeff=np.polynomial.chebyshev.chebfit(xdata,ydata,20)
chebapproxfn(coeff,xdata)-f(xdata)
errs=chebapproxfn(coeff,xdata)-f(xdata)
plot(abs(errs))
show()
semilogy(abs(errs))
show()
semilogy(abs(errs))
semilogy(abs(errs),'ro')
title("Error in the Chebyshev fit of f(x)")
xlabel("Value of x (0 < x < 5)")
ylabel("Error in estimate")
show()
semilogy(xdata,abs(errs))
semilogy(xdata,abs(errs),'ro')
title("Error in the Chebyshev fit of f(x)")
xlabel("Value of x (0 < x < 5)")
ylabel("Error in estimate")
show()
chebder(coeff)
np.polynomial.chebyshev.chebder(coeff)
dercoeff=np.polynomial.chebyshev.chebder(coeff)
chebapproxfn(dercoeff,xdata)
errd=chebapproxfn(dercoeff,xdata)-xdata*jn(0,xdata)
semilogy(xdata,errd)
semilogy(xdata,errd,'ro')
title("Error in the Chebyshev-derivative fit of f'(x)")
xlabel("Value of x (0 < x < 5)")
ylabel("Error in estimate (semilog)")
show()
delta=xdata[1:]-xdata[:-1]
df=ydata[1:]-ydata[:-1]
df/delta
derfn=df/delta
derfn-xdata*jn(0,xdata)
derfn
xdata*jn(0,xdata)
derfn-xdata*jn(0,xdata)[1:]
derfn-xdata[1:]*jn(0,xdata)[1:]
xdata
delta
delta=0.01
derfn=(f(xdata+delta/2)-f(xdata-delta/2))/delta
derfn
derfn-xdata*jn(0,xdata)
errdiff=derfn-xdata*jn(0,xdata)
semilogy(xdata,errdiff,'ro')
semilogy(xdata,errdiff)
show()
xdata
errdiff
semilogy(xdata,abs(errdiff))
semilogy(xdata,abs(errdiff),'ro')
title("Error in the diff-derivative fit of f'(x)")
xlabel("Value of x (0 < x < 5)")
ylabel("Error in estimate (semilog)")
show()
delta=1e-5
derfn=(f(xdata+delta/2)-f(xdata-delta/2))/delta
errdiff=derfn-xdata*jn(0,xdata)
semilogy(xdata,abs(errdiff))
semilogy(xdata,abs(errdiff),'ro')
title("Error in the diff-derivative fit of f'(x)")
xlabel("Value of x (0 < x < 5)")
ylabel("Error in estimate (semilog)")
show()
delta=xdata[1:]-xdata[:-1]
delta
delta=0.10204082
derfn=(f(xdata+delta/2)-f(xdata-delta/2))/delta
errdiff=derfn-xdata*jn(0,xdata)
semilogy(xdata,abs(errdiff))
semilogy(xdata,abs(errdiff),'ro')
title("Error in the diff-derivative fit of f'(x)")
xlabel("Value of x (0 < x < 5)")
ylabel("Error in estimate (semilog)")
show()
xdata=linspace(-1,1,50)
ydata=sin(xdata)
ydata=sin(np.pi*xdata)
np.polynomial.chebyshev.chebfit(xdata,ydata)
np.polynomial.chebyshev.chebfit(xdata,ydata,xdata)
xdata
ydata
len(xdata)
len(ydata)
np.polynomial.chebyshev.chebfit(xdata,ydata,10)
sinco=np.polynomial.chebyshev.chebfit(xdata,ydata,10)
chebapproxfn(sinco,xdata)
chebapproxfn(sinco,xdata)-ydata
x1=linspace(-1,1,200)
y1=sin(np.pi*x1)
chebapproxfn(sinco,x1)-y1
semilogy(chebapproxfn(sinco,x1)-y1)
show()
semilogy(abs(chebapproxfn(sinco,x1)-y1))
show()
xdata=linspace(-1,1,200)
ydata=sin(np.pi*xdata)
sinco=np.polynomial.chebyshev.chebfit(xdata,ydata,10)
semilogy(abs(chebapproxfn(sinco,x1)-y1))
show()
sinco
sinco=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
sinco
semilogy(abs(chebapproxfn(sinco,x1)-y1))
show()
sinco=np.polynomial.chebyshev.chebfit(xdata,ydata,10)
semilogy(abs(chebapproxfn(sinco,x1)-y1))
show()
semilogy(abs(chebapproxfn(sinco,x1)-y1))
semilogy(abs(chebapproxfn(sinco,x1)-y1),'ro')
title("Error in the approximation of sin(x) by 10 Chebyshev terms")
xlabel("Value of x (-1 < x < -1)")
xlabel("Value of x (-1 < x < 1)")
ylabel("Error in estimate (semilog)")
show()
semilogy(x1,abs(chebapproxfn(sinco,x1)-y1))
semilogy(x1,abs(chebapproxfn(sinco,x1)-y1),'ro')
title("Error in the approximation of sin(x) by 10 Chebyshev terms")
xlabel("Value of x (-1 < x < 1)")
ylabel("Error in estimate (semilog)")
show()
def f(x):
    return np.exp(x)
def g(x,d):
    return 1/(x**2+d**2)
def h(x,d):
    return 1/((cos(np.pi*x/2)**2+d**2))
def u(x):
    return exp(-abs(x))
def v(x):
    return sqrt(x+1.1)
xdata=linspace(-1,1,200)
ydata=f(xdata)
np.polynomial.chebyshev.chebfit(xdata,ydata,10)
np.polynomial.chebyshev.chebfit(xdata,ydata,15)
fcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
chebapproxfn(fcoeff,xdata)
ferr=chebapproxfn(fcoeff,xdata)-f(xdata)
plot(abs(ferr))
show()
plot(abs(ferr))
show()
semilogy(abs(ferr))
show()
xdata
len(xdata)
semilogy(xdata,abs(ferr))
semilogy(xdata,abs(ferr),'ro')
title("Error in the approximation of f(x) by 15 Chebyshev terms")
xlabel("Value of x (-1 < x < 1)")
ylabel("Error in estimate (semilog)")
show()
xdata=linspace(-1,1,200)
ydata=u(xdata)
ucoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
ucoeff
chebapproxfn(ucoeff,xdata)
chebapproxfn(ucoeff,xdata)-u(xdata)
ucoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,30)
ucoeff
chebapproxfn(ucoeff,xdata)-u(xdata)
uerr=chebapproxfn(ucoeff,xdata)-u(xdata)
plot(abs(uerr))
show()
semilogy(abs(uerr))
show()
semilogy(xdata,abs(uerr))
semilogy(xdata,abs(uerr),'ro')
title("Error in the approximation of u(x) by 30 Chebyshev terms")
xlabel("Value of x (-1 < x < 1)")
ylabel("Error in estimate (semilog)")
show()
xdata=linspace(-1,0,100)
ydata=u(xdata)
ucoeff1=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
ucoeff1
ucoeff1=np.polynomial.chebyshev.chebfit(xdata,ydata,17)
ucoeff1
ucoeff1=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
ucoeff1
uerr=chebapproxfn(ucoeff1,xdata)-u(xdata)
semilogy(xdata,uerr)
show()
semilogy(xdata,abs(uerr))
show()
semilogy(xdata,abs(uerr))
semilogy(xdata,abs(uerr),'ro')
title("Error in the approximation of u(x) by 15 Chebyshev terms from -1 < x < 0")
xlabel("Value of x (-1 < x < 0)")
ylabel("Error in estimate (semilog)")
show()
xdata=
xdata=linspace(0,1,100)
ydata=u(xdata)
ucoeff2=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
uerr=chebapproxfn(ucoeff2,xdata)-u(xdata)
semilogy(xdata,abs(uerr))
semilogy(xdata,abs(uerr),'ro')
title("Error in the approximation of u(x) by 15 Chebyshev terms from 0 < x < 1")
xlabel("Value of x (0 < x < 1)")
ylabel("Error in estimate (semilog)")
show()
vcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
xdata=linspace(-1,1,200)
ydata=v(xdata)
vcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
vcoeff
vcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,20)
vcoeff
vcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,30)
vcoeff
verr=chebapproxfn(vcoeff,xdata)-v(xdata)
semilogy(xdata,abs(verr))
show()
semilogy(xdata,abs(verr))
semilogy(xdata,abs(verr),'ro')
title("Error in the approximation of v(x) by 30 Chebyshev terms from -1 < x < 1")
xlabel("Value of x (-1 < x < 1)")
ylabel("Error in estimate (semilog)")
show()
v(xdata)
plot(v(xdata))
show()
xdata
plot(xdata,v(xdata))
show()
delta=0.1
d=0.1
xdata=linspace(-1,1,200)
ydata=g(xdata)
ydata=g(xdata,d)
gcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
gcoeff
gcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,30)
gcoeff
gerr=chebapproxfn(gcoeff,xdata)-g(xdata,d)
plot(abs(gerr))
show()
xdata[1]-xdata[0]
gcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,50)
gcoeff
gerr=chebapproxfn(gcoeff,xdata)-g(xdata,d)
plot(abs(gerr))
show()
d=1
gcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
xdata=linspace(-1,1,200)
ydata=g(xdata,d)
gcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
gcoeff
gerr=chebapproxfn(gcoeff,xdata)-g(xdata,d)
plot(abs(gerr))
show()
gerr
semilogy(abs(gerr))
show()
show()
semilogy(abs(gerr))
show()
semilogy(xdata,abs(gerr))
semilogy(xdata,abs(gerr),'ro')
title("Error in the approximation of g(x,del=1) by 15 Chebyshev terms from -1 < x < 1")
xlabel("Value of x (-1 < x < 1)")
ylabel("Error in estimate (semilog)")
show()
d=10
ydata=g(xdata,d)
gcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
gcoeff
gcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,10)
gcoeff
gerr=chebapproxfn(gcoeff,xdata)-g(xdata,d)
semilogy(xdata,abs(gerr))
show()
gcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,7)
gerr=chebapproxfn(gcoeff,xdata)-g(xdata,d)
semilogy(xdata,abs(gerr))
show()
semilogy(xdata,abs(gerr))
semilogy(xdata,abs(gerr),'ro')
title("Error in the approximation of g(x,del=10) by 7 Chebyshev terms from -1 < x < 1")
xlabel("Value of x (-1 < x < 1)")
ylabel("Error in estimate (semilog)")
show()
semilogy(xdata,abs(gerr),'ro')
show()
d=100
ydata=g(xdata,d)
gcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,3)
gcoeff
gerr=chebapproxfn(gcoeff,xdata)-g(xdata,d)
semilogy(xdata,abs(gerr))
semilogy(xdata,abs(gerr),'ro')
title("Error in the approximation of g(x,del=100) by 3 Chebyshev terms from -1 < x < 1")
xlabel("Value of x (-1 < x < 1)")
ylabel("Error in estimate (semilog)")
show()
plot(xdata,g(xdata,0.1))
show()
subplot((1,1,1))
plot(xdata,g(xdata,0.1))
title("Function g(x) with delta=0.1")
xlabel("Value of x (-1 < x < 1)")
ylabel("g(x)")
show()
plot(xdata,g(xdata,1))
title("Function g(x) with delta=1")
xlabel("Value of x (-1 < x < 1)")
ylabel("g(x)")
show()
plot(xdata,g(xdata,10))
title("Function g(x) with delta=10")
xlabel("Value of x (-1 < x < 1)")
ylabel("g(x)")
show()
plot(xdata,g(xdata,100))
xlabel("Value of x (-1 < x < 1)")
ylabel("g(x)")
title("Function g(x) with delta=100")
show()
plot(xdata,g(xdata,100))
xlabel("Value of x (-1 < x < 1)")
ylabel("g(x)")
title("Function g(x) with delta=100")
show()
ydata=h(xdata,d)
d=0.1
ydata=h(xdata,d)
hcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
hcoeff
hcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,30)
hcoeff
max(abs(hcoeff))
hcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,50)
max(abs(hcoeff))
hcoeff
d=1
ydata=h(xdata,d)
plot(xdata,h(xdata,0.1))
show()
plot(xdata,h(xdata,1))
show()
plot(xdata,h(xdata,10))
show()
plot(xdata,h(xdata,100))
show()
hcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
hcoeff
hcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,20)
hcoeff
herr=chebapproxfn(hcoeff,xdata)-h(xdata,d)
herr
semilogy(xdata,abs(herr))
semilogy(xdata,abs(herr),'ro')
title("Function h(x) with delta=1")
xlabel("Value of x (-1 < x < 1)")
ylabel("h(x)")
show()
semilogy(xdata,abs(herr),'ro')
show()
semilogy(xdata,abs(herr))
semilogy(xdata,abs(herr),'ro')
title("Error in h(x,del=1) by 20 Chebyshev terms from -1 < x < 1")
xlabel("Value of x (-1 < x < 1)")
ylabel("h(x)")
show()
d=10
ydata=h(xdata,d)
hcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,15)
hcoeff
herr=chebapproxfn(hcoeff,xdata)-h(xdata,d)
semilogy(xdata,abs(herr))
semilogy(xdata,abs(herr),'ro')
title("Error in h(x,del=10) by 15 Chebyshev terms from -1 < x < 1")
xlabel("Value of x (-1 < x < 1)")
ylabel("h(x)")
show()
d=100
hcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,10)
hcoeff
herr=chebapproxfn(hcoeff,xdata)-h(xdata,d)
semilogy(xdata,abs(herr))
semilogy(xdata,abs(herr),'ro')
xlabel("Value of x (-1 < x < 1)")
ylabel("h(x)")
title("Error in h(x,del=100) by 10 Chebyshev terms from -1 < x < 1")
show()
semilogy(xdata,abs(herr))
show()
ydata=h(xdata,d)
hcoeff=np.polynomial.chebyshev.chebfit(xdata,ydata,10)
hcoeff
herr=chebapproxfn(hcoeff,xdata)-h(xdata,d)
semilogy(xdata,abs(herr))
show()
semilogy(xdata,abs(herr))
semilogy(xdata,abs(herr),'ro')
xlabel("Value of x (-1 < x < 1)")
ylabel("h(x)")
title("Error in h(x,del=100) by 10 Chebyshev terms from -1 < x < 1")
show()
plot(xdata,h(xdata,100))
show()
plot(xdata,h(xdata,0.1))
show()
plot(xdata,h(xdata,100000))
show()
plot(xdata,h(xdata,1e-5))
show()
x=logspace(-10,0,10)
x
for item in x:
    plot(xdata,h(xdata,item))
    
show()
for item in x:
    semilogy(xdata,h(xdata,item))
    
    
show()
for item in x:
    loglog(xdata,h(xdata,item)) 
    
show()
def c(f1,m):
    quad(f1,f2,-1,1)
    
get_ipython().magic(u'pinfo quad')
for i in range(20):
    v=quad(f(x)*cos(i*(x+1)*pi/2),-1,1)
    print v
    
def f1(x,m):
    return f(x)*cos(m*(x+1)*np.pi/2)
quad(f1,args=(0.5),-1,1)
quad(f1,arg=(0.5),-1,1)
quad(f1,arg=0.5,-1,1)
quad(f1,-1,1,args=(0.5))
quad(f1,-1,1,args=(1))
for i in range(20):
    print quad(f1,-1,1,args=(i))
    
for i in range(30):
    print quad(f1,-1,1,args=(i))
    
for i in range(30):
    fcoeff.append(quad(f1,-1,1,args=(i)))
        
fcoeff=[]
for i in range(30):
    fcoeff.append(quad(f1,-1,1,args=(i)))
        
plot(fcoeff)
show()
plot(abs(fcoeff))
plot(abs(np.array(fcoeff)))
show()
plot(abs(np.array(fcoeff)))
show()
fcoeff
fcoeff[:,0]
for i in range(30):
    fcoeff.append(quad(f1,-1,1,args=(i))[0])
        
fcoeff=[]
for i in range(30):
    fcoeff.append(quad(f1,-1,1,args=(i))[0])
        
fcoeff
plot(abs(np.array(fcoeff)))
show()
semilogy(abs(np.array(fcoeff)))
show()
fcoeff=[]
for i in range(100):
    fcoeff.append(quad(f1,-1,1,args=(i))[0])
        
semilogy(abs(np.array(fcoeff)))
show()
def fouriercoeffs(f1,N):
    fcoeff=[]
    for i in range(N):
        fcoeff.append(quad(f1,-1,1,args=(i))[0])
    return np.array(fcoeff)
fouriercoeffs(u,20)
def u1(x,m):
    return u(x)*cos(m*(x+1)*np.pi/2)
fouriercoeffs(u1,20)
fouriercoeffs(u1,30)
plot(abs(fouriercoeffs(u1,30)))
show()
semilogy(abs(fouriercoeffs(u1,30)))
show()
semilogy(abs(fouriercoeffs(u1,100)))
show()
def v1(x,m):
    return v(x)*cos(m*(x+1)*np.pi/2)
semilogy(abs(fouriercoeffs(v1,100)))
show()
def g1_3(x,m):
    return g(x,3)*cos(m*(x+1)*np.pi/2)
semilogy(abs(fouriercoeffs(g1_3,100)))
show()
def g1_1(x,m):
    return g(x,1)*cos(m*(x+1)*np.pi/2)
semilogy(abs(fouriercoeffs(g1_1,100)))
show()
def g1_0p3(x,m):
    return g(x,0.3)*cos(m*(x+1)*np.pi/2)
semilogy(abs(fouriercoeffs(g1_0p3,100)))
show()
def h1_3(x,m):
    return h(x,3)*cos(m*(x+1)*np.pi/2)
semilogy(abs(fouriercoeffs(h1_3,100)))
show()
def h1_1(x,m):
    return h(x,1)*cos(m*(x+1)*np.pi/2)
    
semilogy(abs(fouriercoeffs(h1_1,100)))
show()
def h1_0p3(x,m):
    return h(x,0.3)*cos(m*(x+1)*np.pi/2)
   
semilogy(abs(fouriercoeffs(h1_0p3,100)))
show()
semilogy(abs(fouriercoeffs(h1_3,100)))
semilogy(abs(fouriercoeffs(h1_1,100)))
semilogy(abs(fouriercoeffs(h1_0p3,100)))
show()
semilogy(abs(fouriercoeffs(h1_3,100)),legend="delta=3")
semilogy(abs(fouriercoeffs(h1_3,100)),label="delta=3")
semilogy(abs(fouriercoeffs(h1_1,100)),label="delta=1")
semilogy(abs(fouriercoeffs(h1_0p3,100)),label="delta=0.3")
show()
semilogy(abs(fouriercoeffs(h1_3,100)),label="delta=3")
semilogy(abs(fouriercoeffs(h1_1,100)),label="delta=1")
semilogy(abs(fouriercoeffs(h1_0p3,100)),label="delta=0.3")
legend()
show()
semilogy(abs(fouriercoeffs(h1_3,100)),label="delta=3")
semilogy(abs(fouriercoeffs(h1_1,100)),label="delta=1")
semilogy(abs(fouriercoeffs(h1_0p3,100)),label="delta=0.3")
title("h(x) approximated with Fourier series for varied delta")
legend()
show()
semilogy(abs(fouriercoeffs(h1_0p3,100)),label="delta=0.3")
semilogy(abs(fouriercoeffs(h1_1,100)),label="delta=1")
semilogy(abs(fouriercoeffs(h1_3,100)),label="delta=3")
title("h(x) approximated with Fourier series for varied delta")
xlabel("Coefficient index")
ylabel("Magnitude")
legend()
show()
semilogy(abs(fouriercoeffs(g1_3,100)),label="delta=3")
semilogy(abs(fouriercoeffs(g1_1,100)),label="delta=1")
semilogy(abs(fouriercoeffs(g1_0p3,100)),label="delta=0.3")
legend()
title("g(x) approximated with Fourier series for varied delta")
xlabel("Coefficient index")
ylabel("Magnitude")
show()
semilogy(fcoeff)
show()
fcoeff
semilogy(abs(fcoeff))
fcoeff=np.array(fcoeff)
semilogy(abs(fcoeff))
show()
semilogy(abs(fcoeff))
title("f(x) approximated with Fourier series")
xlabel("Coefficient index")
ylabel("Magnitude")
show()
fouriercoeffs(u1,1000)
fouriercoeffs(h1_1,1000)
hc=fouriercoeffs(h1_1,1000)
loglog(range(1,1001),abs(hc))
show()
hc=fouriercoeffs(h1_3,1000)
loglog(range(1,1001),abs(hc))
show()
hc=fouriercoeffs(h1_0p3,1000)
loglog(range(1,1001),abs(hc))
show()
fouriercoeffs(v1,100)
semilogy(abs(fouriercoeffs(v1,100)))
show()
semilogy(abs(fouriercoeffs(v1,100)))
title("v(x) approximated with Fourier series")
xlabel("Coefficient index")
ylabel("Magnitude")
show()
semilogy(abs(fouriercoeffs(u1,100)))
show()
semilogy(abs(fouriercoeffs(u1,100)))
title("u(x) approximated with Fourier series")
xlabel("Coefficient index")
ylabel("Magnitude")
show()
