# -*- coding: utf-8 -*-
from pylab import *
import scipy.special as sp
from scipy import integrate
import romberg as r

def func(u):
    return 2*u*sp.jv(3,2.7*u)**2   

def func2(u):
    return 2*sp.kv(3,1.2*u)**2*u*abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2

count=0
def integrand(u):
    global count
    count+=1
    if u<1.0:                                                     
        return func(u)
    else: 
        return func2(u)

intg1=np.vectorize(integrand)
x=logspace(-3,7,200)
#semilogx(x,integrand(x))
loglog(x,intg1(x))
title("Dielectric Fibre Electromagnetic Mode integrand function")
xlabel("Value of x ->")
ylabel("Value of function f(x) ->")
grid()
show()

def exact():
    return sp.jv(3,2.7)**2-sp.jv(4,2.7)*sp.jv(2,2.7)+abs(sp.jv(3,2.7)/sp.kv(3,1.2))**2*(sp.kv(4,1.2)*sp.kv(2,1.2)-sp.kv(3,1.2)**2)

I0=exact()
print I0
print integrate.quad(integrand,0,20,full_output=0)[0]-exact()
#n_eval = 567 for non-split quad
#print integrate.quad(integrand,0,1,full_output=1), integrate.quad(integrand,1,20,full_output=1)
#n_eval = 147+21 = 168 for split quad
counts=[]
errs=[]
s=0
"""
for i in range(1,20):
    count=0
    s=r.trapzd(integrand,0,20,s,i)
    print "%1d %.15f %.2e"%(i,s,s-I0)
    print count
    counts.append(count)
    errs.append(abs(s-I0))

loglog(counts,errs)
loglog(counts,errs,'ro')
title("Error in integration vs number of function calls")
ylabel("Error in integration (absolute)")
xlabel("Number of function calls")
grid()
show()

#print r.qromb(integrand,0,20,1e-10)
counts=[]
errs=[]
for i in range(-1,-11,-1):
    count=0
    val= r.qromb(integrand,0,20,10**i)
    print val
    counts.append(count)
    errs.append(abs(val[1]))

loglog(counts,errs)
loglog(counts,errs,'ro')
title("Error vs number of function calls")
ylabel("Error in integration")
xlabel("Number of function calls")
grid()
#show()


counts=[]
errors=[]

for i in range(-1,-12,-1):
    count=0
    x1=r.qromb(integrand,0,1,10**i)
    x2=r.qromb(integrand,1,20,10**i)
    print x1[0]+x2[0],x1[0]+x2[0]-exact(),x1[2]+x2[2]
    counts.append(count)
    errors.append(abs(x1[0]+x2[0]-exact()))

loglog(counts, errors)
loglog(counts,errors,'ro')
title("Error vs number of function calls (split 0-1 and 1-20)")
xlabel("Number of function calls")
ylabel("Error in integration")
#grid()
show()

counts=[]
for i in range(5,21):
    count=0
    x=r.qromb(integrand,0,20,1e-8,k=i)
    print x
    counts.append(count)

semilogy(range(5,21),counts)
semilogy(range(5,21),counts,'ro')
title("Number of function calls vs order")
xlabel("Order of Romberg Integration")
ylabel("Number of function calls")
grid()
show()

import scipy.interpolate as si
err=[]
for i in range(4,20):
    x=linspace(0,20,2**i)
    y=intg1(x)
    tck=si.splrep(x,y)
    I=si.splint(0,20,tck)
    print I-exact()
    err.append(abs(I-exact()))
semilogy(range(4,20),err)
semilogy(range(4,20),err,'ro')
err=[]
for i in range(4,20):
    x1=linspace(0,1,2**(i-1))
    y1=intg1(x1)
    tck1=si.splrep(x1,y1)
    I1=si.splint(0,1,tck1)
    x2=linspace(1,20,2**(i-1))
    y2=intg1(x2)
    tck2=si.splrep(x2,y2)
    I2=si.splint(1,20,tck2)
    print I1+I2-exact()
    err.append(abs(I1+I2-exact()))
semilogy(range(4,20),err)
semilogy(range(4,20),err,'ro')
title("Comparison of both methods")
xlabel("log2(number of points)")
ylabel("Error relative to actual integral value")
grid()
show()
"""

def trap3(func, a, b, n):
    if(n==1):
        return 0.5*(b-a)*(func(a)+func(b))
    else:
        d = (float)(b-a)/3**(n-1)
        sum=0.0
        x=a+d
        while(x<b):
            sum+=func(x)*d; x+=d;
        sum+=0.5*d*(func(a)+func(b))
        return sum

xx=[]; yy=[]
order=12

for i in range(1,order+1):
    #count=0
    xx.append((20.0/3**(i-1)**2))
    yy.append(trap3(integrand,0,1,i)+trap3(integrand,1,20,i))
    print count

y,err=r.polint(xx,yy,0)
print y-exact()
