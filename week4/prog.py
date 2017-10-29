from scipy.special import jv
from pylab import *

def integrand(x):
    return np.exp(-x)/jv(1,np.sqrt(-x**2+4*x-3))

#>>>>>>>Singular Integrals
#q1. Graphing the Integrand
eps=1e-6
x=linspace(1+eps,3-eps,1000)
semilogy(x,integrand(x),'ro')
semilogy(x,integrand(x))
title(r'Graph of the integrand $e^{-x}/J1(\sqrt{-x^2+4x-3})$ from $x=1$ to $x=3$')
xlabel("value of x")
ylabel("value of f(x)")
show()

#q2. Using quad for evaluating the function
from scipy import integrate
print integrate.quad(integrand,1,3,full_output=0)
#OUTPUT: (1.140489938554265, 5.333975483523545e-10)
# n_evals: 567

#q3. Using open Romberg for integration
import sing_intg as si
#print si.qromo(integrand,1,3,eps=1e-5)
#requires 81 function calls

#print si.qromo(integrand,1,3,eps=1e-6)
#requires 2187 function calls

#print si.qromo(integrand,1,3,eps=1e-7)
#requires 531441 function calls

#q4. Singularities at 1 and 3, use a transformation


#>>>>>>>>Gaussian Quadratures
#q1. transform to -1 to 1.
# Let t = x-2
def f(t):
    return np.exp(-t-2)*np.sqrt(1-t*t)/jv(1,sqrt(1-t**2))

#https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
#integral from -1 to 1 f(x)dx/sqrt(1-x^2)
#q2. Gauss-Chebyshev Quadrature
def calcGQ(x,w):
    sum=0.0
    for i in range(len(x)):
        sum+=f(x[i])*w[i]
    return sum

x=np.array(range(20))
x=cos(np.pi*(x-0.5)/20)
print x
w=np.full(20,np.pi/20)
exactVal=calcGQ(x,w)
print exactVal

intvals=[]; errors=[]

for i in range(1,20):
    x=np.array(range(i))
    x=cos(np.pi*(x-0.5)/i)
    w=np.full(i,np.pi/i)
    intVal=calcGQ(x,w)
    print intVal, intVal-exactVal
    intvals.append(intVal)
    errors.append(abs(intVal-exactVal))

plot(range(1,20),errors,'ro')
plot(range(1,20),errors)
title("Error v/s N for the Gauss-Chebyshev Quadratures")
xlabel("Number of points (N)")
ylabel("Error considering N=20 is exact")
show()

#q3. Romberg assignment integral
#q3_1
import scipy.special as sp
def f1(u):
    return u*sp.jv(3,2.7*u)**2

def f2(u):
    return u*sp.kv(3,1.2*u)**2

#q3_2
int1=integrate.quad(f1,0,1,epsabs=1e-12,epsrel=1e-12)
print int1
# num_evals=21, error is order 1e-16

int2=integrate.quad(f2,1,np.inf,epsabs=1e-12,epsrel=1e-12)
print int2
# num_evals=75, error is order 1e-8 initially
# num_evals=105, error is order 8e-13

#q3_3
import gauss_quad as gq
x,w=gq.gauleg(0,1,10)
total=0.0
for i in range(len(x)):
    total+=f1(x[i])*w[i]
print "Gauss-Legendre: ", total, total-int1[0]

#Gauss-Laguerre for I2
def newf2(u):
    return f2(u+1)*np.exp(u)

x,w=gq.gaulag(120,0.0)
total=0.0
for i in range(len(x)):
    total+=newf2(x[i])*w[i]
print "Gauss-Laguerre: ", total, total-int2[0]
#requires N=120 to achieve 1e-12 accuracy... but I did a naive implementation by taking f(u) as f1(u).e**u
#for the whole interval from 1 to inf. Instead if I consider the asymptotic behaviour it may improve

#Using Romberg
import romberg as r
rombval=r.qromb(f1,0,1,eps=1e-12)
print rombval, rombval[0]-int1[0]
#Setting 1e-12 accuracy requires 129 function calls and gives 1e-18 error
#(0.009969186534269642, -1.0555978304531426e-16, 129)

#Transform using u=tan(w)
def newf3(w):
    return np.tan(w)*sp.kv(3,1.2*np.tan(w))**2/(np.cos(w)**2)

rombval2=r.qromb(newf3,np.pi/4,np.pi/2,eps=1e-12)
print rombval2,rombval2[0]-int2[0]
#Setting 1e-12 accuracy requires 257 function calls and gives 1e-14 error
#(3.0924507786178475, -1.5258472381249202e-14, 257)
