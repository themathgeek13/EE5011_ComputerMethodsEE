from pylab import *
import math

def bisection(fd,a,b,tol=1e-15,NMAX=1e4):
    # a < b and f(a).f(b)<0
    N=1
    while(N<NMAX):
        c=(a+b)/2
        if fd(c)==0 or (b-a)/2 < tol:
            print N
            return c
        N=N+1
        if(np.sign(fd(c))==np.sign(fd(a))):
            a=c
        else:
            b=c
    return a,b

def secant(f,x0,x1,tol=1e-15):
    x=[x0,x1]
    while(abs(x[-2]-x[-1])>tol):
        val=x[-1]-f(x[-1])*(x[-1]-x[-2])/(f(x[-1])-f(x[-2]))
        x.append(val)
    return x[-1],len(x)
'''
#q2: Finding roots in 1-D
#Equation: tan(x) = sqrt(pi*alpha-x), alpha > 0
def f1(x):
    return tan(x)-sqrt(pi*al-x)
#bracketing intervals: [0,pi/2], [pi,3pi/2], ...
al=10
x=linspace(0,al*pi,1000)
plot(x,tan(x))
plot(x,sqrt(pi*al-x))
for i in range(0,int(al)+1):
    axvline(x=i*pi)
    axvline(x=i*pi+pi/2)
grid()
show()

xpos=bisection(f1,0,pi/2)
print xpos,f1(xpos)

#Secant method requires more accurate bracketing of the minima, so use bisection
#for a few iterations and then secant on the interval thus obtained
x0,x1=bisection(f1,0,pi/2,NMAX=6)
print x0,x1
xpossec=secant(f1,x0,x1)         #required 9 iterations
print xpossec, f1(xpossec[0])  

def f1_abs(x):
    return abs(f1(x))

#Brent's method (minimizes, so absolute value of the fn can be passed)
from scipy.optimize import brent
x1,fx1,b,c=brent(f1_abs,brack=(x0,x1),full_output=1,tol=1e-15)
print x1,fx1,b,c        #required 32 iterations

#Newton-Raphson method
def f1der(x):
    return 1/(cos(x)**2)+0.5/sqrt(pi*al-x)

def NR(f,fder,init,tol=1e-15,NMAX=1e4):
    x0=init
    x1=x0-f(x0)/fder(x0)
    N=0
    while(abs(x1-x0)>tol):
      x2=x1-f(x1)/fder(x1)
      x1,x0=x2,x1
      N=N+1
      if(N==NMAX):
          return None
    return x1,N

xnr=NR(f1,f1der,0)
print xnr,f1(xnr[0])
#highly optimal, required just 6 iterations, but it gave a value away from the first
#The value varies depending on the starting position...

def f3(x,ep=1e-3):
    return (x-2+ep)*(x-2-ep)*(x-2)

x=linspace(1.95,2.05,1000)
plot(x,f3(x,1e-3))
grid()
show()

xpos=bisection(f3,1,3)
print xpos,f3(xpos)
xpos2=bisection(f3,1,xpos-np.finfo(float).eps)  #subtract eps from interval, obtain zero
print xpos2,f3(xpos2)
'''

#q4: minimization in 1-D
# f(x) = sin(x)+1/(1+x^2)

#The function is somewhat irregular around x=0, but then quickly reverts to sin(x) behaviour
#This means that the minima are around -1. If the function is positive between two zeros of sin(x),
#it is not necessary to search for minima

N=1
def f(x):
    return sin(x)+1.0/(1+x**2)

x=linspace(-N*pi,N*pi,10000)
y=linspace(-2,2,50)
plot(x,f(x))
plot(x,sin(x))
plot(x,np.zeros(len(x)))

for i in range(-N,N+1):
    if i%2==0:
        axvline(x=i*pi,color='k')
    else:
        axvline(x=i*pi)
show()
#BLACK LINE -> BLUE LINE regions
#this shows that the minima can be bracketed by the points [(2N-1)*pi,2*N*pi]
#Whereas maxima could be bracketed using the points [2N*pi, (2N+1)*pi].

#Python code from Wikipedia: https://en.wikipedia.org/wiki/Golden-section_search
gr=(math.sqrt(5)+1)/2
def gss(f,a,b,tol=1e-15):
    count=0
    c=b-(b-a)/gr
    d=a+(b-a)/gr
    while abs(c-d)>tol:
        if f(c)<f(d):
            b=d
        else:
            a=c
        count+=1

        c=b-(b-a)/gr
        d=a+(b-a)/gr

    print count
    return (a+b)/2

print "x=",gss(f,-pi,0)

from scipy.optimize import brent
fx,x1,b,c=brent(f,brack=(-pi,-pi/2,0),full_output=1,tol=1e-15)
print brent(f,brack=(-pi,-pi/2,0),full_output=1,tol=1e-15)

#Brent's method requires 25 fn calls and 24 iterations whereas GS reqs 71 iterations.
#Brent's method thus provides supralinear convergence when possible
#and robustness otherwise (valid bracket maintained)
def f1(x):
    return f(x)+1e-15

fx2,x2,b,c=brent(f1,brack=(-pi,-pi/2,0),full_output=1,tol=1e-15)
print brent(f1,brack=(-pi,-pi/2,0),full_output=1,tol=1e-15)
print "x=",gss(f1,-pi,0)
print "x1-x2 = ",x1-x2

#The dead zone is approximately of the order of the computer resolution itself

def fd(x):
    #derivative of the function f(x)
    return cos(x)-2*x/((1+x**2)**2)

plot(x,fd(x))
grid()
show()

#bisection requires 52 iterations, so better than GS but worse than Brent. Also required derivative
pos=bisection(fd,-pi,0)
print pos,fd(pos),f(pos)
#The accuracy is lower than that of the other two methods, both of x and of f(x)
