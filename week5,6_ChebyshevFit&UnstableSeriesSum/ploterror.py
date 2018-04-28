from pylab import *

f=open("errors.txt").read()

v=f.split("\n")[:-1]

errors1=[]
errors2=[]
alpha=[]
for item in v:
    err=(float(item.split()[1][:-1])+ float(item.split()[2][:-1]))/2
    errors1.append(err)
    err2=(float(item.split()[3][:-1])+ float(item.split()[4][:-1]))/2
    errors2.append(err2)

    x,y=map(float,item.split()[-1].split("+"))
    alpha.append(abs(complex(x,y)))

loglog(alpha,errors1)
loglog(alpha,errors2)
loglog(alpha,errors1,'ro')
loglog(alpha,errors2,'ro')
xlabel(u"Magnitude of $\\alpha$")
ylabel("Magnitude of error")
title("Plot of alpha vs error with N={0}".format(len(errors1)))
show()
