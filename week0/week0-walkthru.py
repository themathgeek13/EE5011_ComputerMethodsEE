
from pylab import *
N=50
M=100
c=1.0/(1.0+arange(N+1)**2)
x=linspace(0,10,M)
def fourier(N,c,x):
	z=zeros(x.shape)  # create and initialize z
	for i in range(M):
		zz=0.0;xx=x[i]
		for k in range(N+1):
			zz += c[k]*cos(k*xx)
		z[i]=zz
	return(z)

from pylab import *
N=50
M=100
c=1.0/(1.0+arange(N+1)**2)
x=linspace(0,10,M)
def fourierv(N,c,x):
	z=zeros(x.shape)  # create and initialize z
	for k in range(N+1):
		z += c[k]*cos(k*x)
	return(z)

from pylab import *
from scipy.weave import inline
N=50
M=100
c=1.0/(1.0+arange(N+1)**2)
x=linspace(0,10,M)
def fourc(N,c,x):
	n=len(x)
	z=zeros(x.shape)
	# now define the C code in a string.
	code="""
	double xx,zz;
	for( int j=0 ; j<n ; j++ ){
        xx=x[j];zz=0;
        for( int k=0 ; k<=N ; k++ )
            zz += c[k]*cos(k*xx);
        z[j]=zz;
    }
	"""
	inline(code,["z","c","x","N","n"],compiler="gcc")
	return(z)

str="""
from pylab import *
N=50
M=100
c=1.0/(1.0+arange(N+1)**2)
x=linspace(0,10,M)
def fourier(N,c,x):
	z=zeros(x.shape)  # create and initialize z
	for i in range(M):
		zz=0.0;xx=x[i]
		for k in range(N+1):
			zz += c[k]*cos(k*xx)
		z[i]=zz
	return(z)
"""
with open("fouriercy0.pyx",mode="w") as f:
	f.write(str)
import pyximport
pyximport.install()

str="""
import numpy as np
N=50
M=100
c=1.0/(1.0+np.arange(N+1)**2)
x=np.linspace(0,10,M)
cpdef fourier(N,c,x):
	cdef np.ndarray z=np.zeros(x.shape)  # create and initialize z
	cdef int i,k
	cdef double zz,xx
	for i in range(M):
		zz=0.0;xx=x[i]
		for k in range(N+1):
			zz += c[k]*np.cos(k*xx)
		z[i]=zz
	return(z)
"""
with open("fouriercy1.pyx",mode="w") as f:
	f.write(str)
import pyximport
pyximport.install()

str="""
import numpy as np
cimport numpy as np
N=50
M=100
c=1.0/(1.0+np.arange(N+1)**2)
x=np.linspace(0,10,M)
cpdef fourier(N,c,x):
	cdef np.ndarray z=np.zeros(x.shape)  # create and initialize z
	cdef int i,k
	cdef double zz,xx
	for i in range(M):
		zz=0.0;xx=x[i]
		for k in range(N+1):
			zz += c[k]*np.cos(k*xx)
		z[i]=zz
	return(z)
"""
with open("fouriercy1.pyx",mode="w") as f:
	f.write(str)
import pyximport
pyximport.install()

str="""
import numpy as np
cimport numpy as np
cdef extern from "<math.h>":
	cdef double cos(double x)
N=50
M=100
c=1.0/(1.0+np.arange(N+1)**2)
x=np.linspace(0,10,M)
cpdef fourier(N,c,x):
	cdef np.ndarray z=np.zeros(x.shape)  # create and initialize z
	cdef int i,k
	cdef double zz,xx
	for i in range(M):
		zz=0.0;xx=x[i]
		for k in range(N+1):
			zz += c[k]*cos(k*xx)
		z[i]=zz
	return(z)
"""
with open("fouriercy2.pyx",mode="w") as f:
	f.write(str)
import pyximport
pyximport.install()

str="""
import numpy as np
cimport numpy as np
cdef extern from "<math.h>":
	cdef double cos(double x)
DTYPE = np.double
ctypedef np.double_t DTYPE_t
N=50
M=100
c=1.0/(1.0+np.arange(N+1)**2)
x=np.linspace(0,10,M)
cpdef fourier(int N,np.ndarray[DTYPE_t,ndim=1] c,np.ndarray[DTYPE_t,ndim=1] x):
	cdef np.ndarray[DTYPE_t,ndim=1] z=np.zeros(M,dtype=DTYPE)
	cdef int i,k
	cdef double zz,xx
	for i in range(M):
		zz=0.0;xx=x[i]
		for k in range(N+1):
			zz += c[k]*cos(k*xx)
		z[i]=zz
	return(z)
"""
with open("fouriercy3.pyx",mode="w") as f:
	f.write(str)
import pyximport
pyximport.install()

from pylab import *
N=50
M=100
c=1.0/(1.0+arange(N+1)**2)
x=linspace(0,10,M)
def fouriervm(N,c,x):
	A=cos(outer(x,arange(N+1)))
	return (dot(A,c))
