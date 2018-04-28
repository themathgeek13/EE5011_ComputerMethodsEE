
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
