from pylab import *
import weave
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
	weave.inline(code,["z","c","x","N","n"],compiler="gcc")
	return(z)
