// #include "nrutil.h"
#include<stdlib.h>

void spline(double *x,double *y,int n,double yp1,double ypn,double *y2,double *u){
  // It is assumed that x,y,y2 and u have been allocated in the calling program. u is a work array of the same size as x.
  int i,k;
  double p,qn,sig,un,h1,h2,del1,del2, hn1, hn2, deln1, deln2;

  x--;y--;y2--;u--; // NR adjustments
  if (yp1 > 0.99e30)
    y2[1]=u[1]=0.0;
  else {
    y2[1] = -0.5;
    h1=x[2]-x[1];
    h2=x[3]-x[2];
    del1=(y[2]-y[1])/h1;
    del2=(y[3]-y[2])/h2;
    u[1]=2*h2*h2+3*h1*h2;
    u[1]=u[1]*del1+5*h1*h1*del2;
    u[1]=u[1]/(h1+h2);
    u[1]=u[1]/h2;
  }
  for (i=2;i<=n-1;i++) {
    sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
    p=sig*y2[i-1]+2.0;
    y2[i]=(sig-1.0)/p;
    u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
    u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
  }
  if (ypn > 0.99e30)
    qn=un=y2[n]=0.0;
  else {
    hn2=x[n-1]-x[n-2];
    hn1=x[n]-x[n-1];
    del1=(y[n]-y[n-1])/hn1;
    del2=(y[n-1]-y[n-2])/hn2;

    y2[n]=2*hn1*hn1+3*hn1*hn2;
    y2[n]=y2[n]*deln2+5*hn2*hn2*deln1;
    y2[n]=y2[n]/(hn1+hn2);
    y2[n]=y2[n]-(hn1+hn2)*u[n-1];
    y2[n]=y2[n]/(hn2+(hn1+hn2)*y2[n-1]);
  }
  for (k=n-1;k>=1;k--)
    y2[k]=y2[k]*y2[k+1]+u[k];
}
void splint(double *xa,double *ya,double *y2a,int n,double x,double *y){
  // void nrerror();
  int klo,khi,k;
  double h,b,a;

  xa--;ya--;y2a--; // NR adjustments
  klo=1;
  khi=n;
  while (khi-klo > 1) {
    k=(khi+klo) >> 1;
    if (xa[k] > x) khi=k;
    else klo=k;
  }
  h=xa[khi]-xa[klo];
  if (h == 0.0){
    puts("Bad xa input to routine splint");
    exit(1);
  }
  a=(xa[khi]-x)/h;
  b=(x-xa[klo])/h;
  *y=a*ya[klo]+b*ya[khi]+( (a*a*a-a)*y2a[klo]
			   +(b*b*b-b)*y2a[khi] )*(h*h)/6.0;
}

