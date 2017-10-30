#include <stdio.h>
#include <math.h>
#include <complex.h>

//Assuming alpha is real and positive (but roots may be complex)

void printc(double complex z)
{	printf("z = %f+%fj\n",creal(z),cimag(z));}

void printcf(float complex z)
{ 	printf("z = %f+%fj\n",creal(z),cimag(z));}

void floatroots(float alpha)
{
	float complex disc;
	if(cabs(alpha)<1)
		disc=I*sqrt(1-alpha*alpha);
	else
		disc=sqrt(alpha*alpha-1);
	float complex z1=-alpha+disc;
	float complex z2=-alpha-disc;
	printcf(z1);
	printcf(z2);
}

int sgn(float x)
{
	if(x>0) return 1;
	if(x<0) return -1;
	return 0;
}

void exactroots(double alpha)
{	
	double complex disc;
	if(cabs(alpha)<1)
		disc=I*sqrt(1-alpha*alpha);
	else	
		disc=sqrt(alpha*alpha-1);
	double complex z1=-alpha+disc;
	double complex z2=-alpha-disc;
	printc(z1);
	printc(z2);
}

void accroots(float alpha)
{	
	printf("\ninside accroots\n");
	float p = -(alpha + sgn(alpha)*sqrt(alpha*alpha-1));
	float complex z1=p;
	float complex z2=1/p;
	printcf(z1);
	printcf(z2);
}

int main()
{
	double alpha_start, alpha_end;
	float asf, aef;
	int N,i;

	printf("Enter alpha_start, alpha_end, N:\n");
	scanf("%lf %lf %d",&alpha_start,&alpha_end,&N);
	double r = pow(alpha_end/alpha_start,1.0/N);
	printf("\n%f, %f, r=%f, %d\n",alpha_start,alpha_end, r, N);
	asf=alpha_start;
	aef=alpha_end;
	for(i=0; i<N; i++)
	{
		printf("%d\n",i);
		double alpha=alpha_start*pow(r,i);
		float alf=asf*pow(r,i);
		if(cabs(alpha)<1)
			exactroots(alpha);	
		else
			accroots(alf);
		floatroots(alf);
		exactroots(alpha);
	}
	printf("%d\n",i);
	exactroots(alpha_end);
	floatroots(aef);

	return 0;
}
