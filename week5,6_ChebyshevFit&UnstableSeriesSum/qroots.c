#include <stdio.h>
#include <math.h>
#include <complex.h>

//Assuming alpha is real and positive (but roots may be complex)

double complex e1, e2;
double complex a1, a2;
double complex f1, f2;

void printc(double complex z)
{	printf("z = %f+%fj\n",creal(z),cimag(z));}

void printcf(float complex z)
{ 	printf("z = %f+%fj\n",creal(z),cimag(z));}

void floatroots(float alpha)
{
	printf("\ninside floatroots\n");
	float complex disc;
	if(cabs(alpha)<1)
		disc=I*sqrt(1-alpha*alpha);
	else
		disc=sqrt(alpha*alpha-1);
	float complex z1=-alpha+disc;
	float complex z2=-alpha-disc;
	
	a1=z1;
	a2=z2;
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
	printf("\ninside exactroots\n");
	double complex disc;
	if(cabs(alpha)<1)
		disc=I*sqrt(1-alpha*alpha);
	else	
		disc=sqrt(alpha*alpha-1);
	double complex z1=-alpha+disc;
	double complex z2=-alpha-disc;
	
	e1=z1;
	e2=z2;
	printc(z1);
	printc(z2);
}

void accroots(float alpha)
{	
	printf("\ninside accroots\n");
	float p = -(alpha + sgn(alpha)*sqrt(alpha*alpha-1));
	float complex z1=1/p;
	float complex z2=p;
	
	f1=z1;
	f2=z2;
	printcf(z1);
	printcf(z2);
}

int main()
{
	double alpha_start_real, alpha_start_comp, alpha_end_real, alpha_end_comp;
	float asf, aef;
	int N,i;
	FILE *fp;
	fp=fopen("errors.txt","w");
	printf("Enter alpha_start, alpha_end, N (example x+jy, both must be specified):\n");
	scanf("%lf+j%lf %lf+j%lf %d",&alpha_start_real, &alpha_start_comp, &alpha_end_real, &alpha_end_comp,&N);
	double complex alpha_start=alpha_start_real+I*alpha_start_comp;
	double complex alpha_end=alpha_end_real+I*alpha_end_comp;
	double r = pow(cabs(alpha_end)/cabs(alpha_start),1.0/N);
	printf("\n%lf+j%lf, %lf+j%lf, r=%f, %d\n",alpha_start_real,alpha_start_comp,alpha_end_real, alpha_end_comp, r, N);
	asf=alpha_start;
	aef=alpha_end;
	for(i=0; i<N; i++)
	{
		printf("%d\n",i);
		double complex alpha=alpha_start*pow(r,i);
		float complex alf=asf*pow(r,i);
		
		//accurate or exact roots are calculated using the best formula
		if(cabs(alpha)<1)
			exactroots(alpha);	
		else
			accroots(alpha);
			
		//float roots use the approximate floating pt. values
		floatroots(alf);
		exactroots(alpha);
		
		printf("Error, val(alpha): %.17f, %.17f, %.17f,%.17f, %.17f+j%.17f\n", cabs(f1-a1), cabs(f2-a2), cabs(f1-e1),cabs(f2-e2), creal(alpha), cimag(alpha));
		fprintf(fp,"Error: %.17f, %.17f,%.17f,%.17f, %.17f+%.17f\n", cabs(f1-a1), cabs(f2-a2), cabs(f1-e1),cabs(f2-e2), creal(alpha),cimag(alpha));
	}
	printf("%d\n",i);
	if(cabs(alpha_end)<1)
			exactroots(alpha_end);	
	else
			floatroots(aef);
	exactroots(alpha_end);
	printf("Error, val(alpha): %.17f, %.17f, %.17f+j%.17f\n", cabs(e1-a1), cabs(e2-a2), creal(alpha_end), cimag(alpha_end));
	fprintf(fp,"Error: %.17f, %.17f, %.17f, %.17f, %.17f+%.17f\n", cabs(f1-a1), cabs(f2-a2), cabs(f1-e1),cabs(f2-e2), creal(alpha_end),cimag(alpha_end));
	fclose(fp);
	return 0;
}
