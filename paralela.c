#include <stdio.h>
#include <math.h>
#include "omp.h"
#define PI 3.1415926535
#define NUM_THREADS 4

static long N = 50000;

FILE *fptr1;
FILE *fptr2;
FILE *fptr3;
FILE *fptr4;


void funcionrk1(){
    fptr1=fopen("Runge_Kutta1_Par.txt","w");
    printf("Numero de pasos:%d Atendido por thread:%d\n", N,omp_get_thread_num());
    fprintf(fptr1, "Datos que encuentra el metodo de Runge Kutta(variable ind.\t variable dep.\t numero de thread)\n");
    double h,t,w, k1= 0.0, k2= 0.0, k3=0.0, k4=0.0, w0=PI/4, t0=0,a=0.0,b=PI, ab=0.0;
    int i;
    w=w0;
    fprintf(fptr1, "%f\t %f\n", a, w);
    for(i=0;i<N;i++){
        h=(b-a)/N;
        t=a+(h*i);
        ab=t*t;
        k1= h*(t*exp(3*t)-2*w);
		k2= h*((t+h/2.0)*exp(3*(t+h/2.0))-2*(w+k1/2.0));
		k3= h*((t+h/2.0)*exp(3*(t+h/2.0))-2*(w+k2/2.0));
		k4= h*((t+h)*exp(3*(t+h))-2*(w+k3));
		w=w+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4);
        fprintf(fptr1, "%f\t %f \t numero de thread:%d\n", t+h, w,omp_get_thread_num());
        }
   fclose(fptr1);
}

void funcionrk2(){
    fptr2=fopen("Runge_Kutta2_Par.txt","w");
    printf("Numero de pasos:%d Atendido por thread:%d\n", N,omp_get_thread_num());
    fprintf(fptr2, "Datos que encuentra el metodo de Runge Kutta(variable ind.\t variable dep.\t numero de thread)\n");
    double h,t,w, k1= 0.0, k2= 0.0, k3=0.0, k4=0.0, w0=PI/4, t0=0,a=0.0,b=PI, ab=0.0;
    int i;
    w=w0;
    fprintf(fptr2, "%f\t %f\n", a, w);
    for(i=0;i<N;i++){
        h=(b-a)/N;
        t=a+(h*i);
        ab=t*t;
        k1= h*(1+pow(t-w,2));
		k2= h*(1+pow((t+h/2.0)-(w+k1/2.0),2));
		k3= h*(1+pow((t+h/2.0)-(w+k2/2.0),2));
		k4= h*(1+pow((t+h)-(w+k3),2));
		w=w+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4);
        fprintf(fptr2, "%f\t %f \t numero de thread:%d\n", t+h, w,omp_get_thread_num());
        }
   fclose(fptr2);
}

void funcionrk3(){
    fptr3=fopen("Runge_Kutta3_Par.txt","w");
    printf("Numero de pasos:%d Atendido por thread:%d\n", N,omp_get_thread_num());
    fprintf(fptr3, "Datos que encuentra el metodo de Runge Kutta(variable ind.\t variable dep.\t numero de thread)\n");
    double h,t,w, k1= 0.0, k2= 0.0, k3=0.0, k4=0.0, w0=PI/4, t0=0,a=0.0,b=PI, ab=0.0;
    int i;
    w=w0;
    fprintf(fptr3, "%f\t %f\n", a, w);
    for(i=0;i<N;i++){
        h=(b-a)/N;
        t=a+(h*i);
        ab=t*t;
        k1= h*(1+t/w);
		k2= h*(1+(t+h/2.0)/(w+k1/2.0));
		k3= h*(1+(t+h/2.0)/(w+k2/2.0));
		k4= h*(1+(t+h)/(w+k3));
		w=w+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4);
        fprintf(fptr3, "%f\t %f \t numero de thread:%d\n", t+h, w,omp_get_thread_num());
        }
   fclose(fptr3);
}

void funcionrk4(){
    fptr4=fopen("Runge_Kutta4_Par.txt","w");
    printf("Numero de pasos:%d Atendido por thread:%d\n", N,omp_get_thread_num());
    fprintf(fptr4, "Datos que encuentra el metodo de Runge Kutta(variable ind.\t variable dep.\t numero de thread)\n");
    double h,t,w, k1= 0.0, k2= 0.0, k3=0.0, k4=0.0, w0=PI/4, t0=0,a=0.0,b=PI, ab=0.0;
    int i;
    w=w0;
    fprintf(fptr4, "%f\t %f\n", a, w);
    for(i=0;i<N;i++){
        h=(b-a)/N;
        t=a+(h*i);
        ab=t*t;
        k1= h*(cos(2*t*w)+sin(3*t*w));
		k2= h*(cos(2*(t+h/2.0)*(w+k1/2.0))+sin(3*(t+h/2.0)*(w+k1/2.0)));
		k3= h*(cos(2*(t+h/2.0)*(w+k2/2.0))+sin(3*(t+h/2.0)*(w+k2/2.0)));
		k4= h*(cos(2*(t+h)*(w+k3))+sin(3*(t+h)*(w+k3)));
		w=w+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4);
        fprintf(fptr4, "%f\t %f \t numero de thread:%d\n", t+h, w,omp_get_thread_num());
        }
   fclose(fptr4);
}

void main(int argc, char const *argv[]){
    const double startTime = omp_get_wtime();


    omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel 
	{
		#pragma omp sections
		{
			#pragma omp section
				funcionrk1();
			#pragma omp section
				funcionrk2();
			#pragma omp section
				funcionrk3();
			#pragma omp section
				funcionrk4();
		}
	}


    const double endTime = omp_get_wtime();
    printf("tomo (%lf) segundos\n",(endTime - startTime));
}