#include "head.h"

#define tpb 256

extern double *d_t;  
extern double *d_it;
extern double *d_V;
extern double *d_dV2;
extern double *d_Vnew;
extern double *d_m;
extern double *d_h;
extern double *d_jj;
extern double *d_d;
extern double *d_f;
extern double *d_X;
extern double *d_cai;



__global__ void comp_dV2(double *d_V ,double *d_dV2 ){
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){

	int j = (int)(k/nx);
	int id = k+(nx+2)+1+(2*j);//这是什么index？

	d_dV2[k] =D*((d_V[id+1] + d_V[id-1] - 2*d_V[id])
             / (dx*dx) +(d_V[id+nx+2] + d_V[id-nx-2]-2*d_V[id])/(dy*dy));

	}
}

void gpu_dV2(){
	int bpg;
	//tpb = 256;
    bpg = (nx*ny+tpb-1)/tpb;
	comp_dV2<<<bpg, tpb>>>(d_V, d_dV2);
	cudaDeviceSynchronize();
}
__global__ void comp_dV2it(double *d_it ,double *d_dV2 ){
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){

	//int j = (int)(k/nx);
	//int id = k+(nx+2)+1+(2*j);//这是什么index？

	d_dV2[k] =-d_it[k];

	}
}

void gpu_dV2it(){
	int bpg;
	//tpb = 256;
    bpg = (nx*ny+tpb-1)/tpb;
	comp_dV2it<<<bpg, tpb>>>(d_it, d_dV2);
	cudaDeviceSynchronize();
}
__global__ void plane_waves(double *d_dV2){
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<ny*5){
	int i, j;
	i = (int)(k/nx);
	j = k-i*nx;

	d_dV2[j*ny+i] = d_dV2[j*ny+i] + (-st);

	}
}

void stimu(){
	int bpg;
        //int tpb;

        //tpb = 256;
        bpg = (ny*5+tpb-1)/tpb;    // 因为刺激5列，所以开5列线程就够了
	plane_waves<<<bpg, tpb>>>(d_dV2);
	cudaDeviceSynchronize();
}

__global__ void Euler(double *d_V, double *d_dV2, double *d_Vnew, double *d_t){
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){

	int j = (int)(k/nx);
	d_Vnew[k] = d_V[k+nx+2+1+2*j] + dt*d_dV2[k];
        d_V[k+nx+2+1+2*j] = d_Vnew[k];

	}

	if(k==0){

	d_t[0] = d_t[0] + dt;

	}
}

void Forward_Euler(){
	int bpg;
        //int tpb;

        //tpb = 256;
        bpg = (nx*ny+tpb-1)/tpb;
	Euler<<<bpg, tpb>>>(d_V, d_dV2, d_Vnew, d_t);
	cudaDeviceSynchronize();
}
__global__ void Euler2(double *d_V, double *d_dV2, double *d_Vnew, double *d_t){
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){

	int j = (int)(k/nx);
	d_Vnew[k] = d_V[k+nx+2+1+2*j] + (dt/2)*d_dV2[k];
        d_V[k+nx+2+1+2*j] = d_Vnew[k];

	}

	if(k==0){

	d_t[0] = d_t[0] + dt/2;

	}
}

void Forward_Euler2(){
	int bpg;
        //int tpb;

        //tpb = 256;
        bpg = (nx*ny+tpb-1)/tpb;
	Euler2<<<bpg, tpb>>>(d_V, d_dV2, d_Vnew, d_t);
	cudaDeviceSynchronize();
}
__global__ void boundary(double *d_V){
	int k = blockDim.x * blockIdx.x + threadIdx.x;//这是global index

	if(k<nx){

	d_V[(k+1)*(nx+2)] = d_V[(k+1)*(nx+2)+1];//这些index是怎么对应？是no flux边界，这是扩充后的情况，(nx+2)*(ny+2)
        d_V[(k+1)*(nx+2)+(nx+1)] = d_V[(k+1)*(nx+2)+nx];
        d_V[k+1] = d_V[k+1+(nx+2)];
        d_V[(ny+1)*(nx+2)+k+1] = d_V[ny*(nx+2)+k+1];

	}
}

void gpu_Boun(){
	int bpg;
	//tpb = 256;
	bpg = (nx+tpb-1)/tpb;  // 边界条件只需要1列线程,算四条语句
	boundary<<<bpg, tpb>>>(d_V);
	cudaDeviceSynchronize();
}

void gpu_step123(int ncount,int stimtime){
	int bpg;
	//tpb = 256;
	//---1---
	bpg = (nx+tpb-1)/tpb;  // 边界条件只需要1列线程,算四条语句
	boundary<<<bpg, tpb>>>(d_V);
	bpg = (nx*ny+tpb-1)/tpb;
	comp_dV2<<<bpg, tpb>>>(d_V, d_dV2);
	Euler2<<<bpg, tpb>>>(d_V, d_dV2, d_Vnew, d_t);
	//---2---
	gpu_Ion();
	comp_dV2it<<<bpg, tpb>>>(d_it, d_dV2);
	if (ncount >= 1 && ncount <= stimtime) {
            bpg = (ny*5+tpb-1)/tpb;    // 因为刺激5列，所以开5列线程就够了
			plane_waves<<<bpg, tpb>>>(d_dV2);
                }
	bpg = (nx*ny+tpb-1)/tpb;
	Euler<<<bpg, tpb>>>(d_V, d_dV2, d_Vnew, d_t);
	//---3---
	bpg = (nx+tpb-1)/tpb;  // 边界条件只需要1列线程,算四条语句	
	boundary<<<bpg, tpb>>>(d_V);
	bpg = (nx*ny+tpb-1)/tpb;
	comp_dV2<<<bpg, tpb>>>(d_V, d_dV2);
	Euler2<<<bpg, tpb>>>(d_V, d_dV2, d_Vnew, d_t);
	cudaDeviceSynchronize();
}
