#include "head.h"

double *h_t;
double *d_t;
double *h_dt;
double *d_dt;

double *h_V;
double *d_V;
double *h_dVdt;
double *d_dVdt;
double *d_dV2;
double *h_Vnew;
double *d_Vnew;

double *h_m;
double *h_m0;
double *d_m;
double *d_m0;
double *h_h;
double *h_h0;
double *d_h;
double *d_h0;
double *h_jj;
double *h_jj0;
double *d_jj;
double *d_jj0;
double *h_d;
double *h_d0;
double *d_d;
double *d_d0;
double *h_f;
double *h_f0;
double *d_f;
double *d_f0;
double *h_X;
double *h_X0;
double *d_X;
double *d_X0;
double *h_cai;
double *d_cai;

double *isi;
double *esi;
double *ina;
double *ik;
double *ik1;
double *ikp;
double *ib;
double *h_it;
double *d_it;

double *f;//double f[nx + 1][ny + 1];
double *belta;//double belta[nx + 1];
double *y_temp;//double y_temp[nx + 1];

int *d_kk0, *d_kk, *h_kk, *h_kk0;

void Allocate(){
	cudaError_t Error;
	size_t size = nx*ny*sizeof(double);
	size_t size2 = nx*ny*sizeof(int);

	h_t = (double*)malloc(sizeof(double));
	Error = cudaMalloc((void**)&d_t, sizeof(double));
	printf("CUDA error = %s\n",cudaGetErrorString(Error));	
	h_dt = (double*)malloc(size);
		cudaMalloc((void**)&d_dt, size);
			
	h_V = (double*)malloc((nx+2)*(ny+2)*sizeof(double));
		cudaMalloc((void**)&d_V, (nx+2)*(ny+2)*sizeof(double));
	h_dVdt = (double*)malloc(size);
		cudaMalloc((void**)&d_dVdt, size);
	h_Vnew = (double*)malloc(size);
		Error = cudaMalloc((void**)&d_Vnew, size);
		printf("CUDA d_Vnew error = %s\n",cudaGetErrorString(Error));
		cudaMalloc((void**)&d_dV2, size);		
		
	h_m = (double*)malloc(size);
	h_m0= (double*)malloc(size);
		cudaMalloc((void**)&d_m, size);
		cudaMalloc((void**)&d_m0, size);
	h_h = (double*)malloc(size);
	h_h0 = (double*)malloc(size);
		cudaMalloc((void**)&d_h, size);
		cudaMalloc((void**)&d_h0, size);
	h_jj = (double*)malloc(size);
	h_jj0 = (double*)malloc(size);
		cudaMalloc((void**)&d_jj, size);
		cudaMalloc((void**)&d_jj0, size);
	h_d = (double*)malloc(size);
	h_d0 = (double*)malloc(size);
        cudaMalloc((void**)&d_d, size);
		cudaMalloc((void**)&d_d0, size);
	h_f = (double*)malloc(size);
	h_f0 = (double*)malloc(size);
        cudaMalloc((void**)&d_f, size);
		cudaMalloc((void**)&d_f0, size);
	h_X = (double*)malloc(size);
	h_X0 = (double*)malloc(size);
        cudaMalloc((void**)&d_X, size);
		cudaMalloc((void**)&d_X0, size);	
	h_cai = (double*)malloc(size);	
        cudaMalloc((void**)&d_cai, size);
		
		cudaMalloc((void**)&isi, size);
		cudaMalloc((void**)&esi, size);
		cudaMalloc((void**)&ina, size);
		cudaMalloc((void**)&ik, size);
		cudaMalloc((void**)&ik1, size);
		cudaMalloc((void**)&ikp, size);
		cudaMalloc((void**)&ib, size);
	h_it = (double*)malloc(size);
		cudaMalloc((void**)&d_it, size);
	
	cudaMalloc((void**)&f, size);
	cudaMalloc((void**)&belta, nx*sizeof(double));
	cudaMalloc((void**)&y_temp, nx*sizeof(double));
	
	h_kk = (int*)malloc(size2);
	h_kk0 = (int*)malloc(size2);
		cudaMalloc((void**)&d_kk, size2);
		cudaMalloc((void**)&d_kk0, size2);		
}

void free(){

	free(h_t);free(h_dt);cudaFree(d_dt);cudaFree(d_t);
	
	free(h_V);cudaFree(d_V);free(h_dVdt);cudaFree(d_dVdt);//free(h_Vnew);cudaFree(d_Vnew);
	free(h_Vnew);cudaFree(d_Vnew);cudaFree(d_dV2);
	
	free(h_m);free(h_h);free(h_jj);
	cudaFree(d_m);cudaFree(d_h);cudaFree(d_jj);
	free(h_m0);free(h_h0);free(h_jj0);
	cudaFree(d_m0);cudaFree(d_h0);cudaFree(d_jj0);
	free(h_d);free(h_f);free(h_X);
	cudaFree(d_d);cudaFree(d_f);cudaFree(d_X);
	free(h_d0);free(h_f0);free(h_X0);
	cudaFree(d_d0);cudaFree(d_f0);cudaFree(d_X0);
	free(h_cai);cudaFree(d_cai);
	
	cudaFree(isi);cudaFree(esi);cudaFree(ina);
	cudaFree(ik);cudaFree(ik1);cudaFree(ikp);cudaFree(ib);
	free(h_it);cudaFree(d_it);
	
	cudaFree(f);cudaFree(belta);cudaFree(y_temp);
	
	free(h_kk0);free(h_kk);cudaFree(d_kk);cudaFree(d_kk0);			
}



void Manage_Comms(int phase){
        cudaError_t Error;
		size_t size = nx*ny*sizeof(double);
		size_t size2 = nx*ny*sizeof(int);
if (phase==1){
        Error = cudaMemcpy(d_dt, h_dt, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_dt->d_dt) = %s\n",cudaGetErrorString(Error));
		Error = cudaMemcpy(d_kk, h_kk, size2, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_kk->d_kk) = %s\n",cudaGetErrorString(Error));
		Error = cudaMemcpy(d_kk0, h_kk0, size2, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_kk0->d_kk0) = %s\n",cudaGetErrorString(Error));
}

if (phase==2){   
        Error = cudaMemcpy(h_dt, d_dt, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_dt->h_dt) = %s\n",cudaGetErrorString(Error));
}

if (phase==3){
        Error = cudaMemcpy(h_kk, d_kk, size2, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_Kk->h_Kk) = %s\n",cudaGetErrorString(Error));
}
if (phase==4){
        Error = cudaMemcpy(h_Vnew,d_Vnew,size,cudaMemcpyDeviceToHost);
       if (Error != cudaSuccess)printf("CUDA error(copy d_Vnew->h_Vnew) = %s\n",cudaGetErrorString(Error));
}
if (phase==5){
		cudaError_t Error;     
        Error = cudaMemcpy(h_V, d_V, (nx+2)*(ny+2)*sizeof(double), cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_V->h_V) = %s\n",cudaGetErrorString(Error));      
}
}



void Send_to_Device(){
        cudaError_t Error;
        size_t size;
        size = nx*ny*sizeof(double);

	Error = cudaMemcpy(d_t, h_t, sizeof(double), cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_t->d_t) = %s\n",cudaGetErrorString(Error));
        Error = cudaMemcpy(d_V, h_V, (nx+2)*(ny+2)*sizeof(double), cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_V->d_V) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_m, h_m, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_m->d_m) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_h, h_h, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_h->d_h) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_jj, h_jj, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_jj->d_jj) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_d, h_d, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_d->d_d) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_f, h_f, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_f->d_f) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_X->d_X) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_cai, h_cai, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_cai->d_cai) = %s\n",cudaGetErrorString(Error));
}

void Send_to_Host(){
	cudaError_t Error;
        size_t size;
        size = nx*ny*sizeof(double);

        Error = cudaMemcpy(h_V, d_V, (nx+2)*(ny+2)*sizeof(double), cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_V->h_V) = %s\n",cudaGetErrorString(Error));
        Error = cudaMemcpy(h_m, d_m, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_m->h_m) = %s\n",cudaGetErrorString(Error));
        Error = cudaMemcpy(h_h, d_h, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_h->h_h) = %s\n",cudaGetErrorString(Error));
        Error = cudaMemcpy(h_jj, d_jj, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_jj->h_jj) = %s\n",cudaGetErrorString(Error));
        Error = cudaMemcpy(h_d, d_d, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_d->h_d) = %s\n",cudaGetErrorString(Error));
        Error = cudaMemcpy(h_f, d_f, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_f->h_f) = %s\n",cudaGetErrorString(Error));
        Error = cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_X->h_X) = %s\n",cudaGetErrorString(Error));
        Error = cudaMemcpy(h_cai, d_cai, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
	printf("CUDA error(copy d_cai->h_cai) = %s\n",cudaGetErrorString(Error));
}

void Send_V(){
        cudaError_t Error;
        size_t size;
        //size = nx*ny*sizeof(double);
		//Error = cudaMemcpy(h_V, d_dV2, size, cudaMemcpyDeviceToHost);
		size = (nx+2)*(ny+2)*sizeof(double);
        Error = cudaMemcpy(h_V, d_V, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_Vnew->Vnew) = %s\n",cudaGetErrorString(Error));
}

void Send_dVdt(){
        cudaError_t Error;
        size_t size;
        size = nx*ny*sizeof(double);
		Error = cudaMemcpy(h_dVdt, d_dVdt, size, cudaMemcpyDeviceToHost);
		//size = (nx+2)*(ny+2)*sizeof(double);
        //Error = cudaMemcpy(h_V, d_V, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_dVdt->d_dVdt) = %s\n",cudaGetErrorString(Error));
}

