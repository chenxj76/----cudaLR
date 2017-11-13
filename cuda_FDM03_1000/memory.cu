#include "head.h"

double *h_t;
double *d_t;
double *h_dt;
double *d_dt;
double *h_V;
double *d_V;
double *d_dVdt;
double *h_Vnew;
double *d_Vnew;
double *d_it;

double *h_m;
double *d_m;
double *d_m0;
double *h_h;
double *d_h;
double *d_h0;
double *h_jj;
double *d_jj;
double *d_jj0;
double *h_d;
double *d_d;
double *d_d0;
double *h_f;
double *d_f;
double *d_f0;
double *h_X;
double *d_X;
double *d_X0;
double *h_cai;
double *d_cai;

double *h_it;

double *f;//double f[nx + 1][ny + 1];
double *belta;//double belta[nx + 1];
double *y_temp;//double y_temp[nx + 1];

int *d_kk0, *d_kk, *h_kk;

void Allocate(){
	cudaError_t Error;
	size_t size = nx*ny*sizeof(double);

	h_t = (double*)malloc(sizeof(double));
	Error = cudaMalloc((void**)&d_t, sizeof(double));
	printf("CUDA error = %s\n",cudaGetErrorString(Error));	
	h_dt = (double*)malloc(sizeof(double));
		cudaMalloc((void**)&d_dt, sizeof(double));
	h_kk = (int*)malloc(sizeof(int));
		cudaMalloc((void**)&d_kk, sizeof(int));
	h_V = (double*)malloc((nx+2)*(ny+2)*sizeof(double));
		cudaMalloc((void**)&d_V, (nx+2)*(ny+2)*sizeof(double));
		cudaMalloc((void**)&d_dVdt, size);
	h_Vnew = (double*)malloc(size);
		cudaMalloc((void**)&d_Vnew, size);

		cudaMalloc((void**)&d_it, size);

	h_m = (double*)malloc(size);
		cudaMalloc((void**)&d_m, size);
		cudaMalloc((void**)&d_m0, size);
	h_h = (double*)malloc(size);
		cudaMalloc((void**)&d_h, size);
		cudaMalloc((void**)&d_h0, size);
	h_jj = (double*)malloc(size);
		cudaMalloc((void**)&d_jj, size);
		cudaMalloc((void**)&d_jj0, size);
	h_d = (double*)malloc(size);
        cudaMalloc((void**)&d_d, size);
		cudaMalloc((void**)&d_d0, size);
	h_f = (double*)malloc(size);
        cudaMalloc((void**)&d_f, size);
		cudaMalloc((void**)&d_f0, size);
	h_X = (double*)malloc(size);
        cudaMalloc((void**)&d_X, size);
		cudaMalloc((void**)&d_X0, size);
	h_cai = (double*)malloc(size);
        cudaMalloc((void**)&d_cai, size);

	h_it = (double*)malloc(size);
	
	cudaMalloc((void**)&f, size);
	cudaMalloc((void**)&belta, nx*sizeof(double));
	cudaMalloc((void**)&y_temp, nx*sizeof(double));
}

void free(){

	free(h_t);free(h_V);free(h_m);free(h_h);
	free(h_jj);free(h_d);free(h_f);free(h_X);free(h_cai);
	free(h_Vnew);
	free(h_it);
	free(h_dt);cudaFree(d_dt);
	free(h_kk);cudaFree(d_kk);
	cudaFree(d_t);cudaFree(d_V);cudaFree(d_dVdt);cudaFree(d_Vnew);cudaFree(d_it);
	cudaFree(d_m);cudaFree(d_h);cudaFree(d_jj);cudaFree(d_d);
	cudaFree(d_m0);cudaFree(d_h0);cudaFree(d_jj0);cudaFree(d_d0);
	cudaFree(d_f);cudaFree(d_X);cudaFree(d_f0);cudaFree(d_X0);cudaFree(d_cai);
	cudaFree(f);cudaFree(belta);cudaFree(y_temp);
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
/*
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
*/
void Send_V(){
        cudaError_t Error;
        size_t size;
        size = nx*ny*sizeof(double);

        Error = cudaMemcpy(h_Vnew,d_Vnew,size,cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)printf("CUDA error(copy d_Vnew->h_Vnew) = %s\n",cudaGetErrorString(Error));
}
void Send2deviceT(){
        cudaError_t Error;
        size_t size;
        size = sizeof(double);

        Error = cudaMemcpy(d_dt, h_dt, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_dt->d_dt) = %s\n",cudaGetErrorString(Error));
}
void Send2hostT(){
        cudaError_t Error;
        size_t size;
        size = sizeof(double);

        Error = cudaMemcpy(h_dt, d_dt, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_dt->h_dt) = %s\n",cudaGetErrorString(Error));
}
void Send2hostK(){
        cudaError_t Error;
        size_t size;
        size = sizeof(int);

        Error = cudaMemcpy(h_kk, d_kk, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_Kk->h_Kk) = %s\n",cudaGetErrorString(Error));
}

