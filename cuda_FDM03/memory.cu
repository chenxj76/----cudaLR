#include "head.h"

double *h_t,*d_t;
double *h_V,*d_V,*d_dV2,*h_Vnew,*d_Vnew;
double *h_m,*d_m,*h_h,*d_h,*h_jj,*d_jj;
double *h_d,*d_d,*h_f,*d_f;
double *h_X,*d_X,*h_cai,*d_cai,*h_it,*d_it;
//double *d_esi,*d_isi,*h_esi,*h_isi;

void Allocate(){
	cudaError_t Error;
	size_t size = nx*ny*sizeof(double);

	h_t = (double*)malloc(sizeof(double));
	Error = cudaMalloc((void**)&d_t, sizeof(double));
	printf("CUDA error = %s\n",cudaGetErrorString(Error));

	h_V = (double*)malloc((nx+2)*(ny+2)*sizeof(double));
	cudaMalloc((void**)&d_V, (nx+2)*(ny+2)*sizeof(double));
	cudaMalloc((void**)&d_dV2, size);
	h_Vnew = (double*)malloc(size);
	cudaMalloc((void**)&d_Vnew, size);

	

	h_m = (double*)malloc(size);
	cudaMalloc((void**)&d_m, size);
	h_h = (double*)malloc(size);
        cudaMalloc((void**)&d_h, size);
	h_jj = (double*)malloc(size);
        cudaMalloc((void**)&d_jj, size);
	h_d = (double*)malloc(size);
        cudaMalloc((void**)&d_d, size);
	h_f = (double*)malloc(size);
        cudaMalloc((void**)&d_f, size);
	h_X = (double*)malloc(size);
        cudaMalloc((void**)&d_X, size);
	h_cai = (double*)malloc(size);
        cudaMalloc((void**)&d_cai, size);
		/*
		h_esi= (double*)malloc(size);
		cudaMalloc((void**)&d_esi, size);
		h_isi= (double*)malloc(size);
		cudaMalloc((void**)&d_isi, size);
*/
	h_it = (double*)malloc(size);
	cudaMalloc((void**)&d_it, size);
}

void free(){

	free(h_t);free(h_V);free(h_m);free(h_h);
	free(h_jj);free(h_d);free(h_f);free(h_X);free(h_cai);
	free(h_Vnew);
	free(h_it);
	//free(h_esi);free(h_isi);

	cudaFree(d_t);cudaFree(d_V);cudaFree(d_dV2);cudaFree(d_Vnew);cudaFree(d_it);
	cudaFree(d_m);cudaFree(d_h);cudaFree(d_jj);cudaFree(d_d);
	cudaFree(d_f);cudaFree(d_X);cudaFree(d_cai);
	//cudaFree(d_esi);cudaFree(d_isi);
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
	/*
	Error = cudaMemcpy(d_esi, h_esi, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_esi->d_esi) = %s\n",cudaGetErrorString(Error));
	Error = cudaMemcpy(d_isi, h_isi, size, cudaMemcpyHostToDevice);
        if (Error != cudaSuccess)
        printf("CUDA error(copy h_isi->d_isi) = %s\n",cudaGetErrorString(Error));
	*/
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
        size = nx*ny*sizeof(double);

        Error = cudaMemcpy(h_Vnew, d_Vnew, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_Vnew->Vnew) = %s\n",cudaGetErrorString(Error));
}

void Save_Result(){

        FILE *pFile;
        int i,j;
        int index;
        //int n;
        //n = nx;
        pFile = fopen("V.txt","w+");
        // Save the matrix V
        for (i = 0; i < ny; i++) {
                for (j = 0; j < nx; j++) {
                        index = i*nx + j;
                        fprintf(pFile, "%g", h_Vnew[index]);
                        if (j == (nx-1)) {
                                fprintf(pFile, "\n");
                        }else{
                                fprintf(pFile, "\t");
                        }
                }
        }
        fclose(pFile);

}

