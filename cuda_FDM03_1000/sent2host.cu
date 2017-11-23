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
-arch=sm_61
__global__ void comp_ina(double *d_V, double *d_m, double *d_h, double *d_jj, 
		double *d_m0, double *d_h0, double *d_jj0, double *d_dt, double *d_it);
__global__ void comp_ical(double *d_V, double *d_d, double *d_f, double *d_d0, 
						double *d_f0, double *d_cai, double *d_dt, double *d_it);
__global__ void comp_ik(double *d_V, double *d_X, double *d_X0, double *d_dt,double *d_it);
__global__ void comp_ik1(double *d_V, double *d_it);
__global__ void comp_ikp(double *d_V, double *d_it);
__global__ void comp_ib(double *d_V, double *d_it);
__global__ void new_gate(double *d_m, double *d_h, double *d_jj,
						double *d_m0, double *d_h0, double *d_jj0,
						double *d_d, double *d_f, double *d_d0, double *d_f0, 
						double *d_X,double *d_X0);