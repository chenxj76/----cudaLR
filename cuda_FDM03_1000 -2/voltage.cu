#include "head.h"


extern double *d_t;
extern double *d_dt;
extern double *d_it;
extern double *d_V;
extern double *d_dVdt;
extern double *d_Vnew;
extern double *belta;
extern double *y_temp;
extern double *f;
extern int *d_kk0, *d_kk;

__global__ void boundary(double *d_V){
	int k = blockDim.x * blockIdx.x + threadIdx.x;//这是global index

	if(k<nx){

	    d_V[(k+1)*(nx+2)] = d_V[(k+1)*(nx+2)+1];
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

//*********** step 1,  --- sweep in x-direction, Thomas algorithm used to solve tridiagonal linear equations ADI method*******
__global__ void step_1(double *d_V ,double *belta ,double *y_temp ,double *f){
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){
		int j = (int)(k/nx);//j =0,1,2,...,n-1。 
		int id = k+(nx+2)+1+(2*j);
		int i = (int)(k/nx);//i =0,1,2,...,n-1。此处i会不会有问题？

		//double belta[nx + 1];
		double eps = D / (dx*dx);
		double eta= eps*dt_max;//这里的时间步长为什么一直是dt_max??是否应该跟着v变而变？
		double b = 1+eta;
		double c = -eta/2; 
		double c_1 = -eta;
		double a = c;
		double a_n = c_1;
		//double f[nx + 1][ny + 1];
		//for (int j = 1; j < ny + 1; j++){
			//for (int i = 1; i < nx + 1; i++){
				if (j==0){
					//f[i][j] = V[i][j]  + (eta/2)*(V[i][j] - 2 * V[i][j] + V[i][j + 1]);
					f[j*nx+i] = d_V[id]  + (eta/2)*(d_V[id] - 2 * d_V[id] + d_V[id+(nx+2)]);
				}else if (j==ny-1){
					//f[i][j] = V[i][j] + (eta/2)*(V[i][j - 1] - 2 * V[i][j] + V[i][j]);
					f[j*nx+i] = d_V[id] + (eta/2)*(d_V[id-(nx+2)] - 2 * d_V[id] + d_V[id]);
				}else{
					//f[i][j] = V[i][j] + (eta/2)*(V[i][j - 1] - 2 * V[i][j] + V[i][j + 1]);
					f[j*nx+i] = d_V[id] + (eta/2)*(d_V[id-(nx+2)] - 2 * d_V[id] + d_V[id+(nx+2)]);
				}
			//}
		//}
		//double y_temp[nx + 1];
		//for (int j = 1; j < ny + 1; j++){
			belta[0] = c_1 / b;
			//y_temp[1] = f[1][j] / b;
			y_temp[0] = f[0+nx*j] / b;    //j =0,1,2,...,n-1。                  
			//for (int i = 2; i < nx; i++){ //i = 2,3,...,n-1
			if(i>0&&i<nx-1){
				//belta[i] = c/(b-a*belta[i-1]);
				belta[i] = c/(b-a*belta[i-1]);
				//y_temp[i] = (f[i][j] - a*y_temp[i - 1]) / (b-a*belta[i-1]);
				y_temp[i] = (f[j*nx+i] - a*y_temp[i - 1]) / (b-a*belta[i-1]);
			}
			//}
			//y_temp[nx] = (f[nx][j] - a_n*y_temp[nx - 1]) / (b - a_n*belta[nx - 1]);
			y_temp[nx-1] = (f[(nx-1)+nx*j] - a_n*y_temp[(nx - 1)-1]) / (b - a_n*belta[(nx - 1)-1]);
			//V[nx][j] = y_temp[nx];
			d_V[(nx+2)+1+(2*j)+(nx-1)+nx*j] = y_temp[nx-1];
			//for (i = nx-1; i >=1; i--){
				//V[i][j] = y_temp[i] - belta[i] * V[i+1][j];
				if(id!=((nx+2)+1+(2*j)+(nx-1)+nx*j))d_V[id] = y_temp[i] - belta[i] * d_V[id+1];//此处k/nx与i有区别吗？
			//}    //int id = k+(nx+2)+1+(2*j);
		//}
	}
}
		//*********** step 1 *******		
void gpuStep_1(){
	int bpg;
	//tpb = 256;	
    bpg = (nx*ny+tpb-1)/tpb;
	step_1<<<bpg, tpb>>>(d_V ,belta ,y_temp ,f);
	cudaDeviceSynchronize();
}	
			//*********** part of the step 2 *******	
		//dt = dt_max;//这里的时间步长为什么一直是dt_max??是否应该跟着v变而变？
	/*	for (i = 1; i < nx + 1; i++){
			for (j = 1; j < ny + 1; j++){
				it[i][j] = get_it(i, j);
				dVdt[i][j] = -it[i][j];
			}
		}*/
__global__ void comp_dVdt(double *d_dVdt  ,double *d_it){
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	if(k<nx*ny){
	//int j = (int)(k/nx);
	//int id = k+(nx+2)+1+(2*j);//这是什么index？
	//		dVdt[i][j] = -it[i][j];
	d_dVdt[k] = -d_it[k];
	}
}	
void gpu_dVdt(){
	int bpg;
	//tpb = 256;	
    bpg = (nx*ny+tpb-1)/tpb;
	comp_dVdt<<<bpg, tpb>>>(d_dVdt, d_it);
	cudaDeviceSynchronize();
}			
		//*****stimulation with a plane waves****
		/*if (ncount >= 1 && ncount <= stimtime) { //stimulus is hold with 0.6 ms
			for (i = 1; i < nx + 1; i++){
				for (j = 1; j <= 5; j++){
					dVdt[i][j] = dVdt[i][j] + (-st);
				}
			}
		}*/
__global__ void plane_waves(double *d_dVdt){
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<ny*5){
	int i, j;
	i = (int)(k/nx);
	j = k-i*nx;

	d_dVdt[j*ny+i] = d_dVdt[j*ny+i] + (-st);

	}
}

void gpu_stimu(){
	int bpg;
        //int tpb;
        //tpb = 256;
    bpg = (ny*5+tpb-1)/tpb;    // 因为刺激5列，所以开5列线程就够了
	plane_waves<<<bpg, tpb>>>(d_dVdt);
	cudaDeviceSynchronize();
}	
	
__global__ void adaptiveT(double *d_dVdt  ,double *d_dt,int *d_kk,int *d_kk0){
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	if(k<nx*ny){
		
		//int d_k0, d_k;
		//for (i = 1; i < nx + 1; i++){
			//for (j = 1; j < ny + 1; j++){
				// adaptive time step
				if (d_dVdt[k] > 0){
					d_kk0[0] = 5;
				}else{
					d_kk0[0] = 1;
				}
				d_kk[0] = d_kk0[0] + (int)(fabs(d_dVdt[k]) + 0.5); //round the value此处(dVdt[k])+0.5是为了四舍五入
				if (d_kk[0] >(int)(dt_max / dt_min)){
					d_kk[0] = (int)(dt_max / dt_min);
				}
				d_dt[0] = dt_max / d_kk[0];
			//}
		//}
	}
}	
void gpu_adaptiveT(){
	int bpg;
        //int tpb;
        //tpb = 256;
    bpg = (nx*ny+tpb-1)/tpb;
	adaptiveT<<<bpg, tpb>>>(d_dVdt, d_dt,d_kk,d_kk0);
	cudaDeviceSynchronize();
}	
__global__ void Euler(double *d_V, double *d_dVdt, double *d_Vnew, double *d_dt, double *d_t){
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	if(k<nx*ny){

	int j = (int)(k/nx);
	d_Vnew[k] = d_V[k+nx+2+1+2*j] + d_dt[0]*d_dVdt[k];
    d_V[k+nx+2+1+2*j] = d_Vnew[k];

	}

	if(k==0){

	d_t[0] = d_t[0] + d_dt[0];//此处必须加上数值类型，因为d_dt是一个指针，与d_t[0]是不同类型，否则报错。

	}
	
}

void Forward_Euler(){
	int bpg;
        //int tpb;
        //tpb = 256;
        bpg = (nx*ny+tpb-1)/tpb;
	Euler<<<bpg, tpb>>>(d_V, d_dVdt, d_Vnew, d_dt, d_t);
	cudaDeviceSynchronize();
}
		//*********** part of the step 2 *******

		//*********** step 3, sweep in y-direction, Thomas algorithm used to solve tridiagonal linear equations ADI method*******
__global__ void step_3(double *d_V ,double *belta ,double *y_temp ,double *f){
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int j = (int)(k/nx);//j =0,1,2,...,n-1。 
	int id = k+(nx+2)+1+(2*j);
	int i = (int)(k/nx);//i =0,1,2,...,n-1。
	if(k<nx*ny){
		
		
		//belta[nx + 1];
		double eta = dt_max*D / (dy*dy);//这里的时间步长为什么一直是dt_max??
		double b = 1+eta;
		double c = -eta/2;
		double c_1 = -eta;
		double a = c;
		double a_n = c_1;
		//for (i = 1; i < nx + 1; i++){
			//for (j = 1; j < ny + 1; j++){
				if (i==1){
					//f[i][j] = V[i][j] + (eta / 2)*(V[i][j] - 2 * V[i][j] + V[i + 1][j]);
					f[j*nx+i] = d_V[id]  + (eta/2)*(d_V[id] - 2 * d_V[id] + d_V[id+1]);
				}else if (i==nx){
					//f[i][j] = V[i][j] + (eta / 2)*(V[i - 1][j] - 2 * V[i][j] + V[i][j]);
					f[j*nx+i] = d_V[id]  + (eta/2)*(d_V[id-1] - 2 * d_V[id] + d_V[id]);
				}else{
					//f[i][j] = V[i][j] + (eta / 2)*(V[i - 1][j] - 2 * V[i][j] + V[i + 1][j]);
					f[j*nx+i] = d_V[id]  + (eta/2)*(d_V[id-1] - 2 * d_V[id] + d_V[id+1]);
				}
			//}
		//}

		//y_temp[nx + 1] ;
		//for (i = 1; i < nx + 1; i++){
			//belta[1] = c_1 / b;
			belta[0] = c_1 / b;
			//y_temp[1] = f[i][1] / b;
			y_temp[0] = f[i+nx*0] / b; 
			//for (j = 2; j < ny; j++){ 
			if(j>0&&j<nx-1){
				//belta[j] = c / (b - a*belta[j - 1]);
				belta[j] = c/(b-a*belta[j-1]);
				//y_temp[j] = (f[i][j] - a*y_temp[j - 1]) / (b - a*belta[j - 1]);
				y_temp[j] = (f[j*nx+i] - a*y_temp[j - 1]) / (b-a*belta[j-1]);
			}
			//y_temp[ny] = (f[i][ny] - a_n*y_temp[ny - 1]) / (b - a_n*belta[ny - 1]);
			y_temp[ny-1] = (f[i+nx*ny] - a_n*y_temp[(ny - 1)-1]) / (b - a_n*belta[(ny - 1)-1]);
			//V[i][ny] = y_temp[ny];
			d_V[(nx+2)+1+(2*j)+i+nx*(ny-1)] = y_temp[ny-1];
			//for (j = ny - 1; j >= 1; j--){
				//V[i][j] = y_temp[j] - belta[j] * V[i][j + 1];
				if(id!=((nx+2)+1+(2*j)+i+nx*(ny-1)))d_V[id] = y_temp[j] - belta[j] * d_V[id+(nx+2)];
			//}
		}
}
void gpuStep_3(){
	int bpg;
	//tpb = 256;	
    bpg = (nx*ny+tpb-1)/tpb;
	step_3<<<bpg, tpb>>>(d_V ,belta ,y_temp ,f);
	cudaDeviceSynchronize();
}	
	
		//*********** step 3 *******

		//t = t + dt_max;