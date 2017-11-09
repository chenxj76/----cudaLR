#include "head.h"

extern double *d_t;
extern double *d_V,*d_dV2,*d_Vnew;
extern double *d_m,*d_h,*d_jj;
extern double *d_d,*d_f;
extern double *d_X,*d_cai,*d_it;
//extern double *d_esi,*d_isi;

__global__ void boundary(double *d_V);
__global__ void comp_ina(double *d_V, double *d_m, double *d_h, double *d_jj, double *d_it) ;
//__global__ void comp_ical(double *d_V, double *d_d, double *d_f, double *d_cai, double *d_it);//,double *d_esi,double *d_isi
//__global__ void comp_ik(double *d_V, double *d_X, double *d_it);
//__global__ void comp_ik1(double *d_V, double *d_it) ;
//__global__ void comp_ikp(double *d_V, double *d_it) ;
//__global__ void comp_ib(double *d_V, double *d_it) ;
//__global__ void comp_dV2(double *d_V ,double *d_dV2  ,double *d_it);
__global__ void plane_waves(double *d_dV2);
__global__ void Euler(double *d_V, double *d_dV2, double *d_Vnew, double *d_t);
//------------------------------------------------------

__global__ void boundary(double *d_V){
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//matrixIdx
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;	
//unsigned int id = iy*(nx+2)+ ix;//globalIdx，包括4条边
if(ix<nx&&iy<ny){//条件控制有多少个点赋初值
	d_V[(iy+1)*(nx+2)+0] = d_V[(iy+1)*(nx+2)+1];
	d_V[(iy+1)*(nx+2)+(nx+1)] = d_V[(iy+1)*(nx+2)+nx];			
	d_V[(ix+1)+0*(nx+2)] = d_V[(ix+1)+1*(nx+2)];
	d_V[(ix+1)+(ny+1)*(nx+2)] = d_V[(ix+1)+ny*(nx+2)];
	//printf("ix=%d",ix);
	}
}
void gpu_boun(){
    dim3 block(BLOCK_SIZE,1);
	dim3 grid((nx+block.x-1)/block.x,1);// 边界条件只需要1列线程,算四条语句	
	boundary<<<grid,block>>>(d_V);
	cudaError_t Error;
	Error=cudaDeviceSynchronize();
	if (Error != cudaSuccess)printf("gpu_bounSynchronize:%s\n",cudaGetErrorString(Error));
}
//--------------------------------------------------------
__global__ void comp_ina(double *d_V, double *d_m, double *d_h, double *d_jj, double *d_it) {
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//matrixIdx
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//unsigned int id=iy*(nx+2)+ix;//扩充了的global index
unsigned	int idx = (iy+1)*(nx+2)+ (ix +1);//V的globalIdx,不要计算4条边界
unsigned	int id = iy*nx+ ix;//除了V其他变量的index
	if(ix<nx&&iy<ny){
		//printf("iy=%d",iy);

	d_it[id] = 0.0;

	double gna = 23.0;
    double ena = ((R*temp) / frdy)*log(nao / nai);

    double am = 0.32*(d_V[idx] + 47.13) / (1 - exp(-0.1*(d_V[idx] + 47.13)));
    double bm = 0.08*exp(-d_V[idx] / 11);
	double ah, bh, aj ,bj;
        if (d_V[idx] < -40.0) {
                ah = 0.135*exp((80 + d_V[idx]) / -6.8);
                bh = 3.56*exp(0.079*d_V[idx]) + 310000 * exp(0.35*d_V[idx]);
                aj = (-127140 * exp(0.2444*d_V[idx]) - 0.00003474*exp(-0.04391*d_V[idx]))*
                        ((d_V[idx] + 37.78)/(1 + exp(0.311*(d_V[idx] + 79.23))));
                bj = (0.1212*exp(-0.01052*d_V[idx])) / (1 + exp(-0.1378*(d_V[idx] + 40.14)));
        }
        else {
                ah = 0;
                bh = 1 / (0.13*(1 + exp((d_V[idx] + 10.66) / -11.1)));
                aj = 0;
                bj = (0.3*exp(-0.0000002535*d_V[idx])) / (1 + exp(-0.1*(d_V[idx] + 32)));
        }
        double mtau = 1 / (am + bm);
        double htau = 1 / (ah + bh);
		double jtau = 1 / (aj + bj);

        double mss = am*mtau;
        double hss = ah*htau;
        double jss = aj*jtau;

        d_m[id] = mss - (mss - d_m[id])*exp(-dt / mtau);
        d_h[id] = hss - (hss - d_h[id])*exp(-dt / htau);
        d_jj[id] = jss - (jss - d_jj[id])*exp(-dt / jtau);

        d_it[id] += gna*d_m[id] * d_m[id] * d_m[id] * d_h[id] * d_jj[id] * (d_V[i dx] - ena);
		//printf("d_it[%d]=%f\t",id,d_it[id]);
	}
	
}

__global__ void comp_ical(double *d_V, double *d_d, double *d_f, double *d_cai, double *d_it){/*,double *d_esi,double *d_isi*/
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//matrixIdx
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//int i = threadIdx.x+threadIdx.y*blockDim.x;//一个block内的index
//unsigned int id=iy*(nx+2)+ix;//扩充了的global index
unsigned 	int idx = (iy+1)*(nx+2)+ (ix +1);//globalIdx,不要计算4条边界
unsigned     int id = iy*nx+ ix;//除了V其他变量的index  
	if(ix<nx&&iy<ny){	
//__shared__ double d_esi[256];//这两个变量设置共享内存，也可以不设置，不影响
//__shared__ double d_isi[256];
		double d_esi = 7.7 - 13.0287*log(d_cai[id]);

        double ad = 0.095*exp(-0.01*(d_V[idx] - 5)) / (1 + exp(-0.072*(d_V[idx] - 5)));
        double bd = 0.07*exp(-0.017*(d_V[idx] + 44)) / (1 + exp(0.05*(d_V[idx] + 44)));
        double af = 0.012*exp(-0.008*(d_V[idx] + 28)) / (1 + exp(0.15*(d_V[idx] + 28)));
        double bf = 0.0065*exp(-0.02*(d_V[idx] + 30)) / (1 + exp(-0.2*(d_V[idx] + 30)));

        double taud = 1 / (ad + bd);
        double tauf = 1 / (af + bf);

        double dss = ad*taud;
        double fss = af*tauf;

        d_d[id] = dss - (dss - d_d[id])*exp(-dt / taud);
        d_f[id] = fss - (fss - d_f[id])*exp(-dt / tauf);

        double d_isi = 0.09*d_d[id] * d_f[id] * (d_V[idx] - d_esi);

        double dcai = -0.0001*d_isi + 0.07*(0.0001 - d_cai[id]);

        d_cai[id] = d_cai[id] + dcai*dt;
	    d_it[id] = d_it[id] + d_isi;

	}
}

__global__ void comp_ik(double *d_V, double *d_X, double *d_it){
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//matrixIdx
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//unsigned int id=iy*(nx+2)+ix;//扩充了的global index
unsigned	int idx = (iy+1)*(nx+2)+ (ix +1);//globalIdx,不要计算4条边界
unsigned    int id = iy*nx+ ix;//除了V其他变量的index 
	if(ix<nx&&iy<ny){

		double gk = 0.282*sqrt(ko / 5.4);
        double ek = ((R*temp) / frdy)*log(ko / ki);
        //double prnak = 0.01833;
        //ek = ((R*temp) / frdy)*log((ko + prnak*nao) / (ki + prnak*nai));

        double ax = 0.0005*exp(0.083*(d_V[idx] + 50)) / (1 + exp(0.057*(d_V[idx] + 50)));
        double bx = 0.0013*exp(-0.06*(d_V[idx] + 20)) / (1 + exp(-0.04*(d_V[idx] + 20)));

        double taux = 1 / (ax + bx);
        double xss = ax*taux;
        d_X[id] = xss - (xss - d_X[id])*exp(-dt / taux);

	    double Xi;
        if (d_V[idx] > -100) {
                Xi = 2.837*(exp(0.04*(d_V[idx] + 77)) - 1)/
			((d_V[idx] + 77)*exp(0.04*(d_V[idx] + 35)));
        }
        else {
                Xi = 1;
        }
        d_it[id] += gk*d_X[id] * Xi*(d_V[idx] - ek);

	}
}

__global__ void comp_ik1(double *d_V, double *d_it) {
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//matrixIdx
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//unsigned int id=iy*(nx+2)+ix;//扩充了的global index
unsigned	int idx = (iy+1)*(nx+2)+ (ix +1);//globalIdx,不要计算4条边界
unsigned    int id = iy*nx+ ix;//除了V其他变量的index  
	if(ix<nx&&iy<ny){

		double gk1 = 0.6047*(sqrt(ko / 5.4));
        double ek1 = ((R*temp) / frdy)*log(ko / ki);

        double ak1 = 1.02 / (1 + exp(0.2385*(d_V[idx] - ek1 - 59.215)));
        double bk1 = (0.49124*exp(0.08032*(d_V[idx] - ek1 + 5.476))+
			exp(0.06175*(d_V[idx] - ek1 - 594.31)))
                	/(1 + exp(-0.5143*(d_V[idx] - ek1 + 4.753)));
        double K1ss = ak1 / (ak1 + bk1);

        d_it[id] += gk1*K1ss*(d_V[idx] - ek1);

	}
}

__global__ void comp_ikp(double *d_V, double *d_it) {
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//matrixIdx
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//unsigned int id=iy*(nx+2)+ix;//扩充了的global index
unsigned	int idx = (iy+1)*(nx+2)+ (ix +1);//globalIdx,不要计算4条边界
unsigned    int id = iy*nx+ ix;//除了V其他变量的index
	if(ix<nx&&iy<ny){

		double gkp = 0.0183;
        double ekp = ((R*temp) / frdy)*log(ko / ki);
        double kp = 1 / (1 + exp((7.488 - d_V[idx]) / 5.98));

        d_it[id] += gkp*kp*(d_V[idx] - ekp);

	}
}

__global__ void comp_ib(double *d_V, double *d_it) {
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//matrixIdx
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//unsigned int id=iy*(nx+2)+ix;//扩充了的global index
unsigned 	int idx = (iy+1)*(nx+2)+ (ix +1);//globalIdx,不要计算4条边界
unsigned    int id = iy*nx+ ix;//除了V其他变量的index
	if(ix<nx&&iy<ny){

		d_it[id] += 0.03921*(d_V[idx] + 59.87);
		
	}
}

__global__ void comp_dV2(double *d_V ,double *d_dV2  ,double *d_it){
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//matrixIdx
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//unsigned int id=iy*(nx+2)+ix;//扩充了的global index
unsigned int idx = (iy+1)*(nx+2)+ (ix +1);//globalIdx,不要计算4条边界
unsigned int id = iy*nx+ ix;//除了V其他变量的index
	if(ix<nx&&iy<ny){
	
	d_dV2[id] = -d_it[id] + D*((d_V[idx+1] + d_V[idx-1] - 2*d_V[idx])
                                 / (dx*dx) +(d_V[idx+(nx+2)] + d_V[idx-(nx+2)]-2*d_V[idx])/(dy*dy));
    //printf("d_dV2=%f\n",d_it[id]);
	}
}


void gpu_comp_ina(){   	
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);	
	dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
	comp_ina<<<grid,block>>>(d_V, d_m, d_h, d_jj, d_it);
	//cudaError_t Error;
	//Error=cudaDeviceSynchronize();
	//if (Error != cudaSuccess)printf("comp_inaSynchronize:%s\n",cudaGetErrorString(Error));
//}
//void gpu_comp_ical(){ 	
 //   dim3 block(BLOCK_SIZE,BLOCK_SIZE);	
//	dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);	
	comp_ical<<<grid,block>>>(d_V, d_d, d_f, d_cai, d_it);//,d_esi,d_isi
//	cudaError_t Error;
//	Error=cudaDeviceSynchronize();
//	if (Error != cudaSuccess)printf("comp_icalSynchronize:%s\n",cudaGetErrorString(Error));
//}	
//void gpu_comp_ik(){ 	
//	dim3 block(BLOCK_SIZE,BLOCK_SIZE);	
//	dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
	comp_ik<<<grid,block>>>(d_V, d_X, d_it);
//	cudaError_t Error;
//	Error=cudaDeviceSynchronize();
//	if (Error != cudaSuccess)printf("comp_ikSynchronize:%s\n",cudaGetErrorString(Error));
//}
//void gpu_comp_ik1(){ 
//	dim3 block(BLOCK_SIZE,BLOCK_SIZE);	
//	dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
	comp_ik1<<<grid,block>>>(d_V, d_it);
//	cudaError_t Error;
//	Error=cudaDeviceSynchronize();
//	if (Error != cudaSuccess)printf("comp_ik1Synchronize:%s\n",cudaGetErrorString(Error));
//}
//void gpu_comp_ikp(){ 
//	dim3 block(BLOCK_SIZE,BLOCK_SIZE);	
//	dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
	comp_ikp<<<grid,block>>>(d_V, d_it);
//	cudaError_t Error;
//	Error=cudaDeviceSynchronize();
//	if (Error != cudaSuccess)printf("comp_ikpSynchronize:%s\n",cudaGetErrorString(Error));
//}
//void gpu_comp_ib(){ 
//	dim3 block(BLOCK_SIZE,BLOCK_SIZE);	
//	dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
	comp_ib<<<grid,block>>>(d_V, d_it);
//	cudaError_t Error;
//	Error=cudaDeviceSynchronize();
//	if (Error != cudaSuccess)printf("comp_ibSynchronize:%s\n",cudaGetErrorString(Error));
//}
//void gpu_comp_dV2(){ 
//	dim3 block(BLOCK_SIZE,BLOCK_SIZE);	
//	dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
	comp_dV2<<<grid,block>>>(d_V, d_dV2, d_it);
	cudaError_t Error;
	Error=cudaDeviceSynchronize();
	if (Error != cudaSuccess)printf("comp_dV2Synchronize:%s\n",cudaGetErrorString(Error));
}

__global__ void plane_waves(double *d_dV2){
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//matrixIdx
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//unsigned int id=iy*(nx+2)+ix;//扩充了的global index
unsigned int id = iy*nx+ ix;//除了V其他变量的index
	if(ix<5&&iy<nx){
//int idx = (iy+1)*(nx+2)+ (ix +1);//globalIdx,不要计算4条边界
	//int id = iy*nx+ ix;//除了V其他变量的index
	d_dV2[id] = d_dV2[id] + (-st);
	//printf("d_dV2=%18.7e\t st=%f\n",d_dV2[id],st);
    
	}
}

void stimu(){
	dim3 block(BLOCK_SIZE,1);// 因为刺激5列，所以开5列线程就够了
	dim3 grid((nx*5+block.x-1)/block.x,1);
   
	plane_waves<<<grid,block>>>(d_dV2);
	cudaError_t Error;
	Error=cudaDeviceSynchronize();
	if (Error != cudaSuccess)printf("plane_wavesSynchronize:%s\n",cudaGetErrorString(Error));
}

__global__ void Euler(double *d_V, double *d_dV2, double *d_Vnew, double *d_t){
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//matrixIdx
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
//unsigned int id=iy*(nx+2)+ix;//扩充了的global index
unsigned 	int idx = (iy+1)*(nx+2)+ (ix +1);//globalIdx,不要计算4条边界
unsigned 	int id = iy*nx+ ix;//除了V其他变量的index
	if(ix<nx&&iy<ny){
	
	d_Vnew[id] = d_V[idx] + dt*d_dV2[id];
    d_V[idx] = d_Vnew[id];
  
	}

	if(id==0){
	d_t[0] = d_t[0] + dt;//暂时用不上
	}
}

void Forward_Euler(){
	dim3 block(BLOCK_SIZE,BLOCK_SIZE);	
	dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
	Euler<<<grid,block>>>(d_V, d_dV2, d_Vnew, d_t);
	cudaError_t Error;
	Error=cudaDeviceSynchronize();
	if (Error != cudaSuccess)printf("Forward_EulerSynchronize:%s\n",cudaGetErrorString(Error));
}
