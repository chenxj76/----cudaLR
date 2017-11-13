#include "head.h"

extern double *h_Vnew;
extern double *h_dt;
extern double *d_t;
extern double *d_dt;
extern double *d_it;

extern double *d_V;
extern double *d_dVdt;
extern double *d_Vnew;
extern double *belta;
extern double *y_temp;
extern double *f;
extern int *d_kk0, *d_kk,*h_kk;;

extern double *d_m;
extern double *d_m0;
extern double *d_h;
extern double *d_h0;
extern double *d_jj;
extern double *d_jj0;
extern double *d_d;
extern double *d_d0;
extern double *d_f;
extern double *d_f0;
extern double *d_X;
extern double *d_X0;
extern double *d_cai;


__global__ void comp_ina(double *d_V, double *d_m, double *d_h, double *d_jj, 
		double *d_m0, double *d_h0, double *d_jj0, double *d_dt, double *d_it) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){

		int j = (int)(k/nx);//相当于二维矩阵中V[i][j]的i行，这里的j表示矩阵的i。
		//int id = k+(nx+2)+1+2*j;//这是什么index？这是扩充为(nx+2)*(ny+2)后的global index,这里+1是为了不要（0,0）这一点，(nx+2)的意义是不要第0行。
		d_it[k] = 0.0;
		
		double gna = 23.0;
        double ena = ((R*temp) / frdy)*log(nao / nai);

        double am = 0.32*(d_V[k+nx+2+1+2*j] + 47.13) / (1 - exp(-0.1*(d_V[k+nx+2+1+2*j] + 47.13)));
        double bm = 0.08*exp(-d_V[k+nx+2+1+2*j] / 11);
		double ah, bh, aj ,bj;
        if (d_V[k+nx+2+1+2*j] < -40.0) {
                ah = 0.135*exp((80 + d_V[k+nx+2+1+2*j]) / -6.8);
                bh = 3.56*exp(0.079*d_V[k+nx+2+1+2*j]) + 310000 * exp(0.35*d_V[k+nx+2+1+2*j]);
                aj = (-127140 * exp(0.2444*d_V[k+nx+2+1+2*j]) - 0.00003474*exp(-0.04391*d_V[k+nx+2+1+2*j]))*
                        ((d_V[k+nx+2+1+2*j] + 37.78)/(1 + exp(0.311*(d_V[k+nx+2+1+2*j] + 79.23))));
                bj = (0.1212*exp(-0.01052*d_V[k+nx+2+1+2*j])) / (1 + exp(-0.1378*(d_V[k+nx+2+1+2*j] + 40.14)));
        }
        else {
                ah = 0;
                bh = 1 / (0.13*(1 + exp((d_V[k+nx+2+1+2*j] + 10.66) / -11.1)));
                aj = 0;
                bj = (0.3*exp(-0.0000002535*d_V[k+nx+2+1+2*j])) / (1 + exp(-0.1*(d_V[k+nx+2+1+2*j] + 32)));
        }
        double mtau = 1 / (am + bm);
        double htau = 1 / (ah + bh);
		double jtau = 1 / (aj + bj);

        double mss = am*mtau;
        double hss = ah*htau;
        double jss = aj*jtau;
		
        d_m0[k] = mss - (mss - d_m[k])*exp(- d_dt[0] / mtau);//可能是d_dt类型出了问题
        d_h0[k] = hss - (hss - d_h[k])*exp(- d_dt[0] / htau);
        d_jj0[k] = jss - (jss - d_jj[k])*exp(- d_dt[0] / jtau);

        d_it[k] += gna*d_m0[k] * d_m0[k] * d_m0[k] * d_h0[k] * d_jj0[k] * (d_V[k+nx+2+1+2*j] - ena);

	}
}

__global__ void comp_ical(double *d_V, double *d_d, double *d_f, double *d_d0, 
						double *d_f0, double *d_cai, double *d_dt, double *d_it){
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.x;

	if(k<nx*ny){

	int j = (int)(k/nx);

	__shared__ double esi[tpb];//这两个变量设置共享内存，也可以不设置，不影响
	__shared__ double isi[tpb];
        esi[i] = 7.7 - 13.0287*log(d_cai[k]);

        double ad = 0.095*exp(-0.01*(d_V[k+nx+2+1+2*j] - 5)) / (1 + exp(-0.072*(d_V[k+nx+2+1+2*j] - 5)));
        double bd = 0.07*exp(-0.017*(d_V[k+nx+2+1+2*j] + 44)) / (1 + exp(0.05*(d_V[k+nx+2+1+2*j] + 44)));
        double af = 0.012*exp(-0.008*(d_V[k+nx+2+1+2*j] + 28)) / (1 + exp(0.15*(d_V[k+nx+2+1+2*j] + 28)));
        double bf = 0.0065*exp(-0.02*(d_V[k+nx+2+1+2*j] + 30)) / (1 + exp(-0.2*(d_V[k+nx+2+1+2*j] + 30)));

        double taud = 1 / (ad + bd);
        double tauf = 1 / (af + bf);

        double dss = ad*taud;
        double fss = af*tauf;

        d_d0[k] = dss - (dss - d_d[k])*exp(-d_dt[0] / taud);
        d_f0[k] = fss - (fss - d_f[k])*exp(-d_dt[0] / tauf);

        isi[i] = 0.09*d_d0[k] * d_f0[k] * (d_V[k+nx+2+1+2*j] - esi[i]);

        double dcai = -0.0001*isi[i] + 0.07*(0.0001 - d_cai[k]);

        d_cai[k] = d_cai[k] + dcai*d_dt[0];
		d_it[k] = d_it[k] + isi[i];

	}
}

__global__ void comp_ik(double *d_V, double *d_X, double *d_X0, double *d_dt,double *d_it){
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){

	int j = (int)(k/nx);

        double gk = 0.282*sqrt(ko / 5.4);
        double ek = ((R*temp) / frdy)*log(ko / ki);
        //double prnak = 0.01833;
        //ek = ((R*temp) / frdy)*log((ko + prnak*nao) / (ki + prnak*nai));

        double ax = 0.0005*exp(0.083*(d_V[k+nx+2+1+2*j] + 50)) / (1 + exp(0.057*(d_V[k+nx+2+1+2*j] + 50)));
        double bx = 0.0013*exp(-0.06*(d_V[k+nx+2+1+2*j] + 20)) / (1 + exp(-0.04*(d_V[k+nx+2+1+2*j] + 20)));

        double taux = 1 / (ax + bx);
        double xss = ax*taux;
        d_X0[k] = xss - (xss - d_X[k])*exp(-d_dt[0] / taux);

		double Xi;
        if (d_V[k+nx+2+1+2*j] > -100) {
                Xi = 2.837*(exp(0.04*(d_V[k+nx+2+1+2*j] + 77)) - 1)/
			((d_V[k+nx+2+1+2*j] + 77)*exp(0.04*(d_V[k+nx+2+1+2*j] + 35)));
        }
        else {
                Xi = 1;
        }
        d_it[k] += gk*d_X0[k] * Xi*(d_V[k+nx+2+1+2*j] - ek);

	}
}

__global__ void comp_ik1(double *d_V, double *d_it) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){

	int j = (int)(k/nx);

        double gk1 = 0.6047*(sqrt(ko / 5.4));
        double ek1 = ((R*temp) / frdy)*log(ko / ki);

        double ak1 = 1.02 / (1 + exp(0.2385*(d_V[k+nx+2+1+2*j] - ek1 - 59.215)));
        double bk1 = (0.49124*exp(0.08032*(d_V[k+nx+2+1+2*j] - ek1 + 5.476))+
			exp(0.06175*(d_V[k+nx+2+1+2*j] - ek1 - 594.31)))
                	/(1 + exp(-0.5143*(d_V[k+nx+2+1+2*j] - ek1 + 4.753)));
        double K1ss = ak1 / (ak1 + bk1);

        d_it[k] += gk1*K1ss*(d_V[k+nx+2+1+2*j] - ek1);

	}
}

__global__ void comp_ikp(double *d_V, double *d_it) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){

	int j = (int)(k/nx);

        double gkp = 0.0183;
        double ekp = ((R*temp) / frdy)*log(ko / ki);

        double kp = 1 / (1 + exp((7.488 - d_V[k+nx+2+1+2*j]) / 5.98));

        d_it[k] += gkp*kp*(d_V[k+nx+2+1+2*j] - ekp);

	}
}

__global__ void comp_ib(double *d_V, double *d_it) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){

	int j = (int)(k/nx);

        d_it[k] += 0.03921*(d_V[k+nx+2+1+2*j] + 59.87);

	}
}
/*
__global__ void renew_cai(double *d_cai) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(k<nx*ny){
	int j = (int)(k/nx);	
	d_cai[k] = d_cai[k] + dcai*d_dt;	//renew Cai //是否会与comp_ical（）中倒数第二行重复？
	}
}
void gpu_renew_cai(){
	int bpg;
	//tpb = 256;	
    bpg = (nx*ny+tpb-1)/tpb;
	renew_cai<<<bpg, tpb>>>(d_cai);
	cudaDeviceSynchronize();
}
*/
__global__ void new_gate(double *d_m, double *d_h, double *d_jj,
						double *d_m0, double *d_h0, double *d_jj0,
						double *d_d, double *d_f, double *d_d0, double *d_f0, 
						double *d_X,double *d_X0){
int k = threadIdx.x + blockIdx.x * blockDim.x;

	if(k<nx*ny){

	//int j = (int)(k/nx);
	d_m[k] = d_m0[k];
	d_h[k] = d_h0[k];
	d_jj[k] = d_jj0[k];

	d_d[k] = d_d0[k];
	d_f[k] = d_f0[k];

	d_X[k] = d_X0[k];
	}
}
void gpu_new_gate(){
	int bpg;
	//tpb = 256;	
    bpg = (nx*ny+tpb-1)/tpb;
	new_gate<<<bpg, tpb>>>(d_m, d_h, d_jj,d_m0, d_h0, d_jj0,
						d_d, d_f, d_d0, d_f0, d_X,d_X0);
	cudaDeviceSynchronize();
}
void gpu_ion(){
	int bpg;
	//tpb = 256;	
    bpg = (nx*ny+tpb-1)/tpb;
	comp_ina<<<bpg, tpb>>>(d_V, d_m, d_h, d_jj, d_m0, d_h0, d_jj0, d_dt,d_it);
	comp_ical<<<bpg, tpb>>>(d_V, d_d, d_f, d_d0, d_f0,d_cai, d_dt, d_it);
	comp_ik<<<bpg, tpb>>>(d_V, d_X,  d_X0, d_dt,d_it);
	comp_ik1<<<bpg, tpb>>>(d_V, d_it);
	comp_ikp<<<bpg, tpb>>>(d_V, d_it);
	comp_ib<<<bpg, tpb>>>(d_V, d_it);
	cudaDeviceSynchronize();
}


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
	step_1<<<bpg, tpb>>>(d_V ,belta ,y_temp ,f);//d_V是否要赋给d_Vnew?
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
	step_3<<<bpg, tpb>>>(d_V ,belta ,y_temp ,f);//d_V是否要赋给d_Vnew？
	cudaDeviceSynchronize();
}	
void Send_V(){
        cudaError_t Error;
        size_t size;
        size = nx*ny*sizeof(double);

        Error = cudaMemcpy(h_Vnew, d_Vnew, size, cudaMemcpyDeviceToHost);
        if (Error != cudaSuccess)
        printf("CUDA error(copy d_Vnew->h_Vnew) = %s\n",cudaGetErrorString(Error));
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
