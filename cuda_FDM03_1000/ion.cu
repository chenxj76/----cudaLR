#include "head.h"



extern double *d_t;
extern double *d_dt;
extern double *d_it;
extern double *d_V;

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


