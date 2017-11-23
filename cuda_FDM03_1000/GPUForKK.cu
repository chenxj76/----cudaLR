#include "head.h"
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

extern double *d_t;
extern double *d_dt;
extern double *d_it;
extern double *d_V;
extern double *d_dVdt;

extern int *d_kk0, *d_kk;
extern double *isi;
extern double *esi;
extern double *ina;
extern double *ik;
extern double *ik1;
extern double *ikp;
extern double *ib;

__global__ void adaptiveT(double *d_dVdt  ,double *d_dt,int *d_kk,int *d_kk0){
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	if(k<nx*ny){
		
		//int d_k0, d_k;
		//for (i = 1; i < nx + 1; i++){
			//for (j = 1; j < ny + 1; j++){
				// adaptive time step
				if (d_dVdt[k] > 0){
					d_kk0[k] = 5;
				}else{
					d_kk0[k] = 1;
				}
				d_kk[k] = d_kk0[k] + (int)(fabs(d_dVdt[k]) + 0.5); //round the value此处(dVdt[k])+0.5是为了四舍五入
				if (d_kk[k] >(int)(dt_max / dt_min)){
					d_kk[k] = (int)(dt_max / dt_min);//最多kk=40
				}
				d_dt[k] = dt_max / d_kk[k];//这里的d_dt[0],d_kk[0]需要改为每个点的dt[k],d_kk[k].
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

__global__ void gpu_ForKK(double *d_V, double *d_dVdt,int *d_kk,int *d_kk0,
						  double *d_dt, double *d_it,int ncount,
						  double *d_m, double *d_h, double *d_jj, 
						  double *d_m0, double *d_h0, double *d_jj0,  
						  double *d_d, double *d_f, double *d_d0, double *d_f0, 
						  double *d_cai, double *isi,double *esi,double *ina,
						  double *ik,double *ik1,double *ikp,double *ib,
						  double *d_X, double *d_X0){

	int stimtime = (int)(0.6/dt_max+0.6); 
	int k = threadIdx.x + blockIdx.x * blockDim.x;//这里出错
	if(k<nx*ny){	
	int j = (int)(k/nx);//相当于二维矩阵中V[i][j]的i行，这里的j表示矩阵的i。
		//int id = k+(nx+2)+1+2*j;//这是什么index？这是扩充为(nx+2)*(ny+2)后的global index,
		//这里+1是为了不要（0,0）这一点，(nx+2)的意义是不要第0行。
	    double dcai; 
		d_dt[k]=dt_max;
		for (int ttt = 1; ttt <= d_kk[k]; ttt++){ 
//-------------ina----------------------------
		//d_it[k] = 0.0;		
		double gna = 23.0;
        double ena = ((R*temp) / frdy)*log(nao / nai);
        double am = 0.32*(d_V[k+nx+2+1+2*j] + 47.13) / (1.0 - exp(-0.1*(d_V[k+nx+2+1+2*j] + 47.13)));
        double bm = 0.08*exp(-d_V[k+nx+2+1+2*j] / 11.0);
		double ah, bh, aj ,bj;
        if (d_V[k+nx+2+1+2*j] < -40.0) {
                ah = 0.135*exp((80 + d_V[k+nx+2+1+2*j]) / -6.8);
                bh = 3.56*exp(0.079*d_V[k+nx+2+1+2*j]) + 310000 * exp(0.35*d_V[k+nx+2+1+2*j]);
                aj = (-127140.0 * exp(0.2444*d_V[k+nx+2+1+2*j]) - 0.00003474*exp(-0.04391*d_V[k+nx+2+1+2*j]))*
                        ((d_V[k+nx+2+1+2*j] + 37.78)/(1.0 + exp(0.311*(d_V[k+nx+2+1+2*j] + 79.23))));
                bj = (0.1212*exp(-0.01052*d_V[k+nx+2+1+2*j])) / (1.0 + exp(-0.1378*(d_V[k+nx+2+1+2*j] + 40.14)));
        }
        else {
                ah = 0.0;
                bh = 1.0 / (0.13*(1 + exp((d_V[k+nx+2+1+2*j] + 10.66) / -11.1)));
                aj = 0.0;
                bj = (0.3*exp(-0.0000002535*d_V[k+nx+2+1+2*j])) / (1.0 + exp(-0.1*(d_V[k+nx+2+1+2*j] + 32.0)));
        }
        double mtau = 1 / (am + bm);
        double htau = 1 / (ah + bh);
		double jtau = 1 / (aj + bj);
        double mss = am*mtau;
        double hss = ah*htau;
        double jss = aj*jtau;
		
        d_m0[k] = mss - (mss - d_m[k])*exp(- d_dt[k] / mtau);//可能是d_dt类型出了问题
        d_h0[k] = hss - (hss - d_h[k])*exp(- d_dt[k] / htau);
        d_jj0[k] = jss - (jss - d_jj[k])*exp(- d_dt[k] / jtau);

        ina[k] = gna*d_m0[k] * d_m0[k] * d_m0[k] * d_h0[k] * d_jj0[k] * (d_V[k+nx+2+1+2*j] - ena);
//--------ical----------------------------------
		//int i = threadIdx.x;
		//__shared__ double esi[tpb];//这两个变量设置共享内存，也可以不设置，不影响
		//__shared__ double isi[tpb];//需要改一下，改为global
        esi[k] = 7.7 - 13.0287*log(d_cai[k]);
        double ad = 0.095*exp(-0.01*(d_V[k+nx+2+1+2*j] - 5)) / (1.0 + exp(-0.072*(d_V[k+nx+2+1+2*j] - 5)));
        double bd = 0.07*exp(-0.017*(d_V[k+nx+2+1+2*j] + 44)) / (1.0 + exp(0.05*(d_V[k+nx+2+1+2*j] + 44)));
        double af = 0.012*exp(-0.008*(d_V[k+nx+2+1+2*j] + 28)) / (1.0 + exp(0.15*(d_V[k+nx+2+1+2*j] + 28)));
        double bf = 0.0065*exp(-0.02*(d_V[k+nx+2+1+2*j] + 30)) / (1.0 + exp(-0.2*(d_V[k+nx+2+1+2*j] + 30)));
        double taud = 1.0 / (ad + bd);
        double tauf = 1.0 / (af + bf);
        double dss = ad*taud;
        double fss = af*tauf;
        d_d0[k] = dss - (dss - d_d[k])*exp(-d_dt[k] / taud);
        d_f0[k] = fss - (fss - d_f[k])*exp(-d_dt[k] / tauf);
        isi[k] = 0.09*d_d0[k] * d_f0[k] * (d_V[k+nx+2+1+2*j] - esi[k]);        
		//d_it[k] = d_it[k] + isi[k];
//-----------------ik------------------------------
        double gk = 0.282*sqrt(ko / 5.4);
        double ek = ((R*temp) / frdy)*log(ko / ki);
        double ax = 0.0005*exp(0.083*(d_V[k+nx+2+1+2*j] + 50)) / (1.0 + exp(0.057*(d_V[k+nx+2+1+2*j] + 50)));
        double bx = 0.0013*exp(-0.06*(d_V[k+nx+2+1+2*j] + 20)) / (1.0 + exp(-0.04*(d_V[k+nx+2+1+2*j] + 20)));
        double taux = 1.0 / (ax + bx);
        double xss = ax*taux;
        d_X0[k] = xss - (xss - d_X[k])*exp(-d_dt[k] / taux);
		double Xi;
        if (d_V[k+nx+2+1+2*j] > -100.0) {
                Xi = 2.837*(exp(0.04*(d_V[k+nx+2+1+2*j] + 77)) - 1.0)/
			((d_V[k+nx+2+1+2*j] + 77.0)*exp(0.04*(d_V[k+nx+2+1+2*j] + 35.0)));
        }
        else {
                Xi = 1.0;
        }
        ik[k] = gk*d_X0[k] * Xi*(d_V[k+nx+2+1+2*j] - ek);
//--------ik1-----------------------------
        double gk1 = 0.6047*(sqrt(ko / 5.4));
        double ek1 = ((R*temp) / frdy)*log(ko / ki);
        double ak1 = 1.02 / (1.0 + exp(0.2385*(d_V[k+nx+2+1+2*j] - ek1 - 59.215)));
        double bk1 = (0.49124*exp(0.08032*(d_V[k+nx+2+1+2*j] - ek1 + 5.476))+
			exp(0.06175*(d_V[k+nx+2+1+2*j] - ek1 - 594.31)))
                	/(1.0 + exp(-0.5143*(d_V[k+nx+2+1+2*j] - ek1 + 4.753)));
        double K1ss = ak1 / (ak1 + bk1);
        ik1[k] = gk1*K1ss*(d_V[k+nx+2+1+2*j] - ek1);
//---------ikp-----------------------
        double gkp = 0.0183;
        double ekp = ((R*temp) / frdy)*log(ko / ki);
        double kp = 1.0 / (1.0 + exp((7.488 - d_V[k+nx+2+1+2*j]) / 5.98));
        ikp[k] = gkp*kp*(d_V[k+nx+2+1+2*j] - ekp);
//----------ib-------------------------
        ib[k] = 0.03921*(d_V[k+nx+2+1+2*j] + 59.87);
//-----------d_it--------------------
		d_it[k]=ina[k]+isi[k]+ik[k]+ik1[k]+ikp[k]+ib[k];
//---------newgate--------------
		d_m[k] = d_m0[k];
		d_h[k] = d_h0[k];
		d_jj[k] = d_jj0[k];
		d_d[k] = d_d0[k];
		d_f[k] = d_f0[k];
		d_X[k] = d_X0[k];
//------------update cai---------------------
		dcai = -0.0001*isi[k] + 0.07*(0.0001 - d_cai[k]);
        d_cai[k] = d_cai[k] + dcai * d_dt[k];
//----------stimu()---dVdt()----Forward_Euler()--------------
		if (ncount >= 1 && ncount <= stimtime ) {
			if(k<ny*5){
				int i, j;
				i = (int)(k/nx);//因为看ny*5 ，所以i 的取值范围是（0~4）。
				j = k-i*nx;//k=ny，2ny，3ny，4ny，5ny，相邻区间总是能保证j的取值范围是（0~ny-1）。
				d_dVdt[j*ny+i] = d_dVdt[j*ny+i] + (-st);
			}else{
				d_dVdt[k] = -d_it[k];				
			}
		d_V[k+nx+2+1+2*j]= d_V[k+nx+2+1+2*j] + d_dt[k]*d_dVdt[k];

		}				
	}							
}
}		
		
			
void gpuForKK(int ncount){	
	int bpg;
	//tpb = 256;
	bpg = (nx*ny+tpb-1)/tpb;		
	gpu_ForKK<<<bpg, tpb>>>(d_V, d_dVdt,d_kk,d_kk0,d_dt, d_it,ncount,
						    d_m, d_h, d_jj, d_m0, d_h0, d_jj0,  
						    d_d, d_f, d_d0, d_f0, d_cai,isi, esi,ina,
						    ik, ik1, ikp, ib, d_X, d_X0);
	cudaDeviceSynchronize();									
}
	