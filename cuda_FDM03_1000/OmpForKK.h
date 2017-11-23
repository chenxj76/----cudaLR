#include "head.h"
#include <omp.h>
extern double *h_m;
extern double *h_m0;
extern double *h_h;
extern double *h_h0;
extern double *h_jj;
extern double *h_jj0;
extern double *h_d;
extern double *h_d0;
extern double *h_f;
extern double *h_f0;
extern double *h_X;
extern double *h_X0;
extern double *h_cai;

extern double *h_t;
extern double *h_dt;
extern double *h_it;
extern double *h_V;
extern double *h_dVdt;

extern int *h_kk0, *h_kk;
double *esi;
double *isi;
int k;

void OmpForKK(int ncount){
	int stimtime = (int)(0.6/dt_max+0.6); 
	//set the number of threads
	int tid;
	omp_set_num_threads(8);
#pragma omp parallel private(tid,k)
{	
	tid=omp_get_thread_num();	
	#pragma omp for private(k) schedule(static)
	for(k=0;k<nx*ny;k++){	//opm
	int j = (int)(k/nx);//相当于二维矩阵中V[i][j]的i行，这里的j表示矩阵的i。
		//int id = k+(nx+2)+1+2*j;//这是什么index？这是扩充为(nx+2)*(ny+2)后的global index,
		//这里+1是为了不要（0,0）这一点，(nx+2)的意义是不要第0行。
		for (int ttt = 1; ttt <= h_kk[k]; ttt++){ 
//-------------ina----------------------------
		//h_it[k] = 0.0;		
		double gna = 23.0;
        double ena = ((R*temp) / frdy)*log(nao / nai);
        double am = 0.32*(h_V[k+nx+2+1+2*j] + 47.13) / (1.0 - exp(-0.1*(h_V[k+nx+2+1+2*j] + 47.13)));
        double bm = 0.08*exp(-h_V[k+nx+2+1+2*j] / 11.0);
		double ah, bh, aj ,bj;
        if (h_V[k+nx+2+1+2*j] < -40.0) {
                ah = 0.135*exp((80 + h_V[k+nx+2+1+2*j]) / -6.8);
                bh = 3.56*exp(0.079*h_V[k+nx+2+1+2*j]) + 310000 * exp(0.35*h_V[k+nx+2+1+2*j]);
                aj = (-127140.0 * exp(0.2444*h_V[k+nx+2+1+2*j]) - 0.00003474*exp(-0.04391*h_V[k+nx+2+1+2*j]))*
                        ((h_V[k+nx+2+1+2*j] + 37.78)/(1.0 + exp(0.311*(h_V[k+nx+2+1+2*j] + 79.23))));
                bj = (0.1212*exp(-0.01052*h_V[k+nx+2+1+2*j])) / (1.0 + exp(-0.1378*(h_V[k+nx+2+1+2*j] + 40.14)));
        }
        else {
                ah = 0.0;
                bh = 1.0 / (0.13*(1 + exp((h_V[k+nx+2+1+2*j] + 10.66) / -11.1)));
                aj = 0.0;
                bj = (0.3*exp(-0.0000002535*h_V[k+nx+2+1+2*j])) / (1.0 + exp(-0.1*(h_V[k+nx+2+1+2*j] + 32.0)));
        }
        double mtau = 1 / (am + bm);
        double htau = 1 / (ah + bh);
		double jtau = 1 / (aj + bj);
        double mss = am*mtau;
        double hss = ah*htau;
        double jss = aj*jtau;
		
        h_m0[k] = mss - (mss - h_m[k])*exp(- h_dt[k] / mtau);//可能是d_dt类型出了问题
        h_h0[k] = hss - (hss - h_h[k])*exp(- h_dt[k] / htau);
        h_jj0[k] = jss - (jss - h_jj[k])*exp(- h_dt[k] / jtau);

        h_it[k] += gna*h_m0[k] * h_m0[k] * h_m0[k] * h_h0[k] * h_jj0[k] * (h_V[k+nx+2+1+2*j] - ena);
//--------ical----------------------------------

		esi[k] = 7.7 - 13.0287*log(h_cai[k]);
        double ad = 0.095*exp(-0.01*(h_V[k+nx+2+1+2*j] - 5)) / (1.0 + exp(-0.072*(h_V[k+nx+2+1+2*j] - 5)));
        double bd = 0.07*exp(-0.017*(h_V[k+nx+2+1+2*j] + 44)) / (1.0 + exp(0.05*(h_V[k+nx+2+1+2*j] + 44)));
        double af = 0.012*exp(-0.008*(h_V[k+nx+2+1+2*j] + 28)) / (1.0 + exp(0.15*(h_V[k+nx+2+1+2*j] + 28)));
        double bf = 0.0065*exp(-0.02*(h_V[k+nx+2+1+2*j] + 30)) / (1.0 + exp(-0.2*(h_V[k+nx+2+1+2*j] + 30)));
        double taud = 1.0 / (ad + bd);
        double tauf = 1.0 / (af + bf);
        double dss = ad*taud;
        double fss = af*tauf;
        h_d0[k] = dss - (dss - h_d[k])*exp(-h_dt[k] / taud);
        h_f0[k] = fss - (fss - h_f[k])*exp(-h_dt[k] / tauf);
        isi[k] = 0.09*h_d0[k] * h_f0[k] * (h_V[k+nx+2+1+2*j] - esi[k]);
        double dcai = -0.0001*isi[k] + 0.07*(0.0001 - h_cai[k]);
        h_cai[k] = h_cai[k] + dcai*h_dt[k];
		h_it[k] = h_it[k] + isi[k];
//-----------------ik------------------------------
        double gk = 0.282*sqrt(ko / 5.4);
        double ek = ((R*temp) / frdy)*log(ko / ki);
        double ax = 0.0005*exp(0.083*(h_V[k+nx+2+1+2*j] + 50)) / (1.0 + exp(0.057*(h_V[k+nx+2+1+2*j] + 50)));
        double bx = 0.0013*exp(-0.06*(h_V[k+nx+2+1+2*j] + 20)) / (1.0 + exp(-0.04*(h_V[k+nx+2+1+2*j] + 20)));
        double taux = 1.0 / (ax + bx);
        double xss = ax*taux;
        h_X0[k] = xss - (xss - h_X[k])*exp(-h_dt[k] / taux);
		double Xi;
        if (h_V[k+nx+2+1+2*j] > -100.0) {
                Xi = 2.837*(exp(0.04*(h_V[k+nx+2+1+2*j] + 77)) - 1.0)/
			((h_V[k+nx+2+1+2*j] + 77.0)*exp(0.04*(h_V[k+nx+2+1+2*j] + 35.0)));
        }
        else {
                Xi = 1.0;
        }
        h_it[k] += gk*h_X0[k] * Xi*(h_V[k+nx+2+1+2*j] - ek);
//--------ik1-----------------------------
        double gk1 = 0.6047*(sqrt(ko / 5.4));
        double ek1 = ((R*temp) / frdy)*log(ko / ki);
        double ak1 = 1.02 / (1.0 + exp(0.2385*(h_V[k+nx+2+1+2*j] - ek1 - 59.215)));
        double bk1 = (0.49124*exp(0.08032*(h_V[k+nx+2+1+2*j] - ek1 + 5.476))+
			exp(0.06175*(h_V[k+nx+2+1+2*j] - ek1 - 594.31)))
                	/(1.0 + exp(-0.5143*(h_V[k+nx+2+1+2*j] - ek1 + 4.753)));
        double K1ss = ak1 / (ak1 + bk1);
        h_it[k] += gk1*K1ss*(h_V[k+nx+2+1+2*j] - ek1);
//---------ikp-----------------------
        double gkp = 0.0183;
        double ekp = ((R*temp) / frdy)*log(ko / ki);
        double kp = 1.0 / (1.0 + exp((7.488 - h_V[k+nx+2+1+2*j]) / 5.98));
        h_it[k] += gkp*kp*(h_V[k+nx+2+1+2*j] - ekp);
//----------ib-------------------------
        h_it[k] += 0.03921*(h_V[k+nx+2+1+2*j] + 59.87);
//---------newgate--------------
		h_m[k] = h_m0[k];
		h_h[k] = h_h0[k];
		h_jj[k] = h_jj0[k];
		h_d[k] = h_d0[k];
		h_f[k] = h_f0[k];
		h_X[k] = h_X0[k];
//----------stimu()---dVdt()----Forward_Euler()--------------
		if (ncount >= 1 && ncount <= stimtime ) {
			if(k<ny*5){
				int i, j;
				i = (int)(k/nx);//因为看ny*5 ，所以i 的取值范围是（0~4）。
				j = k-i*nx;//k=ny，2ny，3ny，4ny，5ny，相邻区间总是能保证j的取值范围是（0~ny-1）。
				h_dVdt[j*ny+i] = h_dVdt[j*ny+i] + (-st);
			}	
		}else{
				h_dVdt[k] = -h_it[k];				
			}
		h_V[k+nx+2+1+2*j]= h_V[k+nx+2+1+2*j] + h_dt[k]*h_dVdt[k];
		}				
	}
#pragma omp barrier	
}//End of parallel section	
		
}			

	