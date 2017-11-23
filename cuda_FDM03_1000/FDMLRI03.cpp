#include "head.h"
//#include "OmpForKK.h"
#include <omp.h>
extern double *h_t;
extern double *h_dt;
extern double *h_V;
//extern double *h_Vnew;
//extern double *d_Vnew;
extern double *h_m;
extern double *h_h;
extern double *h_jj;
extern double *h_d;
extern double *h_f;
extern double *h_X;
extern double *h_cai;
extern int  *h_kk;
extern int  *h_kk0;
double t; // Time (ms)
int stimtime = (int)(0.6/dt_max+0.6); //Time period during which stimulus is applied 
void init();
void init_dtK();
void Save_Result(int ncount);

int main(){
	t = 0.0; // Time (ms)
	double dt[10000];
	double kk[10000];
	FILE *tstep;   
	tstep = fopen("tstep", "w");
	int nstep = 4 / dt_max; // snapshot interval 4 ms to save data files 
	
	Allocate();
	init();
	init_dtK();
	
	struct timeb start, end;
    int diff;
	ftime(&start);
	//h_dt[0] = dt_max;//放进了init()中
	Send_to_Device();
	
	int ncount, i, j;
	
	for (ncount = 0; ncount < 200/dt_max; ncount++){//30000 steps, 600ms
		gpu_Boun();
		gpuStep_1();
//*********** step 2 *******
		init_dtK();
		//h_dt[0] = dt_max;
		Manage_Comms(1);//copy h_dt->d_dt,//h_Kk->d_Kk,copy h_kk0->d_kk0
		gpu_ion();
		gpu_dVdt();
		//*****stimulation with a plane waves****
		if (ncount >= 1 && ncount <= stimtime) { //stimulus is hold with 0.6 ms			
			gpu_stimu();
		}		
		gpu_adaptiveT();			
		Manage_Comms(2);//copy d_dt->h_dt
		dt[ncount]=	h_dt[nx*ny/2+nx/2];	//只看v(nx/2,ny/2)这一点的时间步长改变
		Manage_Comms(3);//d_Kk->h_Kk
		kk[ncount]=	h_kk[nx*ny/2+nx/2];
		gpuForKK(ncount);//
		//OmpForKK(ncount);
		/*for (int ttt = 1; ttt <= h_kk[0]; ttt++){ //from t to t+dt_max, t=t+dt
			gpu_ion();//已经做了gpu_renew_Cai()的动作	
			gpu_new_gate();	//不影响算结果，因为使用RK（4）方法是必须用到，一次for循环里需要多次使用m的初始值，保证一次for里的m值不变。			
			//gpu_renew_Cai();		
			if (ncount >= 1 && ncount <= stimtime ) {
				gpu_stimu();	
			}else{
				gpu_dVdt();					
			}
				Forward_Euler();			
		}*/
		// ------------------------------------------------------
		gpuStep_3();
		t = t + dt_max;//计算performance()时用到，目前程序没有涉及这一步。
		if (ncount%nstep == 0){		
			//Manage_Comms(4);//copy d_Vnew->h_Vnew		
			Manage_Comms(5);//copy d_V->h_V
			Save_Result(ncount);		
		}
		fprintf(tstep, "%g\t %g\t\n", dt[ncount],kk[ncount]);
	}
	ftime(&end);
    diff = (int)(1000.0*(end.time-start.time)+(end.millitm-start.millitm));
    printf("\nTime = %d ms\n", diff);
	fclose(tstep);
	free();
	return 0;	
}


	
void init(){
	int i, j;
	h_t[0] = 0.0;
	/*
	h_V[0] = 0.0;h_V[nx+1] = 0.0;
	h_V[(nx+2)*(ny+1)] = 0.0;h_V[(nx+2)*(ny+1)+nx+1] = 0.0;
	for (i = 1; i < nx + 1; i++){
                h_V[i] = 0.0;
                h_V[i+(nx+2)*(ny+1)] = 0.0;
				h_V[0+(nx+2)*i] = 0.0;
                h_V[nx+1+i*(nx+2)] = 0.0;
     }
	 */
	 for (i = 0; i < nx+2; i++){
		for (j = 0; j < ny+2; j++){
			h_V[i*(nx+2)+j] = -88.654973; // Initial Voltage (mv)
		}
	}
	for (i = 0; i < nx; i++){
		for (j = 0; j < ny; j++){
			//h_V[(i+1)*(nx+2)+j+1] = -88.654973; // Initial Voltage (mv)
			h_m[i*nx+j] = 0.000838;
			h_h[i*nx+j] = 0.993336;
			h_jj[i*nx+j] = 0.995484;
			h_d[i*nx+j] = 0.000003;
			h_f[i*nx+j] = 0.999745;
			h_X[i*nx+j] = 0.000129;
			h_cai[i*nx+j] = 0.0002; // Initial Intracellular Ca (mM)
		}
	}
}
void init_dtK(){
for (int i = 0; i < nx; i++){
		for (int j = 0; j < ny; j++){
			h_dt[i*nx+j] = dt_max;
			h_kk[i*nx+j]=1;
			h_kk0[i*nx+j]=1;
		}
	}
}
void Save_Result(int ncount){
		
		FILE *ap;    		
        int i,j;
        int index;
		int nstep = 4 / dt_max;
		int idx = ncount/nstep;// filename index
		char filename[idx];
		int fileflag = 0;			
				//if (ncount%nstep == 0){//save data every 4 ms
		if (fileflag == 0){
					sprintf(filename, "ap%d", idx);
					ap = fopen(filename, "w");
					fileflag = 1;
					idx++;
		}
					//fprintf(ap, "%g\t", V[i][j]);
		for (i = 0; i < nx ; i++){
			for (j = 0; j < ny ; j++){
					index = (i+1)*(nx+2)+j+1;
                    fprintf(ap, "%g\t", h_V[index]);
					if (j == ny-1){
						fprintf(ap, "\n");
					}
				//}
			}
		}
		if (fileflag == 1){
			fclose(ap);
		}

}