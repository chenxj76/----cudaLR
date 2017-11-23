#include "head.h"

extern double *h_t;
extern double *h_V;
extern double *h_Vnew;
extern double *h_dVdt;
extern double *h_m;
extern double *h_h;
extern double *h_jj;
extern double *h_d;
extern double *h_f;
extern double *h_X;
extern double *h_cai;
int stimtime = (int)(0.6 / dt_max + 0.6);
void init();
void Save_Result();
void Save_ResultDVDT();
int main(){	
	int ncount, i, j;
	Allocate();	
	init();
	struct timeb start, end;
        int diff;
	ftime(&start);

	Send_to_Device();
	for (ncount = 0; ncount < 20; ncount++){
		//---1--- 
		gpu_Boun();	
		gpu_dV2();
		Forward_Euler2();
		
		//---2---
		//dt = dt_max;//改一下		
		gpu_Ion();
		gpu_dV2it();	
		if (ncount >= 1 && ncount <= stimtime) {
                       stimu();
					   Send_dVdt();
					   Save_ResultDVDT();
                }
				
		gpu_adaptiveT();			
		//Manage_Comms(2);//copy d_dt->h_dt
		//dt[ncount]=	h_dt[nx*ny/2+nx/2];	//只看v(nx/2,ny/2)这一点的时间步长改变
		//Manage_Comms(3);//d_Kk->h_Kk
		//kk[ncount]=	h_kk[nx*ny/2+nx/2];
		gpuForKK(ncount);//主要问题出在这

		//---3---
		gpu_Boun();
		gpu_dV2();//here must use the boundary.
		Forward_Euler2();
		//gpu_step123(ncount,stimtime);
	}
	Send_V();
	ftime(&end);
        diff = (int)(1000.0*(end.time-start.time)+(end.millitm-start.millitm));
        printf("\nTime = %d ms\n", diff);
	Save_Result();
	free();
	return 0;
	//double time_used = (double)(end - start) / CLK_TCK;
	//fprintf(fevaluation, "%g", time_used);
}

void init(){	
	int i,j;
	h_t[0] = 0.0;
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
//return 0;
}
void Save_Result(){

        FILE *pFile;
        int i,j;
        int index;
        //int n;
        //n = nx;
        pFile = fopen("V.txt","w+");
        // Save the matrix V
        for (i = 0; i < ny+2; i++) {
                for (j = 0; j < nx+2; j++) {
                        index = i*(nx+2) + j;
                        fprintf(pFile, "%g", h_V[index]);
                        if (j == (nx-1)) {
                                fprintf(pFile, "\n");
                        }else{
                                fprintf(pFile, "\t");
                        }
                }
        }
        fclose(pFile);

}
void Save_ResultDVDT(){

        FILE *pFile;
        int i,j;
        int index;
        //int n;
        //n = nx;
        pFile = fopen("dVdt.txt","w+");
        // Save the matrix V
        for (i = 0; i < ny; i++) {
                for (j = 0; j < nx; j++) {
                        index = i*nx + j;
                        fprintf(pFile, "%g", h_dVdt[index]);
                        if (j == (nx-1)) {
                                fprintf(pFile, "\n");
                        }else{
                                fprintf(pFile, "\t");
                        }
                }
        }
        fclose(pFile);

}
/*
		if (ncount == 20000){
			Send_to_Host();
                        for (i = 0; i < nx/2-1; i++){
                                for (j = 0; j < ny-1; j++){
                                h_V[(i+1)*(nx+2)+j+1] = -88.654973; // Initial Voltage (mv)
                        		h_m[i*nx+j] = 0.000838;
                        		h_h[i*nx+j] = 0.993336;
                 	 	      	h_jj[i*nx+j] = 0.995484;
                	 	       	h_d[i*nx+j] = 0.000003;
								h_f[i*nx+j] = 0.999745;
                    	   		h_X[i*nx+j] = 0.000129;
								h_cai[i*nx+j] = 0.0002; // Initial Intracellular Ca (mM)
                                }
                        }
			Send_to_Device();
                }
*/