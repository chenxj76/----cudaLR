#include "head.h"

extern double *h_t;
extern double *h_V,*h_Vnew;
extern double *h_m,*h_h,*h_jj;
extern double *h_d,*h_f;
extern double *h_X,*h_cai;
//extern double *h_esi,*h_isi;


int main(){

	int ncount, i, j;

	Allocate();
//***init boundary conditions*****			
	/*	
	for (j = 0; j < ny+2 ; j++){//包括4个角顶点
	h_V[j*(nx+2)+0] = 0.0; //左端一列
	h_V[j*(nx+2)+(nx+1)] = 0.0;//右端一列
	}
    for (i = 0; i < nx ; i++){	
	h_V[(i+1)+0*(nx+2)] = 0.0; //上端一行
	h_V[(i+1)+(ny+1)*(nx+2)] = 0.0;//下端一行
	}*/
//---------------------------------------------------
	h_t[0] = 0.0;
	
//***init 1~1000 conditions*****
//unsigned	int idx =(i+1)*(nx+2)+ (j+1);//没有四条边的index

unsigned	int id =i*nx+ j;//包括四条边的index	
for (i = 0; i < nx+2; i++){
		for (j = 0; j < ny+2; j++){
			h_V[id] = -88.654973; // Initial Voltage (mv)
		}
}
	for (i = 0; i < nx; i++){
		for (j = 0; j < ny; j++){
			//h_V[idx] = -88.654973; // Initial Voltage (mv)
			h_m[id] = 0.000838;
			h_h[id] = 0.993336;
			h_jj[id] = 0.995484;
			h_d[id] = 0.000003;
			h_f[id] = 0.999745;
			h_X[id] = 0.000129;
			h_cai[id] = 0.0002;
			//h_esi[id]=0.00000;
	        //h_isi[id]=0.00000;// Initial Intracellular Ca (mM)
		}
	}
	/*for (i = 0; i < nx; i++){
		for (j = 0; j < ny; j++){
			printf("h_V=%g",h_V[idx]); 
		}
	}*/
//------------------------------------------------
	//int nstep = 500; // snapshot interval to save data files 500*0.02=10 ms
	//int index = 0;// filename index from 1-5

	struct timeb start, end;
        int diff;
	ftime(&start);

	Send_to_Device();//9个变量，多了一个时间t
	for (ncount = 0; ncount <= 500; ncount++){//30000 steps, 600ms
		if(ncount==1)Send_V();
		gpu_boun();
		gpu_comp_ina();
		//gpu_comp_ical();
		//gpu_comp_ik();
		//gpu_comp_ik1();
		//gpu_comp_ikp();
		//gpu_comp_ib();
		//gpu_comp_dV2();
		
		if (ncount >= 0 && ncount <= 100) {
                        stimu();
                }

		Forward_Euler();

		if (ncount == 20000){
			Send_to_Host();//8个变量
                    for (i = 0; i < nx/2-1; i++){
                        for (j = 0; j < ny-1; j++){
                            //h_V[idx] = -88.654973; // Initial Voltage (mv)
                        	h_m[id] = 0.000838;
                        	h_h[id] = 0.993336;
                 	 	    h_jj[id] = 0.995484;
                	 	    h_d[id] = 0.000003;
               	  	  		h_f[id] = 0.999745;
                    	   	h_X[id] = 0.000129;
                   		 	h_cai[id] = 0.0002; // Initial Intracellular Ca (mM)
                        }
                    }
			Send_to_Device();//9个变量，多了一个时间t
                }
		
	}
	//Send_V();
	ftime(&end);
        diff = (int)(1000.0*(end.time-start.time)+(end.millitm-start.millitm));
        printf("\nTime = %d ms\n", diff);
	Save_Result();
	free();
	return 0;
}
