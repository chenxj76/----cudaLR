

#include <math.h>
//#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <sys/timeb.h>

//-*******FDM parameters for LR91 *******
//int const nx = 1000, ny = 1000;//grid numbers
//double dx = 0.015, dy = 0.015;//space step, 3cm*3cm
//double D = 0.001;//D: diffusion coefficient cm^2/ms
#define nx 1000
#define ny 1000
#define dx 0.015
#define dy 0.015
#define D 0.001

// Time Step
//double dt =(0.001~0.04)ms; 
#define dt_max  0.04
#define dt_min  0.001

// Total Current and Stimulus
//double st; // Constant Stimulus (uA/cm^2)
#define st (-80.0)

// Terms for Solution of Conductance and Reversal Potential
//const double R = 8314; // Universal Gas Constant (J/kmol*K)
//const double frdy = 96485; // Faraday's Constant (C/mol)
//double temp = 310; // Temperature (K)
#define R 8314.0
#define frdy 96485.0
#define temp 310.0

// Ion Concentrations
//double nai; // Intracellular Na Concentration (mM)
//double nao; // Extracellular Na Concentration (mM)
//double cao; // Extracellular Ca Concentration (mM)
//double ki; // Intracellular K Concentration (mM)
//double ko; // Extracellular K Concentration (mM)
#define nai 18.0
#define nao 140.0
#define ki 145.0
#define ko 5.4
#define cao 1.8

#define tpb 256
//----ion.cu--------
void gpu_ion();
void gpu_new_gate();
void gpu_renew_cai();
//----voltage.cu--------
void gpu_Boun();
void gpuStep_1();
void gpu_dVdt();
void gpu_stimu();
void gpu_adaptiveT();
void Forward_Euler();
void gpuStep_3();
//----memory.cu--------
void Allocate();
void free();
void Send_to_Device();
void Manage_Comms(int phase);
//void Send_V();
//void Send2deviceT();
//void Send2hostT();
//void Send2hostK();
//void Send_to_Host();
