// BarModel1.cpp : 定義控制台應用程式的入口點。

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

#define tpb 256
// Time Step
//double dt = 0.02; // Time step (ms)
//double t; // Time (ms)
//int steps; // Number of Steps
//int increment; // Loop Control Variable
//#define dt 0.1//dt的改???致激?不起?，如改?0.1就不行了,有??出?。
#define dt_max 0.04 // Time step (ms)
#define dt_min 0.001
// Voltage
//double V[nx + 2][nx + 2]; // Initial Voltage (mv)
//double dV2[nx + 2][nx + 2]; // second order derivatives of Voltage (mv)
//double Vnew[nx + 2][nx + 2];// New Voltage (mV)
//double dvdt; // Change in Voltage / Change in Time (mV/ms)
//double dvdtnew; // New dv/dt (mV/ms)

// Total Current and Stimulus
//double st; // Constant Stimulus (uA/cm^2)
//double tstim; //Time Stimulus is Applied (ms)//Time to begin stimulus
//double stimtime; //Time period during which stimulus is applied (ms)
//double it[nx + 1][nx + 1]; // Total current (uA/cm^2)
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
//double cai[nx + 1][nx + 1]; // Intracellular Ca Concentration (mM)
//double cao; // Extracellular Ca Concentration (mM)
//double ki; // Intracellular K Concentration (mM)
//double ko; // Extracellular K Concentration (mM)
#define nai 18.0
#define nao 140.0
#define ki 145.0
#define ko 5.4
#define cao 1.8

/*
// Fast Sodium Current (time dependant)
double ina[nx + 1][nx + 1]; // Fast Na Current (uA/uF)
double gna; // Max. Conductance of the Na Channel (mS/uF)
double ena; // Reversal Potential of Na (mV)
double am; // Na alpha-m rate constant (ms^-1)
double bm; // Na beta-m rate constant (ms^-1)
double ah; // Na alpha-h rate constant (ms^-1)
double bh; // Na beta-h rate constant (ms^-1)
double aj; // Na alpha-j rate constant (ms^-1)
double bj; // Na beta-j rate constant (ms^-1)
double mtau; // Na activation
double htau; // Na inactivation
double jtau; // Na inactivation
double mss; // Na activation
double hss; // Na inactivation
double jss; // Na slow inactivation
double m[nx + 1][nx + 1]; // Na activation
double h[nx + 1][nx + 1]; // Na inactivation
double jj[nx + 1][nx + 1]; // Na slow inactivation

// Current through L-type Ca Channel
double dcai; // Change in myoplasmic Ca concentration (mM)
double isi[nx + 1][nx + 1]; // Slow inward current (uA/uF)
double esi[nx + 1][nx + 1]; // Reversal Potential of si (mV)
double ad; // Ca alpha-d rate constant (ms^-1)
double bd; // Ca beta-d rate constant (ms^-1)
double af; // Ca alpha-f rate constant (ms^-1)
double bf; // Ca beta-f rate constant (ms^-1)

double d[nx + 1][nx + 1]; // Voltage dependant activation gate
double dss; // Steady-state value of activation gate d
double taud; // Time constant of gate d (ms^-1)----mistake ???？ms？
double f[nx + 1][nx + 1]; // Voltage dependant inactivation gate
double fss; // Steady-state value of inactivation gate f
double tauf; // Time constant of gate f (ms^-1)
double fca[nx + 1][nx + 1]; // Ca dependant inactivation gate -from LR94

// Time-dependent potassium current
double ik[nx + 1][nx + 1]; // Rapidly Activating K Current (uA/uF)
double gk; // Channel Conductance of Rapidly Activating K Current (mS/uF)
double ek; // Reversal Potential of Rapidly Activating K Current (mV)
double ax; // K alpha-x rate constant (ms^-1)
double bx; // K beta-x rate constant (ms^-1)
double X[nx + 1][nx + 1]; // Rapidly Activating K time-dependant activation  --gate X in LR91
double xss; // Steady-state value of inactivation gate xr  --gate X in LR91
double taux; // Time constant of gate xr (ms^-1) --gate X in LR91
double Xi; // K time-independent inactivation --gate Xi in LR91

// Potassium Current (time-independent)
double ik1[nx + 1][nx + 1]; // Time-independent K current (uA/uF)
double gk1; // Channel Conductance of Time Independant K Current (mS/uF)
double ek1; // Reversal Potential of Time Independant K Current (mV)
double ak1; // K alpha-ki rate constant (ms^-1)
double bk1; // K beta-ki rate constant (ms^-1)
double K1ss; // Steady-state value of K inactivation gate K1

// Plateau Potassium Current
double ikp[nx + 1][nx + 1]; // Plateau K current (uA/uF)
double gkp; // Channel Conductance of Plateau K Current (mS/uF)
double ekp; // Reversal Potential of Plateau K Current (mV)
double kp; // K plateau factor

// Background Current
double ib[nx + 1][nx + 1]; // Background current (uA/uF)

//performance compared
double Vmax, dvdt_max = 0, APD90, TNP, CPUtime;
double Vold, v_onset;
int flag = 0;
long f_count = 0;

// Ion Current Functions
void comp_ina(int i, int j); // Calculates Fast Na Current
void comp_ical(int i, int j); // Calculates Currents through L-Type Ca Channel
void comp_ik(int i, int j); // Calculates Time-dependent K Current
void comp_ik1(int i, int j); // Calculates Time-Independent K Current
void comp_ikp(int i, int j); // Calculates Plateau K Current
void comp_ib(int i, int j); // Calculates Background Current
void comp_it(int i, int j); // Calculates Total Current
*/
//-----memory.cu-------
void Allocate();
void Send_to_Device();
void Send_to_Host();
void Send_V();
void Send_dVdt();
//void Save_Result();
void free();
//----ion.cu-------
void gpu_Ion();
//--------forward.cu-------
void gpu_Boun();
void gpu_dV2();
void gpu_dV2it();
void stimu();
void Forward_Euler();
void Forward_Euler2();
void gpu_step123(int ncount,int stimtime);
//-----GPUForKK.cu-------
void gpu_adaptiveT();
void gpuForKK(int ncount);

