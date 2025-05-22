/* msline_common.h
 * common definitions for microstrip devices
 * (c) Vadim Kuznetsov 2025
 */


#ifndef MSLINE_COMMON_H
#define MSLINE_COMMON_H

//TRAN model
#define TRAN_DC 0
#define TRAN_FULL 1

// MS line model
#define HAMMERSTAD 0
#define KIRSCHING 1
#define WHEELER 2
#define SCHNEIDER 3


// Dispersion model
#define DISP_KIRSCHING 0
#define KOBAYASHI 1
#define YAMASHITA 2
#define DISP_HAMMERSTAD 3
#define GETSINGER 4
#define DISP_SCHNEIDER 5
#define PRAMANICK 6

void Hammerstad_ab (double, double,
                    double*, double*);
void Hammerstad_er (double, double, double,
                    double, double*);
void Hammerstad_zl (double, double*);
void Getsinger_disp (double, double, double,
                     double, double,
                     double*, double*);
void Kirschning_er (double, double, double,
                    double, double*);
void Kirschning_zl (double, double, double,
                    double, double, double,
                    double*, double*);

void mslineAnalyseQuasiStatic (double W, double h, double t,
		double er, int Model,
		double *ZlEff, double *ErEff,
		double *WEff);

void mslineAnalyseDispersion (double W, double h, double er,
		double ZlEff, double ErEff,
		double frequency, int Model,
		double* ZlEffFreq,
		double* ErEffFreq);

void analyseLoss (double, double, double, double,
                         double, double, double, double,
                         double, double, int,
                         double*, double*);

#endif
