/* ===========================================================================
 FILE    msline_common.h - common definitions for microstrip devices
 Copyright 2025 Vadim Kuznetsov

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

void cpmslineAnalyseQuasiStatic (double W, double h, double s,
				   double t, double er,
				   int SModel, double* Zle,
				   double* Zlo, double* ErEffe,
				   double* ErEffo);

void cpmslineAnalyseDispersion (double W, double h, double s,
				   double t, double er, double Zle,
				   double Zlo, double ErEffe,
				   double ErEffo, double frequency,
				   int  DModel, double *ZleFreq,
				   double  *ZloFreq,
				   double  *ErEffeFreq,
				   double  *ErEffoFreq);


#endif
