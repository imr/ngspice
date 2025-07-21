/* ===========================================================================
 FILE    cfunc.mod for cm_msopen
 Copyright 2025 Vadim Kuznetsov

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


#include <math.h>
#include <complex.h>
#include <stdio.h>


#include "msline_common.h"
#include "tline_common.h"

#ifdef _MSC_VER
typedef _Dcomplex DoubleComplex;  // double complex
#else
typedef double complex DoubleComplex;
#endif

#ifdef _MSC_VER
static DoubleComplex divide(DoubleComplex n1, DoubleComplex n2)
    {
        DoubleComplex rez;
        rez._Val[0] = (n1._Val[0] * n2._Val[0] + n1._Val[1] * n2._Val[1]) / (n2._Val[0] * n2._Val[0] + n2._Val[1] * n2._Val[1]);
        rez._Val[1] = (n1._Val[1] * n2._Val[0] - n1._Val[0] * n2._Val[1]) / (n2._Val[0] * n2._Val[0] + n2._Val[1] * n2._Val[1]);
        return rez;
    }

static DoubleComplex rdivide(double n1, DoubleComplex n2)
    {
        DoubleComplex rez;
        rez._Val[0] = (n1 * n2._Val[0] + n1 * n2._Val[1]) / (n2._Val[0] * n2._Val[0] + n2._Val[1] * n2._Val[1]);
        rez._Val[1] = (n1 * n2._Val[0] - n1 * n2._Val[1]) / (n2._Val[0] * n2._Val[0] + n2._Val[1] * n2._Val[1]);
        return rez;
    }
#endif

#define MSOPEN_KIRSCHNING 0
#define MSOPEN_HAMMERSTAD 1
#define MSOPEN_ALEXOPOULOS 2


// Returns the microstrip open end capacitance.
static double calcCend (double frequency, double W,
			      double h, double t, double er,
			      int SModel, int DModel,
			      int Model) {

  double ZlEff, ErEff, WEff, ZlEffFreq, ErEffFreq;
  mslineAnalyseQuasiStatic (W, h, t, er, SModel, &ZlEff, &ErEff, &WEff);
  mslineAnalyseDispersion  (WEff, h, er, ZlEff, ErEff, frequency, DModel,
			      &ZlEffFreq, &ErEffFreq);

  W /= h;
  double dl = 0;
  /* Kirschning, Jansen and Koster */
  if (Model == MSOPEN_KIRSCHNING) {
    double Q6 = pow (ErEffFreq, 0.81);
    double Q7 = pow (W, 0.8544);
    double Q1 = 0.434907 *
      (Q6 + 0.26) / (Q6 - 0.189) * (Q7 + 0.236) / (Q7 + 0.87);
    double Q2 = pow (W, 0.371) / (2.358 * er + 1.0) + 1.0;
    double Q3 = atan (0.084 * pow (W, 1.9413 / Q2)) *
      0.5274 / pow (ErEffFreq, 0.9236) + 1.0;
    double Q4 = 0.0377 * (6.0 - 5.0 * exp (0.036 * (1.0 - er))) *
      atan (0.067 * pow (W, 1.456)) + 1.0;
    double Q5 = 1.0 - 0.218 * exp (-7.5 * W);
    dl = Q1 * Q3 * Q5 / Q4;
  }
  /* Hammerstad */
  else if (Model == MSOPEN_HAMMERSTAD) {
    dl = 0.102 * (W + 0.106) / (W + 0.264) *
      (1.166 + (er + 1) / er * (0.9 + log (W + 2.475)));
  }
  return dl * h * sqrt (ErEffFreq) / C0 / ZlEffFreq;
}


void cm_msopen (ARGS)
{
    Complex_t   ac_gain;

	/* how to get properties of this component, e.g. L, W */
	double W = PARAM(w);
	int SModel = PARAM(model);
	int DModel = PARAM(disp);
	int Model = PARAM(msopen_model);

	/* how to get properties of the substrate, e.g. Er, H */
	double er    = PARAM(er);
	double h     = PARAM(h);
	double t     = PARAM(t);


    /* Compute the output */
    if(ANALYSIS == AC) {
		if (Model == MSOPEN_ALEXOPOULOS) {
			double ZlEff, ErEff, WEff, ZlEffFreq, ErEffFreq;
			mslineAnalyseQuasiStatic (W, h, t, er, SModel, &ZlEff, &ErEff, &WEff);
			mslineAnalyseDispersion  (WEff, h, er, ZlEff, ErEff, RAD_FREQ/(2*M_PI), DModel,
					&ZlEffFreq, &ErEffFreq);

			if (fabs (er - 9.9) > 0.2) {
				fprintf (stderr, "WARNING: Model for microstrip open end defined "
						"for er = 9.9 (er = %g)\n", er);
			}

			double c1, c2, l2, r2;
			c1 = (1.125 * tanh (1.358 * W / h) - 0.315) *
				h / 2.54e-5 / 25 / ZlEffFreq * 1e-12;
			c2 = (6.832 * tanh (0.0109 * W / h) + 0.919) *
				h / 2.54e-5 / 25 / ZlEffFreq * 1e-12;
			l2 = (0.008285 * tanh (0.5665 * W / h) + 0.0103) *
				h / 2.54e-5 / 25 * ZlEffFreq * 1e-9;
			r2 = (1.024 * tanh (2.025 * W / h)) * ZlEffFreq;
#ifdef _MSC_VER
			DoubleComplex d1 = _Cbuild(0, c1 * RAD_FREQ);
			DoubleComplex d2 = _Cbuild(r2, (l2 * RAD_FREQ - 1.0 / c2 / RAD_FREQ));
			DoubleComplex y;
			y._Val[0]= d1._Val[0] + (rdivide(1.0, d2))._Val[0];
			y._Val[1]= d1._Val[1] + (rdivide(1.0, d2))._Val[1];
#else
			DoubleComplex d1 = 0 + I*c1 * RAD_FREQ;
			DoubleComplex d2 = r2 + I*(l2 * RAD_FREQ - 1.0 / c2 / RAD_FREQ);
			DoubleComplex y = d1 + 1.0/d2;
#endif
            ac_gain.real = creal(y);
			ac_gain.imag = cimag(y);
			AC_GAIN(p1, p1) = ac_gain;
		} else {
			double Ce = calcCend(RAD_FREQ/(2*M_PI), W, h, t, er, SModel, DModel, Model);
			ac_gain.real = 0.0;
			ac_gain.imag = RAD_FREQ * Ce;
			AC_GAIN(p1, p1) = ac_gain;
		}
	}
}

