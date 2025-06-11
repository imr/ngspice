/* ===========================================================================
	FILE    cfunc.mod
	Copyright 2025 Vadim Kuznetsov

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

	2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

	3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#include <stdio.h>
#include <math.h>
#include <complex.h>

#include "tline_common.h"

#include "msline_common.h"

//static tline_state_t *sim_points = NULL;

static double zl, alpha, beta, ereff;

static void cm_mline_callback(ARGS, Mif_Callback_Reason_t reason);

static void calcPropagation (double W, int SModel, int DModel,
                             double er, double h, double t, double tand, double rho, double D,
                             double frequency) {

	/* local variables */
	double ac, ad;
	double ZlEff, ErEff, WEff, ZlEffFreq, ErEffFreq;

	// quasi-static effective dielectric constant of substrate + line and
	// the impedance of the microstrip line
	mslineAnalyseQuasiStatic (W, h, t, er, SModel, &ZlEff, &ErEff, &WEff);

	// analyse dispersion of Zl and Er (use WEff here?)
	mslineAnalyseDispersion (W, h, er, ZlEff, ErEff, frequency, DModel,
			&ZlEffFreq, &ErEffFreq);

	// analyse losses of line
	analyseLoss (W, t, er, rho, D, tand, ZlEff, ZlEff, ErEff,
			frequency, HAMMERSTAD, &ac, &ad);

	// calculate propagation constants and reference impedance
	zl    = ZlEffFreq;
	ereff = ErEffFreq;
	alpha = ac + ad;
	beta  = sqrt (ErEffFreq) * 2 * M_PI * frequency / C0;
}

void cm_mlin (ARGS)
{
	Complex_t   z11, z21;
    void **sim_points;


	/* how to get properties of this component, e.g. L, W */
	double W = PARAM(w);
	double l = PARAM(l);
	int SModel = PARAM(model);
	int DModel = PARAM(disp);
    int TModel = PARAM(tranmodel);

	/* how to get properties of the substrate, e.g. Er, H */
	double er    = PARAM(er);
	double h     = PARAM(h);
	double t     = PARAM(t);
	double tand  = PARAM(tand);
	double rho   = PARAM(rho);
	double D     = PARAM(d);



	/* Initialize/access instance specific storage for capacitor voltage */
	if(INIT) {
        CALLBACK = cm_mline_callback;
        STATIC_VAR(sim_points_data) = NULL;
	}

	/* Compute the output */
	if(ANALYSIS == DC) {

		calcPropagation(W,SModel,DModel,er,h,t,tand,rho,D,0);
	    double V1 = INPUT(V1sens);
		double V2 = INPUT(V2sens);
		double I1 = INPUT(port1);
		double I2 = INPUT(port2);
		double V2out = V1 + zl*I1;
		double V1out = V2 + zl*I2;
		OUTPUT(port1) = V1out + I1*zl;
		OUTPUT(port2) = V2out + I2*zl;

		cm_analog_auto_partial();
	}
	else if(ANALYSIS == AC) {
	    double frequency = RAD_FREQ/(2.0*M_PI);
		calcPropagation(W,SModel,DModel,er,h,t,tand,rho,D,frequency);

		double complex g = alpha + beta*I;
		double complex _Z11 = zl / ctanh(g*l);
		double complex _Z21 = zl / csinh(g*l);

		z11.real = creal(_Z11); z11.imag = cimag(_Z11);
		z21.real = creal(_Z21); z21.imag = cimag(_Z21);

		AC_GAIN(port1,port1) = z11; AC_GAIN(port2,port2) = z11;
		AC_GAIN(port1,port2) = z21; AC_GAIN(port2,port1) = z21;
	}
	else if(ANALYSIS == TRANSIENT) {
		calcPropagation(W,SModel,DModel,er,h,t,tand,rho,D,0);
        sim_points = &(STATIC_VAR(sim_points_data));
        double t = TIME;
		double V1 = INPUT(V1sens);
		double V2 = INPUT(V2sens);
		double I1 = INPUT(port1);
		double I2 = INPUT(port2);
		double delay = l/(C0) * sqrt(ereff);
		if (TModel == TRAN_FULL) {

            tline_state_t *last = get_tline_last_state(*(tline_state_t **)sim_points);
			double last_time = 0;
			if (last != NULL) last_time = last->time;

			if (TIME < last_time) {
                //fprintf(stderr,"Rollbacki time=%g\n",TIME);
				delete_tline_last_state((tline_state_t **)sim_points);
			}
			append_state((tline_state_t **)sim_points, t, V1, V2, I1, I2, 1.2*delay);
		}
		if (t > delay && TModel == TRAN_FULL) {
			tline_state_t *pp = get_state(*(tline_state_t **)sim_points, t - delay);
			if (pp != NULL) {
				double V2out = pp->V1 + zl*(pp->I1);
				double V1out = pp->V2 + zl*(pp->I2);
				OUTPUT(port1) = V1out + I1*zl;
				OUTPUT(port2) = V2out + I2*zl;
			}
			cm_analog_auto_partial();
		} else {
            double V2out = V1 + zl*I1;
			double V1out = V2 + zl*I2;
			OUTPUT(port1) = V1out + I1*zl;
			OUTPUT(port2) = V2out + I2*zl;
			cm_analog_auto_partial();
		}

	}
}

static void cm_mline_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY:
            delete_tline_states((tline_state_t **)&(STATIC_VAR(sim_points_data)));
            break;
        default: break;
    }
}

