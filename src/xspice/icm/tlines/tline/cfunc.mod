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

#include "msline_common.h"
#include "tline_common.h"

//static tline_state_t *sim_points = NULL;

static void cm_tline_callback(ARGS, Mif_Callback_Reason_t reason);

void cm_tline (ARGS)
{
	Complex_t   z11, z21;
    void **sim_points;


	/* how to get properties of this component, e.g. L, W */
	double z = PARAM(z);
	double l = PARAM(l);
	double a = PARAM(a);

	double alpha = pow(10,0.05*a);
	alpha = log(alpha)/2.0;

	/* Initialize/access instance specific storage for capacitor voltage */
	if(INIT) {
        CALLBACK = cm_tline_callback;
        STATIC_VAR(sim_points_data) = NULL;
	}

	/* Compute the output */
	if(ANALYSIS == DC) {

		double V1 = INPUT(V1sens);
		double V2 = INPUT(V2sens);
		double I1 = INPUT(in);
		double I2 = INPUT(out);
		double V2out = V1 + z*I1;
		double V1out = V2 + z*I2;
		OUTPUT(in) = V1out + I1*z;
		OUTPUT(out) = V2out + I2*z;

		cm_analog_auto_partial();
	}
	else if(ANALYSIS == AC) {
		double beta = RAD_FREQ/C0;
		double complex g = alpha + beta*I;
		double complex _Z11 = z / ctanh(g*l);
		double complex _Z21 = z / csinh (g*l);

		z11.real = creal(_Z11);
		z11.imag = cimag(_Z11);
		z21.real = creal(_Z21);
		z21.imag = cimag(_Z21);


		AC_GAIN(in, in) = z11; AC_GAIN(out,out) = z11;
		AC_GAIN(in,out) = z21; AC_GAIN(out,in) = z21;
	}
	else if(ANALYSIS == TRANSIENT) {
        sim_points = &(STATIC_VAR(sim_points_data));
		double t = TIME;
		double V1 = INPUT(V1sens);
		double V2 = INPUT(V2sens);
		double I1 = INPUT(in);
		double I2 = INPUT(out);
		double delay = l/(C0);

        tline_state_t *last = get_tline_last_state(*(tline_state_t **)sim_points);
        double last_time = 0;
        if (last != NULL) last_time = last->time;

        if (TIME < last_time) {
            //fprintf(stderr,"Rollback time=%g\n",TIME);
			delete_tline_last_state((tline_state_t **)sim_points);
		}

		append_state((tline_state_t **)sim_points, t, V1, V2, I1, I2, 1.2*delay);
		if (t > delay) {
			tline_state_t *pp = get_state(*(tline_state_t **)sim_points, t - delay);
			if (pp != NULL) {
				double V2out = pp->V1 + z*(pp->I1);
				double V1out = pp->V2 + z*(pp->I2);
				OUTPUT(in) = V1out + I1*z;
				OUTPUT(out) = V2out + I2*z;
			}
			cm_analog_auto_partial();
		} else  {
			double V2out = V1 + z*I1;
			double V1out = V2 + z*I2;
			OUTPUT(in) = V1out + I1*z;
			OUTPUT(out) = V2out + I2*z;
			cm_analog_auto_partial();
		}
	}
}


static void cm_tline_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY:
            delete_tline_states((tline_state_t **)&(STATIC_VAR(sim_points_data)));
            break;
        default: break;
    }
}
