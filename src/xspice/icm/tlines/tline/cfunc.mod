/* ===========================================================================
   FILE    cfunc.mod

   (c) Vadim Kuznetsov 2025

 */

#include <stdio.h>
#include <math.h>
#include <complex.h>

#include "msline_common.h"
#include "tline_common.h"

static tline_state_t *sim_points = NULL;


void cm_tline (ARGS)
{
	Complex_t   z11, z21;


	/* how to get properties of this component, e.g. L, W */
	double z = PARAM(z);
	double l = PARAM(l);
	double a = PARAM(a);

	double alpha = pow(10,0.05*a);
	alpha = log(alpha)/2.0;

	/* Initialize/access instance specific storage for capacitor voltage */
	if(INIT) {

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
		double t = TIME;
		double V1 = INPUT(V1sens);
		double V2 = INPUT(V2sens);
		double I1 = INPUT(in);
		double I2 = INPUT(out);
		double delay = l/(C0);
		append_state(&sim_points, t, V1, V2, I1, I2, 1.2*delay);
		if (t > delay) {
			tline_state_t *pp = get_state(sim_points, t - delay);
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

