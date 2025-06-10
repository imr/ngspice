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
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "msline_common.h"
#include "tline_common.h"


static void copy_complex(double complex s, Complex_t *d)
{
    d->real = creal(s);
	d->imag = cimag(s);
}

cpline_state_t *sim_points = NULL;


void cm_cpline (ARGS)
{
	Complex_t   z11, z12, z13, z14;

	/* how to get properties of this component, e.g. L, W */
	double l   = PARAM(l);
	double ze  = PARAM(ze);
	double zo  = PARAM(zo);
	double ere = PARAM(ere);
	double ero = PARAM(ero);
	double ae  = PARAM(ae);
	double ao  = PARAM(ao);
	ae = pow(10, 0.05*ae);
	ao = pow(10, 0.05*ao);

	if(INIT) {

	}

	/* Compute the output */
	if(ANALYSIS == DC) {

		  double V1 = INPUT(p1s);
		  double V2 = INPUT(p2s);
		  double V3 = INPUT(p3s);
		  double V4 = INPUT(p4s);
		  double I1 = INPUT(p1);
		  double I2 = INPUT(p2);
		  double I3 = INPUT(p3);
		  double I4 = INPUT(p4);

		  double z = sqrt(ze*zo);

		  double V2out = V1 + z*I1;
		  double V1out = V2 + z*I2;
		  OUTPUT(p1) = V1out + I1*z;
		  OUTPUT(p2) = V2out + I2*z;

		  double V3out = V4 + z*I4;
		  double V4out = V3 + z*I3;
		  OUTPUT(p3) = V3out + I3*z;
		  OUTPUT(p4) = V4out + I4*z;

		  cm_analog_auto_partial();
	}
	else if(ANALYSIS == AC) {
		double o = RAD_FREQ;

		double complex _Z11, _Z12, _Z13, _Z14;
		double complex arg_e =  log(ae)*l/2.0 + I*o*l/C0*sqrt(ere);
		double complex arg_o =  log(ao)*l/2.0 + I*o*l/C0*sqrt(ero);

		_Z11 = zo / (2*ctanh(arg_o)) + ze / (2*ctanh(arg_e));
		_Z12 = zo / (2*csinh(arg_o)) + ze / (2*csinh(arg_e));
		_Z13 = ze / (2*csinh(arg_e)) - zo / (2*csinh(arg_o));
		_Z14 = ze / (2*ctanh(arg_e)) - zo / (2*ctanh(arg_o));

		copy_complex(_Z11,&z11);
		copy_complex(_Z12,&z12);
		copy_complex(_Z13,&z13);
		copy_complex(_Z14,&z14);

        AC_GAIN(p1,p1) = z11; AC_GAIN(p2,p2) = z11;
        AC_GAIN(p3,p3) = z11; AC_GAIN(p4,p4) = z11;

		AC_GAIN(p1,p2) = z12; AC_GAIN(p2,p1) = z12;
        AC_GAIN(p3,p4) = z12; AC_GAIN(p4,p3) = z12;

		AC_GAIN(p1,p3) = z13; AC_GAIN(p3,p1) = z13;
        AC_GAIN(p2,p4) = z13; AC_GAIN(p4,p2) = z13;

		AC_GAIN(p1,p4) = z14; AC_GAIN(p4,p1) = z14;
        AC_GAIN(p2,p3) = z14; AC_GAIN(p3,p2) = z14;
	}
	else if(ANALYSIS == TRANSIENT) {
		double t = TIME;
		double Vp[PORT_NUM];
		double Ip[PORT_NUM];

		Vp[0] = INPUT(p1s);
		Vp[1] = INPUT(p2s);
		Vp[2] = INPUT(p3s);
		Vp[3] = INPUT(p4s);
		Ip[0] = INPUT(p1);
		Ip[1] = INPUT(p2);
		Ip[2] = INPUT(p3);
		Ip[3] = INPUT(p4);
		double delay = l/(C0);
		append_cpline_state(&sim_points, t, Vp, Ip, 1.2*delay);
		if (t > delay) {
			cpline_state_t *pp = find_cpline_state(sim_points, t - delay);
			if (pp != NULL) {

				double J1e = 0.5*(Ip[3] + Ip[0]);
				double J1o = 0.5*(Ip[0] - Ip[3]);
				double J2e = 0.5*(Ip[1] + Ip[2]);
				double J2o = 0.5*(Ip[1] - Ip[2]);


				double J1et = 0.5*(pp->Ip[3] + pp->Ip[0]);
				double J1ot = 0.5*(pp->Ip[0] - pp->Ip[3]);
				double J2et = 0.5*(pp->Ip[1] + pp->Ip[2]);
				double J2ot = 0.5*(pp->Ip[1] - pp->Ip[2]);


				double V1et = 0.5*(pp->Vp[3] + pp->Vp[0]);
				double V1ot = 0.5*(pp->Vp[0] - pp->Vp[3]);
				double V2et = 0.5*(pp->Vp[1] + pp->Vp[2]);
				double V2ot = 0.5*(pp->Vp[1] - pp->Vp[2]);

				double V1e = ze*J1e + V2et + ze*J2et;
				double V1o = zo*J1o + V2ot + zo*J2ot;
				double V2e = ze*J2e + V1et + ze*J1et;
				double V2o = zo*J2o + V1ot + zo*J1ot;

				double V1 = V1o + V1e;
				double V2 = V2o + V2e;
				double V3 = V2e - V2o;
				double V4 = V1e - V1o;

				OUTPUT(p1) = V1;
				OUTPUT(p2) = V2;
				OUTPUT(p3) = V3;
				OUTPUT(p4) = V4;
			}
			cm_analog_auto_partial();
		} else {
			cm_analog_auto_partial();
		}
	}
}

