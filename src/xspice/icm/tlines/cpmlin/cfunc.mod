/* ===========================================================================
	FILE    cfunc.mod for cm_cpmline
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

#ifdef _MSC_VER
typedef _Dcomplex DoubleComplex;  // double complex
#else
typedef double complex DoubleComplex;
#endif

static double ae, ao, be, bo, ze, zo, ee, eo;

static void copy_complex(DoubleComplex s, Complex_t *d)
{
	d->real = creal(s);
	d->imag = cimag(s);
}

#ifdef _MSC_VER
static DoubleComplex divide(DoubleComplex n1, DoubleComplex n2)
    {
        DoubleComplex rez;
        double denom = n2._Val[0] * n2._Val[0] + n2._Val[1] * n2._Val[1];
        rez._Val[0] = (n1._Val[0] * n2._Val[0] + n1._Val[1] * n2._Val[1]) / denom;
        rez._Val[1] = (n1._Val[1] * n2._Val[0] - n1._Val[0] * n2._Val[1]) / denom;
        return rez;
    }

static DoubleComplex rdivide(double n1, DoubleComplex n2)
    {
        DoubleComplex rez;
        double denom = n2._Val[0] * n2._Val[0] + n2._Val[1] * n2._Val[1];
        rez._Val[0] = (n1 * n2._Val[0]) / denom;
        rez._Val[1] = (-1. * n1 * n2._Val[1]) / denom;
        return rez;
    }
#endif

//static cpline_state_t *state = NULL;
static void cm_cpmline_callback(ARGS, Mif_Callback_Reason_t reason);

static void calcPropagation (double W, double s,
								 double er, double h, double t, double tand, double rho, double D,
								 int SModel, int DModel, double frequency)
{

	// quasi-static analysis
	double Zle, ErEffe, Zlo, ErEffo;
	cpmslineAnalyseQuasiStatic (W, h, s, t, er, SModel, &Zle, &Zlo, &ErEffe, &ErEffo);

	// analyse dispersion of Zl and Er
	double ZleFreq, ErEffeFreq, ZloFreq, ErEffoFreq;
	cpmslineAnalyseDispersion (W, h, s, t, er, Zle, Zlo, ErEffe, ErEffo, frequency, DModel,
					   &ZleFreq, &ZloFreq, &ErEffeFreq, &ErEffoFreq);

	// analyse losses of line
	double ace, aco, ade, ado;
	analyseLoss (W, t, er, rho, D, tand, Zle, Zlo, ErEffe,
						 frequency, HAMMERSTAD, &ace, &ade);
	analyseLoss (W, t, er, rho, D, tand, Zlo, Zle, ErEffo,
						 frequency, HAMMERSTAD, &aco, &ado);

	// compute propagation constants for even and odd mode
	double k0 = 2 * M_PI * frequency / C0;
	ae = ace + ade;
	ao = aco + ado;
	be = sqrt (ErEffeFreq) * k0;
	bo = sqrt (ErEffoFreq) * k0;
	ze = ZleFreq;
	zo = ZloFreq;
	ee = ErEffeFreq;
	eo = ErEffoFreq;
}



void cm_cpmline (ARGS)
{
	Complex_t   z11, z12, z13, z14;

	/* how to get properties of this component, e.g. L, W */
	double W = PARAM(w);
	double l = PARAM(l);
	double s = PARAM(s);
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

    if(INIT) {
        CALLBACK = cm_cpmline_callback;
        STATIC_VAR(sim_points_data) = NULL;
	}


	/* Compute the output */
	if(ANALYSIS == DC) {
          calcPropagation(W,s,er,h,t,tand,rho,D,SModel,DModel,0);

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
        calcPropagation(W,s,er,h,t,tand,rho,D,SModel,DModel, o/(2*M_PI));
		DoubleComplex _Z11, _Z12, _Z13, _Z14;
#ifdef _MSC_VER
		DoubleComplex ge = _Cbuild(ae, be);
		DoubleComplex go = _Cbuild(ao, bo);
        DoubleComplex tango = _Cmulcr(ctanh(_Cmulcr(go, l)), 2.);
        DoubleComplex tange = _Cmulcr(ctanh(_Cmulcr(ge, l)), 2.);
        DoubleComplex singo = _Cmulcr(csinh(_Cmulcr(go, l)), 2.);
        DoubleComplex singe = _Cmulcr(csinh(_Cmulcr(ge, l)), 2.);
        DoubleComplex zotango = rdivide(zo, tango);
        DoubleComplex zetange = rdivide(ze, tange);
        DoubleComplex zosingo = rdivide(zo, singo);
        DoubleComplex zesinge = rdivide(ze, singe);

		_Z11._Val[0] = zotango._Val[0] + zetange._Val[0];
        _Z11._Val[1] = zotango._Val[1] + zetange._Val[1];
        _Z12._Val[0] = zosingo._Val[0] + zesinge._Val[0];
        _Z12._Val[1] = zosingo._Val[1] + zesinge._Val[1];
		_Z13._Val[0] = zesinge._Val[0] - zosingo._Val[0];
		_Z13._Val[1] = zesinge._Val[1] - zosingo._Val[1];
		_Z14._Val[0] = zetange._Val[0] - zotango._Val[0];
		_Z14._Val[1] = zetange._Val[1] - zotango._Val[1];
#else
		DoubleComplex ge =  ae + I*be;
		DoubleComplex go =  ao + I*bo;

		_Z11 = zo / (2*ctanh(go*l)) + ze / (2*ctanh(ge*l));
		_Z12 = zo / (2*csinh(go*l)) + ze / (2*csinh(ge*l));
		_Z13 = ze / (2*csinh(ge*l)) - zo / (2*csinh(go*l));
		_Z14 = ze / (2*ctanh(ge*l)) - zo / (2*ctanh(go*l));
#endif
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
        calcPropagation(W,s,er,h,t,tand,rho,D,SModel,DModel,0);
		double time = TIME;
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
        void **sim_points = &(STATIC_VAR(sim_points_data));
		if (TModel == TRAN_FULL) {
			cpline_state_t *last = get_cpline_last_state(*(cpline_state_t **)sim_points);
			double last_time = 0;
			if (last != NULL) last_time = last->time;

            if (TIME < last_time) {
				delete_cpline_last_state((cpline_state_t **)sim_points);
			}
			append_cpline_state((cpline_state_t **)sim_points, time, Vp, Ip, 1.2*delay);
		}
        if (time > delay && TModel == TRAN_FULL) {
			cpline_state_t *pp = find_cpline_state(*(cpline_state_t **)sim_points, time - delay);
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
		    double z = sqrt(ze*zo);
			double V2out = Vp[0] + z*Ip[0];
			double V1out = Vp[1] + z*Ip[1];
			OUTPUT(p1) = V1out + Ip[0]*z;
			OUTPUT(p2) = V2out + Ip[1]*z;

			double V3out = Vp[3] + z*Ip[3];
			double V4out = Vp[2] + z*Ip[2];
			OUTPUT(p3) = V3out + Ip[2]*z;
			OUTPUT(p4) = V4out + Ip[3]*z;

			cm_analog_auto_partial();
		}
	}
}

static void cm_cpmline_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY:
            delete_cpline_states((cpline_state_t **)&(STATIC_VAR(sim_points_data)));
            break;
        default: break;
    }
}
