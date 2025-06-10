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

static double ae, ao, be, bo, ze, zo, ee, eo;

static void copy_complex(double complex s, Complex_t *d)
{
	d->real = creal(s);
	d->imag = cimag(s);
}


static cpline_state_t *state = NULL;

static void analyseQuasiStatic (double W, double h, double s,
				   double t, double er,
				   int SModel, double* Zle,
				   double* Zlo, double* ErEffe,
				   double* ErEffo);

static void analyseDispersion (double W, double h, double s,
				   double t, double er, double Zle,
				   double Zlo, double ErEffe,
				   double ErEffo, double frequency,
				   int  DModel, double *ZleFreq,
				   double  *ZloFreq,
				   double  *ErEffeFreq,
				   double  *ErEffoFreq);

static void calcPropagation (double W, double s,
								 double er, double h, double t, double tand, double rho, double D,
								 int SModel, int DModel, double frequency)
{

	// quasi-static analysis
	double Zle, ErEffe, Zlo, ErEffo;
	analyseQuasiStatic (W, h, s, t, er, SModel, &Zle, &Zlo, &ErEffe, &ErEffo);

	// analyse dispersion of Zl and Er
	double ZleFreq, ErEffeFreq, ZloFreq, ErEffoFreq;
	analyseDispersion (W, h, s, t, er, Zle, Zlo, ErEffe, ErEffo, frequency, DModel,
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



/* The function calculates the quasi-static dielectric constants and
   characteristic impedances for the even and odd mode based upon the
   given line and substrate properties for parallel coupled microstrip
   lines. */
static void analyseQuasiStatic (double W, double h, double s,
				   double t, double er,
				   int SModel, double* Zle,
				   double* Zlo, double* ErEffe,
				   double* ErEffo) {
  // initialize default return values
  *ErEffe = er; *ErEffo = er;
  *Zlo = 42.2; *Zle = 55.7;

  // normalized width and gap
  double u = W / h;
  double g = s / h;

  // HAMMERSTAD and JENSEN
  if (SModel == HAMMERSTAD) {
    double Zl1, Fe, Fo, a, b, fo, Mu, Alpha, Beta, ErEff;
    double Pe, Po, r, fo1, q, p, n, Psi, Phi, m, Theta;

    // modifying equations for even mode
    m = 0.2175 + pow (4.113 + pow (20.36 / g, 6.), -0.251) +
      log (pow (g, 10.) / (1 + pow (g / 13.8, 10.))) / 323;
    Alpha = 0.5 * exp (-g);
    Psi = 1 + g / 1.45 + pow (g, 2.09) / 3.95;
    Phi = 0.8645 * pow (u, 0.172);
    Pe = Phi / (Psi * (Alpha * pow (u, m) + (1 - Alpha) * pow (u, -m)));
    // TODO: is this ... Psi * (Alpha ... or ... Psi / (Alpha ... ?

    // modifying equations for odd mode
    n = (1 / 17.7 + exp (-6.424 - 0.76 * log (g) - pow (g / 0.23, 5.))) *
      log ((10 + 68.3 * sqr (g)) / (1 + 32.5 * pow (g, 3.093)));
    Beta = 0.2306 + log (pow (g, 10.) / (1 + pow (g / 3.73, 10.))) / 301.8 +
      log (1 + 0.646 * pow (g, 1.175)) / 5.3;
    Theta = 1.729 + 1.175 * log (1 + 0.627 / (g + 0.327 * pow (g, 2.17)));
    Po = Pe - Theta / Psi * exp (Beta * pow (u, -n) * log (u));

    // further modifying equations
    r = 1 + 0.15 * (1 - exp (1 - sqr (er - 1) / 8.2) / (1 + pow (g, -6.)));
    fo1 = 1 - exp (-0.179 * pow (g, 0.15) -
		   0.328 * pow (g, r) / log (M_E + pow (g / 7, 2.8)));
    q = exp (-1.366 - g);
    p = exp (-0.745 * pow (g, 0.295)) / cosh (pow (g, 0.68));
    fo = fo1 * exp (p * log (u) + q * sin (M_PI * log10 (u)));

    Mu = g * exp (-g) + u * (20 + sqr (g)) / (10 + sqr (g));
    Hammerstad_ab (Mu, er, &a, &b);
    Fe = pow (1 + 10 / Mu, -a * b);
    Hammerstad_ab (u, er, &a, &b);
    Fo = fo * pow (1 + 10 / u, -a * b);

    // finally compute effective dielectric constants and impedances
    *ErEffe = (er + 1) / 2 + (er - 1) / 2 * Fe;
    *ErEffo = (er + 1) / 2 + (er - 1) / 2 * Fo;

    Hammerstad_er (u, er, a, b, &ErEff);  // single microstrip

    // first variant
    Zl1 = Z0 / (u + 1.98 * pow (u, 0.172));
    Zl1 /= sqrt (ErEff);

    // second variant
    Hammerstad_zl (u, &Zl1);
    Zl1 /= sqrt (ErEff);

    *Zle = Zl1 / (1 - Zl1 * Pe / Z0);
    *Zlo = Zl1 / (1 - Zl1 * Po / Z0);
  }
  // KIRSCHNING and JANSEN
  else if (SModel == KIRSCHING) {
    double a, b, ae, be, ao, bo, v, co, d, ErEff, Zl1;
    double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

    // consider effect of finite strip thickness (JANSEN only)
    double ue = u;
    double uo = u;
    if (t != 0 && s > 10 * (2 * t)) {
      double dW = 0;
      // SCHNEIDER, referred by JANSEN
      if (u >= M_1_PI / 2 && M_1_PI / 2 > 2 * t / h)
	dW = t * (1 + log (2 * h / t)) / M_PI;
      else if (W > 2 * t)
	dW = t * (1 + log (4 * M_PI * W / t)) / M_PI;
      // JANSEN
      double dt = 2 * t * h / s / er;
      double We = W + dW * (1 - 0.5 * exp (-0.69 * dW / dt));
      double Wo = We + dt;
      ue = We / h;
      uo = Wo / h;
    }

    // even relative dielectric constant
    v = ue * (20 + sqr (g)) / (10 + sqr (g)) + g * exp (-g);
    Hammerstad_ab (v, er, &ae, &be);
    Hammerstad_er (v, er, ae, be, ErEffe);

    // odd relative dielectric constant
    Hammerstad_ab (uo, er, &a, &b);
    Hammerstad_er (uo, er, a, b, &ErEff);
    d = 0.593 + 0.694 * exp (-0.562 * uo);
    bo = 0.747 * er / (0.15 + er);
    co = bo - (bo - 0.207) * exp (-0.414 * uo);
    ao = 0.7287 * (ErEff - (er + 1) / 2) * (1 - exp (-0.179 * uo));
    *ErEffo = ((er + 1) / 2 + ao - ErEff) * exp (-co * pow (g, d)) + ErEff;

    // characteristic impedance of single line
    Hammerstad_zl (u, &Zl1);
    Zl1 /= sqrt (ErEff);

    // even characteristic impedance
    q1 = 0.8695 * pow (ue, 0.194);
    q2 = 1 + 0.7519 * g + 0.189 * pow (g, 2.31);
    q3 = 0.1975 + pow (16.6 + pow (8.4 / g, 6.), -0.387) +
      log (pow (g, 10.) / (1 + pow (g / 3.4, 10.))) / 241;
    q4 = q1 / q2 * 2 /
      (exp (-g) * pow (ue, q3) + (2 - exp (-g)) * pow (ue, -q3));
    *Zle = sqrt (ErEff / *ErEffe) * Zl1 / (1 - Zl1 * sqrt (ErEff) * q4 / Z0);

    // odd characteristic impedance
    q5 = 1.794 + 1.14 * log (1 + 0.638 / (g + 0.517 * pow (g, 2.43)));
    q6 = 0.2305 + log (pow (g, 10.) / (1 + pow (g / 5.8, 10.))) / 281.3 +
      log (1 + 0.598 * pow (g, 1.154)) / 5.1;
    q7 = (10 + 190 * sqr (g)) / (1 + 82.3 * cubic (g));
    q8 = exp (-6.5 - 0.95 * log (g) - pow (g / 0.15, 5.));
    q9 = log (q7) * (q8 + 1 / 16.5);
    q10 = (q2 * q4 - q5 * exp (log (uo) * q6 * pow (uo, -q9))) / q2;
    *Zlo = sqrt (ErEff / *ErEffo) * Zl1 / (1 - Zl1 * sqrt (ErEff) * q10 / Z0);
  }
}

/* The function computes the dispersion effects on the dielectric
   constants and characteristic impedances for the even and odd mode
   of parallel coupled microstrip lines. */
static void analyseDispersion (double W, double h, double s,
				   double t, double er, double Zle,
				   double Zlo, double ErEffe,
				   double ErEffo, double frequency,
				   int  DModel, double *ZleFreq,
				   double  *ZloFreq,
				   double  *ErEffeFreq,
				   double  *ErEffoFreq) {

  // initialize default return values
  *ZleFreq = Zle;
  *ErEffeFreq = ErEffe;
  *ZloFreq = Zlo;
  *ErEffoFreq = ErEffo;

  // normalized width and gap
  double u = W / h;
  double g = s / h;
  double ue, uo;
  double B, dW, dt;

  // compute u_odd, u_even
  if (t > 0.0) {
    if (u < 0.1592) {
      B = 2 * M_PI * W;
    } else {
      B = h;
    }
    dW = t * (1.0 + log(2 * B / t)) / M_PI;
    dt = t / (er * g);
    ue = (W + dW * (1.0 - 0.5 * exp( -0.69 * dW / dt ))) / h;
    uo = ue + dt / h;
  } else {
    ue = u;
    uo = u;
  }

  // GETSINGER
  if (DModel == GETSINGER) {
    // even mode dispersion
    Getsinger_disp (h, er, ErEffe, Zle / 2,
			    frequency, ErEffeFreq, ZleFreq);
    *ZleFreq *= 2;
    // odd mode dispersion
    Getsinger_disp (h, er, ErEffo, Zlo * 2,
			    frequency, ErEffoFreq, ZloFreq);
    *ZloFreq /= 2;
  }
  // KIRSCHNING and JANSEN
  else if (DModel == DISP_KIRSCHING) {
    double p1, p2, p3, p4, p5, p6, p7, Fe;
    double fn = frequency * h * 1e-6;

    // even relative dielectric constant dispersion
    p1 = 0.27488 * (0.6315 + 0.525 / pow (1 + 0.0157 * fn, 20.)) * ue -
      0.065683 * exp (-8.7513 * ue);
    p2 = 0.33622 * (1 - exp (-0.03442 * er));
    p3 = 0.0363 * exp (-4.6 * ue) * (1 - exp (- pow (fn / 38.7, 4.97)));
    p4 = 1 + 2.751 * (1 - exp (- pow (er / 15.916, 8.)));
    p5 = 0.334 * exp (-3.3 * cubic (er / 15)) + 0.746;
    p6 = p5 * exp (- pow (fn / 18, 0.368));
    p7 = 1 + 4.069 * p6 * pow (g, 0.479) *
      exp (-1.347 * pow (g, 0.595) - 0.17 * pow (g, 2.5));
    Fe = p1 * p2 * pow ((p3 * p4 + 0.1844 * p7) * fn, 1.5763);
    *ErEffeFreq = er - (er - ErEffe) / (1 + Fe);

    // odd relative dielectric constant dispersion
    double p8, p9, p10, p11, p12, p13, p14, p15, Fo;
    p1 = 0.27488 * (0.6315 + 0.525 / pow (1 + 0.0157 * fn, 20.)) * uo -
      0.065683 * exp (-8.7513 * uo);
    p3 = 0.0363 * exp (-4.6 * uo) * (1 - exp (- pow (fn / 38.7, 4.97)));
    p8 = 0.7168 * (1 + 1.076 / (1 + 0.0576 * (er - 1)));
    p9 = p8 - 0.7913 * (1 - exp (- pow (fn / 20, 1.424))) *
      atan (2.481 * pow (er / 8, 0.946));
    p10 = 0.242 * pow (er - 1, 0.55);
    p11 = 0.6366 * (exp (-0.3401 * fn) - 1) *
      atan (1.263 * pow (uo / 3, 1.629));
    p12 = p9 + (1 - p9) / (1 + 1.183 * pow (uo, 1.376));
    p13 = 1.695 * p10 / (0.414 + 1.605 * p10);
    p14 = 0.8928 + 0.1072 * (1 - exp (-0.42 * pow (fn / 20, 3.215)));
    p15 = fabs (1 - 0.8928 * (1 + p11) *
		exp (-p13 * pow (g, 1.092)) * p12 / p14);
    Fo = p1 * p2 * pow ((p3 * p4 + 0.1844) * fn * p15, 1.5763);
    *ErEffoFreq = er - (er - ErEffo) / (1 + Fo);

    // dispersion of even characteristic impedance
    double t, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21;
    q11 = 0.893 * (1 - 0.3 / (1 + 0.7 * (er - 1)));
    t = pow (fn / 20, 4.91);
    q12 = 2.121 * t / (1 + q11 * t) * exp (-2.87 * g) * pow (g, 0.902);
    q13 = 1 + 0.038 * pow (er / 8, 5.1);
    t = quadr (er / 15);
    q14 = 1 + 1.203 * t / (1 + t);
    q15 = 1.887 * exp (-1.5 * pow (g, 0.84)) * pow (g, q14) /
      (1 + 0.41 * pow (fn / 15, 3.) *
       pow (u, 2 / q13) / (0.125 + pow (u, 1.626 / q13)));
    q16 = q15 * (1 + 9 / (1 + 0.403 * sqr (er - 1)));
    q17 = 0.394 * (1 - exp (-1.47 * pow (u / 7, 0.672))) *
      (1 - exp (-4.25 * pow (fn / 20, 1.87)));
    q18 = 0.61 * (1 - exp (-2.31 * pow (u / 8, 1.593))) /
      (1 + 6.544 * pow (g, 4.17));
    q19 = 0.21 * quadr (g) / (1 + 0.18 * pow (g, 4.9)) / (1 + 0.1 * sqr (u)) /
      (1 + pow (fn / 24, 3.));
    q20 = q19 * (0.09 + 1 / (1 + 0.1 * pow (er - 1, 2.7)));
    t = pow (u, 2.5);
    q21 = fabs (1 - 42.54 * pow (g, 0.133) * exp (-0.812 * g) * t /
		(1 + 0.033 * t));

    double re, qe, pe, de, Ce, q0, ZlFreq, ErEffFreq;
    Kirschning_er (u, fn, er, ErEffe, &ErEffFreq);
    Kirschning_zl (u, fn, er, ErEffe, ErEffFreq, Zle, &q0, &ZlFreq);
    re = pow (fn / 28.843, 12.);
    qe = 0.016 + pow (0.0514 * er * q21, 4.524);
    pe = 4.766 * exp (-3.228 * pow (u, 0.641));
    t = pow (er - 1, 6.);
    de = 5.086 * qe * re / (0.3838 + 0.386 * qe) *
      exp (-22.2 * pow (u, 1.92)) / (1 + 1.2992 * re) * t / (1 + 10 * t);
    Ce = 1 + 1.275 * (1 - exp (-0.004625 * pe * pow (er, 1.674) *
	 pow (fn / 18.365, 2.745))) - q12 + q16 - q17 + q18 + q20;
    *ZleFreq = Zle * pow ((0.9408 * pow (ErEffFreq, Ce) - 0.9603) /
			 ((0.9408 - de) * pow (ErEffe, Ce) - 0.9603), q0);

    // dispersion of odd characteristic impedance
    double q22, q23, q24, q25, q26, q27, q28, q29;
    Kirschning_er (u, fn, er, ErEffo, &ErEffFreq);
    Kirschning_zl (u, fn, er, ErEffo, ErEffFreq, Zlo, &q0, &ZlFreq);
    q29 = 15.16 / (1 + 0.196 * sqr (er - 1));
    t = sqr (er - 1);
    q25 = 0.3 * sqr (fn) / (10 + sqr (fn)) * (1 + 2.333 * t / (5 + t));
    t = pow ((er - 1) / 13, 12.);
    q26 = 30 - 22.2 * t / (1 + 3 * t) - q29;
    t = pow (er - 1, 1.5);
    q27 = 0.4 * pow (g, 0.84) * (1 + 2.5 * t / (5 + t));
    t = pow (er - 1, 3.);
    q28 = 0.149 * t / (94.5 + 0.038 * t);
    q22 = 0.925 * pow (fn / q26, 1.536) / (1 + 0.3 * pow (fn / 30, 1.536));
    q23 = 1 + 0.005 * fn * q27 / (1 + 0.812 * pow (fn / 15, 1.9)) /
      (1 + 0.025 * sqr (u));
    t = pow (u, 0.894);
    q24 = 2.506 * q28 * t / (3.575 + t) *
      pow ((1 + 1.3 * u) * fn / 99.25, 4.29);
    *ZloFreq = ZlFreq + (Zlo * pow (*ErEffoFreq / ErEffo, q22) - ZlFreq * q23) /
      (1 + q24 + pow (0.46 * g, 2.2) * q25);

  }
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

	/* how to get properties of the substrate, e.g. Er, H */
	double er    = PARAM(er);
	double h     = PARAM(h);
	double t     = PARAM(t);
	double tand  = PARAM(tand);
	double rho   = PARAM(rho);
	double D     = PARAM(d);



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
		double complex _Z11, _Z12, _Z13, _Z14;
		double complex ge =  ae + I*be;
		double complex go =  ao + I*bo;

		_Z11 = zo / (2*ctanh(go*l)) + ze / (2*ctanh(ge*l));
		_Z12 = zo / (2*csinh(go*l)) + ze / (2*csinh(ge*l));
		_Z13 = ze / (2*csinh(ge*l)) - zo / (2*csinh(go*l));
		_Z14 = ze / (2*ctanh(ge*l)) - zo / (2*ctanh(go*l));

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
		double t = TIME;
		double Vp[PORT_NUM];
		double Ip[PORT_NUM];
		double Vnew[PORT_NUM];
		Vp[0] = INPUT(p1s);
		Vp[1] = INPUT(p2s);
		Vp[2] = INPUT(p3s);
		Vp[3] = INPUT(p4s);
		Ip[0] = INPUT(p1);
		Ip[1] = INPUT(p2);
		Ip[2] = INPUT(p3);
		Ip[3] = INPUT(p4);
		double delay = l/(C0);
		append_cpline_state(&state, t, Vp, Ip, 1.2*delay);
		if (t > delay) {
			cpline_state_t *pp = find_cpline_state(state, t - delay);
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

