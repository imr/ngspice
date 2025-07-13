/* ===========================================================================
 FILE    msline_common.c - common definitions for microstrip devices
 Copyright 2025 Vadim Kuznetsov

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


#include <stdio.h>
#include <math.h>

#include "tline_common.h"
#include "msline_common.h"

/* This function calculates the quasi-static impedance of a microstrip
 *  line, the value of the effective dielectric constant and the
 *  effective width due to the finite conductor thickness for the given
 *  microstrip line and substrate properties. */
void mslineAnalyseQuasiStatic (double W, double h, double t,
		double er, int Model,
		double *ZlEff, double *ErEff,
		double *WEff) {

	double z, e;

	// default values
	e = er;
	z = z0;
	*WEff = W;

	// WHEELER
	if (Model == WHEELER) {
		double a, b, c, d, x, dW1, dWr, Wr;

		// compute strip thickness effect
		if (t != 0) {
			dW1 = t / M_PI * log (4 * M_E / sqrt (sqr (t / h) +
						sqr (M_1_PI / (W / t + 1.10))));
		}
		else dW1 = 0;
		dWr = (1 + 1 / er) / 2 * dW1;
		Wr  =  W + dWr; *WEff =Wr;

		// compute characteristic impedance
		if (W / h < 3.3) {
			c = log (4 * h / Wr + sqrt (sqr (4 * h / Wr) + 2));
			b = (er - 1) / (er + 1) / 2 * (log (M_PI_2) + log (2 * M_2_PI) / er);
			z = (c - b) * Z0 / M_PI / sqrt (2 * (er + 1));
		}
		else {
			c = 1 + log (M_PI_2) + log (Wr / h / 2 + 0.94);
			d = M_1_PI / 2 * (1 + log (sqr (M_PI) / 16)) * (er - 1) / sqr (er);
			x = 2 * M_LN2 / M_PI + Wr / h / 2 + (er + 1) / 2 / M_PI / er * c + d;
			z = Z0 / 2 / x / sqrt (er);
		}

		// compute effective dielectric constant
		if (W / h < 1.3) {
			a = log (8 * h / Wr) + sqr (Wr / h) / 32;
			b = (er - 1) / (er + 1) / 2 * (log (M_PI_2) + log (2 * M_2_PI) / er);
			e = (er + 1) / 2 * sqr (a / (a - b));
		}
		else {
			a = (er - 1) / 2 / M_PI / er * (log (2.1349 * Wr / h + 4.0137) -
					0.5169 / er);
			b = Wr / h / 2 + M_1_PI * log (8.5397 * Wr / h + 16.0547);
			e = er * sqr ((b - a) / b);
		}
	}
	// SCHNEIDER
	else if (Model == SCHNEIDER) {

		double dW = 0, u = W / h;

		// consider strip thickness equations
		if (t != 0 && t < W / 2) {
			double arg = (u < M_1_PI / 2) ? 2 * M_PI * W / t : h / t;
			dW = t / M_PI * (1 + log (2 * arg));
			if (t / dW >= 0.75) dW = 0;
		}
		*WEff = W + dW; u = *WEff / h;

		// effective dielectric constant
		e = (er + 1) / 2 + (er - 1) / 2 / sqrt (1 + 10 / u);

		// characteristic impedance
		if (u < 1.0) {
			z = M_1_PI / 2 * log (8 / u + u / 4);
		}
		else {
			z = 1 / (u + 2.42 - 0.44 / u + pow ((1. - 1. / u), 6.));
		}
		z = Z0 * z / sqrt (e);
	}
	// HAMMERSTAD and JENSEN
	else if (Model == HAMMERSTAD) {
		double a, b, du1, du, u, ur, u1, zr, z1;

		u = W / h; // normalized width
		t = t / h; // normalized thickness

		// compute strip thickness effect
		if (t != 0) {
			du1 = t / M_PI * log (1 + 4 * M_E / t / sqr (coth (sqrt (6.517 * u))));
		}
		else du1 = 0;
		du = du1 * (1 + sech (sqrt (er - 1))) / 2;
		u1 = u + du1;
		ur = u + du;
		*WEff = ur * h;

		// compute impedances for homogeneous medium
		Hammerstad_zl (ur, &zr);
		Hammerstad_zl (u1, &z1);

		// compute effective dielectric constant
		Hammerstad_ab (ur, er, &a, &b);
		Hammerstad_er (ur, er, a, b, &e);

		// compute final characteristic impedance and dielectric constant
		// including strip thickness effects
		z = zr / sqrt (e);
		e = e * sqr (z1 / zr);
	}

	*ZlEff = z;
	*ErEff = e;
}

/* This function calculates the frequency dependent value of the
 *  effective dielectric constant and the microstrip line impedance for
 *  the given frequency. */
void mslineAnalyseDispersion (double W, double h, double er,
		double ZlEff, double ErEff,
		double frequency, int Model,
		double* ZlEffFreq,
		double* ErEffFreq) {

	double e, z;

	// default values
	z = *ZlEffFreq = ZlEff;
	e = *ErEffFreq = ErEff;

	// GETSINGER
	if (Model == GETSINGER) {
		Getsinger_disp (h, er, ErEff, ZlEff, frequency, &e, &z);
	}
	// SCHNEIDER
	else if (Model == DISP_SCHNEIDER) {
		double k, f;
		k = sqrt (ErEff / er);
		f = 4 * h * frequency / C0 * sqrt (er - 1);
		f = sqr (f);
		e = ErEff * sqr ((1 + f) / (1 + k * f));
		z = ZlEff * sqrt (ErEff / e);
	}
	// YAMASHITA
	else if (Model == YAMASHITA) {
		double k, f;
		k = sqrt (er / ErEff);
		f = 4 * h * frequency / C0 * sqrt (er - 1) *
			(0.5 + sqr (1 + 2 * log10 (1 + W / h)));
		e = ErEff * sqr ((1 + k * pow (f, 1.5) / 4) / (1 + pow (f, 1.5) / 4));
	}
	// KOBAYASHI
	else if (Model == KOBAYASHI) {
		double n, no, nc, fh, fk;
		fk = C0 * atan (er * sqrt ((ErEff - 1) / (er - ErEff))) /
			(2 * M_PI * h * sqrt (er - ErEff));
		fh = fk / (0.75 + (0.75 - 0.332 / pow (er, 1.73)) * W / h);
		no = 1 + 1 / (1 + sqrt (W / h)) + 0.32 * cubic (1 / (1 + sqrt (W / h)));
		if (W / h < 0.7) {
			nc = 1 + 1.4 / (1 + W / h) * (0.15 - 0.235 *
					exp (-0.45 * frequency / fh));
		}
		else nc = 1;
		n = no * nc < 2.32 ? no * nc : 2.32;
		e = er - (er - ErEff) / (1 + pow (frequency / fh, n));
	}
	// PRAMANICK and BHARTIA
	else if (Model == PRAMANICK) {
		double Weff, We, f;
		f = 2 * MU0 * h * frequency * sqrt (ErEff / er) / ZlEff;
		e = er - (er - ErEff) / (1 + sqr (f));
		Weff = Z0 * h / ZlEff / sqrt (ErEff);
		We = W + (Weff - W) / (1 + sqr (f));
		z = Z0 * h / We / sqrt (e);
	}
	// HAMMERSTAD and JENSEN
	else if (Model == DISP_HAMMERSTAD) {
		double f, g;
		g = sqr (M_PI) / 12 * (er - 1) / ErEff * sqrt (2 * M_PI * ZlEff / Z0);
		f = 2 * MU0 * h * frequency / ZlEff;
		e = er - (er - ErEff) / (1 + g * sqr (f));
		z = ZlEff * sqrt (ErEff / e) * (e - 1) / (ErEff - 1);
	}
	// KIRSCHNING and JANSEN
	else if (Model == DISP_KIRSCHING) {
		double r17, u  = W / h, fn = frequency * h / 1e6;

		// dispersion of dielectric constant
		Kirschning_er (u, fn, er, ErEff, &e);

		// dispersion of characteristic impedance
		Kirschning_zl (u, fn, er, ErEff, e, ZlEff, &r17, &z);
	}

	*ZlEffFreq = z;
	*ErEffFreq = e;
}


/* Computes the exponent factors a(u) and b(er) used within the
 *  effective relative dielectric constant calculations for single and
 *  coupled microstrip lines by Hammerstad and Jensen. */
void Hammerstad_ab (double u, double er, double *a,
		double *b) {
	*a = 1 + log ((quadr (u) + sqr (u / 52)) / (quadr (u) + 0.432)) / 49 +
		log (1 + cubic (u / 18.1)) / 18.7;
	*b = 0.564 * pow ((er - 0.9) / (er + 3), 0.053);
}

/* The function computes the effective dielectric constant of a single
 *  microstrip.  The equation is used in single and coupled microstrip
 *  calculations. */
void Hammerstad_er (double u, double er, double a,
		double b, double* e) {
	*e = (er + 1) / 2 + (er - 1) / 2 * pow (1 + 10 / u, -a * b);
}

/* This function computes the characteristic impedance of single
 *  microstrip line based upon the given width-height ratio.  The
 *  equation is used in single and coupled microstrip calculations as
 *  well. */
void Hammerstad_zl (double u, double *zl) {
	double fu = 6 + (2 * M_PI - 6) * exp (- pow (30.666 / u, 0.7528));
	*zl = Z0 / 2 / M_PI * log (fu / u + sqrt (1 + sqr (2 / u)));
}

/* Calculates dispersion effects for effective dielectric constant and
 *  characteristic impedance as defined by Getsinger (for single and
 *  coupled microstrips). */
void Getsinger_disp (double h, double er, double ErEff,
		double ZlEff, double frequency,
		double *e, double *z) {
	double g, f, d;
	g = 0.6 + 0.009 * ZlEff;
	f = frequency * 2 * MU0 * h / ZlEff;
	*e = er - (er - ErEff) / (1 + g * sqr (f));
	d = (er - *e) * (*e - ErEff) / *e / (er - ErEff);
	*z = ZlEff * sqrt (*e / ErEff) / (1 + d);  // group delay model
}

/* This function computes the dispersion of the effective dielectric
 *  constant of a single microstrip line.  It is defined in a separate
 *  function because it is used within the coupled microstrip lines as
 *  well. */
void Kirschning_er (double u, double fn, double er,
		double ErEff, double* ErEffFreq) {
	double p, p1, p2, p3, p4;
	p1 = 0.27488 + (0.6315 + 0.525 / pow (1. + 0.0157 * fn, 20.)) * u -
		0.065683 * exp (-8.7513 * u);
	p2 = 0.33622 * (1 - exp (-0.03442 * er));
	p3 = 0.0363 * exp (-4.6 * u) * (1 - exp (- pow (fn / 38.7, 4.97)));
	p4 = 1 + 2.751 * (1 - exp (- pow (er / 15.916, 8.)));
	p  = p1 * p2 * pow ((0.1844 + p3 * p4) * fn, 1.5763);
	*ErEffFreq  = er - (er - ErEff) / (1 + p);
}

/* Computes dispersion effects of characteristic impedance of a single
 *  microstrip line according to Kirschning and Jansen.  Also used in
 *  coupled microstrip lines calculations. */
void Kirschning_zl (double u, double fn, double er,
		double ErEff, double ErEffFreq,
		double ZlEff, double* r17,
		double* ZlEffFreq) {
	double r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;
	double r11, r12, r13, r14, r15, r16;
	r1 = 0.03891 * pow (er, 1.4);
	r2 = 0.267 * pow (u, 7.);
	r3 = 4.766 * exp (-3.228 * pow (u, 0.641));
	r4 = 0.016 + pow (0.0514 * er, 4.524);
	r5 = pow (fn / 28.843, 12.);
	r6 = 22.20 * pow (u, 1.92);
	r7 = 1.206 - 0.3144 * exp (-r1) * (1 - exp (-r2));
	r8 = 1 + 1.275 * (1 - exp (-0.004625 * r3 *
				pow (er, 1.674) * pow (fn / 18.365, 2.745)));
	r9 = 5.086 * r4 * r5 / (0.3838 + 0.386 * r4) *
		exp (-r6) / (1 + 1.2992 * r5) *
		pow (er - 1., 6.) / (1 + 10 * pow (er - 1., 6.));
	r10 = 0.00044 * pow (er, 2.136) + 0.0184;
	r11 = pow (fn / 19.47, 6.) / (1 + 0.0962 * pow (fn / 19.47, 6.));
	r12 = 1 / (1 + 0.00245 * sqr (u));
	r13 = 0.9408 * pow (ErEffFreq, r8) - 0.9603;
	r14 = (0.9408 - r9) * pow (ErEff, r8) - 0.9603;
	r15 = 0.707 * r10 * pow (fn / 12.3, 1.097);
	r16 = 1 + 0.0503 * sqr (er) * r11 * (1 - exp (- pow (u / 15., 6.)));
	*r17 = r7 * (1 - 1.1241 * r12 / r16 *
			exp (-0.026 * pow (fn, 1.15656) - r15));
	*ZlEffFreq = ZlEff * pow (r13 / r14, *r17);
}

/* The function calculates the conductor and dielectric losses of a
 *  single microstrip line. */
void analyseLoss (double W, double t, double er,
		double rho, double D, double tand,
		double ZlEff1, double ZlEff2,
		double ErEff,
		double frequency, int Model,
		double* ac, double* ad) {
	*ac = *ad = 0;

	// HAMMERSTAD and JENSEN
	if (Model == HAMMERSTAD) {
		double Rs, ds, l0, Kr, Ki;

		// conductor losses
		if (t != 0.0) {
			Rs = sqrt (M_PI * frequency * MU0 * rho); // skin resistance
			ds = rho / Rs;                            // skin depth
								  // valid for t > 3 * ds
			if (t < 3 * ds && frequency != 0) {
				fprintf (stderr,
						"WARNING: conductor loss calculation invalid for line "
						"height t (%g) < 3 * skin depth (%g)\n", t, 3 * ds);
			}
			// current distribution factor
			Ki = exp (-1.2 * pow ((ZlEff1 + ZlEff2) / 2 / Z0, 0.7));
			// D is RMS surface roughness
			Kr = 1 + M_2_PI * atan (1.4 * sqr (D / ds));
			*ac = Rs / (ZlEff1 * W) * Ki * Kr;
		}

		// dielectric losses
		l0 = C0 / frequency;
		*ad = M_PI * er / (er - 1) * (ErEff - 1) / sqrt (ErEff) * tand / l0;
	}
}

