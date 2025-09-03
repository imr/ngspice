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

/* The function calculates the quasi-static dielectric constants and
   characteristic impedances for the even and odd mode based upon the
   given line and substrate properties for parallel coupled microstrip
   lines. */
void cpmslineAnalyseQuasiStatic (double W, double h, double s,
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
void cpmslineAnalyseDispersion (double W, double h, double s,
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
    double tn, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21;
    q11 = 0.893 * (1 - 0.3 / (1 + 0.7 * (er - 1)));
    tn = pow (fn / 20, 4.91);
    q12 = 2.121 * tn / (1 + q11 * tn) * exp (-2.87 * g) * pow (g, 0.902);
    q13 = 1 + 0.038 * pow (er / 8, 5.1);
    tn = quadr (er / 15);
    q14 = 1 + 1.203 * tn / (1 + tn);
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
    tn = pow (u, 2.5);
    q21 = fabs (1 - 42.54 * pow (g, 0.133) * exp (-0.812 * g) * tn /
		(1 + 0.033 * tn));

    double re, qe, pe, de, Ce, q0, ZlFreq, ErEffFreq;
    Kirschning_er (u, fn, er, ErEffe, &ErEffFreq);
    Kirschning_zl (u, fn, er, ErEffe, ErEffFreq, Zle, &q0, &ZlFreq);
    re = pow (fn / 28.843, 12.);
    qe = 0.016 + pow (0.0514 * er * q21, 4.524);
    pe = 4.766 * exp (-3.228 * pow (u, 0.641));
    tn = pow (er - 1, 6.);
    de = 5.086 * qe * re / (0.3838 + 0.386 * qe) *
      exp (-22.2 * pow (u, 1.92)) / (1 + 1.2992 * re) * tn / (1 + 10 * tn);
    Ce = 1 + 1.275 * (1 - exp (-0.004625 * pe * pow (er, 1.674) *
	 pow (fn / 18.365, 2.745))) - q12 + q16 - q17 + q18 + q20;
    *ZleFreq = Zle * pow ((0.9408 * pow (ErEffFreq, Ce) - 0.9603) /
			 ((0.9408 - de) * pow (ErEffe, Ce) - 0.9603), q0);

    // dispersion of odd characteristic impedance
    double q22, q23, q24, q25, q26, q27, q28, q29;
    Kirschning_er (u, fn, er, ErEffo, &ErEffFreq);
    Kirschning_zl (u, fn, er, ErEffo, ErEffFreq, Zlo, &q0, &ZlFreq);
    q29 = 15.16 / (1 + 0.196 * sqr (er - 1));
    tn = sqr (er - 1);
    q25 = 0.3 * sqr (fn) / (10 + sqr (fn)) * (1 + 2.333 * tn / (5 + tn));
    tn = pow ((er - 1) / 13, 12.);
    q26 = 30 - 22.2 * tn / (1 + 3 * tn) - q29;
    tn = pow (er - 1, 1.5);
    q27 = 0.4 * pow (g, 0.84) * (1 + 2.5 * tn / (5 + tn));
    tn = pow (er - 1, 3.);
    q28 = 0.149 * tn / (94.5 + 0.038 * tn);
    q22 = 0.925 * pow (fn / q26, 1.536) / (1 + 0.3 * pow (fn / 30, 1.536));
    q23 = 1 + 0.005 * fn * q27 / (1 + 0.812 * pow (fn / 15, 1.9)) /
      (1 + 0.025 * sqr (u));
    tn = pow (u, 0.894);
    q24 = 2.506 * q28 * tn / (3.575 + tn) *
      pow ((1 + 1.3 * u) * fn / 99.25, 4.29);
    *ZloFreq = ZlFreq + (Zlo * pow (*ErEffoFreq / ErEffo, q22) - ZlFreq * q23) /
      (1 + q24 + pow (0.46 * g, 2.2) * q25);

  }
}



