/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/pzdefs.h"
#include "ngspice/complex.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"


#ifdef PZDEBUG
#define DEBUG(N)  if (Debug >= (unsigned) (N))
static unsigned int Debug = 3;
#endif

extern void zaddeq(double *a, int *amag, double x, int xmag, double y, int ymag);

extern	int CKTpzTrapped;
double	NIpzK;
int	NIpzK_mag;

int
NIpzSym(PZtrial **set, PZtrial *new)
{
    return NIpzSym2(set, new);
}

int
NIpzComplex(PZtrial **set, PZtrial *new)
{
    return NIpzSym2(set, new);
}

int
NIpzMuller(PZtrial **set, PZtrial *newtry)
{
    SPcomplex	A, B, C, D, E;
    SPcomplex	h0, h1;
    SPcomplex	lambda_i, delta_i;
    double	scale[3];
    double	q;
    int		mag[3], magx, min;
    int		i, j , total;

    min = -999999;
    j = 0;
    total = 0;
    for (i = 0; i < 3; i++) {
	if (set[i]->f_def.real != 0.0 || set[i]->f_def.imag != 0.0) {
	    if (min < set[i]->mag_def - 50)
		min = set[i]->mag_def - 50;
	    total += set[i]->mag_def;
	    j += 1;
	}
    }

    magx = total / j;
    if (magx < min)
	magx = min;

#ifdef PZDEBUG
    DEBUG(2) fprintf(stderr, "Base scale: %d / %d (0: %d, 1: %d, 2: %d)\n",
	magx, min, set[0]->mag_def, set[1]->mag_def, set[2]->mag_def);
#endif

    for (i = 0; i < 3; i++) {
	mag[i] = set[i]->mag_def - magx;
	scale[i] = 1.0;
	while (mag[i] > 0) {
		scale[i] *= 2.0;
		mag[i] -= 1;
	}
	if (mag[i] < -90)
	    scale[i] = 0.0;
	else {
	    while (mag[i] < 0) {
		scale[i] /= 2.0;
		mag[i] += 1;
	    }
	}
    }

    C_SUBEQ(h0,set[0]->s,set[1]->s);
    C_SUBEQ(h1,set[1]->s,set[2]->s);
    C_DIVEQ(lambda_i,h0,h1);

    /* Quadratic interpolation (Muller's method) */

    C_EQ(delta_i,lambda_i);
    delta_i.real += 1.0;

    /* Quadratic coefficients A, B, C (Note: reciprocal form of eqn) */

    /* A = lambda_i * (f[i-2] * lambda_i - f[i-1] * delta_i + f[i]) */
    C_MULEQ(A,scale[2] * set[2]->f_def,lambda_i);
    C_MULEQ(C,scale[1] * set[1]->f_def,delta_i);
    C_SUB(A,C);
    C_ADD(A,scale[0] * set[0]->f_def);
    C_MUL(A,lambda_i);

    /* B = f[i-2] * lambda_i * lambda_1 - f[i-1] * delta_i * delta_i
	+ f[i] * (lambda_i + delta_i) */
    C_MULEQ(B,lambda_i,lambda_i);
    C_MUL(B,scale[2] * set[2]->f_def);
    C_MULEQ(C,delta_i,delta_i);
    C_MUL(C,scale[1] * set[1]->f_def);
    C_SUB(B,C);
    C_ADDEQ(C,lambda_i,delta_i);
    C_MUL(C,scale[0] * set[0]->f_def);
    C_ADD(B,C);

    /* C = delta_i * f[i] */
    C_MULEQ(C,delta_i,scale[0] * set[0]->f_def);

    while (fabs(A.real) > 1.0 || fabs(A.imag) > 1.0
	|| fabs(B.real) > 1.0 || fabs(B.imag) > 1.0
	|| fabs(C.real) > 1.0 || fabs(C.imag) > 1.0) {
	A.real /= 2.0;
	B.real /= 2.0;
	C.real /= 2.0;
	A.imag /= 2.0;
	B.imag /= 2.0;
	C.imag /= 2.0;
    }

    /* discriminate = B * B - 4 * A * C */
    C_MULEQ(D,B,B);
    C_MULEQ(E,4.0 * A,C);
    C_SUB(D,E);

#ifdef PZDEBUG
    DEBUG(2) fprintf(stderr, "  Discr: (%g,%g)\n",D.real, D.imag);
#endif
    C_SQRT(D);
#ifdef PZDEBUG
    DEBUG(1) fprintf(stderr, "  sqrtDiscr: (%g,%g)\n",D.real, D.imag);
#endif

#ifndef notdef
    /* Maximize denominator */

#ifdef PZDEBUG
    DEBUG(1) fprintf(stderr, "  B: (%g,%g)\n",B.real, B.imag);
#endif
    /* Dot product */
    q = B.real * D.real + B.imag * D.imag;
    if (q > 0.0) {
#ifdef PZDEBUG
	DEBUG(1) fprintf(stderr, "       add\n");
#endif
	C_ADD(B,D);
    } else {
#ifdef PZDEBUG
	DEBUG(1) fprintf(stderr, "       sub\n");
#endif
	C_SUB(B,D);
    }

#else
    /* For trapped zeros, the step should always be positive */
    if (C.real >= 0.0) {
	if (B.real < D.real) {
	    C_SUB(B,D);
	} else {
	    C_ADD(B,D);
	}
    } else {
	if (B.real > D.real) {
	    C_SUB(B,D);
	} else {
	    C_ADD(B,D);
	}
    }
#endif

#ifdef PZDEBUG
    DEBUG(1) fprintf(stderr, "  C: (%g,%g)\n", C.real, C.imag);
#endif
    C_DIV(C,-0.5 * B);

    newtry->next = NULL;

#ifdef PZDEBUG
    DEBUG(1) fprintf(stderr, "  lambda: (%g,%g)\n",C.real, C.imag);
#endif
    C_MULEQ(newtry->s,h0,C);

#ifdef PZDEBUG
    DEBUG(1) fprintf(stderr, "  h: (%g,%g)\n", newtry->s.real, newtry->s.imag);
#endif

    C_ADD(newtry->s,set[0]->s);

#ifdef PZDEBUG
    DEBUG(1) fprintf(stderr, "New try: (%g,%g)\n",
	newtry->s.real, newtry->s.imag);
#endif

    return OK;

}


int
NIpzSym2(PZtrial **set, PZtrial *new)
{
    double	a, b, c, x0;
    double	dx0, dx1, d2x, diff;
    double	disc;
    int		a_mag, b_mag, c_mag;
    int		tmag;
    int		error;
    int		disc_mag;
    int		new_mag = 0;

    error = OK;

    /*
    NIpzK = 0.0;
    NIpzK_mag = 0;
    */

    /* Solve for X = the distance from set[1], where X > 0 => root < set[1] */
    dx0 = set[1]->s.real - set[0]->s.real;
    dx1 = set[2]->s.real - set[1]->s.real;

    x0 = (set[0]->s.real + set[1]->s.real) / 2.0;
  /*  x1 = (set[1]->s.real + set[2]->s.real) / 2.0;
      d2x = x1 - x0; */
    d2x = (set[2]->s.real - set[0]->s.real) / 2.0;

    zaddeq(&a, &a_mag, set[1]->f_def.real, set[1]->mag_def,
	-set[0]->f_def.real, set[0]->mag_def);

    tmag = 0;
    R_NORM(dx0,tmag);
    a /= dx0;
    a_mag -= tmag;
    R_NORM(a,a_mag);

    zaddeq(&b, &b_mag, set[2]->f_def.real, set[2]->mag_def,
	-set[1]->f_def.real, set[1]->mag_def);

    tmag = 0;
    R_NORM(dx1,tmag);
    b /= dx1;
    b_mag -= tmag;
    R_NORM(b,b_mag);

    zaddeq(&c, &c_mag, b, b_mag, -a, a_mag);

    tmag = 0;
    R_NORM(d2x,tmag);
    c /= d2x;	/* = f'' */
    c_mag -= tmag;
    R_NORM(c,c_mag);

    if (c == 0.0 || ((a == 0.0 || c_mag < a_mag - 40)
	    && (b = 0.0 ||c_mag < b_mag - 40))) {
	/*fprintf(stderr, "\t- linear (%g, %d)\n", c, c_mag);*/
	if (a == 0.0) {
	    a = b;
	    a_mag = b_mag;
	}
	if (a != 0.0) {
	    new->s.real = - set[1]->f_def.real / a;
	    a_mag -= set[1]->mag_def;
	    while (a_mag > 0) {
		new->s.real /= 2.0;
		a_mag -= 1;
	    }
	    while (a_mag < 0) {
		new->s.real *= 2.0;
		a_mag += 1;
	    }
	    new->s.real += set[1]->s.real;
	} else
	    new->s.real = set[1]->s.real;
    } else {

	/* Quadratic power series about set[1]->s.real			*/
	/* c : d2f/dx2 @ s1 (assumed constant for all s), or "2A"	*/

	/* a : (df/dx) / (d2f/dx2) @ s1, or "B/2A"			*/
	a /= c;
	R_NORM(a,a_mag);
	a_mag -= c_mag;

	diff = set[1]->s.real - x0;
	tmag = 0;
	R_NORM(diff,tmag);

	zaddeq(&a, &a_mag, a, a_mag, diff, tmag);

	/* b : f(s1) / (1/2 d2f/ds2), or "C / A"			*/
	b = 2.0 * set[1]->f_def.real / c;
	b_mag = set[1]->mag_def - c_mag;
	R_NORM(b,b_mag);

	disc = a * a;
	disc_mag = 2 * a_mag;

	/* disc = a^2  - b  :: (B/2A)^2 - C/A */
	zaddeq(&disc, &disc_mag, disc, disc_mag, - b, b_mag);

	if (disc < 0.0) {
	    /* Look for minima instead, but save radical for later work */
	    disc *= -1;
	    new_mag = 1;
	}

	if (disc_mag % 2 == 0)
	    disc = sqrt(disc);
	else {
	    disc = sqrt(2.0 * disc);
	    disc_mag -= 1;
	}
	disc_mag /= 2;

	if (new_mag != 0) {
	    if (NIpzK == 0.0) {
		NIpzK = disc;
		NIpzK_mag = disc_mag;
#ifdef PZDEBUG
		DEBUG(1) fprintf(stderr, "New NIpzK: %g*2^%d\n",
		    NIpzK, NIpzK_mag);
#endif
	    } else {
#ifdef PZDEBUG
		DEBUG(1) fprintf(stderr,
		    "Ignore NIpzK: %g*2^%d for previous value of %g*2^%d\n",
		    disc, disc_mag,
		    NIpzK, NIpzK_mag);
#endif
	    }
	    disc = 0.0;
	    disc_mag = 0;
	}

	/* NOTE: c & b get reused here-after */

	if (a * disc >= 0.0) {
	    zaddeq(&c, &c_mag, a, a_mag, disc, disc_mag);
	} else {
	    zaddeq(&c, &c_mag, a, a_mag, -disc, disc_mag);
	}

	/* second root = C / (first root) */
	if (c != 0.0) {
	    b /= c;
	    b_mag -= c_mag;
	} else {
	    /* special case */
	    b = 0.0;
	    b_mag = 0;
	}

	zaddeq(&b, &b_mag, set[1]->s.real, 0, -b, b_mag);
	zaddeq(&c, &c_mag, set[1]->s.real, 0, -c, c_mag);

	while (b_mag > 0) {
	    b *= 2.0;
	    b_mag -= 1;
	}
	while (b_mag < 0) {
	    b /= 2.0;
	    b_mag += 1;
	}

	while (c_mag > 0) {
	    c *= 2.0;
	    c_mag -= 1;
	}
	while (c_mag < 0) {
	    c /= 2.0;
	    c_mag += 1;
	}

#ifdef PZDEBUG
	DEBUG(1) fprintf(stderr, "@@@ (%.15g) -vs- (%.15g)\n", b, c);
#endif
	/* XXXX */
	if (b < set[0]->s.real || b > set[2]->s.real) {
	    /* b not in range */
	    if (c < set[0]->s.real || c > set[2]->s.real) {
#ifdef PZDEBUG
		DEBUG(1) fprintf(stderr, "@@@ both are junk\n");
#endif
		if (CKTpzTrapped == 1)
		    new->s.real = (set[0]->s.real + set[1]->s.real) / 2.0;
		else if (CKTpzTrapped == 2)
		    new->s.real = (set[1]->s.real + set[2]->s.real) / 2.0;
		else if (CKTpzTrapped == 3) {
		    if (fabs(set[1]->s.real - c) < fabs(set[1]->s.real - b)) {
#ifdef PZDEBUG
			DEBUG(1) fprintf(stderr, "@@@ mix w/second (c)\n");
#endif
			new->s.real = (set[1]->s.real + c) / 2.0;
		    } else {
#ifdef PZDEBUG
			DEBUG(1) fprintf(stderr, "@@@ mix w/first (b)\n");
#endif
			new->s.real = (set[1]->s.real + b) / 2.0;
		    }
		} else
		    MERROR(E_PANIC,"Lost numerical stability");
	    } else {
#ifdef PZDEBUG
		DEBUG(1) fprintf(stderr, "@@@ take second (c)\n");
#endif
		new->s.real = c;
	    }
	} else {
	    /* b in range */
	    if (c < set[0]->s.real || c > set[2]->s.real) {
#ifdef PZDEBUG
		DEBUG(1) fprintf(stderr, "@@@ take first (b)\n");
#endif
		new->s.real = b;
	    } else {
		/* Both in range -- take the smallest mag */
		if (a > 0.0) {
#ifdef PZDEBUG
		    DEBUG(1) fprintf(stderr, "@@@ push -- first (b)\n");
#endif
		    new->s.real = b;
		} else {
#ifdef PZDEBUG
		    DEBUG(1) fprintf(stderr, "@@@ push -- first (b)\n");
#endif
		    new->s.real = c;
		}
	    }
	}

    }

    new->s.imag = 0.0;

    return error;
}
