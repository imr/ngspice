/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * CubeDeriv computes the partial derivatives of the cube
 * function where the argument to the function is itself a
 * function of three variables p, q, and r.
 */

void
CubeDeriv(Dderivs *new, Dderivs *old)
{
    Dderivs temp;

    EqualDeriv(&temp, old);

    new->value = temp.value * temp.value * temp.value;
    new->d1_p = 3*temp.value*temp.value*temp.d1_p;
    new->d1_q = 3*temp.value*temp.value*temp.d1_q;
    new->d1_r = 3*temp.value*temp.value*temp.d1_r;
    new->d2_p2 = 3*(2*temp.value*temp.d1_p*temp.d1_p
		    + temp.value*temp.value*temp.d2_p2);
    new->d2_q2 = 3*(2*temp.value*temp.d1_q*temp.d1_q
		    + temp.value*temp.value*temp.d2_q2);
    new->d2_r2 = 3*(2*temp.value*temp.d1_r*temp.d1_r
		    + temp.value*temp.value*temp.d2_r2);
    new->d2_pq = 3*(2*temp.value*temp.d1_p*temp.d1_q
		    + temp.value*temp.value*temp.d2_pq);
    new->d2_qr = 3*(2*temp.value*temp.d1_q*temp.d1_r
		    + temp.value*temp.value*temp.d2_qr);
    new->d2_pr = 3*(2*temp.value*temp.d1_p*temp.d1_r
		    + temp.value*temp.value*temp.d2_pr);
    new->d3_p3 = 3*(2*(temp.d1_p*temp.d1_p*temp.d1_p
		       + temp.value*(temp.d2_p2*temp.d1_p
				     + temp.d2_p2*temp.d1_p
				     + temp.d2_p2*temp.d1_p))
		    + temp.value*temp.value*temp.d3_p3);
    new->d3_q3 = 3*(2*(temp.d1_q*temp.d1_q*temp.d1_q
		       + temp.value*(temp.d2_q2*temp.d1_q
				     + temp.d2_q2*temp.d1_q
				     + temp.d2_q2*temp.d1_q))
		    + temp.value*temp.value*temp.d3_q3);
    new->d3_r3 = 3*(2*(temp.d1_r*temp.d1_r*temp.d1_r
		       + temp.value*(temp.d2_r2*temp.d1_r
				     + temp.d2_r2*temp.d1_r
				     + temp.d2_r2*temp.d1_r))
		    + temp.value*temp.value*temp.d3_r3);
    new->d3_p2r = 3*(2*(temp.d1_p*temp.d1_p*temp.d1_r
			+ temp.value*(temp.d2_p2*temp.d1_r
				      + temp.d2_pr*temp.d1_p
				      + temp.d2_pr*temp.d1_p))
		     + temp.value*temp.value*temp.d3_p2r);
    new->d3_p2q = 3*(2*(temp.d1_p*temp.d1_p*temp.d1_q
			+ temp.value*(temp.d2_p2*temp.d1_q
				      + temp.d2_pq*temp.d1_p
				      + temp.d2_pq*temp.d1_p))
		     + temp.value*temp.value*temp.d3_p2q);
    new->d3_q2r = 3*(2*(temp.d1_q*temp.d1_q*temp.d1_r
			+ temp.value*(temp.d2_q2*temp.d1_r
				      + temp.d2_qr*temp.d1_q
				      + temp.d2_qr*temp.d1_q))
		     + temp.value*temp.value*temp.d3_q2r);
    new->d3_pq2 = 3*(2*(temp.d1_q*temp.d1_q*temp.d1_p
			+ temp.value*(temp.d2_q2*temp.d1_p
				      + temp.d2_pq*temp.d1_q
				      + temp.d2_pq*temp.d1_q))
		     + temp.value*temp.value*temp.d3_pq2);
    new->d3_pr2 = 3*(2*(temp.d1_r*temp.d1_r*temp.d1_p
			+ temp.value*(temp.d2_r2*temp.d1_p
				      + temp.d2_pr*temp.d1_r
				      + temp.d2_pr*temp.d1_r))
		     + temp.value*temp.value*temp.d3_pr2);
    new->d3_qr2 = 3*(2*(temp.d1_r*temp.d1_r*temp.d1_q
			+ temp.value*(temp.d2_r2*temp.d1_q
				      + temp.d2_qr*temp.d1_r
				      + temp.d2_qr*temp.d1_r))
		     + temp.value*temp.value*temp.d3_qr2);
    new->d3_pqr = 3*(2*(temp.d1_p*temp.d1_q*temp.d1_r
			+ temp.value*(temp.d2_pq*temp.d1_r
				      + temp.d2_qr*temp.d1_p
				      + temp.d2_pr*temp.d1_q))
		     + temp.value*temp.value*temp.d3_pqr);
}
