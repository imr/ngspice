/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * SqrtDeriv computes the partial derivatives of the sqrt
 * function where the argument to the function is itself a
 * function of three variables p, q, and r.
 */

void
SqrtDeriv(Dderivs *new, Dderivs *old)
{

    Dderivs temp;

    EqualDeriv(&temp, old);
    new->value = sqrt(temp.value);
    if (temp.value == 0.0)
    {
	new->d1_p = 0.0;
	new->d1_q = 0.0;
	new->d1_r = 0.0;
	new->d2_p2 = 0.0;
	new->d2_q2 = 0.0;
	new->d2_r2 = 0.0;
	new->d2_pq = 0.0;
	new->d2_qr = 0.0;
	new->d2_pr = 0.0;
	new->d3_p3 = 0.0;
	new->d3_q3 = 0.0;
	new->d3_r3 = 0.0;
	new->d3_p2r = 0.0;
	new->d3_p2q = 0.0;
	new->d3_q2r = 0.0;
	new->d3_pq2 = 0.0;
	new->d3_pr2 = 0.0;
	new->d3_qr2 = 0.0;
	new->d3_pqr = 0.0;
    } else {
	new->d1_p = 0.5*temp.d1_p/new->value;
	new->d1_q = 0.5*temp.d1_q/new->value;
	new->d1_r = 0.5*temp.d1_r/new->value;
	new->d2_p2 = 0.5/new->value * (temp.d2_p2 - 0.5 * temp.d1_p *
				       temp.d1_p/ temp.value);
	new->d2_q2 = 0.5/new->value*(temp.d2_q2 -0.5 * temp.d1_q *
				     temp.d1_q/ temp.value);
	new->d2_r2 = 0.5/new->value*(temp.d2_r2 -0.5 * temp.d1_r *
				     temp.d1_r/ temp.value);
	new->d2_pq = 0.5/new->value*(temp.d2_pq -0.5 * temp.d1_p *
				     temp.d1_q/ temp.value);
	new->d2_qr = 0.5/new->value*(temp.d2_qr -0.5 * temp.d1_q *
				     temp.d1_r/ temp.value);
	new->d2_pr = 0.5/new->value*(temp.d2_pr -0.5 * temp.d1_p *
				     temp.d1_r/ temp.value);
	new->d3_p3 = 0.5 *
	    (temp.d3_p3 / new->value - 0.5 / (temp.value*new->value) *
	     (-1.5 / temp.value * temp.d1_p * temp.d1_p * temp.d1_p
	      + temp.d1_p*temp.d2_p2
	      + temp.d1_p*temp.d2_p2
	      + temp.d1_p*temp.d2_p2));
	new->d3_q3 = 0.5 *
	    (temp.d3_q3 / new->value - 0.5 / (temp.value*new->value) *
	     (-1.5 / temp.value * temp.d1_q * temp.d1_q * temp.d1_q
	      + temp.d1_q*temp.d2_q2
	      + temp.d1_q*temp.d2_q2
	      + temp.d1_q* temp.d2_q2));
	new->d3_r3 = 0.5 *
	    (temp.d3_r3 / new->value - 0.5 / (temp.value*new->value) *
	     (-1.5 / temp.value * temp.d1_r * temp.d1_r * temp.d1_r
	      + temp.d1_r*temp.d2_r2
	      + temp.d1_r*temp.d2_r2 
	      + temp.d1_r* temp.d2_r2));
	new->d3_p2r = 0.5 *
	    (temp.d3_p2r / new->value - 0.5 / (temp.value*new->value) *
	     (-1.5 / temp.value * temp.d1_p * temp.d1_p * temp.d1_r
	      + temp.d1_p * temp.d2_pr
	      + temp.d1_p * temp.d2_pr
	      + temp.d1_r * temp.d2_p2));
	new->d3_p2q = 0.5 *
	    (temp.d3_p2q / new->value - 0.5 / (temp.value * new->value) *
	     (-1.5/temp.value*temp.d1_p*temp.d1_p*temp.d1_q
	      + temp.d1_p * temp.d2_pq
	      + temp.d1_p * temp.d2_pq
	      + temp.d1_q * temp.d2_p2));
	new->d3_q2r = 0.5 *
	    (temp.d3_q2r / new->value - 0.5 / (temp.value * new->value) *
	     (-1.5 / temp.value * temp.d1_q * temp.d1_q * temp.d1_r
	      + temp.d1_q*temp.d2_qr
	      + temp.d1_q*temp.d2_qr
	      + temp.d1_r* temp.d2_q2));
	new->d3_pq2 = 0.5 *
	    (temp.d3_pq2 / new->value - 0.5 / (temp.value * new->value) *
	     (-1.5 / temp.value * temp.d1_q * temp.d1_q * temp.d1_p
	      + temp.d1_q*temp.d2_pq
	      + temp.d1_q*temp.d2_pq
	      + temp.d1_p* temp.d2_q2));
	new->d3_pr2 = 0.5 *
	    (temp.d3_pr2 / new->value - 0.5 / (temp.value * new->value) *
	     (-1.5/temp.value * temp.d1_r * temp.d1_r * temp.d1_p
	      + temp.d1_r*temp.d2_pr
	      + temp.d1_r*temp.d2_pr
	      + temp.d1_p* temp.d2_r2));
	new->d3_qr2 = 0.5 *
	    (temp.d3_qr2 / new->value - 0.5 / (temp.value * new->value) *
	     (-1.5/temp.value * temp.d1_r * temp.d1_r * temp.d1_q
	      + temp.d1_r*temp.d2_qr
	      + temp.d1_r*temp.d2_qr
	      + temp.d1_q* temp.d2_r2));
	new->d3_pqr = 0.5 *
	    (temp.d3_pqr / new->value - 0.5 / (temp.value * new->value) *
	     (-1.5/temp.value * temp.d1_p * temp.d1_q * temp.d1_r
	      + temp.d1_p*temp.d2_qr
	      + temp.d1_q*temp.d2_pr
	      + temp.d1_r* temp.d2_pq));
    }
}
