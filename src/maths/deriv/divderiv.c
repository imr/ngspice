/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * DivDeriv computes the partial derivatives of the division
 * function where the arguments to the function are
 * functions of three variables p, q, and r.
 */

void
DivDeriv(Dderivs *new, Dderivs *old1, Dderivs *old2)
{

    Dderivs num, den;

    EqualDeriv(&num, old1);
    EqualDeriv(&den, old2);

    new->value = num.value / den.value;
    new->d1_p = (num.d1_p - num.value * den.d1_p / den.value) / den.value;
    new->d1_q = (num.d1_q - num.value * den.d1_q / den.value) / den.value;
    new->d1_r = (num.d1_r - num.value * den.d1_r / den.value) / den.value;
    new->d2_p2 = (num.d2_p2 - den.d1_p * new->d1_p
		  - new->value * den.d2_p2
		  + (den.d1_p * (new->value * den.d1_p - num.d1_p)
		     / den.value)) / den.value;
    new->d2_q2 = (num.d2_q2 - den.d1_q * new->d1_q
		  - new->value * den.d2_q2
		  + (den.d1_q * (new->value * den.d1_q - num.d1_q)
		     / den.value)) / den.value;
    new->d2_r2 = (num.d2_r2 - den.d1_r * new->d1_r - new->value *
		  den.d2_r2 + den.d1_r * (new->value * den.d1_r -
					  num.d1_r) / den.value) /
	den.value;
    new->d2_pq = (num.d2_pq - den.d1_q * new->d1_p - new->value *
		  den.d2_pq + den.d1_p * (new->value * den.d1_q -
					  num.d1_q) / den.value) /
	den.value; 
    new->d2_qr = (num.d2_qr - den.d1_r * new->d1_q - new->value *
		  den.d2_qr + den.d1_q * (new->value * den.d1_r -
					  num.d1_r) / den.value) /
	den.value;
    new->d2_pr = (num.d2_pr - den.d1_r * new->d1_p - new->value *
		  den.d2_pr + den.d1_p * (new->value * den.d1_r -
					  num.d1_r) / den.value) /
	den.value;
    new->d3_p3 = (-den.d1_p * new->d2_p2 + num.d3_p3 -den.d2_p2 *
		  new->d1_p - den.d1_p * new->d2_p2 - new->d1_p *
		  den.d2_p2 - new->value * den.d3_p3
		  + (den.d1_p * (new->d1_p * den.d1_p +
				 new->value * den.d2_p2 -
				 num.d2_p2) +
		     (new->value * den.d1_p - num.d1_p) *
		     (den.d2_p2 - den.d1_p * den.d1_p / den.value))
		  / den.value) / den.value;
    new->d3_q3 = (-den.d1_q * new->d2_q2 + num.d3_q3 -den.d2_q2 *
		  new->d1_q - den.d1_q * new->d2_q2 - new->d1_q *
		  den.d2_q2 - new->value * den.d3_q3 +
		  (den.d1_q * (new->d1_q * den.d1_q + new->value *
			       den.d2_q2 - num.d2_q2) +
		   (new->value * den.d1_q - num.d1_q) *
		   (den.d2_q2 - den.d1_q * den.d1_q / den.value)) /
		  den.value) / den.value;
    new->d3_r3 = (-den.d1_r * new->d2_r2 + num.d3_r3 -den.d2_r2 *
		  new->d1_r - den.d1_r * new->d2_r2 - new->d1_r *
		  den.d2_r2 - new->value * den.d3_r3 +
		  (den.d1_r * (new->d1_r * den.d1_r + new->value *
			       den.d2_r2 - num.d2_r2) +
		   (new->value * den.d1_r - num.d1_r) *
		   (den.d2_r2 - den.d1_r * den.d1_r / den.value)) /
		  den.value) / den.value;
    new->d3_p2r = (-den.d1_r * new->d2_p2 + num.d3_p2r -den.d2_pr *
		   new->d1_p - den.d1_p * new->d2_pr - new->d1_r *
		   den.d2_p2 - new->value * den.d3_p2r +
		   (den.d1_p * (new->d1_r * den.d1_p + new->value *
				den.d2_pr - num.d2_pr) +
		    (new->value * den.d1_p - num.d1_p) *
		    (den.d2_pr - den.d1_p * den.d1_r / den.value))
		   / den.value) / den.value;
    new->d3_p2q = (-den.d1_q * new->d2_p2 + num.d3_p2q -den.d2_pq *
		   new->d1_p - den.d1_p * new->d2_pq - new->d1_q *
		   den.d2_p2 - new->value * den.d3_p2q +
		   (den.d1_p * (new->d1_q * den.d1_p + new->value *
				den.d2_pq - num.d2_pq) +
		    (new->value * den.d1_p - num.d1_p) *
		    (den.d2_pq - den.d1_p * den.d1_q / den.value)) /
		   den.value) / den.value;
    new->d3_q2r = (-den.d1_r * new->d2_q2 + num.d3_q2r -den.d2_qr *
		   new->d1_q - den.d1_q * new->d2_qr - new->d1_r *
		   den.d2_q2 - new->value * den.d3_q2r +
		   (den.d1_q * (new->d1_r * den.d1_q + new->value *
				den.d2_qr - num.d2_qr) +
		    (new->value * den.d1_q - num.d1_q) *
		    (den.d2_qr - den.d1_q * den.d1_r / den.value)) /
		   den.value) / den.value;
    new->d3_pq2 = (-den.d1_p * new->d2_q2 + num.d3_pq2 -den.d2_pq *
		   new->d1_q - den.d1_q * new->d2_pq - new->d1_p *
		   den.d2_q2 - new->value * den.d3_pq2 +
		   (den.d1_q * (new->d1_p * den.d1_q + new->value *
				den.d2_pq - num.d2_pq) +
		    (new->value * den.d1_q - num.d1_q) *
		    (den.d2_pq - den.d1_q * den.d1_p / den.value)) /
		   den.value) / den.value;
    new->d3_pr2 = (-den.d1_p * new->d2_r2 + num.d3_pr2 -den.d2_pr *
		   new->d1_r - den.d1_r * new->d2_pr - new->d1_p *
		   den.d2_r2 - new->value * den.d3_pr2 +
		   (den.d1_r * (new->d1_p * den.d1_r + new->value *
				den.d2_pr - num.d2_pr) +
		    (new->value * den.d1_r - num.d1_r) *
		    (den.d2_pr - den.d1_r * den.d1_p / den.value)) /
		   den.value) / den.value;
    new->d3_qr2 = (-den.d1_q * new->d2_r2 + num.d3_qr2 -den.d2_qr *
		   new->d1_r - den.d1_r * new->d2_qr - new->d1_q *
		   den.d2_r2 - new->value * den.d3_qr2 +
		   (den.d1_r * (new->d1_q * den.d1_r + new->value *
				den.d2_qr - num.d2_qr) +
		    (new->value * den.d1_r - num.d1_r) *
		    (den.d2_qr - den.d1_r * den.d1_q / den.value)) /
		   den.value) / den.value;
    new->d3_pqr = (-den.d1_r * new->d2_pq + num.d3_pqr -den.d2_qr *
		   new->d1_p - den.d1_q * new->d2_pr - new->d1_r *
		   den.d2_pq - new->value * den.d3_pqr +
		   (den.d1_p * (new->d1_r * den.d1_q + new->value *
				den.d2_qr - num.d2_qr) +
		    (new->value * den.d1_q - num.d1_q) *
		    (den.d2_pr - den.d1_p * den.d1_r / den.value)) /
		   den.value) / den.value;
}
