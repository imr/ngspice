/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * AtanDeriv computes the partial derivatives of the arctangent
 * function where the argument to the atan function is itself a
 * function of three variables p, q, and r.
 */

void AtanDeriv(Dderivs *new, Dderivs *old)
{

    Dderivs temp;

    EqualDeriv(&temp, old);

    new->value = atan( temp.value);
    new->d1_p = temp.d1_p / (1 + temp.value * temp.value);
    new->d1_q = temp.d1_q / (1 + temp.value * temp.value);
    new->d1_r = temp.d1_r / (1 + temp.value * temp.value);
    new->d2_p2 = temp.d2_p2 / (1 + temp.value * temp.value) - 2 *
	temp.value * new->d1_p * new->d1_p;
    new->d2_q2 = temp.d2_q2 / (1 + temp.value * temp.value) - 2 *
	temp.value * new->d1_q * new->d1_q;
    new->d2_r2 = temp.d2_r2 / (1 + temp.value * temp.value) - 2 *
	temp.value * new->d1_r * new->d1_r;
    new->d2_pq = temp.d2_pq / (1 + temp.value * temp.value) - 2 *
	temp.value * new->d1_p * new->d1_q;
    new->d2_qr = temp.d2_qr / (1 + temp.value * temp.value) - 2 *
	temp.value * new->d1_q * new->d1_r;
    new->d2_pr = temp.d2_pr / (1 + temp.value * temp.value) - 2 *
	temp.value * new->d1_p * new->d1_r;
    new->d3_p3 = (temp.d3_p3 - temp.d2_p2 * new->d1_p * 2 *
		  temp.value) / (1 + temp.value * temp.value) - 2 *
	(new->d1_p * new->d1_p * temp.d1_p + temp.value *
	 ( new->d2_p2 * new->d1_p + new->d2_p2 * new->d1_p));

    new->d3_q3 = (temp.d3_q3 - temp.d2_q2 * new->d1_q * 2 *
		  temp.value) / (1 + temp.value * temp.value) - 2 *
	(new->d1_q * new->d1_q * temp.d1_q + temp.value *
	 ( new->d2_q2 * new->d1_q + new->d2_q2 * new->d1_q));
    new->d3_r3 = (temp.d3_r3 - temp.d2_r2 * new->d1_r * 2 *
		  temp.value) / (1 + temp.value * temp.value) - 2 *
	(new->d1_r * new->d1_r * temp.d1_r + temp.value *
	 ( new->d2_r2 * new->d1_r + new->d2_r2 * new->d1_r));
    new->d3_p2r = (temp.d3_p2r - temp.d2_p2 * new->d1_r * 2 *
		   temp.value) / (1 + temp.value * temp.value) - 2 *
	(new->d1_p * new->d1_p * temp.d1_r + temp.value *
	 ( new->d2_pr * new->d1_p + new->d2_pr * new->d1_p));
    new->d3_p2q = (temp.d3_p2q - temp.d2_p2 * new->d1_q * 2 *
		   temp.value) / (1 + temp.value * temp.value) - 2 *
	(new->d1_p * new->d1_p * temp.d1_q + temp.value *
	 ( new->d2_pq * new->d1_p + new->d2_pq * new->d1_p));
    new->d3_q2r = (temp.d3_q2r - temp.d2_q2 * new->d1_r * 2 *
		   temp.value) / (1 + temp.value * temp.value) - 2 *
	(new->d1_q * new->d1_q * temp.d1_r + temp.value *
	 ( new->d2_qr * new->d1_q + new->d2_qr * new->d1_q));
    new->d3_pq2 = (temp.d3_pq2 - temp.d2_q2 * new->d1_p * 2 *
		   temp.value) / (1 + temp.value * temp.value) - 2 *
	(new->d1_q * new->d1_q * temp.d1_p + temp.value *
	 ( new->d2_pq * new->d1_q + new->d2_pq * new->d1_q));
    new->d3_pr2 = (temp.d3_pr2 - temp.d2_r2 * new->d1_p * 2 *
		   temp.value) / (1 + temp.value * temp.value) - 2 *
	(new->d1_r * new->d1_r * temp.d1_p + temp.value *
	 ( new->d2_pr * new->d1_r + new->d2_pr * new->d1_r));
    new->d3_qr2 = (temp.d3_qr2 - temp.d2_r2 * new->d1_q * 2 *
		   temp.value) / (1 + temp.value * temp.value) - 2 *
	(new->d1_r * new->d1_r * temp.d1_q + temp.value *
	 ( new->d2_qr * new->d1_r + new->d2_qr * new->d1_r));
    new->d3_pqr = (temp.d3_pqr - temp.d2_pq * new->d1_r * 2 *
		   temp.value) / (1 + temp.value * temp.value) - 2 *
	(new->d1_p * new->d1_q * temp.d1_r + temp.value *
	 ( new->d2_pr * new->d1_q + new->d2_qr * new->d1_p));
}
