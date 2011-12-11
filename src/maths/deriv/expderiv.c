/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * ExpDeriv computes the partial derivatives of the exponential
 * function where the argument to the function is itself a
 * function of three variables p, q, and r.
 */

void
ExpDeriv(Dderivs *new, Dderivs *old)
{

    Dderivs temp;

    EqualDeriv(&temp, old);
    new->value = exp(temp.value);
    new->d1_p = new->value*temp.d1_p;
    new->d1_q = new->value*temp.d1_q;
    new->d1_r = new->value*temp.d1_r;
    new->d2_p2 = new->value*temp.d2_p2 + temp.d1_p*new->d1_p;
    new->d2_q2 = new->value*temp.d2_q2 + temp.d1_q*new->d1_q;
    new->d2_r2 = new->value*temp.d2_r2 + temp.d1_r*new->d1_r;
    new->d2_pq = new->value*temp.d2_pq + temp.d1_p*new->d1_q;
    new->d2_qr = new->value*temp.d2_qr + temp.d1_q*new->d1_r;
    new->d2_pr = new->value*temp.d2_pr + temp.d1_p*new->d1_r;
    new->d3_p3 = new->value*temp.d3_p3 + temp.d2_p2*new->d1_p
	+ temp.d2_p2*new->d1_p
	+ new->d2_p2*temp.d1_p;
    new->d3_q3 = new->value*temp.d3_q3 + temp.d2_q2*new->d1_q
	+ temp.d2_q2*new->d1_q
	+ new->d2_q2*temp.d1_q;
    new->d3_r3 = new->value*temp.d3_r3 + temp.d2_r2*new->d1_r
	+ temp.d2_r2*new->d1_r
	+ new->d2_r2*temp.d1_r;
    new->d3_p2r = new->value*temp.d3_p2r + temp.d2_p2*new->d1_r
	+ temp.d2_pr*new->d1_p
	+ new->d2_pr*temp.d1_p;
    new->d3_p2q = new->value*temp.d3_p2q + temp.d2_p2*new->d1_q
	+ temp.d2_pq*new->d1_p
	+ new->d2_pq*temp.d1_p;
    new->d3_q2r = new->value*temp.d3_q2r + temp.d2_q2*new->d1_r
	+ temp.d2_qr*new->d1_q
	+ new->d2_qr*temp.d1_q;
    new->d3_pq2 = new->value*temp.d3_pq2 + temp.d2_q2*new->d1_p
	+ temp.d2_pq*new->d1_q
	+ new->d2_pq*temp.d1_q;
    new->d3_pr2 = new->value*temp.d3_pr2 + temp.d2_r2*new->d1_p
	+ temp.d2_pr*new->d1_r
	+ new->d2_pr*temp.d1_r;
    new->d3_qr2 = new->value*temp.d3_qr2 + temp.d2_r2*new->d1_q
	+ temp.d2_qr*new->d1_r
	+ new->d2_qr*temp.d1_r;
    new->d3_pqr = new->value*temp.d3_pqr + temp.d2_pq*new->d1_r
	+ temp.d2_pr*new->d1_q
	+ new->d2_qr*temp.d1_p;

}
