/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * MultDeriv computes the partial derivatives of the multiplication
 * function where the arguments to the function are
 * functions of three variables p, q, and r.
 */

void
MultDeriv(Dderivs *new, Dderivs *old1, Dderivs *old2)
{
    Dderivs temp1, temp2;

    EqualDeriv(&temp1, old1);
    EqualDeriv(&temp2, old2);

    new->value = temp1.value * temp2.value;
    new->d1_p = temp1.d1_p*temp2.value + temp1.value*temp2.d1_p;
    new->d1_q = temp1.d1_q*temp2.value + temp1.value*temp2.d1_q;
    new->d1_r = temp1.d1_r*temp2.value + temp1.value*temp2.d1_r;
    new->d2_p2 = temp1.d2_p2*temp2.value + temp1.d1_p*temp2.d1_p
	+ temp1.d1_p*temp2.d1_p + temp1.value*temp2.d2_p2;
    new->d2_q2 = temp1.d2_q2*temp2.value + temp1.d1_q*temp2.d1_q
	+ temp1.d1_q*temp2.d1_q + temp1.value*temp2.d2_q2;
    new->d2_r2 = temp1.d2_r2*temp2.value + temp1.d1_r*temp2.d1_r
	+ temp1.d1_r*temp2.d1_r + temp1.value*temp2.d2_r2;
    new->d2_pq = temp1.d2_pq*temp2.value + temp1.d1_p*temp2.d1_q
	+ temp1.d1_q*temp2.d1_p + temp1.value*temp2.d2_pq;
    new->d2_qr = temp1.d2_qr*temp2.value + temp1.d1_q*temp2.d1_r
	+ temp1.d1_r*temp2.d1_q + temp1.value*temp2.d2_qr;
    new->d2_pr = temp1.d2_pr*temp2.value + temp1.d1_p*temp2.d1_r
	+ temp1.d1_r*temp2.d1_p + temp1.value*temp2.d2_pr;
    new->d3_p3 = temp1.d3_p3*temp2.value + temp1.d2_p2*temp2.d1_p
	+ temp1.d2_p2*temp2.d1_p + temp2.d2_p2*temp1.d1_p
	+ temp2.d2_p2*temp1.d1_p + temp1.d2_p2*temp2.d1_p
	+ temp2.d2_p2*temp1.d1_p + temp1.value*temp2.d3_p3;
    
    new->d3_q3 = temp1.d3_q3*temp2.value + temp1.d2_q2*temp2.d1_q
	+ temp1.d2_q2*temp2.d1_q + temp2.d2_q2*temp1.d1_q
	+ temp2.d2_q2*temp1.d1_q + temp1.d2_q2*temp2.d1_q
	+ temp2.d2_q2*temp1.d1_q + temp1.value*temp2.d3_q3;
    new->d3_r3 = temp1.d3_r3*temp2.value + temp1.d2_r2*temp2.d1_r
	+ temp1.d2_r2*temp2.d1_r + temp2.d2_r2*temp1.d1_r
	+ temp2.d2_r2*temp1.d1_r + temp1.d2_r2*temp2.d1_r
	+ temp2.d2_r2*temp1.d1_r + temp1.value*temp2.d3_r3;
    new->d3_p2r = temp1.d3_p2r*temp2.value + temp1.d2_p2*temp2.d1_r
	+ temp1.d2_pr*temp2.d1_p + temp2.d2_p2*temp1.d1_r
	+ temp2.d2_pr*temp1.d1_p + temp1.d2_pr*temp2.d1_p
	+ temp2.d2_pr*temp1.d1_p + temp1.value*temp2.d3_p2r;
    new->d3_p2q = temp1.d3_p2q*temp2.value + temp1.d2_p2*temp2.d1_q
	+ temp1.d2_pq*temp2.d1_p + temp2.d2_p2*temp1.d1_q
	+ temp2.d2_pq*temp1.d1_p + temp1.d2_pq*temp2.d1_p
	+ temp2.d2_pq*temp1.d1_p + temp1.value*temp2.d3_p2q;
    new->d3_q2r = temp1.d3_q2r*temp2.value + temp1.d2_q2*temp2.d1_r
	+ temp1.d2_qr*temp2.d1_q + temp2.d2_q2*temp1.d1_r
	+ temp2.d2_qr*temp1.d1_q + temp1.d2_qr*temp2.d1_q
	+ temp2.d2_qr*temp1.d1_q + temp1.value*temp2.d3_q2r;
    new->d3_pq2 = temp1.d3_pq2*temp2.value + temp1.d2_q2*temp2.d1_p
	+ temp1.d2_pq*temp2.d1_q + temp2.d2_q2*temp1.d1_p
	+ temp2.d2_pq*temp1.d1_q + temp1.d2_pq*temp2.d1_q
	+ temp2.d2_pq*temp1.d1_q + temp1.value*temp2.d3_pq2;
    new->d3_pr2 = temp1.d3_pr2*temp2.value + temp1.d2_r2*temp2.d1_p
	+ temp1.d2_pr*temp2.d1_r + temp2.d2_r2*temp1.d1_p
	+ temp2.d2_pr*temp1.d1_r + temp1.d2_pr*temp2.d1_r
	+ temp2.d2_pr*temp1.d1_r + temp1.value*temp2.d3_pr2;
    new->d3_qr2 = temp1.d3_qr2*temp2.value + temp1.d2_r2*temp2.d1_q
	+ temp1.d2_qr*temp2.d1_r + temp2.d2_r2*temp1.d1_q
	+ temp2.d2_qr*temp1.d1_r + temp1.d2_qr*temp2.d1_r
	+ temp2.d2_qr*temp1.d1_r + temp1.value*temp2.d3_qr2;
    new->d3_pqr = temp1.d3_pqr*temp2.value + temp1.d2_pq*temp2.d1_r
	+ temp1.d2_pr*temp2.d1_q + temp2.d2_pq*temp1.d1_r
	+ temp2.d2_qr*temp1.d1_p + temp1.d2_qr*temp2.d1_p
	+ temp2.d2_pr*temp1.d1_q + temp1.value*temp2.d3_pqr;
}
