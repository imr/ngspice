/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * TanDeriv computes the partial derivatives of the tangent
 * function where the argument to the function is itself a
 * function of three variables p, q, and r.
 */

void TanDeriv(
Dderivs *new, Dderivs *old)
{

Dderivs temp;

EqualDeriv(&temp, old);
new->value = tan(temp.value);

new->d1_p = (1 + new->value*new->value)*temp.d1_p;
new->d1_q = (1 + new->value*new->value)*temp.d1_q;
new->d1_r = (1 + new->value*new->value)*temp.d1_r;
new->d2_p2 = (1 + new->value*new->value)*temp.d2_p2 + 2*new->value*temp.d1_p*new->d1_p;
new->d2_q2 = (1 + new->value*new->value)*temp.d2_q2 + 2*new->value*temp.d1_q*new->d1_q;
new->d2_r2 = (1 + new->value*new->value)*temp.d2_r2 + 2*new->value*temp.d1_r*new->d1_r;
new->d2_pq = (1 + new->value*new->value)*temp.d2_pq + 2*new->value*temp.d1_p*new->d1_q;
new->d2_qr = (1 + new->value*new->value)*temp.d2_qr + 2*new->value*temp.d1_q*new->d1_r;
new->d2_pr = (1 + new->value*new->value)*temp.d2_pr + 2*new->value*temp.d1_p*new->d1_r;
new->d3_p3 = (1 + new->value*new->value)*temp.d3_p3 +2*( new->value*(
	temp.d2_p2*new->d1_p + temp.d2_p2*new->d1_p + new->d2_p2*
	temp.d1_p) + temp.d1_p*new->d1_p*new->d1_p);
new->d3_q3 = (1 + new->value*new->value)*temp.d3_q3 +2*( new->value*(
	temp.d2_q2*new->d1_q + temp.d2_q2*new->d1_q + new->d2_q2*
	temp.d1_q) + temp.d1_q*new->d1_q*new->d1_q);
new->d3_r3 = (1 + new->value*new->value)*temp.d3_r3 +2*( new->value*(
	temp.d2_r2*new->d1_r + temp.d2_r2*new->d1_r + new->d2_r2*
	temp.d1_r) + temp.d1_r*new->d1_r*new->d1_r);
new->d3_p2r = (1 + new->value*new->value)*temp.d3_p2r +2*( new->value*(
	temp.d2_p2*new->d1_r + temp.d2_pr*new->d1_p + new->d2_pr*
	temp.d1_p) + temp.d1_p*new->d1_p*new->d1_r);
new->d3_p2q = (1 + new->value*new->value)*temp.d3_p2q +2*( new->value*(
	temp.d2_p2*new->d1_q + temp.d2_pq*new->d1_p + new->d2_pq*
	temp.d1_p) + temp.d1_p*new->d1_p*new->d1_q);
new->d3_q2r = (1 + new->value*new->value)*temp.d3_q2r +2*( new->value*(
	temp.d2_q2*new->d1_r + temp.d2_qr*new->d1_q + new->d2_qr*
	temp.d1_q) + temp.d1_q*new->d1_q*new->d1_r);
new->d3_pq2 = (1 + new->value*new->value)*temp.d3_pq2 +2*( new->value*(
	temp.d2_q2*new->d1_p + temp.d2_pq*new->d1_q + new->d2_pq*
	temp.d1_q) + temp.d1_q*new->d1_q*new->d1_p);
new->d3_pr2 = (1 + new->value*new->value)*temp.d3_pr2 +2*( new->value*(
	temp.d2_r2*new->d1_p + temp.d2_pr*new->d1_r + new->d2_pr*
	temp.d1_r) + temp.d1_r*new->d1_r*new->d1_p);
new->d3_qr2 = (1 + new->value*new->value)*temp.d3_qr2 +2*( new->value*(
	temp.d2_r2*new->d1_q + temp.d2_qr*new->d1_r + new->d2_qr*
	temp.d1_r) + temp.d1_r*new->d1_r*new->d1_q);
new->d3_pqr = (1 + new->value*new->value)*temp.d3_pqr +2*( new->value*(
	temp.d2_pq*new->d1_r + temp.d2_pr*new->d1_q + new->d2_qr*
	temp.d1_p) + temp.d1_p*new->d1_q*new->d1_r);
	}
