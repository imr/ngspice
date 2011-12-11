/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * InvDeriv computes the partial derivatives of the 1/x
 * function where the argument to the function is itself a
 * function of three variables p, q, and r.
 */

void
InvDeriv(Dderivs *new, Dderivs *old)
{
    Dderivs temp;

    EqualDeriv(&temp, old);

    new->value = 1/temp.value;
    new->d1_p = -new->value*new->value*temp.d1_p;
    new->d1_q = -new->value*new->value*temp.d1_q;
    new->d1_r = -new->value*new->value*temp.d1_r;
    new->d2_p2 = -new->value*(2*new->d1_p*temp.d1_p + new->value*temp.d2_p2);
    new->d2_q2 = -new->value*(2*new->d1_q*temp.d1_q + new->value*temp.d2_q2);
    new->d2_r2 = -new->value*(2*new->d1_r*temp.d1_r + new->value*temp.d2_r2);
    new->d2_pq = -new->value*(2*new->d1_q*temp.d1_p + new->value*temp.d2_pq);
    new->d2_qr = -new->value*(2*new->d1_r*temp.d1_q + new->value*temp.d2_qr);
    new->d2_pr = -new->value*(2*new->d1_r*temp.d1_p + new->value*temp.d2_pr);
    new->d3_p3 = -(2*(temp.d1_p*new->d1_p*new->d1_p + new->value*(
	new->d2_p2*temp.d1_p + new->d1_p*temp.d2_p2 +
	new->d1_p*temp.d2_p2)) + new->value*new->value*temp.d3_p3);
    new->d3_q3 = -(2*(temp.d1_q*new->d1_q*new->d1_q + new->value*(
	new->d2_q2*temp.d1_q + new->d1_q*temp.d2_q2 +
	new->d1_q*temp.d2_q2)) + new->value*new->value*temp.d3_q3);
    new->d3_r3 = -(2*(temp.d1_r*new->d1_r*new->d1_r + new->value*(
	new->d2_r2*temp.d1_r + new->d1_r*temp.d2_r2 +
	new->d1_r*temp.d2_r2)) + new->value*new->value*temp.d3_r3);
    new->d3_p2r = -(2*(temp.d1_p*new->d1_p*new->d1_r + new->value*(
	new->d2_pr*temp.d1_p + new->d1_p*temp.d2_pr +
	new->d1_r*temp.d2_p2)) + new->value*new->value*temp.d3_p2r);
    new->d3_p2q = -(2*(temp.d1_p*new->d1_p*new->d1_q + new->value*(
	new->d2_pq*temp.d1_p + new->d1_p*temp.d2_pq +
	new->d1_q*temp.d2_p2)) + new->value*new->value*temp.d3_p2q);
    new->d3_q2r = -(2*(temp.d1_q*new->d1_q*new->d1_r + new->value*(
	new->d2_qr*temp.d1_q + new->d1_q*temp.d2_qr +
	new->d1_r*temp.d2_q2)) + new->value*new->value*temp.d3_q2r);
    new->d3_pq2 = -(2*(temp.d1_q*new->d1_q*new->d1_p + new->value*(
	new->d2_pq*temp.d1_q + new->d1_q*temp.d2_pq +
	new->d1_p*temp.d2_q2)) + new->value*new->value*temp.d3_pq2);
    new->d3_pr2 = -(2*(temp.d1_r*new->d1_r*new->d1_p + new->value*(
	new->d2_pr*temp.d1_r + new->d1_r*temp.d2_pr +
	new->d1_p*temp.d2_r2)) + new->value*new->value*temp.d3_pr2);
    new->d3_qr2 = -(2*(temp.d1_r*new->d1_r*new->d1_q + new->value*(
	new->d2_qr*temp.d1_r + new->d1_r*temp.d2_qr +
	new->d1_q*temp.d2_r2)) + new->value*new->value*temp.d3_qr2);
    new->d3_pqr = -(2*(temp.d1_p*new->d1_q*new->d1_r + new->value*(
	new->d2_qr*temp.d1_p + new->d1_q*temp.d2_pr +
	new->d1_r*temp.d2_pq)) + new->value*new->value*temp.d3_pqr);

}
