/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * TimesDeriv computes the partial derivatives of the x*k
 * function where the argument to the function is itself a
 * function of three variables p, q, and r. k is a constant.
 */

void
TimesDeriv(Dderivs *new, Dderivs *old, double k)
{
    new->value = k* old->value;
    new->d1_p = k*old->d1_p;
    new->d1_q = k*old->d1_q;
    new->d1_r = k*old->d1_r;
    new->d2_p2 = k*old->d2_p2;
    new->d2_q2 = k*old->d2_q2;
    new->d2_r2 = k*old->d2_r2;
    new->d2_pq = k*old->d2_pq;
    new->d2_qr = k*old->d2_qr;
    new->d2_pr = k*old->d2_pr;
    new->d3_p3 = k*old->d3_p3;
    new->d3_q3 = k*old->d3_q3;
    new->d3_r3 = k*old->d3_r3;
    new->d3_p2r = k*old->d3_p2r;
    new->d3_p2q = k*old->d3_p2q;
    new->d3_q2r = k*old->d3_q2r;
    new->d3_pq2 = k*old->d3_pq2;
    new->d3_pr2 = k*old->d3_pr2;
    new->d3_qr2 = k*old->d3_qr2;
    new->d3_pqr = k*old->d3_pqr;
}
