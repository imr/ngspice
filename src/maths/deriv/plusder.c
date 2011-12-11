/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * PlusDeriv computes the partial derivatives of the addition
 * function where the arguments to the function are
 * functions of three variables p, q, and r.
 */

void
PlusDeriv(Dderivs *new, Dderivs *old1, Dderivs *old2)
{
    new->value = old1->value + old2->value;
    new->d1_p = old1->d1_p  + old2->d1_p;
    new->d1_q = old1->d1_q  + old2->d1_q;
    new->d1_r = old1->d1_r  + old2->d1_r;
    new->d2_p2 = old1->d2_p2  + old2->d2_p2;
    new->d2_q2 = old1->d2_q2  + old2->d2_q2;
    new->d2_r2 = old1->d2_r2  + old2->d2_r2;
    new->d2_pq = old1->d2_pq  + old2->d2_pq;
    new->d2_qr = old1->d2_qr  + old2->d2_qr;
    new->d2_pr = old1->d2_pr  + old2->d2_pr;
    new->d3_p3 = old1->d3_p3 + old2->d3_p3;
    new->d3_q3 = old1->d3_q3 + old2->d3_q3;
    new->d3_r3 = old1->d3_r3 + old2->d3_r3;
    new->d3_p2r = old1->d3_p2r + old2->d3_p2r;
    new->d3_p2q = old1->d3_p2q + old2->d3_p2q;
    new->d3_q2r = old1->d3_q2r + old2->d3_q2r;
    new->d3_pq2 = old1->d3_pq2 + old2->d3_pq2;
    new->d3_pr2 = old1->d3_pr2 + old2->d3_pr2;
    new->d3_qr2 = old1->d3_qr2 + old2->d3_qr2;
    new->d3_pqr = old1->d3_pqr + old2->d3_pqr;
}
