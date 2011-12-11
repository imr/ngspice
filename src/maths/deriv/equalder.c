/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * EqualDeriv equates partial derivatives.
 */

void 
EqualDeriv(Dderivs *new, Dderivs *old)
{

new->value = old->value;
new->d1_p = old->d1_p;
new->d1_q = old->d1_q;
new->d1_r = old->d1_r;
new->d2_p2 = old->d2_p2 ;
new->d2_q2 = old->d2_q2 ;
new->d2_r2 = old->d2_r2 ;
new->d2_pq = old->d2_pq ;
new->d2_qr = old->d2_qr ;
new->d2_pr = old->d2_pr ;
new->d3_p3 = old->d3_p3  ;
new->d3_q3 = old->d3_q3  ;
new->d3_r3 = old->d3_r3  ;
new->d3_p2r = old->d3_p2r  ;
new->d3_p2q = old->d3_p2q  ;
new->d3_q2r = old->d3_q2r  ;
new->d3_pq2 = old->d3_pq2  ;
new->d3_pr2 = old->d3_pr2  ;
new->d3_qr2 = old->d3_qr2  ;
new->d3_pqr = old->d3_pqr  ;
}
