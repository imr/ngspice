/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * PowDeriv computes the partial derivatives of the x^^m
 * function where the argument to the function is itself a
 * function of three variables p, q, and r. m is a constant.
 */

void 
PowDeriv(Dderivs *new, Dderivs *old, double emm)
{
    Dderivs temp;

    EqualDeriv(&temp, old);

    new->value = pow(temp.value, emm);
    new->d1_p = emm * new->value / temp.value * temp.d1_p;
    new->d1_q = emm * new->value / temp.value * temp.d1_q;
    new->d1_r = emm * new->value / temp.value * temp.d1_r;
    new->d2_p2 = emm * new->value / temp.value *
	((emm-1) / temp.value * temp.d1_p * temp.d1_p + temp.d2_p2);
    new->d2_q2 = emm * new->value / temp.value *
	((emm-1) / temp.value * temp.d1_q * temp.d1_q + temp.d2_q2);
    new->d2_r2 = emm * new->value / temp.value *
	((emm-1) / temp.value * temp.d1_r * temp.d1_r + temp.d2_r2);
    new->d2_pq = emm * new->value / temp.value *
	((emm-1) / temp.value * temp.d1_p * temp.d1_q + temp.d2_pq);
    new->d2_qr = emm * new->value / temp.value *
	((emm-1) / temp.value * temp.d1_q * temp.d1_r + temp.d2_qr);
    new->d2_pr = emm * new->value / temp.value *
	((emm-1) / temp.value * temp.d1_p * temp.d1_r + temp.d2_pr);
    new->d3_p3 = emm * (emm-1) * new->value / (temp.value * temp.value) *
	((emm-2) / temp.value * temp.d1_p * 
	 temp.d1_p * temp.d1_p + temp.d1_p * temp.d2_p2 + temp.d1_p *
	 temp.d2_p2 + temp.d1_p * temp.d2_p2) + emm * new->value /
	temp.value * temp.d3_p3;
    new->d3_q3 = emm * (emm-1) * new->value / (temp.value * temp.value) *
	((emm-2) / temp.value * temp.d1_q * 
	 temp.d1_q * temp.d1_q + temp.d1_q * temp.d2_q2 + temp.d1_q *
	 temp.d2_q2 + temp.d1_q * temp.d2_q2) + emm * new->value /
	temp.value * temp.d3_q3;
    new->d3_r3 = emm * (emm-1) * new->value / (temp.value * temp.value) *
	((emm-2) / temp.value * temp.d1_r *temp.d1_r * temp.d1_r +
	 temp.d1_r * temp.d2_r2 + temp.d1_r * temp.d2_r2 + temp.d1_r *
	 temp.d2_r2) + emm * new->value / temp.value * temp.d3_r3;
    new->d3_p2r = emm * (emm-1) * new->value / (temp.value * temp.value) *
	((emm-2) / temp.value * temp.d1_p * temp.d1_p * temp.d1_r +
	 temp.d1_p * temp.d2_pr + temp.d1_p * temp.d2_pr + temp.d1_r *
	 temp.d2_p2) + emm * new->value / temp.value * temp.d3_p2r;
    new->d3_p2q = emm * (emm-1) * new->value / (temp.value * temp.value) *
	((emm-2) / temp.value * temp.d1_p * temp.d1_p * temp.d1_q +
	 temp.d1_p * temp.d2_pq + temp.d1_p * temp.d2_pq + temp.d1_q *
	 temp.d2_p2) + emm * new->value / temp.value * temp.d3_p2q;
    new->d3_q2r = emm * (emm-1) * new->value / (temp.value * temp.value) *
	((emm-2) / temp.value * temp.d1_q * temp.d1_q * temp.d1_r +
	 temp.d1_q * temp.d2_qr + temp.d1_q * temp.d2_qr + temp.d1_r *
	 temp.d2_q2) + emm * new->value / temp.value * temp.d3_q2r;
    new->d3_pq2 = emm * (emm-1) * new->value / (temp.value * temp.value) *
	((emm-2) / temp.value * temp.d1_q * temp.d1_q * temp.d1_p +
	 temp.d1_q * temp.d2_pq + temp.d1_q * temp.d2_pq + temp.d1_p *
	 temp.d2_q2) + emm * new->value / temp.value * temp.d3_pq2;
    new->d3_pr2 = emm * (emm-1) * new->value / (temp.value * temp.value) *
	((emm-2) / temp.value * temp.d1_r * temp.d1_r * temp.d1_p +
	 temp.d1_r * temp.d2_pr + temp.d1_r * temp.d2_pr + temp.d1_p *
	 temp.d2_r2) + emm * new->value / temp.value * temp.d3_pr2;
    new->d3_qr2 = emm * (emm-1) * new->value / (temp.value * temp.value) *
	((emm-2) / temp.value * temp.d1_r * temp.d1_r * temp.d1_q +
	 temp.d1_r * temp.d2_qr + temp.d1_r * temp.d2_qr + temp.d1_q *
	 temp.d2_r2) + emm * new->value / temp.value * temp.d3_qr2;
    new->d3_pqr = emm * (emm-1) * new->value / (temp.value * temp.value) *
	((emm-2) / temp.value * temp.d1_p * temp.d1_q * temp.d1_r +
	 temp.d1_p * temp.d2_qr + temp.d1_q * temp.d2_pr + temp.d1_r *
	 temp.d2_pq) + emm * new->value / temp.value * temp.d3_pqr;
}
