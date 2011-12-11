/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

/*
 * CosDeriv computes the partial derivatives of the cosine
 * function where the argument to the function is itself a
 * function of three variables p, q, and r.
 */

void
CosDeriv(Dderivs *new, Dderivs *old)
{

    Dderivs temp;

    EqualDeriv(&temp, old);

    new->value = cos (temp.value);
    new->d1_p = - sin(temp.value)*temp.d1_p;
    new->d1_q = - sin(temp.value)*temp.d1_q;
    new->d1_r = - sin(temp.value)*temp.d1_r;
    new->d2_p2 = -(cos(temp.value)*temp.d1_p*temp.d1_p +
		   sin(temp.value)*temp.d2_p2);
    new->d2_q2 = -(cos(temp.value)*temp.d1_q*temp.d1_q +
		   sin(temp.value)*temp.d2_q2);
    new->d2_r2 = -(cos(temp.value)*temp.d1_r*temp.d1_r +
		   sin(temp.value)*temp.d2_r2);
    new->d2_pq = -(cos(temp.value)*temp.d1_p*temp.d1_q +
		   sin(temp.value)*temp.d2_pq);
    new->d2_qr = -(cos(temp.value)*temp.d1_q*temp.d1_r +
		   sin(temp.value)*temp.d2_qr);
    new->d2_pr = -(cos(temp.value)*temp.d1_p*temp.d1_r +
		   sin(temp.value)*temp.d2_pr);
    new->d3_p3 = -(sin(temp.value)*(temp.d3_p3 - temp.d1_p*temp.d1_p*temp.d1_p)
		   + cos(temp.value)*(temp.d1_p*temp.d2_p2 +
				      temp.d1_p*temp.d2_p2
				      + temp.d1_p*temp.d2_p2));
    new->d3_q3 = -(sin(temp.value)*(temp.d3_q3 - temp.d1_q*temp.d1_q*temp.d1_q)
		   + cos(temp.value)*(temp.d1_q*temp.d2_q2 +
				      temp.d1_q*temp.d2_q2
				      + temp.d1_q*temp.d2_q2));
    new->d3_r3 = -(sin(temp.value)*(temp.d3_r3 - temp.d1_r*temp.d1_r*temp.d1_r)
		   + cos(temp.value)*(temp.d1_r*temp.d2_r2 +
				      temp.d1_r*temp.d2_r2
				      + temp.d1_r*temp.d2_r2));
    new->d3_p2r = -(sin(temp.value)*(temp.d3_p2r -
				     temp.d1_r*temp.d1_p*temp.d1_p)
		    + cos(temp.value)*(temp.d1_p*temp.d2_pr +
				       temp.d1_p*temp.d2_pr
				       + temp.d1_r*temp.d2_p2));
    new->d3_p2q = -(sin(temp.value)*(temp.d3_p2q -
				     temp.d1_q*temp.d1_p*temp.d1_p)
		    + cos(temp.value)*(temp.d1_p*temp.d2_pq +
				       temp.d1_p*temp.d2_pq
				       + temp.d1_q*temp.d2_p2));
    new->d3_q2r = -(sin(temp.value)*(temp.d3_q2r -
				     temp.d1_r*temp.d1_q*temp.d1_q)
		    + cos(temp.value)*(temp.d1_q*temp.d2_qr +
				       temp.d1_q*temp.d2_qr
				       + temp.d1_r*temp.d2_q2));
    new->d3_pq2 = -(sin(temp.value)*(temp.d3_pq2 -
				     temp.d1_p*temp.d1_q*temp.d1_q)
		    + cos(temp.value)*(temp.d1_q*temp.d2_pq +
				       temp.d1_q*temp.d2_pq
				       + temp.d1_p*temp.d2_q2));
    new->d3_pr2 = -(sin(temp.value)*(temp.d3_pr2 -
				     temp.d1_p*temp.d1_r*temp.d1_r)
		    + cos(temp.value)*(temp.d1_r*temp.d2_pr +
				       temp.d1_r*temp.d2_pr
				       + temp.d1_p*temp.d2_r2));
    new->d3_qr2 = -(sin(temp.value)*(temp.d3_qr2 -
				     temp.d1_q*temp.d1_r*temp.d1_r)
		    + cos(temp.value)*(temp.d1_r*temp.d2_qr +
				       temp.d1_r*temp.d2_qr
				       + temp.d1_q*temp.d2_r2));
    new->d3_pqr = -(sin(temp.value)*(temp.d3_pqr -
				     temp.d1_r*temp.d1_p*temp.d1_q)
		    + cos(temp.value)*(temp.d1_q*temp.d2_pr +
				       temp.d1_p*temp.d2_qr
				       + temp.d1_r*temp.d2_pq));
}
