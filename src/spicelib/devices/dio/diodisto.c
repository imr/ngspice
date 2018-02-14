/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author:  1988 Jaijeet S Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

int
DIOdisto( int mode, GENmodel *genmodel, CKTcircuit *ckt)
/* assuming here that ckt->CKTomega has been initialised to 
 * the correct value
 */
{
 DIOmodel *model = (DIOmodel *) genmodel;
 DISTOAN* job = (DISTOAN*) ckt->CKTcurJob;
 double g2,g3;
 double cdiff2,cdiff3;
 double cjunc2,cjunc3;
 double r1h1x,i1h1x;
 double r1h2x, i1h2x;
 double i1hm2x;
 double r2h11x,i2h11x;
 double r2h1m2x,i2h1m2x;
 double temp, itemp;
 DIOinstance *here;

if (mode == D_SETUP)
 return(DIOdSetup(model,ckt));

if ((mode == D_TWOF1) || (mode == D_THRF1) || 
 (mode == D_F1PF2) || (mode == D_F1MF2) ||
 (mode == D_2F1MF2)) {

 /* loop through all the DIO models */
for( ; model != NULL; model = DIOnextModel(model)) {

  /* loop through all the instances of the model */
  for (here = DIOinstances(model); here != NULL ;
       here=DIOnextInstance(here)) {

    /* loading starts here */

    switch (mode) {
    case D_TWOF1:
            g2=here->id_x2;

	    cdiff2=here->cdif_x2;
	     
	    cjunc2=here->cjnc_x2;

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->DIOposPrimeNode)) -
			*(job->r1H1ptr + (here->DIOnegNode));
	    i1h1x = *(job->i1H1ptr + (here->DIOposPrimeNode)) -
			*(job->i1H1ptr + (here->DIOnegNode));

	    /* formulae start here */

	    temp = D1n2F1(g2,r1h1x,i1h1x);
	    itemp = D1i2F1(g2,r1h1x,i1h1x);

	    /* the above are for the memoryless nonlinearity */

	    if ((cdiff2 + cjunc2) != 0.0) {
	    temp +=  - ckt->CKTomega * D1i2F1
				(cdiff2+cjunc2,r1h1x,i1h1x);
	    itemp += ckt->CKTomega * D1n2F1
				((cdiff2 + cjunc2),r1h1x,i1h1x);
	    }

	    *(ckt->CKTrhs + (here->DIOposPrimeNode)) -= temp;
	    *(ckt->CKTirhs + (here->DIOposPrimeNode)) -= itemp;
	    *(ckt->CKTrhs + (here->DIOnegNode)) += temp;
	    *(ckt->CKTirhs + (here->DIOnegNode)) += itemp;
      break;

    case D_THRF1:
            g2=here->id_x2;
	    g3=here->id_x3;

	    cdiff2=here->cdif_x2;
	    cdiff3=here->cdif_x3;
	     
	    cjunc2=here->cjnc_x2;
	    cjunc3=here->cjnc_x3;

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->DIOposPrimeNode)) -
			*(job->r1H1ptr + (here->DIOnegNode));
	    i1h1x = *(job->i1H1ptr + (here->DIOposPrimeNode)) -
			*(job->i1H1ptr + (here->DIOnegNode));

	    /* getting second order kernel at (F1_F1) */
	    r2h11x = *(job->r2H11ptr + (here->DIOposPrimeNode)) -
			*(job->r2H11ptr + (here->DIOnegNode));
	    i2h11x = *(job->i2H11ptr + (here->DIOposPrimeNode)) -
			*(job->i2H11ptr + (here->DIOnegNode));

	    /* formulae start here */

	    temp = D1n3F1(g2,g3,r1h1x,i1h1x,r2h11x,
						i2h11x);
	    itemp = D1i3F1(g2,g3,r1h1x,i1h1x,r2h11x,
						i2h11x);

	    /* the above are for the memoryless nonlinearity */
	    /* the following are for the capacitors */

	    if ((cdiff2 + cjunc2) != 0.0) {
	    temp += -ckt->CKTomega * D1i3F1
				(cdiff2+cjunc2,cdiff3+cjunc3,r1h1x,
						i1h1x,r2h11x,i2h11x);

	    itemp += ckt->CKTomega * D1n3F1
				(cdiff2+cjunc2,cdiff3+cjunc3,r1h1x,
						i1h1x,r2h11x,i2h11x);
	    }

	    /* end of formulae */

	    *(ckt->CKTrhs + (here->DIOposPrimeNode)) -= temp;
	    *(ckt->CKTirhs + (here->DIOposPrimeNode)) -= itemp;
	    *(ckt->CKTrhs + (here->DIOnegNode)) += temp;
	    *(ckt->CKTirhs + (here->DIOnegNode)) += itemp;


      break;
    case D_F1PF2:
            g2=here->id_x2;
	    g3=here->id_x3;

	    cdiff2=here->cdif_x2;
	    cdiff3=here->cdif_x3;
	     
	    cjunc2=here->cjnc_x2;
	    cjunc3=here->cjnc_x3;

	    /* getting first order (linear) Volterra kernel for F1*/
	    r1h1x = *(job->r1H1ptr + (here->DIOposPrimeNode)) -
			*(job->r1H1ptr + (here->DIOnegNode));
	    i1h1x = *(job->i1H1ptr + (here->DIOposPrimeNode)) -
			*(job->i1H1ptr + (here->DIOnegNode));

	    /* getting first order (linear) Volterra kernel for F2*/
	    r1h2x = *(job->r1H2ptr + (here->DIOposPrimeNode)) -
			*(job->r1H2ptr + (here->DIOnegNode));
	    i1h2x = *(job->i1H2ptr + (here->DIOposPrimeNode)) -
			*(job->i1H2ptr + (here->DIOnegNode));

	    /* formulae start here */

	    temp = D1nF12(g2,r1h1x,i1h1x,r1h2x,i1h2x);
	    itemp = D1iF12(g2,r1h1x,i1h1x,r1h2x,i1h2x);

	    /* the above are for the memoryless nonlinearity */
	    /* the following are for the capacitors */

	    if ((cdiff2 + cjunc2) != 0.0) {
	    temp += - ckt->CKTomega * D1iF12
				(cdiff2+cjunc2,r1h1x,i1h1x,r1h2x,i1h2x);
	    itemp += ckt->CKTomega * D1nF12
				(cdiff2+cjunc2,r1h1x,i1h1x,r1h2x,i1h2x);
	    }

	    /* end of formulae */

	    *(ckt->CKTrhs + (here->DIOposPrimeNode)) -= temp;
	    *(ckt->CKTirhs + (here->DIOposPrimeNode)) -= itemp;
	    *(ckt->CKTrhs + (here->DIOnegNode)) += temp;
	    *(ckt->CKTirhs + (here->DIOnegNode)) += itemp;


      break;
    case D_F1MF2:
            g2=here->id_x2;
	    g3=here->id_x3;

	    cdiff2=here->cdif_x2;
	    cdiff3=here->cdif_x3;
	     
	    cjunc2=here->cjnc_x2;
	    cjunc3=here->cjnc_x3;

	    /* getting first order (linear) Volterra kernel for F1*/
	    r1h1x = *(job->r1H1ptr + (here->DIOposPrimeNode)) -
			*(job->r1H1ptr + (here->DIOnegNode));
	    i1h1x = *(job->i1H1ptr + (here->DIOposPrimeNode)) -
			*(job->i1H1ptr + (here->DIOnegNode));

	    /* getting first order (linear) Volterra kernel for F2*/
	    r1h2x = *(job->r1H2ptr + (here->DIOposPrimeNode)) -
			*(job->r1H2ptr + (here->DIOnegNode));
	    i1hm2x = -(*(job->i1H2ptr + (here->DIOposPrimeNode)) -
			*(job->i1H2ptr + (here->DIOnegNode)));

	    /* formulae start here */

	    temp = D1nF12(g2,r1h1x,i1h1x,r1h2x,i1hm2x);
	    itemp = D1iF12(g2,r1h1x,i1h1x,r1h2x,i1hm2x);

	    /* the above are for the memoryless nonlinearity */
	    /* the following are for the capacitors */

	    if ((cdiff2 + cjunc2) != 0.0) {
	    temp += - ckt->CKTomega * D1iF12
				(cdiff2+cjunc2,r1h1x,i1h1x,r1h2x,i1hm2x);
	    itemp += ckt->CKTomega * D1nF12
				(cdiff2+cjunc2,r1h1x,i1h1x,r1h2x,i1hm2x);
	    }

	    /* end of formulae */

	    *(ckt->CKTrhs + (here->DIOposPrimeNode)) -= temp;
	    *(ckt->CKTirhs + (here->DIOposPrimeNode)) -= itemp;
	    *(ckt->CKTrhs + (here->DIOnegNode)) += temp;
	    *(ckt->CKTirhs + (here->DIOnegNode)) += itemp;


      break;
    case D_2F1MF2:
            g2=here->id_x2;
	    g3=here->id_x3;

	    cdiff2=here->cdif_x2;
	    cdiff3=here->cdif_x3;
	     
	    cjunc2=here->cjnc_x2;
	    cjunc3=here->cjnc_x3;

	    /* getting first order (linear) Volterra kernel at F1*/
	    r1h1x = *(job->r1H1ptr + (here->DIOposPrimeNode)) -
			*(job->r1H1ptr + (here->DIOnegNode));
	    i1h1x = *(job->i1H1ptr + (here->DIOposPrimeNode)) -
			*(job->i1H1ptr + (here->DIOnegNode));

	    /* getting first order (linear) Volterra kernel at minusF2*/
	    r1h2x = *(job->r1H2ptr + (here->DIOposPrimeNode)) -
			*(job->r1H2ptr + (here->DIOnegNode));
	    i1hm2x = -(*(job->i1H2ptr + (here->DIOposPrimeNode)) -
			*(job->i1H2ptr + (here->DIOnegNode)));

	    /* getting second order kernel at (F1_F1) */
	    r2h11x = *(job->r2H11ptr + (here->DIOposPrimeNode)) -
			*(job->r2H11ptr + (here->DIOnegNode));
	    i2h11x = *(job->i2H11ptr + (here->DIOposPrimeNode)) -
			*(job->i2H11ptr + (here->DIOnegNode));

	    /* getting second order kernel at (F1_minusF2) */
	    r2h1m2x = *(job->r2H1m2ptr + (here->DIOposPrimeNode)) -
			*(job->r2H1m2ptr + (here->DIOnegNode));
	    i2h1m2x = *(job->i2H1m2ptr + (here->DIOposPrimeNode)) -
			*(job->i2H1m2ptr + (here->DIOnegNode));

	    /* formulae start here */

	    temp = D1n2F12(g2,g3,r1h1x,i1h1x,r1h2x,
				i1hm2x,r2h11x,i2h11x,
					r2h1m2x,i2h1m2x);
	    itemp = D1i2F12(g2,g3,r1h1x,i1h1x,
				r1h2x,i1hm2x,r2h11x,i2h11x,
					r2h1m2x,i2h1m2x);


	    /* the above are for the memoryless nonlinearity */
	    /* the following are for the capacitors */

	    if ((cdiff2 + cjunc2) != 0.0) {
	    temp += -ckt->CKTomega * 
	    D1i2F12(cdiff2+cjunc2,cdiff3+cjunc3,
			r1h1x,i1h1x,r1h2x,i1hm2x,r2h11x,
				i2h11x,r2h1m2x,i2h1m2x);
	    itemp += ckt->CKTomega *
	    D1n2F12(cdiff2+cjunc2,cdiff3+cjunc3,
			r1h1x,i1h1x,r1h2x,i1hm2x,r2h11x,
				i2h11x,r2h1m2x,i2h1m2x);
	    }


	    /* end of formulae */

	    *(ckt->CKTrhs + (here->DIOposPrimeNode)) -= temp;
	    *(ckt->CKTirhs + (here->DIOposPrimeNode)) -= itemp;
	    *(ckt->CKTrhs + (here->DIOnegNode)) += temp;
	    *(ckt->CKTirhs + (here->DIOnegNode)) += itemp;


      break;
    default:
;
    }
  }
}
return(OK);
}
  else
    return(E_BADPARM);
}
