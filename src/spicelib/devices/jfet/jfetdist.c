/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

int
JFETdisto(int mode, GENmodel *genmodel, CKTcircuit *ckt)
/* assuming here that ckt->CKTomega has been initialised to 
 * the correct value
 */
{
 JFETmodel *model = (JFETmodel *) genmodel;
 DISTOAN* job = (DISTOAN*) ckt->CKTcurJob;
 DpassStr pass;
 double r1h1x,i1h1x;
 double r1h1y,i1h1y;
 double r1h2x, i1h2x;
 double r1h2y, i1h2y;
 double r1hm2x,i1hm2x;
 double r1hm2y,i1hm2y;
 double r2h11x,i2h11x;
 double r2h11y,i2h11y;
 double r2h1m2x,i2h1m2x;
 double r2h1m2y,i2h1m2y;
 double temp, itemp;
 JFETinstance *here;

if (mode == D_SETUP)
  return(JFETdSetup(genmodel,ckt)); /* another hack from similar structures */

if ((mode == D_TWOF1) || (mode == D_THRF1) || 
 (mode == D_F1PF2) || (mode == D_F1MF2) ||
 (mode == D_2F1MF2)) {

 /* loop through all the JFET models */
for( ; model != NULL; model = JFETnextModel(model)) {

  /* loop through all the instances of the model */
  for (here = JFETinstances(model); here != NULL ;
 		here=JFETnextInstance(here)) {

    /* loading starts here */

    switch (mode) {
    case D_TWOF1:
	/* x = vgs, y = vds */

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->JFETgateNode)) -
			*(job->r1H1ptr + (here->JFETsourcePrimeNode));
	    i1h1x = *(job->i1H1ptr + (here->JFETgateNode)) -
			*(job->i1H1ptr + (here->JFETsourcePrimeNode));

	    r1h1y = *(job->r1H1ptr + (here->JFETdrainPrimeNode)) -
			*(job->r1H1ptr + (here->JFETsourcePrimeNode));
	    i1h1y = *(job->i1H1ptr + (here->JFETdrainPrimeNode)) -
			*(job->i1H1ptr + (here->JFETsourcePrimeNode));


	    /* loading starts here */
	    /* loading cdrain term  */

	    temp = DFn2F1(here->cdr_x2,
					here->cdr_y2,
					0.0,
					here->cdr_xy,
					0.0,
					0.0,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					0.0,
					0.0);

	    itemp = DFi2F1(here->cdr_x2,
					here->cdr_y2,
					0.0,
					here->cdr_xy,
					0.0,
					0.0,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					0.0,
					0.0);

	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;
	
	    /* cdrain term over */

		/* loading ggs term */

		temp = D1n2F1(here->ggs2,
						r1h1x,
						i1h1x);

		itemp = D1i2F1(here->ggs2,
						r1h1x,
						i1h1x);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;

		/* ggs over */

		/* loading ggd term */

		temp = D1n2F1(here->ggd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y);

		itemp = D1i2F1(here->ggd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) += itemp;

		/* ggd over */

		/* loading capgs term */

		temp = -ckt->CKTomega *
				D1i2F1(here->capgs2,
						r1h1x,
						i1h1x);

		itemp = ckt->CKTomega *
				D1n2F1(here->capgs2,
						r1h1x,
						i1h1x);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;

		/* capgs over */

		/* loading capgd term */

		temp = -ckt->CKTomega *
				D1i2F1(here->capgd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y);

		itemp = ckt->CKTomega *
				D1n2F1(here->capgd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) += itemp;

		/* capgd over */

	/* all done */

      break;

    case D_THRF1:
	/* x = vgs, y = vds */

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->JFETgateNode)) -
			*(job->r1H1ptr + (here->JFETsourcePrimeNode));
	    i1h1x = *(job->i1H1ptr + (here->JFETgateNode)) -
			*(job->i1H1ptr + (here->JFETsourcePrimeNode));

	    r1h1y = *(job->r1H1ptr + (here->JFETdrainPrimeNode)) -
			*(job->r1H1ptr + (here->JFETsourcePrimeNode));
	    i1h1y = *(job->i1H1ptr + (here->JFETdrainPrimeNode)) -
			*(job->i1H1ptr + (here->JFETsourcePrimeNode));

	  r2h11x = *(job->r2H11ptr + (here->JFETgateNode)) -
		*(job->r2H11ptr + (here->JFETsourcePrimeNode));
	  i2h11x = *(job->i2H11ptr + (here->JFETgateNode)) -
		*(job->i2H11ptr + (here->JFETsourcePrimeNode));

	  r2h11y = *(job->r2H11ptr + (here->JFETdrainPrimeNode)) -
		*(job->r2H11ptr + (here->JFETsourcePrimeNode));
	  i2h11y = *(job->i2H11ptr + (here->JFETdrainPrimeNode)) -
		*(job->i2H11ptr + (here->JFETsourcePrimeNode));

	    /* loading starts here */
	    /* loading cdrain term  */

	    temp = DFn3F1(here->cdr_x2,
					here->cdr_y2,
					0.0,
					here->cdr_xy,
					0.0,
					0.0,
					here->cdr_x3,
					here->cdr_y3,
					0.0,
					here->cdr_x2y,
					0.0,
					here->cdr_xy2,
					0.0,
					0.0,
					0.0,
					0.0,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					0.0,
					0.0,
					r2h11x,
					i2h11x,
					r2h11y,
					i2h11y,
					0.0,
					0.0);

	    itemp = DFi3F1(here->cdr_x2,
					here->cdr_y2,
					0.0,
					here->cdr_xy,
					0.0,
					0.0,
					here->cdr_x3,
					here->cdr_y3,
					0.0,
					here->cdr_x2y,
					0.0,
					here->cdr_xy2,
					0.0,
					0.0,
					0.0,
					0.0,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					0.0,
					0.0,
					r2h11x,
					i2h11x,
					r2h11y,
					i2h11y,
					0.0,
					0.0);

	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;
	
	    /* cdrain term over */

		/* loading ggs term */

		temp = D1n3F1(here->ggs2,
						here->ggs3,
						r1h1x,
						i1h1x,
						r2h11x,
						i2h11x);

		itemp = D1i3F1(here->ggs2,
						here->ggs3,
						r1h1x,
						i1h1x,
						r2h11x,
						i2h11x);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;

		/* ggs over */

		/* loading ggd term */

		temp = D1n3F1(here->ggd2,
						here->ggd3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r2h11x - r2h11y,
						i2h11x - i2h11y);

		itemp = D1i3F1(here->ggd2,
						here->ggd3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r2h11x - r2h11y,
						i2h11x - i2h11y);



	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) += itemp;

		/* ggd over */

		/* loading capgs term */

		temp = -ckt->CKTomega *
				D1i3F1(here->capgs2,
						here->capgs3,
						r1h1x,
						i1h1x,
						r2h11x,
						i2h11x);

		itemp = ckt->CKTomega *
				D1n3F1(here->capgs2,
						here->capgs3,
						r1h1x,
						i1h1x,
						r2h11x,
						i2h11x);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;

		/* capgs over */

		/* loading capgd term */

		temp = -ckt->CKTomega *
				D1i3F1(here->capgd2,
						here->capgd3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r2h11x - r2h11y,
						i2h11x - i2h11y);

		itemp = ckt->CKTomega *
				D1n3F1(here->capgd2,
						here->capgd3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r2h11x - r2h11y,
						i2h11x - i2h11y);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) += itemp;

		/* capgd over */

	/* all done */

      break;
    case D_F1PF2:
	/* x = vgs, y = vds */

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->JFETgateNode)) -
			*(job->r1H1ptr + (here->JFETsourcePrimeNode));
	    i1h1x = *(job->i1H1ptr + (here->JFETgateNode)) -
			*(job->i1H1ptr + (here->JFETsourcePrimeNode));

	    r1h1y = *(job->r1H1ptr + (here->JFETdrainPrimeNode)) -
			*(job->r1H1ptr + (here->JFETsourcePrimeNode));
	    i1h1y = *(job->i1H1ptr + (here->JFETdrainPrimeNode)) -
			*(job->i1H1ptr + (here->JFETsourcePrimeNode));

	    r1h2x = *(job->r1H2ptr + (here->JFETgateNode)) -
			*(job->r1H2ptr + (here->JFETsourcePrimeNode));
	    i1h2x = *(job->i1H2ptr + (here->JFETgateNode)) -
			*(job->i1H2ptr + (here->JFETsourcePrimeNode));

	    r1h2y = *(job->r1H2ptr + (here->JFETdrainPrimeNode)) -
			*(job->r1H2ptr + (here->JFETsourcePrimeNode));
	    i1h2y = *(job->i1H2ptr + (here->JFETdrainPrimeNode)) -
			*(job->i1H2ptr + (here->JFETsourcePrimeNode));

	    /* loading starts here */
	    /* loading cdrain term  */

	    temp = DFnF12(here->cdr_x2,
					here->cdr_y2,
					0.0,
					here->cdr_xy,
					0.0,
					0.0,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					0.0,
					0.0,
					r1h2x,
					i1h2x,
					r1h2y,
					i1h2y,
					0.0,
					0.0);

	    itemp = DFiF12(here->cdr_x2,
					here->cdr_y2,
					0.0,
					here->cdr_xy,
					0.0,
					0.0,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					0.0,
					0.0,
					r1h2x,
					i1h2x,
					r1h2y,
					i1h2y,
					0.0,
					0.0);

	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;
	
	    /* cdrain term over */

		/* loading ggs term */

		temp = D1nF12(here->ggs2,
						r1h1x,
						i1h1x,
						r1h2x,
						i1h2x);

		itemp = D1iF12(here->ggs2,
						r1h1x,
						i1h1x,
						r1h2x,
						i1h2x);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;

		/* ggs over */

		/* loading ggd term */

		temp = D1nF12(here->ggd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1h2x - r1h2y,
						i1h2x - i1h2y);

		itemp = D1iF12(here->ggd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1h2x - r1h2y,
						i1h2x - i1h2y);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) += itemp;

		/* ggd over */

		/* loading capgs term */

		temp = -ckt->CKTomega *
				D1iF12(here->capgs2,
						r1h1x,
						i1h1x,
						r1h2x,
						i1h2x);

		itemp = ckt->CKTomega *
				D1nF12(here->capgs2,
						r1h1x,
						i1h1x,
						r1h2x,
						i1h2x);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;

		/* capgs over */

		/* loading capgd term */

		temp = -ckt->CKTomega *
				D1iF12(here->capgd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1h2x - r1h2y,
						i1h2x - i1h2y);

		itemp = ckt->CKTomega *
				D1nF12(here->capgd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1h2x - r1h2y,
						i1h2x - i1h2y);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) += itemp;

		/* capgd over */

	/* all done */

      break;
    case D_F1MF2:
	/* x = vgs, y = vds */

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->JFETgateNode)) -
			*(job->r1H1ptr + (here->JFETsourcePrimeNode));
	    i1h1x = *(job->i1H1ptr + (here->JFETgateNode)) -
			*(job->i1H1ptr + (here->JFETsourcePrimeNode));

	    r1h1y = *(job->r1H1ptr + (here->JFETdrainPrimeNode)) -
			*(job->r1H1ptr + (here->JFETsourcePrimeNode));
	    i1h1y = *(job->i1H1ptr + (here->JFETdrainPrimeNode)) -
			*(job->i1H1ptr + (here->JFETsourcePrimeNode));

	    r1hm2x = *(job->r1H2ptr + (here->JFETgateNode)) -
		*(job->r1H2ptr + (here->JFETsourcePrimeNode));
	    i1hm2x = -(*(job->i1H2ptr + (here->JFETgateNode)) -
		*(job->i1H2ptr + (here->JFETsourcePrimeNode)));

	    r1hm2y = *(job->r1H2ptr + (here->JFETdrainPrimeNode)) -
		*(job->r1H2ptr + (here->JFETsourcePrimeNode));
	i1hm2y = -(*(job->i1H2ptr + (here->JFETdrainPrimeNode)) -
		*(job->i1H2ptr + (here->JFETsourcePrimeNode)));

	    /* loading starts here */
	    /* loading cdrain term  */

	    temp = DFnF12(here->cdr_x2,
					here->cdr_y2,
					0.0,
					here->cdr_xy,
					0.0,
					0.0,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					0.0,
					0.0,
					r1hm2x,
					i1hm2x,
					r1hm2y,
					i1hm2y,
					0.0,
					0.0);

	    itemp = DFiF12(here->cdr_x2,
					here->cdr_y2,
					0.0,
					here->cdr_xy,
					0.0,
					0.0,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					0.0,
					0.0,
					r1hm2x,
					i1hm2x,
					r1hm2y,
					i1hm2y,
					0.0,
					0.0);

	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;
	
	    /* cdrain term over */

		/* loading ggs term */

		temp = D1nF12(here->ggs2,
						r1h1x,
						i1h1x,
						r1hm2x,
						i1hm2x);

		itemp = D1iF12(here->ggs2,
						r1h1x,
						i1h1x,
						r1hm2x,
						i1hm2x);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;

		/* ggs over */

		/* loading ggd term */

		temp = D1nF12(here->ggd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
						i1hm2x - i1hm2y);

		itemp = D1iF12(here->ggd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
						i1hm2x - i1hm2y);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) += itemp;

		/* ggd over */

		/* loading capgs term */

		temp = -ckt->CKTomega *
				D1iF12(here->capgs2,
						r1h1x,
						i1h1x,
						r1hm2x,
						i1hm2x);

		itemp = ckt->CKTomega *
				D1nF12(here->capgs2,
						r1h1x,
						i1h1x,
						r1hm2x,
						i1hm2x);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;

		/* capgs over */

		/* loading capgd term */

		temp = -ckt->CKTomega *
				D1iF12(here->capgd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
						i1hm2x - i1hm2y);

		itemp = ckt->CKTomega *
				D1nF12(here->capgd2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
						i1hm2x - i1hm2y);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) += itemp;

		/* capgd over */

	/* all done */

      break;
    case D_2F1MF2:
	/* x = vgs, y = vds */

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->JFETgateNode)) -
			*(job->r1H1ptr + (here->JFETsourcePrimeNode));
	    i1h1x = *(job->i1H1ptr + (here->JFETgateNode)) -
			*(job->i1H1ptr + (here->JFETsourcePrimeNode));

	    r1h1y = *(job->r1H1ptr + (here->JFETdrainPrimeNode)) -
			*(job->r1H1ptr + (here->JFETsourcePrimeNode));
	    i1h1y = *(job->i1H1ptr + (here->JFETdrainPrimeNode)) -
			*(job->i1H1ptr + (here->JFETsourcePrimeNode));

	  r2h11x = *(job->r2H11ptr + (here->JFETgateNode)) -
		*(job->r2H11ptr + (here->JFETsourcePrimeNode));
	  i2h11x = *(job->i2H11ptr + (here->JFETgateNode)) -
		*(job->i2H11ptr + (here->JFETsourcePrimeNode));

	  r2h11y = *(job->r2H11ptr + (here->JFETdrainPrimeNode)) -
		*(job->r2H11ptr + (here->JFETsourcePrimeNode));
	  i2h11y = *(job->i2H11ptr + (here->JFETdrainPrimeNode)) -
		*(job->i2H11ptr + (here->JFETsourcePrimeNode));

    r1hm2x = *(job->r1H2ptr + (here->JFETgateNode)) -
		*(job->r1H2ptr + (here->JFETsourcePrimeNode));
   i1hm2x = -(*(job->i1H2ptr + (here->JFETgateNode)) -
	*(job->i1H2ptr + (here->JFETsourcePrimeNode)));

   r1hm2y = *(job->r1H2ptr + (here->JFETdrainPrimeNode)) -
		*(job->r1H2ptr + (here->JFETsourcePrimeNode));
i1hm2y = -(*(job->i1H2ptr + (here->JFETdrainPrimeNode)) -
		*(job->i1H2ptr + (here->JFETsourcePrimeNode)));

  r2h1m2x = *(job->r2H1m2ptr + (here->JFETgateNode)) -
	*(job->r2H1m2ptr + (here->JFETsourcePrimeNode));
  i2h1m2x = *(job->i2H1m2ptr + (here->JFETgateNode)) -
	*(job->i2H1m2ptr + (here->JFETsourcePrimeNode));

  r2h1m2y = *(job->r2H1m2ptr + (here->JFETdrainPrimeNode)) -
	*(job->r2H1m2ptr + (here->JFETsourcePrimeNode));
i2h1m2y = *(job->i2H1m2ptr + (here->JFETdrainPrimeNode))
	- *(job->i2H1m2ptr + (here->JFETsourcePrimeNode));

	    /* loading starts here */
	    /* loading cdrain term  */

pass.cxx = here->cdr_x2;
pass.cyy = here->cdr_y2;
pass.czz = 0.0;
pass.cxy = here->cdr_xy;
pass.cyz = 0.0;
pass.cxz = 0.0;
pass.cxxx = here->cdr_x3;
pass.cyyy = here->cdr_y3;
pass.czzz = 0.0;
pass.cxxy = here->cdr_x2y;
pass.cxxz = 0.0;
pass.cxyy = here->cdr_xy2;
pass.cyyz = 0.0;
pass.cxzz = 0.0;
pass.cyzz = 0.0;
pass.cxyz = 0.0;
pass.r1h1x = r1h1x;
pass.i1h1x = i1h1x;
pass.r1h1y = r1h1y;
pass.i1h1y = i1h1y;
pass.r1h1z = 0.0;
pass.i1h1z = 0.0;
pass.r1h2x = r1hm2x;
pass.i1h2x = i1hm2x;
pass.r1h2y = r1hm2y;
pass.i1h2y = i1hm2y;
pass.r1h2z = 0.0;
pass.i1h2z = 0.0;
pass.r2h11x = r2h11x;
pass.i2h11x = i2h11x;
pass.r2h11y = r2h11y;
pass.i2h11y = i2h11y;
pass.r2h11z = 0.0;
pass.i2h11z = 0.0;
pass.h2f1f2x = r2h1m2x;
pass.ih2f1f2x = i2h1m2x;
pass.h2f1f2y = r2h1m2y;
pass.ih2f1f2y = i2h1m2y;
pass.h2f1f2z = 0.0;
pass.ih2f1f2z = 0.0;
	    temp = DFn2F12(&pass);

	    itemp = DFi2F12(&pass);

	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;
	
	    /* cdrain term over */

		/* loading ggs term */

		temp = D1n2F12(here->ggs2,
						here->ggs3,
						r1h1x,
						i1h1x,
						r1hm2x,
						i1hm2x,
						r2h11x,
						i2h11x,
						r2h1m2x,
						i2h1m2x);

		itemp = D1i2F12(here->ggs2,
						here->ggs3,
						r1h1x,
						i1h1x,
						r1hm2x,
						i1hm2x,
						r2h11x,
						i2h11x,
						r2h1m2x,
						i2h1m2x);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;

		/* ggs over */

		/* loading ggd term */

		temp = D1n2F12(here->ggd2,
						here->ggd3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
						i1hm2x - i1hm2y,
						r2h11x - r2h11y,
						i2h11x - i2h11y,
						r2h1m2x - r2h1m2y,
					i2h1m2x - i2h1m2y);

		itemp = D1i2F12(here->ggd2,
						here->ggd3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
						i1hm2x - i1hm2y,
						r2h11x - r2h11y,
						i2h11x - i2h11y,
						r2h1m2x - r2h1m2y,
					i2h1m2x - i2h1m2y);



	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) += itemp;

		/* ggd over */

		/* loading capgs term */

		temp = -ckt->CKTomega *
				D1i2F12(here->capgs2,
						here->capgs3,
						r1h1x,
						i1h1x,
						r1hm2x,
						i1hm2x,
						r2h11x,
						i2h11x,
						r2h1m2x,
						i2h1m2x);

		itemp = ckt->CKTomega *
				D1n2F12(here->capgs2,
						here->capgs3,
						r1h1x,
						i1h1x,
						r1hm2x,
						i1hm2x,
						r2h11x,
						i2h11x,
						r2h1m2x,
						i2h1m2x);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETsourcePrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETsourcePrimeNode)) += itemp;

		/* capgs over */

		/* loading capgd term */

		temp = -ckt->CKTomega *
				D1i2F12(here->capgd2,
						here->capgd3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
						i1hm2x - i1hm2y,
						r2h11x - r2h11y,
						i2h11x - i2h11y,
						r2h1m2x - r2h1m2y,
					i2h1m2x - i2h1m2y);

		itemp = ckt->CKTomega *
				D1n2F12(here->capgd2,
						here->capgd3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
						i1hm2x - i1hm2y,
						r2h11x - r2h11y,
						i2h11x - i2h11y,
						r2h1m2x - r2h1m2y,
					i2h1m2x - i2h1m2y);


	    *(ckt->CKTrhs + (here->JFETgateNode)) -= temp;
	    *(ckt->CKTirhs + (here->JFETgateNode)) -= itemp;
	    *(ckt->CKTrhs + (here->JFETdrainPrimeNode)) += temp;
	    *(ckt->CKTirhs + (here->JFETdrainPrimeNode)) += itemp;

		/* capgd over */

	/* all done */

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
