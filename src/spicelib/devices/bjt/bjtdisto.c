/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bjtdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"
#include "bjtdset.h"


/* assuming here that ckt->CKTomega has been initialised to 
 * the correct value
 */
int
BJTdisto(int mode, GENmodel *genmodel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel *) genmodel;
    DISTOAN* job = (DISTOAN*) ckt->CKTcurJob;
    double td;
    DpassStr pass;
    double r1h1x = 0.0, i1h1x = 0.0;
    double r1h1y = 0.0, i1h1y = 0.0;
    double r1h1z = 0.0, i1h1z = 0.0;
    double r1h2x = 0.0, i1h2x = 0.0;
    double r1h2y = 0.0, i1h2y = 0.0;
    double r1h2z = 0.0, i1h2z = 0.0;
    double r1hm2x = 0.0, i1hm2x = 0.0;
    double r1hm2y = 0.0, i1hm2y = 0.0;
    double r1hm2z = 0.0, i1hm2z = 0.0;
    double r2h11x = 0.0, i2h11x = 0.0;
    double r2h11y = 0.0, i2h11y = 0.0;
    double r2h11z = 0.0, i2h11z = 0.0;
    double r2h1m2x = 0.0, i2h1m2x = 0.0;
    double r2h1m2y = 0.0, i2h1m2y = 0.0;
    double r2h1m2z = 0.0, i2h1m2z = 0.0;
    double temp, itemp;
    BJTinstance *here;
#ifdef DISTODEBUG
    double time;
#endif

    if (mode == D_SETUP)
	return(BJTdSetup((GENmodel *)model,ckt));

    if ((mode == D_TWOF1) || (mode == D_THRF1) || 
	(mode == D_F1PF2) || (mode == D_F1MF2) ||
	(mode == D_2F1MF2)) {

	/* loop through all the BJT models */
	for( ; model != NULL; model = BJTnextModel(model)) {
	    td = model->BJTexcessPhaseFactor;

	    /* loop through all the instances of the model */
	    for (here = BJTinstances(model); here != NULL ;
		 here=BJTnextInstance(here)) {

		/* getting Volterra kernels */
		/* until further notice x = vbe, y = vbc, z= vbed */

		r1h1x = *(job->r1H1ptr + (here->BJTbasePrimeNode)) -
		    *(job->r1H1ptr + (here->BJTemitPrimeNode));
		i1h1x = *(job->i1H1ptr + (here->BJTbasePrimeNode)) -
		    *(job->i1H1ptr + (here->BJTemitPrimeNode));

		r1h1y = *(job->r1H1ptr + (here->BJTbasePrimeNode)) -
		    *(job->r1H1ptr + (here->BJTcolPrimeNode));
		i1h1y = *(job->i1H1ptr + (here->BJTbasePrimeNode)) -
		    *(job->i1H1ptr + (here->BJTcolPrimeNode));

		if (td != 0) {

		    temp = job->Domega1 * td;

		    /* multiplying r1h1x by exp(-j omega td) */
		    r1h1z = r1h1x*cos(temp) + i1h1x*sin(temp);
		    i1h1z = i1h1x*cos(temp) - r1h1x*sin(temp);
		}
		else {
		    r1h1z = r1h1x;
		    i1h1z = i1h1x;
		}

		if ((mode == D_F1MF2) || 
		    (mode == D_2F1MF2)) {

		    r1hm2x = *(job->r1H2ptr + (here->BJTbasePrimeNode)) -
			*(job->r1H2ptr + (here->BJTemitPrimeNode));
		    i1hm2x = -(*(job->i1H2ptr + (here->BJTbasePrimeNode)) -
			       *(job->i1H2ptr + (here->BJTemitPrimeNode)));

		    r1hm2y = *(job->r1H2ptr + (here->BJTbasePrimeNode)) -
			*(job->r1H2ptr + (here->BJTcolPrimeNode));
		    i1hm2y = -(*(job->i1H2ptr + (here->BJTbasePrimeNode)) -
			       *(job->i1H2ptr + (here->BJTcolPrimeNode)));

		    if (td != 0) {

			temp = -job->Domega2 * td;
			r1hm2z = r1hm2x*cos(temp) + i1hm2x*sin(temp);
			i1hm2z = i1hm2x*cos(temp) - r1hm2x*sin(temp);
		    }
		    else {
			r1hm2z = r1hm2x;
			i1hm2z = i1hm2x;
		    }
		}
		if ((mode == D_THRF1) || (mode == D_2F1MF2)){


		    r2h11x = *(job->r2H11ptr + (here->BJTbasePrimeNode)) -
			*(job->r2H11ptr + (here->BJTemitPrimeNode));
		    i2h11x = *(job->i2H11ptr + (here->BJTbasePrimeNode)) -
			*(job->i2H11ptr + (here->BJTemitPrimeNode));

		    r2h11y = *(job->r2H11ptr + (here->BJTbasePrimeNode)) -
			*(job->r2H11ptr + (here->BJTcolPrimeNode));
		    i2h11y = *(job->i2H11ptr + (here->BJTbasePrimeNode)) -
			*(job->i2H11ptr + (here->BJTcolPrimeNode));

		    if (td != 0) {
			temp = 2*job->Domega1* td ;
			r2h11z = r2h11x*cos(temp) + i2h11x*sin(temp);
			i2h11z = i2h11x*cos(temp) - r2h11x*sin(temp);
		    }
		    else {
			r2h11z = r2h11x;
			i2h11z = i2h11x;
		    }
		}

		if (mode == D_2F1MF2){

		    r2h1m2x = *(job->r2H1m2ptr + (here->BJTbasePrimeNode)) -
			*(job->r2H1m2ptr + (here->BJTemitPrimeNode));
		    i2h1m2x = *(job->i2H1m2ptr + (here->BJTbasePrimeNode))
			- *(job->i2H1m2ptr + (here->BJTemitPrimeNode));

		    r2h1m2y = *(job->r2H1m2ptr + (here->BJTbasePrimeNode)) -
			*(job->r2H1m2ptr + (here->BJTcolPrimeNode));
		    i2h1m2y = *(job->i2H1m2ptr + (here->BJTbasePrimeNode))
			- *(job->i2H1m2ptr + (here->BJTcolPrimeNode));

		    if (td != 0) {

			temp = (job->Domega1 - job->Domega2) * td;
			r2h1m2z = r2h1m2x*cos(temp) 
			    + i2h1m2x*sin(temp);
			i2h1m2z = i2h1m2x*cos(temp) 
			    - r2h1m2x*sin(temp);
		    }
		    else {
			r2h1m2z = r2h1m2x;
			i2h1m2z = i2h1m2x;
		    }
		}
		if (mode == D_F1PF2){

		    r1h2x = *(job->r1H2ptr + (here->BJTbasePrimeNode)) -
			*(job->r1H2ptr + (here->BJTemitPrimeNode));
		    i1h2x = *(job->i1H2ptr + (here->BJTbasePrimeNode)) -
			*(job->i1H2ptr + (here->BJTemitPrimeNode));

		    r1h2y = *(job->r1H2ptr + (here->BJTbasePrimeNode)) -
			*(job->r1H2ptr + (here->BJTcolPrimeNode));
		    i1h2y = *(job->i1H2ptr + (here->BJTbasePrimeNode)) -
			*(job->i1H2ptr + (here->BJTcolPrimeNode));


		    if (td != 0) {
			temp = job->Domega2 * td;
			r1h2z = r1h2x*cos(temp) + i1h2x*sin(temp);
			i1h2z = i1h2x*cos(temp) - r1h2x*sin(temp);
		    }
		    else {
			r1h2z = r1h2x;
			i1h2z = i1h2x;
		    }
		}
		/* loading starts here */

		switch (mode) {
		case D_TWOF1:


		    /* ic term */

#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds();
#endif
		    temp = DFn2F1( here->ic_x2,
				   here->ic_y2,
				   here->ic_w2,
				   here->ic_xy,
				   here->ic_yw,
				   here->ic_xw,
				   r1h1x,
				   i1h1x,
				   r1h1y,
				   i1h1y,
				   r1h1z,
				   i1h1z);

		    itemp = DFi2F1( here->ic_x2,
				    here->ic_y2,
				    here->ic_w2,
				    here->ic_xy,
				    here->ic_yw,
				    here->ic_xw,
				    r1h1x,
				    i1h1x,
				    r1h1y,
				    i1h1y,
				    r1h1z,
				    i1h1z);
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds() - time;
		    printf("Time for DFn2F1: %g seconds \n", time);
#endif

		    *(ckt->CKTrhs + here->BJTcolPrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* finish ic term */
		    /* loading ib term */
		    /* x and y still the same */
		    temp = DFn2F1( here->ib_x2,
				   here->ib_y2,
				   0.0,
				   here->ib_xy,
				   0.0,
				   0.0,
				   r1h1x,
				   i1h1x,
				   r1h1y,
				   i1h1y,
				   0.0,
				   0.0);

		    itemp = DFi2F1( here->ib_x2,
				    here->ib_y2,
				    0.0,
				    here->ib_xy,
				    0.0,
				    0.0,
				    r1h1x,
				    i1h1x,
				    r1h1y,
				    i1h1y,
				    0.0,
				    0.0);

		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* ib term over */
		    /* loading ibb term */
		    /* now x = vbe, y = vbc, z = vbb */
		    if ( !((model->BJTminBaseResist == 0.0) &&
			   (model->BJTbaseResist == model->BJTminBaseResist))) {

			r1h1z = *(job->r1H1ptr + (here->BJTbaseNode)) -
			    *(job->r1H1ptr + (here->BJTbasePrimeNode));
			i1h1z = *(job->i1H1ptr + (here->BJTbaseNode)) -
			    *(job->i1H1ptr + (here->BJTbasePrimeNode));

			temp = DFn2F1( here->ibb_x2,
				       here->ibb_y2,
				       here->ibb_z2,
				       here->ibb_xy,
				       here->ibb_yz,
				       here->ibb_xz,
				       r1h1x,
				       i1h1x,
				       r1h1y,
				       i1h1y,
				       r1h1z,
				       i1h1z);

			itemp = DFi2F1( here->ibb_x2,
					here->ibb_y2,
					here->ibb_z2,
					here->ibb_xy,
					here->ibb_yz,
					here->ibb_xz,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					r1h1z,
					i1h1z);

			*(ckt->CKTrhs + here->BJTbaseNode) -= temp;
			*(ckt->CKTirhs + here->BJTbaseNode) -= itemp;
			*(ckt->CKTrhs + here->BJTbasePrimeNode) += temp;
			*(ckt->CKTirhs + here->BJTbasePrimeNode) += itemp;
		    }

		    /* ibb term over */
		    /* loading qbe term */
		    /* x = vbe, y = vbc, z not used */
		    /* (have to multiply by j omega for charge storage 
		     * elements to get the current)
		     */

		    temp = - ckt->CKTomega*
			DFi2F1( here->qbe_x2,
				here->qbe_y2,
				0.0,
				here->qbe_xy,
				0.0,
				0.0,
				r1h1x,
				i1h1x,
				r1h1y,
				i1h1y,
				0.0,
				0.0);

		    itemp = ckt->CKTomega*
			DFn2F1( here->qbe_x2,
				here->qbe_y2,
				0.0,
				here->qbe_xy,
				0.0,
				0.0,
				r1h1x,
				i1h1x,
				r1h1y,
				i1h1y,
				0.0,
				0.0);

		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* qbe term over */
		    /* loading qbx term */
		    /* z = vbx= vb - vcPrime */

		    r1h1z = r1h1z + r1h1y;
		    i1h1z = i1h1z + i1h1y;
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds();
#endif
		    temp = - ckt->CKTomega *
			D1i2F1(here->capbx2,
			       r1h1z,
			       i1h1z);
		    itemp = ckt->CKTomega *
			D1n2F1(here->capbx2,
			       r1h1z,
			       i1h1z);
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds() - time;
		    printf("Time for D1n2F1: %g seconds \n", time);
#endif


		    *(ckt->CKTrhs + here->BJTbaseNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbaseNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qbx term over */

		    /* loading qbc term */

		    temp = - ckt->CKTomega *
			D1i2F1(here->capbc2,
			       r1h1y,
			       i1h1y);
		    itemp = ckt->CKTomega *
			D1n2F1(here->capbc2,
			       r1h1y,
			       i1h1y);


		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qbc term over */


		    /* loading qsc term */
		    /* z = vsc */



		    r1h1z = *(job->r1H1ptr + (here->BJTsubstNode)) -
			*(job->r1H1ptr + (here->BJTcolPrimeNode));
		    i1h1z = *(job->i1H1ptr + (here->BJTsubstNode)) -
			*(job->i1H1ptr + (here->BJTcolPrimeNode));

		    temp = - ckt->CKTomega *
			D1i2F1(here->capsc2,
			       r1h1z,
			       i1h1z);
		    itemp = ckt->CKTomega *
			D1n2F1(here->capsc2,
			       r1h1z,
			       i1h1z);


		    *(ckt->CKTrhs + here->BJTsubstNode) -= temp;
		    *(ckt->CKTirhs + here->BJTsubstNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qsc term over */


		    break;

		case D_THRF1:
		    /* ic term */

#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds();
#endif
		    temp = DFn3F1( here->ic_x2,
				   here->ic_y2,
				   here->ic_w2,
				   here->ic_xy,
				   here->ic_yw,
				   here->ic_xw,
				   here->ic_x3,
				   here->ic_y3,
				   here->ic_w3,
				   here->ic_x2y,
				   here->ic_x2w,
				   here->ic_xy2,
				   here->ic_y2w,
				   here->ic_xw2,
				   here->ic_yw2,
				   here->ic_xyw,
				   r1h1x,
				   i1h1x,
				   r1h1y,
				   i1h1y,
				   r1h1z,
				   i1h1z,
				   r2h11x,
				   i2h11x,
				   r2h11y,
				   i2h11y,
				   r2h11z,
				   i2h11z);

		    itemp = DFi3F1( here->ic_x2,
				    here->ic_y2,
				    here->ic_w2,
				    here->ic_xy,
				    here->ic_yw,
				    here->ic_xw,
				    here->ic_x3,
				    here->ic_y3,
				    here->ic_w3,
				    here->ic_x2y,
				    here->ic_x2w,
				    here->ic_xy2,
				    here->ic_y2w,
				    here->ic_xw2,
				    here->ic_yw2,
				    here->ic_xyw,
				    r1h1x,
				    i1h1x,
				    r1h1y,
				    i1h1y,
				    r1h1z,
				    i1h1z,
				    r2h11x,
				    i2h11x,
				    r2h11y,
				    i2h11y,
				    r2h11z,
				    i2h11z);
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds() - time;
		    printf("Time for DFn3F1: %g seconds \n", time);
#endif

		    *(ckt->CKTrhs + here->BJTcolPrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* finish ic term */
		    /* loading ib term */
		    /* x and y still the same */
		    temp = DFn3F1( here->ib_x2,
				   here->ib_y2,
				   0.0,
				   here->ib_xy,
				   0.0,
				   0.0,
				   here->ib_x3,
				   here->ib_y3,
				   0.0,
				   here->ib_x2y,
				   0.0,
				   here->ib_xy2,
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

		    itemp = DFi3F1( here->ib_x2,
				    here->ib_y2,
				    0.0,
				    here->ib_xy,
				    0.0,
				    0.0,
				    here->ib_x3,
				    here->ib_y3,
				    0.0,
				    here->ib_x2y,
				    0.0,
				    here->ib_xy2,
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

		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* ib term over */
		    /* loading ibb term */
		    if ( !((model->BJTminBaseResist == 0.0) &&
			   (model->BJTbaseResist == model->BJTminBaseResist))) {

			/* now x = vbe, y = vbc, z = vbb */
			r1h1z = *(job->r1H1ptr + (here->BJTbaseNode)) -
			    *(job->r1H1ptr + (here->BJTbasePrimeNode));
			i1h1z = *(job->i1H1ptr + (here->BJTbaseNode)) -
			    *(job->i1H1ptr + (here->BJTbasePrimeNode));

			r2h11z = *(job->r2H11ptr + (here->BJTbaseNode)) -
			    *(job->r2H11ptr + (here->BJTbasePrimeNode));
			i2h11z = *(job->i2H11ptr + (here->BJTbaseNode)) -
			    *(job->i2H11ptr + (here->BJTbasePrimeNode));

			temp = DFn3F1( here->ibb_x2,
				       here->ibb_y2,
				       here->ibb_z2,
				       here->ibb_xy,
				       here->ibb_yz,
				       here->ibb_xz,
				       here->ibb_x3,
				       here->ibb_y3,
				       here->ibb_z3,
				       here->ibb_x2y,
				       here->ibb_x2z,
				       here->ibb_xy2,
				       here->ibb_y2z,
				       here->ibb_xz2,
				       here->ibb_yz2,
				       here->ibb_xyz,
				       r1h1x,
				       i1h1x,
				       r1h1y,
				       i1h1y,
				       r1h1z,
				       i1h1z,
				       r2h11x,
				       i2h11x,
				       r2h11y,
				       i2h11y,
				       r2h11z,
				       i2h11z);

			itemp = DFi3F1( here->ibb_x2,
					here->ibb_y2,
					here->ibb_z2,
					here->ibb_xy,
					here->ibb_yz,
					here->ibb_xz,
					here->ibb_x3,
					here->ibb_y3,
					here->ibb_z3,
					here->ibb_x2y,
					here->ibb_x2z,
					here->ibb_xy2,
					here->ibb_y2z,
					here->ibb_xz2,
					here->ibb_yz2,
					here->ibb_xyz,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					r1h1z,
					i1h1z,
					r2h11x,
					i2h11x,
					r2h11y,
					i2h11y,
					r2h11z,
					i2h11z);

			*(ckt->CKTrhs + here->BJTbaseNode) -= temp;
			*(ckt->CKTirhs + here->BJTbaseNode) -= itemp;
			*(ckt->CKTrhs + here->BJTbasePrimeNode) += temp;
			*(ckt->CKTirhs + here->BJTbasePrimeNode) += itemp;

		    }
		    /* ibb term over */
		    /* loading qbe term */
		    /* x = vbe, y = vbc, z not used */
		    /* (have to multiply by j omega for charge storage 
		     * elements to get the current)
		     */

		    temp = - ckt->CKTomega*
			DFi3F1( here->qbe_x2,
				here->qbe_y2,
				0.0,
				here->qbe_xy,
				0.0,
				0.0,
				here->qbe_x3,
				here->qbe_y3,
				0.0,
				here->qbe_x2y,
				0.0,
				here->qbe_xy2,
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

		    itemp = ckt->CKTomega*
			DFn3F1( here->qbe_x2,
				here->qbe_y2,
				0.0,
				here->qbe_xy,
				0.0,
				0.0,
				here->qbe_x3,
				here->qbe_y3,
				0.0,
				here->qbe_x2y,
				0.0,
				here->qbe_xy2,
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

		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* qbe term over */
		    /* loading qbx term */
		    /* z = vbx= vb - vcPrime */

		    r1h1z = r1h1z + r1h1y;
		    i1h1z = i1h1z + i1h1y;
		    r2h11z = r2h11z + r2h11y;
		    i2h11z = i2h11z + i2h11y;
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds();
#endif
		    temp = - ckt->CKTomega *
			D1i3F1(here->capbx2,
			       here->capbx3,
			       r1h1z,
			       i1h1z,
			       r2h11z,
			       i2h11z);
		    itemp = ckt->CKTomega *
			D1n3F1(here->capbx2,
			       here->capbx3,
			       r1h1z,
			       i1h1z,
			       r2h11z,
			       i2h11z);
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds() - time;
		    printf("Time for D1n3F1: %g seconds \n", time);
#endif


		    *(ckt->CKTrhs + here->BJTbaseNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbaseNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qbx term over */

		    /* loading qbc term */

		    temp = - ckt->CKTomega *
			D1i3F1(here->capbc2,
			       here->capbc3,
			       r1h1y,
			       i1h1y,
			       r2h11y,
			       i2h11y);
		    itemp = ckt->CKTomega *
			D1n3F1(here->capbc2,
			       here->capbc3,
			       r1h1y,
			       i1h1y,
			       r2h11y,
			       i2h11y);



		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qbc term over */


		    /* loading qsc term */
		    /* z = vsc */



		    r1h1z = *(job->r1H1ptr + (here->BJTsubstNode)) -
			*(job->r1H1ptr + (here->BJTcolPrimeNode));
		    i1h1z = *(job->i1H1ptr + (here->BJTsubstNode)) -
			*(job->i1H1ptr + (here->BJTcolPrimeNode));

		    r2h11z = *(job->r2H11ptr + (here->BJTsubstNode)) -
			*(job->r2H11ptr + (here->BJTcolPrimeNode));
		    i2h11z = *(job->i2H11ptr + (here->BJTsubstNode)) -
			*(job->i2H11ptr + (here->BJTcolPrimeNode));

		    temp = - ckt->CKTomega *
			D1i3F1(here->capsc2,
			       here->capsc3,
			       r1h1z,
			       i1h1z,
			       r2h11z,
			       i2h11z);

		    itemp = ckt->CKTomega *
			D1n3F1(here->capsc2,
			       here->capsc3,
			       r1h1z,
			       i1h1z,
			       r2h11z,
			       i2h11z);


		    *(ckt->CKTrhs + here->BJTsubstNode) -= temp;
		    *(ckt->CKTirhs + here->BJTsubstNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qsc term over */


		    break;
		case D_F1PF2:
		    /* ic term */

		    temp = DFnF12( here->ic_x2,
				   here->ic_y2,
				   here->ic_w2,
				   here->ic_xy,
				   here->ic_yw,
				   here->ic_xw,
				   r1h1x,
				   i1h1x,
				   r1h1y,
				   i1h1y,
				   r1h1z,
				   i1h1z,
				   r1h2x,
				   i1h2x,
				   r1h2y,
				   i1h2y,
				   r1h2z,
				   i1h2z);

		    itemp = DFiF12( here->ic_x2,
				    here->ic_y2,
				    here->ic_w2,
				    here->ic_xy,
				    here->ic_yw,
				    here->ic_xw,
				    r1h1x,
				    i1h1x,
				    r1h1y,
				    i1h1y,
				    r1h1z,
				    i1h1z,
				    r1h2x,
				    i1h2x,
				    r1h2y,
				    i1h2y,
				    r1h2z,
				    i1h2z);

		    *(ckt->CKTrhs + here->BJTcolPrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* finish ic term */
		    /* loading ib term */
		    /* x and y still the same */
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds();
#endif
		    temp = DFnF12( here->ib_x2,
				   here->ib_y2,
				   0.0,
				   here->ib_xy,
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

		    itemp = DFiF12( here->ib_x2,
				    here->ib_y2,
				    0.0,
				    here->ib_xy,
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
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds() - time;
		    printf("Time for DFnF12: %g seconds \n", time);
#endif

		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* ib term over */
		    /* loading ibb term */
		    if ( !((model->BJTminBaseResist == 0.0) &&
			   (model->BJTbaseResist == model->BJTminBaseResist))) {

			/* now x = vbe, y = vbc, z = vbb */
			r1h1z = *(job->r1H1ptr + (here->BJTbaseNode)) -
			    *(job->r1H1ptr + (here->BJTbasePrimeNode));
			i1h1z = *(job->i1H1ptr + (here->BJTbaseNode)) -
			    *(job->i1H1ptr + (here->BJTbasePrimeNode));

			r1h2z = *(job->r1H2ptr + (here->BJTbaseNode)) -
			    *(job->r1H2ptr + (here->BJTbasePrimeNode));
			i1h2z = *(job->i1H2ptr + (here->BJTbaseNode)) -
			    *(job->i1H2ptr + (here->BJTbasePrimeNode));

			temp = DFnF12( here->ibb_x2,
				       here->ibb_y2,
				       here->ibb_z2,
				       here->ibb_xy,
				       here->ibb_yz,
				       here->ibb_xz,
				       r1h1x,
				       i1h1x,
				       r1h1y,
				       i1h1y,
				       r1h1z,
				       i1h1z,
				       r1h2x,
				       i1h2x,
				       r1h2y,
				       i1h2y,
				       r1h2z,
				       i1h2z);

			itemp = DFiF12( here->ibb_x2,
					here->ibb_y2,
					here->ibb_z2,
					here->ibb_xy,
					here->ibb_yz,
					here->ibb_xz,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					r1h1z,
					i1h1z,
					r1h2x,
					i1h2x,
					r1h2y,
					i1h2y,
					r1h2z,
					i1h2z);

			*(ckt->CKTrhs + here->BJTbaseNode) -= temp;
			*(ckt->CKTirhs + here->BJTbaseNode) -= itemp;
			*(ckt->CKTrhs + here->BJTbasePrimeNode) += temp;
			*(ckt->CKTirhs + here->BJTbasePrimeNode) += itemp;

		    }
		    /* ibb term over */
		    /* loading qbe term */
		    /* x = vbe, y = vbc, z not used */
		    /* (have to multiply by j omega for charge storage 
		     * elements - to get the current)
		     */

		    temp = - ckt->CKTomega*
			DFiF12( here->qbe_x2,
				here->qbe_y2,
				0.0,
				here->qbe_xy,
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

		    itemp = ckt->CKTomega*
			DFnF12( here->qbe_x2,
				here->qbe_y2,
				0.0,
				here->qbe_xy,
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

		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* qbe term over */
		    /* loading qbx term */
		    /* z = vbx= vb - vcPrime */

		    r1h1z = r1h1z + r1h1y;
		    i1h1z = i1h1z + i1h1y;
		    r1h2z = r1h2z + r1h2y;
		    i1h2z = i1h2z + i1h2y;
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds();
#endif
		    temp = - ckt->CKTomega *
			D1iF12(here->capbx2,
			       r1h1z,
			       i1h1z,
			       r1h2z,
			       i1h2z);
		    itemp = ckt->CKTomega *
			D1nF12(here->capbx2,
			       r1h1z,
			       i1h1z,
			       r1h2z,
			       i1h2z);
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds() - time;
		    printf("Time for D1nF12: %g seconds \n", time);
#endif


		    *(ckt->CKTrhs + here->BJTbaseNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbaseNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qbx term over */

		    /* loading qbc term */

		    temp = - ckt->CKTomega *
			D1iF12(here->capbc2,
			       r1h1y,
			       i1h1y,
			       r1h2y,
			       i1h2y);
		    itemp = ckt->CKTomega *
			D1nF12(here->capbc2,
			       r1h1y,
			       i1h1y,
			       r1h2y,
			       i1h2y);


		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qbc term over */


		    /* loading qsc term */
		    /* z = vsc */



		    r1h1z = *(job->r1H1ptr + (here->BJTsubstNode)) -
			*(job->r1H1ptr + (here->BJTcolPrimeNode));
		    i1h1z = *(job->i1H1ptr + (here->BJTsubstNode)) -
			*(job->i1H1ptr + (here->BJTcolPrimeNode));
		    r1h2z = *(job->r1H2ptr + (here->BJTsubstNode)) -
			*(job->r1H2ptr + (here->BJTcolPrimeNode));
		    i1h2z = *(job->i1H2ptr + (here->BJTsubstNode)) -
			*(job->i1H2ptr + (here->BJTcolPrimeNode));

		    temp = - ckt->CKTomega *
			D1iF12(here->capsc2,
			       r1h1z,
			       i1h1z,
			       r1h2z,
			       i1h2z);
		    itemp = ckt->CKTomega *
			D1nF12(here->capsc2,
			       r1h1z,
			       i1h1z,
			       r1h2z,
			       i1h2z);


		    *(ckt->CKTrhs + here->BJTsubstNode) -= temp;
		    *(ckt->CKTirhs + here->BJTsubstNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qsc term over */


		    break;
		case D_F1MF2:
		    /* ic term */

		    temp = DFnF12( here->ic_x2,
				   here->ic_y2,
				   here->ic_w2,
				   here->ic_xy,
				   here->ic_yw,
				   here->ic_xw,
				   r1h1x,
				   i1h1x,
				   r1h1y,
				   i1h1y,
				   r1h1z,
				   i1h1z,
				   r1hm2x,
				   i1hm2x,
				   r1hm2y,
				   i1hm2y,
				   r1hm2z,
				   i1hm2z);

		    itemp = DFiF12( here->ic_x2,
				    here->ic_y2,
				    here->ic_w2,
				    here->ic_xy,
				    here->ic_yw,
				    here->ic_xw,
				    r1h1x,
				    i1h1x,
				    r1h1y,
				    i1h1y,
				    r1h1z,
				    i1h1z,
				    r1hm2x,
				    i1hm2x,
				    r1hm2y,
				    i1hm2y,
				    r1hm2z,
				    i1hm2z);

		    *(ckt->CKTrhs + here->BJTcolPrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* finish ic term */
		    /* loading ib term */
		    /* x and y still the same */
		    temp = DFnF12( here->ib_x2,
				   here->ib_y2,
				   0.0,
				   here->ib_xy,
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

		    itemp = DFiF12( here->ib_x2,
				    here->ib_y2,
				    0.0,
				    here->ib_xy,
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

		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* ib term over */
		    /* loading ibb term */
		    if ( !((model->BJTminBaseResist == 0.0) &&
			   (model->BJTbaseResist == model->BJTminBaseResist))) {

			/* now x = vbe, y = vbc, z = vbb */
			r1h1z = *(job->r1H1ptr + (here->BJTbaseNode)) -
			    *(job->r1H1ptr + (here->BJTbasePrimeNode));
			i1h1z = *(job->i1H1ptr + (here->BJTbaseNode)) -
			    *(job->i1H1ptr + (here->BJTbasePrimeNode));

			r1hm2z = *(job->r1H2ptr + (here->BJTbaseNode)) -
			    *(job->r1H2ptr + (here->BJTbasePrimeNode));
			i1hm2z = *(job->i1H2ptr + (here->BJTbaseNode)) -
			    *(job->i1H2ptr + (here->BJTbasePrimeNode));

			temp = DFnF12( here->ibb_x2,
				       here->ibb_y2,
				       here->ibb_z2,
				       here->ibb_xy,
				       here->ibb_yz,
				       here->ibb_xz,
				       r1h1x,
				       i1h1x,
				       r1h1y,
				       i1h1y,
				       r1h1z,
				       i1h1z,
				       r1hm2x,
				       i1hm2x,
				       r1hm2y,
				       i1hm2y,
				       r1hm2z,
				       i1hm2z);

			itemp = DFiF12( here->ibb_x2,
					here->ibb_y2,
					here->ibb_z2,
					here->ibb_xy,
					here->ibb_yz,
					here->ibb_xz,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					r1h1z,
					i1h1z,
					r1hm2x,
					i1hm2x,
					r1hm2y,
					i1hm2y,
					r1hm2z,
					i1hm2z);

			*(ckt->CKTrhs + here->BJTbaseNode) -= temp;
			*(ckt->CKTirhs + here->BJTbaseNode) -= itemp;
			*(ckt->CKTrhs + here->BJTbasePrimeNode) += temp;
			*(ckt->CKTirhs + here->BJTbasePrimeNode) += itemp;
		    }

		    /* ibb term over */
		    /* loading qbe term */
		    /* x = vbe, y = vbc, z not used */
		    /* (have to multiply by j omega for charge storage 
		     * elements - to get the current)
		     */

		    temp = - ckt->CKTomega*
			DFiF12( here->qbe_x2,
				here->qbe_y2,
				0.0,
				here->qbe_xy,
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

		    itemp = ckt->CKTomega*
			DFnF12( here->qbe_x2,
				here->qbe_y2,
				0.0,
				here->qbe_xy,
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

		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		    /* qbe term over */
		    /* loading qbx term */
		    /* z = vbx= vb - vcPrime */

		    r1h1z = r1h1z + r1h1y;
		    i1h1z = i1h1z + i1h1y;
		    r1hm2z = r1hm2z + r1hm2y;
		    i1hm2z = i1hm2z + i1hm2y;
		    temp = - ckt->CKTomega *
			D1iF12(here->capbx2,
			       r1h1z,
			       i1h1z,
			       r1hm2z,
			       i1hm2z);
		    itemp = ckt->CKTomega *
			D1nF12(here->capbx2,
			       r1h1z,
			       i1h1z,
			       r1hm2z,
			       i1hm2z);


		    *(ckt->CKTrhs + here->BJTbaseNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbaseNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qbx term over */

		    /* loading qbc term */

		    temp = - ckt->CKTomega *
			D1iF12(here->capbc2,
			       r1h1y,
			       i1h1y,
			       r1hm2y,
			       i1hm2y);
		    itemp = ckt->CKTomega *
			D1nF12(here->capbc2,
			       r1h1y,
			       i1h1y,
			       r1hm2y,
			       i1hm2y);


		    *(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qbc term over */


		    /* loading qsc term */
		    /* z = vsc */



		    r1h1z = *(job->r1H1ptr + (here->BJTsubstNode)) -
			*(job->r1H1ptr + (here->BJTcolPrimeNode));
		    i1h1z = *(job->i1H1ptr + (here->BJTsubstNode)) -
			*(job->i1H1ptr + (here->BJTcolPrimeNode));
		    r1hm2z = *(job->r1H2ptr + (here->BJTsubstNode)) -
			*(job->r1H2ptr + (here->BJTcolPrimeNode));
		    i1hm2z = *(job->i1H2ptr + (here->BJTsubstNode)) -
			*(job->i1H2ptr + (here->BJTcolPrimeNode));

		    temp = - ckt->CKTomega *
			D1iF12(here->capsc2,
			       r1h1z,
			       i1h1z,
			       r1hm2z,
			       i1hm2z);
		    itemp = ckt->CKTomega *
			D1nF12(here->capsc2,
			       r1h1z,
			       i1h1z,
			       r1hm2z,
			       i1hm2z);


		    *(ckt->CKTrhs + here->BJTsubstNode) -= temp;
		    *(ckt->CKTirhs + here->BJTsubstNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		    /* qsc term over */


		    break;
		case D_2F1MF2:
		    /* ic term */

		{
		    pass.cxx =   here->ic_x2;
		    pass.cyy =   here->ic_y2;
		    pass.czz =   here->ic_w2;
		    pass.cxy =   here->ic_xy;
		    pass.cyz =   here->ic_yw;
		    pass.cxz =   here->ic_xw;
		    pass.cxxx =   here->ic_x3;
		    pass.cyyy =   here->ic_y3;
		    pass.czzz =   here->ic_w3;
		    pass.cxxy =   here->ic_x2y;
		    pass.cxxz =   here->ic_x2w;
		    pass.cxyy =   here->ic_xy2;
		    pass.cyyz =   here->ic_y2w;
		    pass.cxzz =   here->ic_xw2;
		    pass.cyzz =   here->ic_yw2;
		    pass.cxyz =   here->ic_xyw;
		    pass.r1h1x =   r1h1x;
		    pass.i1h1x =   i1h1x;
		    pass.r1h1y =   r1h1y;
		    pass.i1h1y =   i1h1y;
		    pass.r1h1z =   r1h1z;
		    pass.i1h1z =   i1h1z;
		    pass.r1h2x =   r1hm2x;
		    pass.i1h2x =   i1hm2x;
		    pass.r1h2y =   r1hm2y;
		    pass.i1h2y =   i1hm2y;
		    pass.r1h2z =   r1hm2z;
		    pass.i1h2z =   i1hm2z;
		    pass.r2h11x =   r2h11x;
		    pass.i2h11x =  i2h11x;
		    pass.r2h11y =   r2h11y;
		    pass.i2h11y =  i2h11y;
		    pass.r2h11z =   r2h11z;
		    pass.i2h11z =  i2h11z;
		    pass.h2f1f2x =   r2h1m2x;
		    pass.ih2f1f2x =  i2h1m2x;
		    pass.h2f1f2y =   r2h1m2y;
		    pass.ih2f1f2y =  i2h1m2y;
		    pass.h2f1f2z =  r2h1m2z;
		    pass.ih2f1f2z =   i2h1m2z;
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds();
#endif
		    temp = DFn2F12(&pass);


		    itemp = DFi2F12(&pass);
#ifdef D_DBG_SMALLTIMES
		    time = SPfrontEnd->IFseconds() - time;
		    printf("Time for DFn2F12: %g seconds \n", time);
#endif
		}

		*(ckt->CKTrhs + here->BJTcolPrimeNode) -= temp;
		*(ckt->CKTirhs + here->BJTcolPrimeNode) -= itemp;
		*(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		*(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		/* finish ic term */
		/* loading ib term */
		/* x and y still the same */
		{
		    pass.cxx = here->ib_x2;
		    pass.cyy = here->ib_y2;
		    pass.czz = 0.0;
		    pass.cxy = here->ib_xy;
		    pass.cyz = 0.0;
		    pass.cxz = 0.0;
		    pass.cxxx = here->ib_x3;
		    pass.cyyy = here->ib_y3;
		    pass.czzz = 0.0;
		    pass.cxxy = here->ib_x2y;
		    pass.cxxz = 0.0;
		    pass.cxyy = here->ib_xy2;
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
		}

		*(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		*(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		*(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		*(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		/* ib term over */
		/* loading ibb term */
		if ( !((model->BJTminBaseResist == 0.0) &&
		       (model->BJTbaseResist == model->BJTminBaseResist))) {

		    /* now x = vbe, y = vbc, z = vbb */
		    r1h1z = *(job->r1H1ptr + (here->BJTbaseNode)) -
			*(job->r1H1ptr + (here->BJTbasePrimeNode));
		    i1h1z = *(job->i1H1ptr + (here->BJTbaseNode)) -
			*(job->i1H1ptr + (here->BJTbasePrimeNode));

		    r1hm2z = *(job->r1H2ptr + (here->BJTbaseNode)) -
			*(job->r1H2ptr + (here->BJTbasePrimeNode));
		    i1hm2z = -(*(job->i1H2ptr + (here->BJTbaseNode)) -
			       *(job->i1H2ptr + (here->BJTbasePrimeNode)));

		    r2h11z = *(job->r2H11ptr + (here->BJTbaseNode)) -
			*(job->r2H11ptr + (here->BJTbasePrimeNode));
		    i2h11z = *(job->i2H11ptr + (here->BJTbaseNode)) -
			*(job->i2H11ptr + (here->BJTbasePrimeNode));

		    r2h1m2z = *(job->r2H1m2ptr + (here->BJTbaseNode)) -
			*(job->r2H1m2ptr + (here->BJTbasePrimeNode));
		    i2h1m2z = *(job->i2H1m2ptr + (here->BJTbaseNode)) -
			*(job->i2H1m2ptr + (here->BJTbasePrimeNode));

		    {
			pass.cxx = here->ibb_x2;
			pass.cyy = here->ibb_y2;
			pass.czz = here->ibb_z2;
			pass.cxy = here->ibb_xy;
			pass.cyz = here->ibb_yz;
			pass.cxz = here->ibb_xz;
			pass.cxxx = here->ibb_x3;
			pass.cyyy = here->ibb_y3;
			pass.czzz = here->ibb_z3;
			pass.cxxy = here->ibb_x2y;
			pass.cxxz = here->ibb_x2z;
			pass.cxyy = here->ibb_xy2;
			pass.cyyz = here->ibb_y2z;
			pass.cxzz = here->ibb_xz2;
			pass.cyzz = here->ibb_yz2;
			pass.cxyz = here->ibb_xyz;
			pass.r1h1x = r1h1x;
			pass.i1h1x = i1h1x;
			pass.r1h1y = r1h1y;
			pass.i1h1y = i1h1y;
			pass.r1h1z = r1h1z;
			pass.i1h1z = i1h1z;
			pass.r1h2x = r1hm2x;
			pass.i1h2x = i1hm2x;
			pass.r1h2y = r1hm2y;
			pass.i1h2y = i1hm2y;
			pass.r1h2z = r1hm2z;
			pass.i1h2z = i1hm2z;
			pass.r2h11x = r2h11x;
			pass.i2h11x = i2h11x;
			pass.r2h11y = r2h11y;
			pass.i2h11y = i2h11y;
			pass.r2h11z = r2h11z;
			pass.i2h11z = i2h11z;
			pass.h2f1f2x = r2h1m2x;
			pass.ih2f1f2x = i2h1m2x;
			pass.h2f1f2y = r2h1m2y;
			pass.ih2f1f2y = i2h1m2y;
			pass.h2f1f2z = r2h1m2z;
			pass.ih2f1f2z = i2h1m2z;
			temp = DFn2F12(&pass);

			itemp = DFi2F12(&pass);
		    }

		    *(ckt->CKTrhs + here->BJTbaseNode) -= temp;
		    *(ckt->CKTirhs + here->BJTbaseNode) -= itemp;
		    *(ckt->CKTrhs + here->BJTbasePrimeNode) += temp;
		    *(ckt->CKTirhs + here->BJTbasePrimeNode) += itemp;

		}
		/* ibb term over */
		/* loading qbe term */
		/* x = vbe, y = vbc, z not used */
		/* (have to multiply by j omega for charge storage 
		 * elements to get the current)
		 */

		{
		    pass.cxx = here->qbe_x2;
		    pass.cyy = here->qbe_y2;
		    pass.czz = 0.0;
		    pass.cxy = here->qbe_xy;
		    pass.cyz = 0.0;
		    pass.cxz = 0.0;
		    pass.cxxx = here->qbe_x3;
		    pass.cyyy = here->qbe_y3;
		    pass.czzz = 0.0;
		    pass.cxxy = here->qbe_x2y;
		    pass.cxxz = 0.0;
		    pass.cxyy = here->qbe_xy2;
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
		    temp = - ckt->CKTomega*
			DFi2F12(&pass);

		    itemp = ckt->CKTomega*
			DFn2F12(&pass);
		}

		*(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		*(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		*(ckt->CKTrhs + here->BJTemitPrimeNode) += temp;
		*(ckt->CKTirhs + here->BJTemitPrimeNode) += itemp;

		/* qbe term over */
		/* loading qbx term */
		/* z = vbx= vb - vcPrime */

		r1h1z = r1h1z + r1h1y;
		i1h1z = i1h1z + i1h1y;
		r1hm2z = r1hm2z + r1hm2y;
		i1hm2z = i1hm2z + i1hm2y;
		r2h11z = r2h11z + r2h11y;
		i2h11z = i2h11z + i2h11y;
		r2h1m2z = r2h1m2z + r2h1m2y;
		i2h1m2z = i2h1m2z + i2h1m2y;
#ifdef D_DBG_SMALLTIMES
		time = SPfrontEnd->IFseconds();
#endif
		temp = - ckt->CKTomega *
		    D1i2F12(here->capbx2,
			    here->capbx3,
			    r1h1z,
			    i1h1z,
			    r1hm2z,
			    i1hm2z,
			    r2h11z,
			    i2h11z,
			    r2h1m2z,
			    i2h1m2z);
		itemp = ckt->CKTomega *
		    D1n2F12(here->capbx2,
			    here->capbx3,
			    r1h1z,
			    i1h1z,
			    r1hm2z,
			    i1hm2z,
			    r2h11z,
			    i2h11z,
			    r2h1m2z,
			    i2h1m2z);
#ifdef D_DBG_SMALLTIMES
		time = SPfrontEnd->IFseconds() - time;
		printf("Time for D1n2F12: %g seconds \n", time);
#endif


		*(ckt->CKTrhs + here->BJTbaseNode) -= temp;
		*(ckt->CKTirhs + here->BJTbaseNode) -= itemp;
		*(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		*(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		/* qbx term over */

		/* loading qbc term */

		temp = - ckt->CKTomega *
		    D1i2F12(here->capbc2,
			    here->capbc3,
			    r1h1y,
			    i1h1y,
			    r1hm2y,
			    i1hm2y,
			    r2h11y,
			    i2h11y,
			    r2h1m2y,
			    i2h1m2y);
		itemp = ckt->CKTomega *
		    D1n2F12(here->capbc2,
			    here->capbc3,
			    r1h1y,
			    i1h1y,
			    r1hm2y,
			    i1hm2y,
			    r2h11y,
			    i2h11y,
			    r2h1m2y,
			    i2h1m2y);




		*(ckt->CKTrhs + here->BJTbasePrimeNode) -= temp;
		*(ckt->CKTirhs + here->BJTbasePrimeNode) -= itemp;
		*(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		*(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		/* qbc term over */


		/* loading qsc term */
		/* z = vsc */



		r1h1z = *(job->r1H1ptr + (here->BJTsubstNode)) -
		    *(job->r1H1ptr + (here->BJTcolPrimeNode));
		i1h1z = *(job->i1H1ptr + (here->BJTsubstNode)) -
		    *(job->i1H1ptr + (here->BJTcolPrimeNode));

		r1hm2z = *(job->r1H2ptr + (here->BJTsubstNode)) -
		    *(job->r1H2ptr + (here->BJTcolPrimeNode));
		i1hm2z = -(*(job->i1H2ptr + (here->BJTsubstNode)) -
			   *(job->i1H2ptr + (here->BJTcolPrimeNode)));

		r2h11z = *(job->r2H11ptr + (here->BJTsubstNode)) -
		    *(job->r2H11ptr + (here->BJTcolPrimeNode));
		i2h11z = *(job->i2H11ptr + (here->BJTsubstNode)) -
		    *(job->i2H11ptr + (here->BJTcolPrimeNode));

		r2h1m2z = *(job->r2H1m2ptr + (here->BJTsubstNode)) -
		    *(job->r2H1m2ptr + (here->BJTcolPrimeNode));
		i2h1m2z = *(job->i2H1m2ptr + (here->BJTsubstNode)) -
		    *(job->i2H1m2ptr + (here->BJTcolPrimeNode));

		temp = - ckt->CKTomega *
		    D1i2F12(here->capsc2,
			    here->capsc3,
			    r1h1z,
			    i1h1z,
			    r1hm2z,
			    i1hm2z,
			    r2h11z,
			    i2h11z,
			    r2h1m2z,
			    i2h1m2z);

		itemp = ckt->CKTomega *
		    D1n2F12(here->capsc2,
			    here->capsc3,
			    r1h1z,
			    i1h1z,
			    r1hm2z,
			    i1hm2z,
			    r2h11z,
			    i2h11z,
			    r2h1m2z,
			    i2h1m2z);


		*(ckt->CKTrhs + here->BJTsubstNode) -= temp;
		*(ckt->CKTirhs + here->BJTsubstNode) -= itemp;
		*(ckt->CKTrhs + here->BJTcolPrimeNode) += temp;
		*(ckt->CKTirhs + here->BJTcolPrimeNode) += itemp;

		/* qsc term over */


		break;
		default:
		    ;
		    ;
		}
	    }
	}
	return(OK);
    }
    else
	return(E_BADPARM);
}
