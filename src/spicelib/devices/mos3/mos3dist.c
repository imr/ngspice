/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

int
MOS3disto(int mode, GENmodel *genmodel, CKTcircuit *ckt)
/* assuming here that ckt->CKTomega has been initialised to 
 * the correct value
 */
{
 MOS3model *model = (MOS3model *) genmodel;
 DISTOAN* job = (DISTOAN*) ckt->CKTcurJob;
 DpassStr pass;
 double r1h1x,i1h1x;
 double r1h1y,i1h1y;
 double r1h1z, i1h1z;
 double r1h2x, i1h2x;
 double r1h2y, i1h2y;
 double r1h2z, i1h2z;
 double r1hm2x,i1hm2x;
 double r1hm2y,i1hm2y;
 double r1hm2z, i1hm2z;
 double r2h11x,i2h11x;
 double r2h11y,i2h11y;
 double r2h11z, i2h11z;
 double r2h1m2x,i2h1m2x;
 double r2h1m2y,i2h1m2y;
 double r2h1m2z, i2h1m2z;
 double temp, itemp;
 MOS3instance *here;

if (mode == D_SETUP)
 return(MOS3dSetup(genmodel,ckt));

if ((mode == D_TWOF1) || (mode == D_THRF1) || 
 (mode == D_F1PF2) || (mode == D_F1MF2) ||
 (mode == D_2F1MF2)) {

 /* loop through all the MOS3 models */
for( ; model != NULL; model = MOS3nextModel(model)) {

  /* loop through all the instances of the model */
  for (here = MOS3instances(model); here != NULL ;
	here=MOS3nextInstance(here)) {

    /* loading starts here */

    switch (mode) {
    case D_TWOF1:
	/* x = vgs, y = vbs z = vds */

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->MOS3gNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1x = *(job->i1H1ptr + (here->MOS3gNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h1y = *(job->r1H1ptr + (here->MOS3bNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1y = *(job->i1H1ptr + (here->MOS3bNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h1z = *(job->r1H1ptr + (here->MOS3dNodePrime)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1z = *(job->i1H1ptr + (here->MOS3dNodePrime)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    /* loading starts here */
	    /* loading cdrain term  */

	    temp = DFn2F1(here->cdr_x2,
					here->cdr_y2,
					here->cdr_z2,
					here->cdr_xy,
					here->cdr_yz,
					here->cdr_xz,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					r1h1z,
					i1h1z);

	    itemp = DFi2F1(here->cdr_x2,
					here->cdr_y2,
					here->cdr_z2,
					here->cdr_xy,
					here->cdr_yz,
					here->cdr_xz,
					r1h1x,
					i1h1x,
					r1h1y,
					i1h1y,
					r1h1z,
					i1h1z);

	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;
	
	    /* cdrain term over */

		/* loading gbs term */

		temp = D1n2F1(here->gbs2,
						r1h1y,
						i1h1y);

		itemp = D1i2F1(here->gbs2,
						r1h1y,
						i1h1y);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* gbs over */

		/* loading gbd term */

		temp = D1n2F1(here->gbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z);

		itemp = D1i2F1(here->gbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* gbd over */

		/* loading capgs term */

		temp = -ckt->CKTomega *
				D1i2F1(here->capgs2,
						r1h1x,
						i1h1x);

		itemp = ckt->CKTomega *
				D1n2F1(here->capgs2,
						r1h1x,
						i1h1x);

	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* capgs over */

		/* loading capgd term */

		temp = -ckt->CKTomega *
				D1i2F1(here->capgd2,
						r1h1x - r1h1z,
						i1h1x - i1h1z);

		itemp = ckt->CKTomega *
				D1n2F1(here->capgd2,
						r1h1x - r1h1z,
						i1h1x - i1h1z);


	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* capgd over */
		/* loading capgb term */

		temp = -ckt->CKTomega *
				D1i2F1(here->capgb2,
						r1h1x - r1h1y,
						i1h1x - i1h1y);

		itemp = ckt->CKTomega *
				D1n2F1(here->capgb2,
						r1h1x - r1h1y,
						i1h1x - i1h1y);

	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3bNode)) += temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) += itemp;

		/* capgb over */

		/* loading capbs term */

		temp = -ckt->CKTomega *
				D1i2F1(here->capbs2,
						r1h1y,
						i1h1y);

		itemp = ckt->CKTomega *
				D1n2F1(here->capbs2,
						r1h1y,
						i1h1y);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* capbs over */

		/* loading capbd term */

		temp = -ckt->CKTomega *
				D1i2F1(here->capbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z);

		itemp = ckt->CKTomega *
				D1n2F1(here->capbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* capbd over */
	/* all done */

      break;

    case D_THRF1:
	/* x = vgs, y = vbs z = vds */

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->MOS3gNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1x = *(job->i1H1ptr + (here->MOS3gNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h1y = *(job->r1H1ptr + (here->MOS3bNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1y = *(job->i1H1ptr + (here->MOS3bNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h1z = *(job->r1H1ptr + (here->MOS3dNodePrime)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1z = *(job->i1H1ptr + (here->MOS3dNodePrime)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	r2h11x = *(job->r2H11ptr + (here->MOS3gNode)) -
		    *(job->r2H11ptr + (here->MOS3sNodePrime));
	i2h11x = *(job->i2H11ptr + (here->MOS3gNode)) -
		    *(job->i2H11ptr + (here->MOS3sNodePrime));

	r2h11y = *(job->r2H11ptr + (here->MOS3bNode)) -
		    *(job->r2H11ptr + (here->MOS3sNodePrime));
	i2h11y = *(job->i2H11ptr + (here->MOS3bNode)) -
		    *(job->i2H11ptr + (here->MOS3sNodePrime));

	r2h11z = *(job->r2H11ptr + (here->MOS3dNodePrime)) -
		    *(job->r2H11ptr + (here->MOS3sNodePrime));
	i2h11z = *(job->i2H11ptr + (here->MOS3dNodePrime)) -
		    *(job->i2H11ptr + (here->MOS3sNodePrime));
		/* loading starts here */
		/* loading cdrain term  */

		temp = DFn3F1(here->cdr_x2,
					    here->cdr_y2,
					    here->cdr_z2,
					    here->cdr_xy,
					    here->cdr_yz,
					    here->cdr_xz,
					    here->cdr_x3,
					    here->cdr_y3,
					    here->cdr_z3,
					here->cdr_x2y,
					here->cdr_x2z,
					here->cdr_xy2,
					here->cdr_y2z,
					here->cdr_xz2,
					here->cdr_yz2,
				    here->cdr_xyz,
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
		itemp = DFi3F1(here->cdr_x2,
					    here->cdr_y2,
					    here->cdr_z2,
					    here->cdr_xy,
					here->cdr_yz,
					here->cdr_xz,
					here->cdr_x3,
					here->cdr_y3,
					here->cdr_z3,
					here->cdr_x2y,
					here->cdr_x2z,
					here->cdr_xy2,
					here->cdr_y2z,
					here->cdr_xz2,
					here->cdr_yz2,
				    here->cdr_xyz,
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


	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;
	
	    /* cdrain term over */

		/* loading gbs term */

		temp = D1n3F1(here->gbs2,
						here->gbs3,
						r1h1y,
						i1h1y,
						r2h11y,
						i2h11y);


		itemp = D1i3F1(here->gbs2,
						here->gbs3,
						r1h1y,
						i1h1y,
						r2h11y,
						i2h11y);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* gbs over */

		/* loading gbd term */

		temp = D1n3F1(here->gbd2,
						here->gbd3,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r2h11y - r2h11z,
						i2h11y - i2h11z);

		itemp = D1i3F1(here->gbd2,
						here->gbd3,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r2h11y - r2h11z,
						i2h11y - i2h11z);

	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* gbd over */

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

	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* capgs over */

		/* loading capgd term */

		temp = -ckt->CKTomega *
			    D1i3F1(here->capgd2,
						here->capgd3,
						r1h1x - r1h1z,
						i1h1x - i1h1z,
						r2h11x - r2h11z,
						i2h11x - i2h11z);

		itemp = ckt->CKTomega *
			    D1n3F1(here->capgd2,
						here->capgd3,
						r1h1x - r1h1z,
						i1h1x - i1h1z,
						r2h11x - r2h11z,
						i2h11x - i2h11z);


	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* capgd over */
		/* loading capgb term */

		temp = -ckt->CKTomega *
			    D1i3F1(here->capgb2,
						here->capgb3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r2h11x - r2h11y,
						i2h11x - i2h11y);

		itemp = ckt->CKTomega *
			    D1n3F1(here->capgb2,
						here->capgb3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r2h11x - r2h11y,
						i2h11x - i2h11y);

	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3bNode)) += temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) += itemp;

		/* capgb over */

		/* loading capbs term */

		temp = -ckt->CKTomega *
			    D1i3F1(here->capbs2,
						here->capbs3,
						r1h1y,
						i1h1y,
						r2h11y,
						i2h11y);

		itemp = ckt->CKTomega *
			    D1n3F1(here->capbs2,
						here->capbs3,
						r1h1y,
						i1h1y,
						r2h11y,
						i2h11y);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* capbs over */

		/* loading capbd term */

		temp = -ckt->CKTomega *
			    D1i3F1(here->capbd2,
						here->capbd3,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r2h11y - r2h11z,
						i2h11y - i2h11z);

		itemp = ckt->CKTomega *
			    D1n3F1(here->capbd2,
						here->capbd3,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r2h11y - r2h11z,
						i2h11y - i2h11z);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* capbd over */
	/* all done */

      break;
    case D_F1PF2:
	/* x = vgs, y = vbs z = vds */

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->MOS3gNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1x = *(job->i1H1ptr + (here->MOS3gNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h1y = *(job->r1H1ptr + (here->MOS3bNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1y = *(job->i1H1ptr + (here->MOS3bNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h1z = *(job->r1H1ptr + (here->MOS3dNodePrime)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1z = *(job->i1H1ptr + (here->MOS3dNodePrime)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h2x = *(job->r1H2ptr + (here->MOS3gNode)) -
			*(job->r1H2ptr + (here->MOS3sNodePrime));
	    i1h2x = *(job->i1H2ptr + (here->MOS3gNode)) -
			*(job->i1H2ptr + (here->MOS3sNodePrime));

	    r1h2y = *(job->r1H2ptr + (here->MOS3bNode)) -
			*(job->r1H2ptr + (here->MOS3sNodePrime));
	    i1h2y = *(job->i1H2ptr + (here->MOS3bNode)) -
			*(job->i1H2ptr + (here->MOS3sNodePrime));

	    r1h2z = *(job->r1H2ptr + (here->MOS3dNodePrime)) -
			*(job->r1H2ptr + (here->MOS3sNodePrime));
	    i1h2z = *(job->i1H2ptr + (here->MOS3dNodePrime)) -
			*(job->i1H2ptr + (here->MOS3sNodePrime));

	    /* loading starts here */
	    /* loading cdrain term  */

	    temp = DFnF12(here->cdr_x2,
					here->cdr_y2,
					here->cdr_z2,
					here->cdr_xy,
					here->cdr_yz,
					here->cdr_xz,
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

	    itemp = DFiF12(here->cdr_x2,
					here->cdr_y2,
					here->cdr_z2,
					here->cdr_xy,
					here->cdr_yz,
					here->cdr_xz,
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

	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;
	
	    /* cdrain term over */

		/* loading gbs term */

		temp = D1nF12(here->gbs2,
						r1h1y,
						i1h1y,
						r1h2y,
						i1h2y);

		itemp = D1iF12(here->gbs2,
						r1h1y,
						i1h1y,
						r1h2y,
						i1h2y);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* gbs over */

		/* loading gbd term */

		temp = D1nF12(here->gbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1h2y - r1h2z,
						i1h2y - i1h2z);

		itemp = D1iF12(here->gbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1h2y - r1h2z,
						i1h2y - i1h2z);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* gbd over */

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

	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* capgs over */

		/* loading capgd term */

		temp = -ckt->CKTomega *
			    D1iF12(here->capgd2,
						r1h1x - r1h1z,
						i1h1x - i1h1z,
						r1h2x - r1h2z,
						i1h2x - i1h2z);

		itemp = ckt->CKTomega *
			    D1nF12(here->capgd2,
						r1h1x - r1h1z,
						i1h1x - i1h1z,
						r1h2x - r1h2z,
						i1h2x - i1h2z);


	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* capgd over */
		/* loading capgb term */

		temp = -ckt->CKTomega *
			    D1iF12(here->capgb2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1h2x - r1h2y,
						i1h2x - i1h2y);

		itemp = ckt->CKTomega *
			    D1nF12(here->capgb2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1h2x - r1h2y,
						i1h2x - i1h2y);

	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3bNode)) += temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) += itemp;

		/* capgb over */

		/* loading capbs term */

		temp = -ckt->CKTomega *
			    D1iF12(here->capbs2,
						r1h1y,
						i1h1y,
						r1h2y,
						i1h2y);

		itemp = ckt->CKTomega *
			    D1nF12(here->capbs2,
						r1h1y,
						i1h1y,
						r1h2y,
						i1h2y);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* capbs over */

		/* loading capbd term */

		temp = -ckt->CKTomega *
			    D1iF12(here->capbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1h2y - r1h2z,
						i1h2y - i1h2z);

		itemp = ckt->CKTomega *
			    D1nF12(here->capbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1h2y - r1h2z,
						i1h2y - i1h2z);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* capbd over */
	/* all done */

      break;
    case D_F1MF2:
	/* x = vgs, y = vbs z = vds */

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->MOS3gNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1x = *(job->i1H1ptr + (here->MOS3gNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h1y = *(job->r1H1ptr + (here->MOS3bNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1y = *(job->i1H1ptr + (here->MOS3bNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h1z = *(job->r1H1ptr + (here->MOS3dNodePrime)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1z = *(job->i1H1ptr + (here->MOS3dNodePrime)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1hm2x = *(job->r1H2ptr + (here->MOS3gNode)) -
			*(job->r1H2ptr + (here->MOS3sNodePrime));
	    i1hm2x = -(*(job->i1H2ptr + (here->MOS3gNode)) -
			*(job->i1H2ptr + (here->MOS3sNodePrime)));

	    r1hm2y = *(job->r1H2ptr + (here->MOS3bNode)) -
			*(job->r1H2ptr + (here->MOS3sNodePrime));
	    i1hm2y = -(*(job->i1H2ptr + (here->MOS3bNode)) -
			*(job->i1H2ptr + (here->MOS3sNodePrime)));

	    r1hm2z = *(job->r1H2ptr + (here->MOS3dNodePrime)) -
			*(job->r1H2ptr + (here->MOS3sNodePrime));
	i1hm2z = -(*(job->i1H2ptr + (here->MOS3dNodePrime)) -
			*(job->i1H2ptr + (here->MOS3sNodePrime)));

	    /* loading starts here */
	    /* loading cdrain term  */

	    temp = DFnF12(here->cdr_x2,
					here->cdr_y2,
					here->cdr_z2,
					here->cdr_xy,
					here->cdr_yz,
					here->cdr_xz,
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

	    itemp = DFiF12(here->cdr_x2,
					here->cdr_y2,
					here->cdr_z2,
					here->cdr_xy,
					here->cdr_yz,
					here->cdr_xz,
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

	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;
	
	    /* cdrain term over */

		/* loading gbs term */

		temp = D1nF12(here->gbs2,
						r1h1y,
						i1h1y,
						r1hm2y,
						i1hm2y);

		itemp = D1iF12(here->gbs2,
						r1h1y,
						i1h1y,
						r1hm2y,
						i1hm2y);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* gbs over */

		/* loading gbd term */

		temp = D1nF12(here->gbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1hm2y - r1hm2z,
						i1hm2y - i1hm2z);

		itemp = D1iF12(here->gbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1hm2y - r1hm2z,
						i1hm2y - i1hm2z);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* gbd over */

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

	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* capgs over */

		/* loading capgd term */

		temp = -ckt->CKTomega *
			    D1iF12(here->capgd2,
						r1h1x - r1h1z,
						i1h1x - i1h1z,
						r1hm2x - r1hm2z,
						i1hm2x - i1hm2z);

		itemp = ckt->CKTomega *
			    D1nF12(here->capgd2,
						r1h1x - r1h1z,
						i1h1x - i1h1z,
						r1hm2x - r1hm2z,
						i1hm2x - i1hm2z);


	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* capgd over */
		/* loading capgb term */

		temp = -ckt->CKTomega *
			    D1iF12(here->capgb2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
						i1hm2x - i1hm2y);

		itemp = ckt->CKTomega *
			    D1nF12(here->capgb2,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
						i1hm2x - i1hm2y);

	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3bNode)) += temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) += itemp;

		/* capgb over */

		/* loading capbs term */

		temp = -ckt->CKTomega *
			    D1iF12(here->capbs2,
						r1h1y,
						i1h1y,
						r1hm2y,
						i1hm2y);

		itemp = ckt->CKTomega *
			    D1nF12(here->capbs2,
						r1h1y,
						i1h1y,
						r1hm2y,
						i1hm2y);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* capbs over */

		/* loading capbd term */

		temp = -ckt->CKTomega *
			    D1iF12(here->capbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1hm2y - r1hm2z,
						i1hm2y - i1hm2z);

		itemp = ckt->CKTomega *
			    D1nF12(here->capbd2,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1hm2y - r1hm2z,
						i1hm2y - i1hm2z);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* capbd over */
	/* all done */

      break;
    case D_2F1MF2:
	/* x = vgs, y = vbs z = vds */

	    /* getting first order (linear) Volterra kernel */
	    r1h1x = *(job->r1H1ptr + (here->MOS3gNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1x = *(job->i1H1ptr + (here->MOS3gNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h1y = *(job->r1H1ptr + (here->MOS3bNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1y = *(job->i1H1ptr + (here->MOS3bNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1h1z = *(job->r1H1ptr + (here->MOS3dNodePrime)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i1h1z = *(job->i1H1ptr + (here->MOS3dNodePrime)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r1hm2x = *(job->r1H2ptr + (here->MOS3gNode)) -
			*(job->r1H2ptr + (here->MOS3sNodePrime));
	    i1hm2x = -(*(job->i1H2ptr + (here->MOS3gNode)) -
			*(job->i1H2ptr + (here->MOS3sNodePrime)));

	    r1hm2y = *(job->r1H2ptr + (here->MOS3bNode)) -
			*(job->r1H2ptr + (here->MOS3sNodePrime));
	    i1hm2y = -(*(job->i1H2ptr + (here->MOS3bNode)) -
			*(job->i1H2ptr + (here->MOS3sNodePrime)));

	    r1hm2z = *(job->r1H2ptr + (here->MOS3dNodePrime)) -
			*(job->r1H2ptr + (here->MOS3sNodePrime));
	i1hm2z = -(*(job->i1H2ptr + (here->MOS3dNodePrime)) -
			*(job->i1H2ptr + (here->MOS3sNodePrime)));

	    r2h11x = *(job->r1H1ptr + (here->MOS3gNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i2h11x = *(job->i1H1ptr + (here->MOS3gNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r2h11y = *(job->r1H1ptr + (here->MOS3bNode)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i2h11y = *(job->i1H1ptr + (here->MOS3bNode)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

	    r2h11z = *(job->r1H1ptr + (here->MOS3dNodePrime)) -
			*(job->r1H1ptr + (here->MOS3sNodePrime));
	    i2h11z = *(job->i1H1ptr + (here->MOS3dNodePrime)) -
			*(job->i1H1ptr + (here->MOS3sNodePrime));

    r2h1m2x = *(job->r2H1m2ptr + (here->MOS3gNode)) -
		*(job->r2H1m2ptr + (here->MOS3sNodePrime));
    i2h1m2x = *(job->i2H1m2ptr + (here->MOS3gNode)) -
		*(job->i2H1m2ptr + (here->MOS3sNodePrime));

    r2h1m2y = *(job->r2H1m2ptr + (here->MOS3bNode)) -
		*(job->r2H1m2ptr + (here->MOS3sNodePrime));
    i2h1m2y = *(job->i2H1m2ptr + (here->MOS3bNode)) -
		*(job->i2H1m2ptr + (here->MOS3sNodePrime));

r2h1m2z = *(job->r2H1m2ptr + (here->MOS3dNodePrime)) -
		*(job->r2H1m2ptr + (here->MOS3sNodePrime));
i2h1m2z = *(job->i2H1m2ptr + (here->MOS3dNodePrime)) -
		*(job->i2H1m2ptr + (here->MOS3sNodePrime));

		/* loading starts here */
		/* loading cdrain term  */

pass.cxx = here->cdr_x2;
pass.cyy = here->cdr_y2;
pass.czz = here->cdr_z2;
pass.cxy = here->cdr_xy;
pass.cyz = here->cdr_yz;
pass.cxz = here->cdr_xz;
pass.cxxx = here->cdr_x3;
pass.cyyy = here->cdr_y3;
pass.czzz = here->cdr_z3;
pass.cxxy = here->cdr_x2y;
pass.cxxz = here->cdr_x2z;
pass.cxyy = here->cdr_xy2;
pass.cyyz = here->cdr_y2z;
pass.cxzz = here->cdr_xz2;
pass.cyzz = here->cdr_yz2;
pass.cxyz = here->cdr_xyz;
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


	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;
	
	    /* cdrain term over */

		/* loading gbs term */

		temp = D1n2F12(here->gbs2,
						here->gbs3,
						r1h1y,
						i1h1y,
						r1hm2y,
						i1hm2y,
						r2h11y,
						i2h11y,
						r2h1m2y,
						i2h1m2y);



		itemp = D1i2F12(here->gbs2,
						here->gbs3,
						r1h1y,
						i1h1y,
						r1hm2y,
						i1hm2y,
						r2h11y,
						i2h11y,
						r2h1m2y,
						i2h1m2y);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* gbs over */

		/* loading gbd term */

		temp = D1n2F12(here->gbd2,
						here->gbd3,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1hm2y - r1hm2z,
						i1hm2y - i1hm2z,
						r2h11y - r2h11z,
						i2h11y - i2h11z,
					    r2h1m2y - r2h1m2z,
					i2h1m2y - i2h1m2z);

		itemp = D1i2F12(here->gbd2,
						here->gbd3,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1hm2y - r1hm2z,
					    i1hm2y - i1hm2z,
						r2h11y - r2h11z,
						i2h11y - i2h11z,
					    r2h1m2y - r2h1m2z,
					i2h1m2y - i2h1m2z);

	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* gbd over */

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

	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* capgs over */

		/* loading capgd term */

		temp = -ckt->CKTomega *
			D1i2F12(here->capgd2,
						here->capgd3,
						r1h1x - r1h1z,
						i1h1x - i1h1z,
						r1hm2x - r1hm2z,
					    i1hm2x - i1hm2z,
						r2h11x - r2h11z,
						i2h11x - i2h11z,
					    r2h1m2x - r2h1m2z,
					i2h1m2x - i2h1m2z);

		itemp = ckt->CKTomega *
			D1n2F12(here->capgd2,
						here->capgd3,
						r1h1x - r1h1z,
						i1h1x - i1h1z,
						r1hm2x - r1hm2z,
					    i1hm2x - i1hm2z,
						r2h11x - r2h11z,
						i2h11x - i2h11z,
					    r2h1m2x - r2h1m2z,
					i2h1m2x - i2h1m2z);


	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* capgd over */
		/* loading capgb term */

		temp = -ckt->CKTomega *
			D1i2F12(here->capgb2,
						here->capgb3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
					    i1hm2x - i1hm2y,
						r2h11x - r2h11y,
						i2h11x - i2h11y,
					    r2h1m2x - r2h1m2y,
					i2h1m2x - i2h1m2y);

		itemp = ckt->CKTomega *
			D1n2F12(here->capgb2,
						here->capgb3,
						r1h1x - r1h1y,
						i1h1x - i1h1y,
						r1hm2x - r1hm2y,
					    i1hm2x - i1hm2y,
						r2h11x - r2h11y,
						i2h11x - i2h11y,
					    r2h1m2x - r2h1m2y,
					i2h1m2x - i2h1m2y);

	    *(ckt->CKTrhs + (here->MOS3gNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3gNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3bNode)) += temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) += itemp;

		/* capgb over */

		/* loading capbs term */

		temp = -ckt->CKTomega *
			D1i2F12(here->capbs2,
						here->capbs3,
						r1h1y,
						i1h1y,
						r1hm2y,
						i1hm2y,
						r2h11y,
						i2h11y,
						r2h1m2y,
						i2h1m2y);

		itemp = ckt->CKTomega *
			D1n2F12(here->capbs2,
						here->capbs3,
						r1h1y,
						i1h1y,
						r1hm2y,
						i1hm2y,
						r2h11y,
						i2h11y,
						r2h1m2y,
						i2h1m2y);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3sNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3sNodePrime)) += itemp;

		/* capbs over */

		/* loading capbd term */

		temp = -ckt->CKTomega *
			D1i2F12(here->capbd2,
						here->capbd3,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1hm2y - r1hm2z,
					    i1hm2y - i1hm2z,
						r2h11y - r2h11z,
						i2h11y - i2h11z,
					    r2h1m2y - r2h1m2z,
					i2h1m2y - i2h1m2z);

		itemp = ckt->CKTomega *
			D1n2F12(here->capbd2,
						here->capbd3,
						r1h1y - r1h1z,
						i1h1y - i1h1z,
						r1hm2y - r1hm2z,
					    i1hm2y - i1hm2z,
						r2h11y - r2h11z,
						i2h11y - i2h11z,
					    r2h1m2y - r2h1m2z,
					i2h1m2y - i2h1m2z);


	    *(ckt->CKTrhs + (here->MOS3bNode)) -= temp;
	    *(ckt->CKTirhs + (here->MOS3bNode)) -= itemp;
	    *(ckt->CKTrhs + (here->MOS3dNodePrime)) += temp;
	    *(ckt->CKTirhs + (here->MOS3dNodePrime)) += itemp;

		/* capbd over */
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
