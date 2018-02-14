/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
Modified: AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
#include "ngspice/sperror.h"
#include "ngspice/distodef.h"
#include "ngspice/suffix.h"

int
B1disto(int mode, GENmodel *genmodel, CKTcircuit *ckt)
/* assuming here that ckt->CKTomega has been initialised to 
 * the correct value
 */
{
 B1model *model = (B1model *) genmodel;
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
 B1instance *here;

if (mode == D_SETUP)
 return(B1dSetup((GENmodel *)model,ckt));

if ((mode == D_TWOF1) || (mode == D_THRF1) || 
 (mode == D_F1PF2) || (mode == D_F1MF2) ||
 (mode == D_2F1MF2)) {

 /* loop through all the B1 models */
for( ; model != NULL; model = B1nextModel(model)) {

  /* loop through all the instances of the model */
  for (here = B1instances(model); here != NULL ;
       here=B1nextInstance(here)) {

    /* loading starts here */

    switch (mode) {
    case D_TWOF1:
      /* from now on, in the 3-var case, x=vgs,y=vbs,z=vds */

      {
        /* draincurrent term */
        r1h1x = *(job->r1H1ptr + here->B1gNode) - 
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1gNode) - 
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h1y = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1y = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h1z = *(job->r1H1ptr + here->B1dNodePrime) -
            *(job->r1H1ptr + here->B1sNodePrime);

        i1h1z = *(job->i1H1ptr + here->B1dNodePrime) -
            *(job->i1H1ptr + here->B1sNodePrime);

        /* draincurrent is a function of vgs,vbs,and vds;
         * have got their linear kernels; now to call
         * load functions 
         */
        
         temp = DFn2F1(here->DrC_x2,
         here->DrC_y2,
         here->DrC_z2,
         here->DrC_xy,
         here->DrC_yz,
         here->DrC_xz,
         r1h1x,
         i1h1x,
         r1h1y,
         i1h1y,
         r1h1z,
         i1h1z);
        
         itemp = DFi2F1(here->DrC_x2,
         here->DrC_y2,
         here->DrC_z2,
         here->DrC_xy,
         here->DrC_yz,
         here->DrC_xz,
         r1h1x,
         i1h1x,
         r1h1y,
         i1h1y,
         r1h1z,
         i1h1z);
        
         *(ckt->CKTrhs + here->B1dNodePrime) -= temp;
         *(ckt->CKTirhs + here->B1dNodePrime) -= itemp;
        
         *(ckt->CKTrhs + here->B1sNodePrime) += temp;
         *(ckt->CKTirhs + here->B1sNodePrime) += itemp;
        
         /* draincurrent term loading over */



        /* loading qg term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFi2F1(here->qg_x2,
        here->qg_y2,
        here->qg_z2,
        here->qg_xy,
        here->qg_yz,
        here->qg_xz,
        r1h1x,
        i1h1x,
        r1h1y,
        i1h1y,
        r1h1z,
        i1h1z);

        itemp = ckt->CKTomega * DFn2F1(here->qg_x2,
        here->qg_y2,
        here->qg_z2,
        here->qg_xy,
        here->qg_yz,
        here->qg_xz,
        r1h1x,
        i1h1x,
        r1h1y,
        i1h1y,
        r1h1z,
        i1h1z);

        *(ckt->CKTrhs + here->B1gNode) -= temp;
        *(ckt->CKTirhs + here->B1gNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qg term over */

        /* loading qb term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFi2F1(here->qb_x2,
        here->qb_y2,
        here->qb_z2,
        here->qb_xy,
        here->qb_yz,
        here->qb_xz,
        r1h1x,
        i1h1x,
        r1h1y,
        i1h1y,
        r1h1z,
        i1h1z);

        itemp = ckt->CKTomega * DFn2F1(here->qb_x2,
        here->qb_y2,
        here->qb_z2,
        here->qb_xy,
        here->qb_yz,
        here->qb_xz,
        r1h1x,
        i1h1x,
        r1h1y,
        i1h1y,
        r1h1z,
        i1h1z);

        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qb term over */

        /* loading qd term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFi2F1(here->qd_x2,
        here->qd_y2,
        here->qd_z2,
        here->qd_xy,
        here->qd_yz,
        here->qd_xz,
        r1h1x,
        i1h1x,
        r1h1y,
        i1h1y,
        r1h1z,
        i1h1z);

        itemp = ckt->CKTomega * DFn2F1(here->qd_x2,
        here->qd_y2,
        here->qd_z2,
        here->qd_xy,
        here->qd_yz,
        here->qd_xz,
        r1h1x,
        i1h1x,
        r1h1y,
        i1h1y,
        r1h1z,
        i1h1z);

        *(ckt->CKTrhs + here->B1dNodePrime) -= temp;
        *(ckt->CKTirhs + here->B1dNodePrime) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qd term over */

        /* loading here->B1gbs term */

        r1h1x = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1sNodePrime);

        /* now r1h1x = vbs */

        temp = D1n2F1(here->gbs2,
        r1h1x,
        i1h1x);

        itemp = D1i2F1(here->gbs2,
        r1h1x,
        i1h1x);

        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* here->B1gbs term over */

        /* loading here->B1gbd term */

        r1h1x = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1dNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1dNodePrime);

        /* now r1h1x = vbd */

        temp = D1n2F1(here->gbd2,
        r1h1x,
        i1h1x);

        itemp = D1i2F1(here->gbd2,
        r1h1x,
        i1h1x);

        *(ckt->CKTrhs + here->B1gNode) -= temp;
        *(ckt->CKTirhs + here->B1gNode) -= itemp;

        *(ckt->CKTrhs + here->B1dNodePrime) += temp;
        *(ckt->CKTirhs + here->B1dNodePrime) += itemp;

        /* here->B1gbd term over */

        /* all done */
      }

      break;

    case D_THRF1:
      /* from now on, in the 3-var case, x=vgs,y=vbs,z=vds */

      {
        /* draincurrent term */
        r1h1x = *(job->r1H1ptr + here->B1gNode) - 
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1gNode) - 
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h1y = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1y = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h1z = *(job->r1H1ptr + here->B1dNodePrime) -
            *(job->r1H1ptr + here->B1sNodePrime);

        i1h1z = *(job->i1H1ptr + here->B1dNodePrime) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r2h11x = *(job->r2H11ptr + here->B1gNode) - 
            *(job->r2H11ptr + here->B1sNodePrime);
        i2h11x = *(job->i2H11ptr + here->B1gNode) - 
            *(job->i2H11ptr + here->B1sNodePrime);

        r2h11y = *(job->r2H11ptr + here->B1bNode) -
            *(job->r2H11ptr + here->B1sNodePrime);
        i2h11y = *(job->i2H11ptr + here->B1bNode) -
            *(job->i2H11ptr + here->B1sNodePrime);

        r2h11z = *(job->r2H11ptr + here->B1dNodePrime) -
            *(job->r2H11ptr + here->B1sNodePrime);

        i2h11z = *(job->i2H11ptr + here->B1dNodePrime) -
            *(job->i2H11ptr + here->B1sNodePrime);

        /* draincurrent is a function of vgs,vbs,and vds;
         * have got their linear kernels; now to call
         * load functions 
         */
        
         temp = DFn3F1(here->DrC_x2,
         here->DrC_y2,
         here->DrC_z2,
         here->DrC_xy,
         here->DrC_yz,
         here->DrC_xz,
         here->DrC_x3,
         here->DrC_y3,
         here->DrC_z3,
         here->DrC_x2y,
         here->DrC_x2z,
         here->DrC_xy2,
         here->DrC_y2z,
         here->DrC_xz2,
         here->DrC_yz2,
         here->DrC_xyz,
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
        
         itemp = DFi3F1(here->DrC_x2,
         here->DrC_y2,
         here->DrC_z2,
         here->DrC_xy,
         here->DrC_yz,
         here->DrC_xz,
         here->DrC_x3,
         here->DrC_y3,
         here->DrC_z3,
         here->DrC_x2y,
         here->DrC_x2z,
         here->DrC_xy2,
         here->DrC_y2z,
         here->DrC_xz2,
         here->DrC_yz2,
         here->DrC_xyz,
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
        
         *(ckt->CKTrhs + here->B1dNodePrime) -= temp;
         *(ckt->CKTirhs + here->B1dNodePrime) -= itemp;
        
         *(ckt->CKTrhs + here->B1sNodePrime) += temp;
         *(ckt->CKTirhs + here->B1sNodePrime) += itemp;
        
         /* draincurrent term loading over */



        /* loading qg term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFi3F1(here->qg_x2,
        here->qg_y2,
        here->qg_z2,
        here->qg_xy,
        here->qg_yz,
        here->qg_xz,
        here->qg_x3,
        here->qg_y3,
        here->qg_z3,
        here->qg_x2y,
        here->qg_x2z,
        here->qg_xy2,
        here->qg_y2z,
        here->qg_xz2,
        here->qg_yz2,
        here->qg_xyz,
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

        itemp = ckt->CKTomega * DFn3F1(here->qg_x2,
        here->qg_y2,
        here->qg_z2,
        here->qg_xy,
        here->qg_yz,
        here->qg_xz,
        here->qg_x3,
        here->qg_y3,
        here->qg_z3,
        here->qg_x2y,
        here->qg_x2z,
        here->qg_xy2,
        here->qg_y2z,
        here->qg_xz2,
        here->qg_yz2,
        here->qg_xyz,
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

        *(ckt->CKTrhs + here->B1gNode) -= temp;
        *(ckt->CKTirhs + here->B1gNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qg term over */

        /* loading qb term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFi3F1(here->qb_x2,
        here->qb_y2,
        here->qb_z2,
        here->qb_xy,
        here->qb_yz,
        here->qb_xz,
        here->qb_x3,
        here->qb_y3,
        here->qb_z3,
        here->qb_x2y,
        here->qb_x2z,
        here->qb_xy2,
        here->qb_y2z,
        here->qb_xz2,
        here->qb_yz2,
        here->qb_xyz,
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

        itemp = ckt->CKTomega * DFn3F1(here->qb_x2,
        here->qb_y2,
        here->qb_z2,
        here->qb_xy,
        here->qb_yz,
        here->qb_xz,
        here->qb_x3,
        here->qb_y3,
        here->qb_z3,
        here->qb_x2y,
        here->qb_x2z,
        here->qb_xy2,
        here->qb_y2z,
        here->qb_xz2,
        here->qb_yz2,
        here->qb_xyz,
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

        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qb term over */

        /* loading qd term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFi3F1(here->qd_x2,
        here->qd_y2,
        here->qd_z2,
        here->qd_xy,
        here->qd_yz,
        here->qd_xz,
        here->qd_x3,
        here->qd_y3,
        here->qd_z3,
        here->qd_x2y,
        here->qd_x2z,
        here->qd_xy2,
        here->qd_y2z,
        here->qd_xz2,
        here->qd_yz2,
        here->qd_xyz,
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

        itemp = ckt->CKTomega * DFn3F1(here->qd_x2,
        here->qd_y2,
        here->qd_z2,
        here->qd_xy,
        here->qd_yz,
        here->qd_xz,
        here->qd_x3,
        here->qd_y3,
        here->qd_z3,
        here->qd_x2y,
        here->qd_x2z,
        here->qd_xy2,
        here->qd_y2z,
        here->qd_xz2,
        here->qd_yz2,
        here->qd_xyz,
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

        *(ckt->CKTrhs + here->B1dNodePrime) -= temp;
        *(ckt->CKTirhs + here->B1dNodePrime) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qd term over */

        /* loading here->B1gbs term */

        r1h1x = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r2h11x = *(job->r2H11ptr + here->B1bNode) -
            *(job->r2H11ptr + here->B1sNodePrime);
        i2h11x = *(job->i2H11ptr + here->B1bNode) -
            *(job->i2H11ptr + here->B1sNodePrime);

        /* now r1h1x = vbs */

        temp = D1n3F1(here->gbs2,
        here->gbs3,
        r1h1x,
        i1h1x,
        r2h11x,
        i2h11x);

        itemp = D1i3F1(here->gbs2,
        here->gbs3,
        r1h1x,
        i1h1x,
        r2h11x,
        i2h11x);


        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* here->B1gbs term over */

        /* loading here->B1gbd term */

        r1h1x = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1dNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1dNodePrime);

        r2h11x = *(job->r2H11ptr + here->B1bNode) -
            *(job->r2H11ptr + here->B1dNodePrime);
        i2h11x = *(job->i2H11ptr + here->B1bNode) -
            *(job->i2H11ptr + here->B1dNodePrime);

        /* now r1h1x = vbd */

        temp = D1n3F1(here->gbd2,
        here->gbd3,
        r1h1x,
        i1h1x,
        r2h11x,
        i2h11x);

        itemp = D1i3F1(here->gbd2,
        here->gbd3,
        r1h1x,
        i1h1x,
        r2h11x,
        i2h11x);


        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1dNodePrime) += temp;
        *(ckt->CKTirhs + here->B1dNodePrime) += itemp;

        /* here->B1gbd term over */

        /* all done */
      }

      break;
    case D_F1PF2:
      /* from now on, in the 3-var case, x=vgs,y=vbs,z=vds */

      {
        /* draincurrent term */
        r1h1x = *(job->r1H1ptr + here->B1gNode) - 
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1gNode) - 
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h1y = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1y = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h1z = *(job->r1H1ptr + here->B1dNodePrime) -
            *(job->r1H1ptr + here->B1sNodePrime);

        i1h1z = *(job->i1H1ptr + here->B1dNodePrime) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h2x = *(job->r1H2ptr + here->B1gNode) - 
            *(job->r1H2ptr + here->B1sNodePrime);
        i1h2x = (*(job->i1H2ptr + here->B1gNode) - 
            *(job->i1H2ptr + here->B1sNodePrime));

        r1h2y = *(job->r1H2ptr + here->B1bNode) -
            *(job->r1H2ptr + here->B1sNodePrime);
        i1h2y = (*(job->i1H2ptr + here->B1bNode) -
            *(job->i1H2ptr + here->B1sNodePrime));

        r1h2z = *(job->r1H2ptr + here->B1dNodePrime) -
            *(job->r1H2ptr + here->B1sNodePrime);

        i1h2z = (*(job->i1H2ptr + here->B1dNodePrime) -
            *(job->i1H2ptr + here->B1sNodePrime));

        /* draincurrent is a function of vgs,vbs,and vds;
         * have got their linear kernels; now to call
         * load functions 
         */
        
         temp = DFnF12(here->DrC_x2,
         here->DrC_y2,
         here->DrC_z2,
         here->DrC_xy,
         here->DrC_yz,
         here->DrC_xz,
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
        
         itemp = DFiF12(here->DrC_x2,
         here->DrC_y2,
         here->DrC_z2,
         here->DrC_xy,
         here->DrC_yz,
         here->DrC_xz,
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
        
         *(ckt->CKTrhs + here->B1dNodePrime) -= temp;
         *(ckt->CKTirhs + here->B1dNodePrime) -= itemp;
        
         *(ckt->CKTrhs + here->B1sNodePrime) += temp;
         *(ckt->CKTirhs + here->B1sNodePrime) += itemp;
        
         /* draincurrent term loading over */



        /* loading qg term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFiF12(here->qg_x2,
        here->qg_y2,
        here->qg_z2,
        here->qg_xy,
        here->qg_yz,
        here->qg_xz,
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

        itemp = ckt->CKTomega * DFnF12(here->qg_x2,
        here->qg_y2,
        here->qg_z2,
        here->qg_xy,
        here->qg_yz,
        here->qg_xz,
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

        *(ckt->CKTrhs + here->B1gNode) -= temp;
        *(ckt->CKTirhs + here->B1gNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qg term over */

        /* loading qb term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFiF12(here->qb_x2,
        here->qb_y2, /* XXX Bug fixed: fewer arguments passed than declared */
        here->qb_z2,
        here->qb_xy,
        here->qb_yz,
        here->qb_xz,
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

        itemp = ckt->CKTomega * DFnF12(here->qb_x2,
        here->qb_y2,
        here->qb_z2,
        here->qb_xy,
        here->qb_yz,
        here->qb_xz,
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

        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qb term over */

        /* loading qd term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFiF12(here->qd_x2,
        here->qd_y2,
        here->qd_z2,
        here->qd_xy,
        here->qd_yz,
        here->qd_xz,
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

        itemp = ckt->CKTomega * DFnF12(here->qd_x2,
        here->qd_y2,
        here->qd_z2,
        here->qd_xy,
        here->qd_yz,
        here->qd_xz,
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

        *(ckt->CKTrhs + here->B1dNodePrime) -= temp;
        *(ckt->CKTirhs + here->B1dNodePrime) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qd term over */

        /* loading here->B1gbs term */

        r1h1x = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h2x = *(job->r1H2ptr + here->B1bNode) -
            *(job->r1H2ptr + here->B1sNodePrime);
        i1h2x = *(job->i1H2ptr + here->B1bNode) -
            *(job->i1H2ptr + here->B1sNodePrime);

        /* now r1h1x = vbs */

        temp = D1nF12(here->gbs2,
        r1h1x,
        i1h1x,
        r1h2x,
        i1h2x);

        itemp = D1iF12(here->gbs2,
        r1h1x,
        i1h1x,
        r1h2x,
        i1h2x);

        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* here->B1gbs term over */

        /* loading here->B1gbd term */

        r1h1x = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1dNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1dNodePrime);

        r1h2x = *(job->r1H2ptr + here->B1bNode) -
            *(job->r1H2ptr + here->B1dNodePrime);
        i1h2x = *(job->i1H2ptr + here->B1bNode) -
            *(job->i1H2ptr + here->B1dNodePrime);

        /* now r1h1x = vbd */

        temp = D1nF12(here->gbd2,
        r1h1x,
        i1h1x,
        r1h2x,
        i1h2x);

        itemp = D1iF12(here->gbd2,
        r1h1x,
        i1h1x,
        r1h2x,
        i1h2x);

        *(ckt->CKTrhs + here->B1gNode) -= temp;
        *(ckt->CKTirhs + here->B1gNode) -= itemp;

        *(ckt->CKTrhs + here->B1dNodePrime) += temp;
        *(ckt->CKTirhs + here->B1dNodePrime) += itemp;

        /* here->B1gbd term over */

        /* all done */
      }

      break;
    case D_F1MF2:
      /* from now on, in the 3-var case, x=vgs,y=vbs,z=vds */

      {
        /* draincurrent term */
        r1h1x = *(job->r1H1ptr + here->B1gNode) - 
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1gNode) - 
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h1y = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1y = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h1z = *(job->r1H1ptr + here->B1dNodePrime) -
            *(job->r1H1ptr + here->B1sNodePrime);

        i1h1z = *(job->i1H1ptr + here->B1dNodePrime) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r1hm2x = *(job->r1H2ptr + here->B1gNode) - 
            *(job->r1H2ptr + here->B1sNodePrime);
        i1hm2x = -(*(job->i1H2ptr + here->B1gNode) - 
            *(job->i1H2ptr + here->B1sNodePrime));

        r1hm2y = *(job->r1H2ptr + here->B1bNode) -
            *(job->r1H2ptr + here->B1sNodePrime);
        i1hm2y = -(*(job->i1H2ptr + here->B1bNode) -
            *(job->i1H2ptr + here->B1sNodePrime));

        r1hm2z = *(job->r1H2ptr + here->B1dNodePrime) -
            *(job->r1H2ptr + here->B1sNodePrime);

        i1hm2z = -(*(job->i1H2ptr + here->B1dNodePrime) -
            *(job->i1H2ptr + here->B1sNodePrime));

        /* draincurrent is a function of vgs,vbs,and vds;
         * have got their linear kernels; now to call
         * load functions 
         */
        
         temp = DFnF12(here->DrC_x2,
         here->DrC_y2,
         here->DrC_z2,
         here->DrC_xy,
         here->DrC_yz,
         here->DrC_xz,
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
        
         itemp = DFiF12(here->DrC_x2,
         here->DrC_y2,
         here->DrC_z2,
         here->DrC_xy,
         here->DrC_yz,
         here->DrC_xz,
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
        
         *(ckt->CKTrhs + here->B1dNodePrime) -= temp;
         *(ckt->CKTirhs + here->B1dNodePrime) -= itemp;
        
         *(ckt->CKTrhs + here->B1sNodePrime) += temp;
         *(ckt->CKTirhs + here->B1sNodePrime) += itemp;
        
         /* draincurrent term loading over */



        /* loading qg term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFiF12(here->qg_x2,
        here->qg_y2,
        here->qg_z2,
        here->qg_xy,
        here->qg_yz,
        here->qg_xz,
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

        itemp = ckt->CKTomega * DFnF12(here->qg_x2,
        here->qg_y2,
        here->qg_z2,
        here->qg_xy,
        here->qg_yz,
        here->qg_xz,
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

        *(ckt->CKTrhs + here->B1gNode) -= temp;
        *(ckt->CKTirhs + here->B1gNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qg term over */

        /* loading qb term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFiF12(here->qb_x2,
        here->qb_y2, /* XXX Bug fixed: fewer arguments passed than declared */
        here->qb_z2,
        here->qb_xy,
        here->qb_yz,
        here->qb_xz,
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

        itemp = ckt->CKTomega * DFnF12(here->qb_x2,
        here->qb_y2,
        here->qb_z2,
        here->qb_xy,
        here->qb_yz,
        here->qb_xz,
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

        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qb term over */

        /* loading qd term */

        /* kernels for vgs,vbs and vds already set up */

        temp = -ckt->CKTomega * DFiF12(here->qd_x2,
        here->qd_y2,
        here->qd_z2,
        here->qd_xy,
        here->qd_yz,
        here->qd_xz,
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

        itemp = ckt->CKTomega * DFnF12(here->qd_x2,
        here->qd_y2,
        here->qd_z2,
        here->qd_xy,
        here->qd_yz,
        here->qd_xz,
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

        *(ckt->CKTrhs + here->B1dNodePrime) -= temp;
        *(ckt->CKTirhs + here->B1dNodePrime) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qd term over */

        /* loading here->B1gbs term */

        r1h1x = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r1hm2x = *(job->r1H2ptr + here->B1bNode) -
            *(job->r1H2ptr + here->B1sNodePrime);
        i1hm2x = -(*(job->i1H2ptr + here->B1bNode) -
            *(job->i1H2ptr + here->B1sNodePrime));

        /* now r1h1x = vbs */

        temp = D1nF12(here->gbs2,
        r1h1x,
        i1h1x,
        r1hm2x,
        i1hm2x);

        itemp = D1iF12(here->gbs2,
        r1h1x,
        i1h1x,
        r1hm2x,
        i1hm2x);

        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* here->B1gbs term over */

        /* loading here->B1gbd term */

        r1h1x = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1dNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1dNodePrime);

        r1hm2x = *(job->r1H2ptr + here->B1bNode) -
            *(job->r1H2ptr + here->B1dNodePrime);
        i1hm2x = -(*(job->i1H2ptr + here->B1bNode) -
            *(job->i1H2ptr + here->B1dNodePrime));

        /* now r1h1x = vbd */

        temp = D1nF12(here->gbd2,
        r1h1x,
        i1h1x,
        r1hm2x,
        i1hm2x);

        itemp = D1iF12(here->gbd2,
        r1h1x,
        i1h1x,
        r1hm2x,
        i1hm2x);

        *(ckt->CKTrhs + here->B1gNode) -= temp;
        *(ckt->CKTirhs + here->B1gNode) -= itemp;

        *(ckt->CKTrhs + here->B1dNodePrime) += temp;
        *(ckt->CKTirhs + here->B1dNodePrime) += itemp;

        /* here->B1gbd term over */

        /* all done */
      }

      break;
    case D_2F1MF2:
      /* from now on, in the 3-var case, x=vgs,y=vbs,z=vds */

      {
        /* draincurrent term */
        r1h1x = *(job->r1H1ptr + here->B1gNode) - 
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1gNode) - 
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h1y = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1y = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r1h1z = *(job->r1H1ptr + here->B1dNodePrime) -
            *(job->r1H1ptr + here->B1sNodePrime);

        i1h1z = *(job->i1H1ptr + here->B1dNodePrime) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r1hm2x = *(job->r1H2ptr + here->B1gNode) - 
            *(job->r1H2ptr + here->B1sNodePrime);
        i1hm2x = -(*(job->i1H2ptr + here->B1gNode) - 
            *(job->i1H2ptr + here->B1sNodePrime));

        r1hm2y = *(job->r1H2ptr + here->B1bNode) -
            *(job->r1H2ptr + here->B1sNodePrime);
        i1hm2y = -(*(job->i1H2ptr + here->B1bNode) -
            *(job->i1H2ptr + here->B1sNodePrime));

        r1hm2z = *(job->r1H2ptr + here->B1dNodePrime) -
            *(job->r1H2ptr + here->B1sNodePrime);

        i1hm2z = -(*(job->i1H2ptr + here->B1dNodePrime) -
            *(job->i1H2ptr + here->B1sNodePrime));

        r2h11x = *(job->r2H11ptr + here->B1gNode) - 
            *(job->r2H11ptr + here->B1sNodePrime);
        i2h11x = *(job->i2H11ptr + here->B1gNode) - 
            *(job->i2H11ptr + here->B1sNodePrime);

        r2h11y = *(job->r2H11ptr + here->B1bNode) -
            *(job->r2H11ptr + here->B1sNodePrime);
        i2h11y = *(job->i2H11ptr + here->B1bNode) -
            *(job->i2H11ptr + here->B1sNodePrime);

        r2h11z = *(job->r2H11ptr + here->B1dNodePrime) -
            *(job->r2H11ptr + here->B1sNodePrime);

        i2h11z = *(job->i2H11ptr + here->B1dNodePrime) -
            *(job->i2H11ptr + here->B1sNodePrime);

        r2h1m2x = *(job->r2H1m2ptr + here->B1gNode) - 
            *(job->r2H1m2ptr + here->B1sNodePrime);
        i2h1m2x = *(job->i2H1m2ptr + here->B1gNode) - 
            *(job->i2H1m2ptr + here->B1sNodePrime);

        r2h1m2y = *(job->r2H1m2ptr + here->B1bNode) -
            *(job->r2H1m2ptr + here->B1sNodePrime);
        i2h1m2y = *(job->i2H1m2ptr + here->B1bNode) -
            *(job->i2H1m2ptr + here->B1sNodePrime);

        r2h1m2z = *(job->r2H1m2ptr + here->B1dNodePrime) -
            *(job->r2H1m2ptr + here->B1sNodePrime);

        i2h1m2z = *(job->i2H1m2ptr + here->B1dNodePrime) -
            *(job->i2H1m2ptr + here->B1sNodePrime);

        /* draincurrent is a function of vgs,vbs,and vds;
         * have got their linear kernels; now to call
         * load functions 
         */
        
        pass.cxx = here->DrC_x2;
        pass.cyy = here->DrC_y2;
        pass.czz = here->DrC_z2;
        pass.cxy = here->DrC_xy;
        pass.cyz = here->DrC_yz;
        pass.cxz = here->DrC_xz;
        pass.cxxx = here->DrC_x3;
        pass.cyyy = here->DrC_y3;
        pass.czzz = here->DrC_z3;
        pass.cxxy = here->DrC_x2y;
        pass.cxxz = here->DrC_x2z;
        pass.cxyy = here->DrC_xy2;
        pass.cyyz = here->DrC_y2z;
        pass.cxzz = here->DrC_xz2;
        pass.cyzz = here->DrC_yz2;
        pass.cxyz = here->DrC_xyz;
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
        
         *(ckt->CKTrhs + here->B1dNodePrime) -= temp;
         *(ckt->CKTirhs + here->B1dNodePrime) -= itemp;
        
         *(ckt->CKTrhs + here->B1sNodePrime) += temp;
         *(ckt->CKTirhs + here->B1sNodePrime) += itemp;
        
         /* draincurrent term loading over */



        /* loading qg term */

        /* kernels for vgs,vbs and vds already set up */

        pass.cxx = here->qg_x2;
        pass.cyy = here->qg_y2;
        pass.czz = here->qg_z2;
        pass.cxy = here->qg_xy;
        pass.cyz = here->qg_yz;
        pass.cxz = here->qg_xz;
        pass.cxxx = here->qg_x3;
        pass.cyyy = here->qg_y3;
        pass.czzz = here->qg_z3;
        pass.cxxy = here->qg_x2y;
        pass.cxxz = here->qg_x2z;
        pass.cxyy = here->qg_xy2;
        pass.cyyz = here->qg_y2z;
        pass.cxzz = here->qg_xz2;
        pass.cyzz = here->qg_yz2;
        pass.cxyz = here->qg_xyz;
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
        temp = -ckt->CKTomega * DFi2F12(&pass);

        itemp = ckt->CKTomega * DFn2F12(&pass);

        *(ckt->CKTrhs + here->B1gNode) -= temp;
        *(ckt->CKTirhs + here->B1gNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qg term over */

        /* loading qb term */

        /* kernels for vgs,vbs and vds already set up */

        pass.cxx = here->qb_x2;
        pass.cyy = here->qb_y2;
        pass.czz = here->qb_z2;
        pass.cxy = here->qb_xy;
        pass.cyz = here->qb_yz;
        pass.cxz = here->qb_xz;
        pass.cxxx = here->qb_x3;
        pass.cyyy = here->qb_y3;
        pass.czzz = here->qb_z3;
        pass.cxxy = here->qb_x2y;
        pass.cxxz = here->qb_x2z;
        pass.cxyy = here->qb_xy2;
        pass.cyyz = here->qb_y2z;
        pass.cxzz = here->qb_xz2;
        pass.cyzz = here->qb_yz2;
        pass.cxyz = here->qb_xyz;
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
        temp = -ckt->CKTomega * DFi2F12(&pass);

        itemp = ckt->CKTomega * DFn2F12(&pass);

        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qb term over */

        /* loading qd term */

        /* kernels for vgs,vbs and vds already set up */

        pass.cxx = here->qd_x2;
        pass.cyy = here->qd_y2;
        pass.czz = here->qd_z2;
        pass.cxy = here->qd_xy;
        pass.cyz = here->qd_yz;
        pass.cxz = here->qd_xz;
        pass.cxxx = here->qd_x3;
        pass.cyyy = here->qd_y3;
        pass.czzz = here->qd_z3;
        pass.cxxy = here->qd_x2y;
        pass.cxxz = here->qd_x2z;
        pass.cxyy = here->qd_xy2;
        pass.cyyz = here->qd_y2z;
        pass.cxzz = here->qd_xz2;
        pass.cyzz = here->qd_yz2;
        pass.cxyz = here->qd_xyz;
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
        temp = -ckt->CKTomega * DFi2F12(&pass);

        itemp = ckt->CKTomega * DFn2F12(&pass);

        *(ckt->CKTrhs + here->B1dNodePrime) -= temp;
        *(ckt->CKTirhs + here->B1dNodePrime) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* qd term over */

        /* loading here->B1gbs term */

        r1h1x = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1sNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1sNodePrime);

        r2h11x = *(job->r2H11ptr + here->B1bNode) -
            *(job->r2H11ptr + here->B1sNodePrime);
        i2h11x = *(job->i2H11ptr + here->B1bNode) -
            *(job->i2H11ptr + here->B1sNodePrime);

        r1hm2x = *(job->r1H2ptr + here->B1bNode) -
            *(job->r1H2ptr + here->B1sNodePrime);
        i1hm2x = -(*(job->i1H2ptr + here->B1bNode) -
            *(job->i1H2ptr + here->B1sNodePrime));

        r2h1m2x = *(job->r2H1m2ptr + here->B1bNode) -
            *(job->r2H1m2ptr + here->B1sNodePrime);
        i2h1m2x = *(job->i2H1m2ptr + here->B1bNode) -
            *(job->i2H1m2ptr + here->B1sNodePrime);

        /* now r1h1x = vbs */

        temp = D1n2F12(here->gbs2,
        here->gbs3,
        r1h1x,
        i1h1x,
        r1hm2x,
        i1hm2x,
        r2h11x,
        i2h11x,
        r2h1m2x,
        i2h1m2x);

        itemp = D1i2F12(here->gbs2,
        here->gbs3,
        r1h1x,
        i1h1x,
        r1hm2x,
        i1hm2x,
        r2h11x,
        i2h11x,
        r2h1m2x,
        i2h1m2x);


        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1sNodePrime) += temp;
        *(ckt->CKTirhs + here->B1sNodePrime) += itemp;

        /* here->B1gbs term over */

        /* loading here->B1gbd term */

        r1h1x = *(job->r1H1ptr + here->B1bNode) -
            *(job->r1H1ptr + here->B1dNodePrime);
        i1h1x = *(job->i1H1ptr + here->B1bNode) -
            *(job->i1H1ptr + here->B1dNodePrime);

        r2h11x = *(job->r2H11ptr + here->B1bNode) -
            *(job->r2H11ptr + here->B1dNodePrime);
        i2h11x = *(job->i2H11ptr + here->B1bNode) -
            *(job->i2H11ptr + here->B1dNodePrime);

        r1hm2x = *(job->r1H2ptr + here->B1bNode) -
            *(job->r1H2ptr + here->B1dNodePrime);
        i1hm2x = -(*(job->i1H2ptr + here->B1bNode) -
            *(job->i1H2ptr + here->B1dNodePrime));

        r2h1m2x = *(job->r2H1m2ptr + here->B1bNode) -
            *(job->r2H1m2ptr + here->B1dNodePrime);
        i2h1m2x = *(job->i2H1m2ptr + here->B1bNode) -
            *(job->i2H1m2ptr + here->B1dNodePrime);

        /* now r1h1x = vbd */

        temp = D1n2F12(here->gbd2,
        here->gbd3,
        r1h1x,
        i1h1x,
        r1hm2x,
        i1hm2x,
        r2h11x,
        i2h11x,
        r2h1m2x,
        i2h1m2x);

        itemp = D1i2F12(here->gbd2,
        here->gbd3,
        r1h1x,
        i1h1x,
        r1hm2x,
        i1hm2x,
        r2h11x,
        i2h11x,
        r2h1m2x,
        i2h1m2x);


        *(ckt->CKTrhs + here->B1bNode) -= temp;
        *(ckt->CKTirhs + here->B1bNode) -= itemp;

        *(ckt->CKTrhs + here->B1dNodePrime) += temp;
        *(ckt->CKTirhs + here->B1dNodePrime) += itemp;

        /* here->B1gbd term over */

        /* all done */
      }

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
