/***  B4SOI 12/16/2010 Released by Tanvir Morshed   ***/


/**********
 * Copyright 2010 Regents of the University of California.  All rights reserved.
 * Authors: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
 * Authors: 1999-2004 Pin Su, Hui Wan, Wei Jin, b3soicvtest.c
 * Authors: 2005- Hui Wan, Xuemei Xi, Ali Niknejad, Chenming Hu.
 * Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu.
 * File: b4soicvtest.c
 * Modified by Hui Wan, Xuemei Xi 11/30/2005
 * Modified by Wenwei Yang, Chung-Hsun Lin, Darsen Lu 03/06/2009
 * Modified by Tanvir Morshed 09/22/2009
 * Modified by Tanvir Morshed 12/31/2009
 **********/

#include "ngspice/ngspice.h"

#include "ngspice/cktdefs.h"
#include "b4soidef.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B4SOIconvTest(
GENmodel *inModel,
CKTcircuit *ckt)
{
register B4SOImodel *model = (B4SOImodel*)inModel;
register B4SOIinstance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the B4SOI device models */
    for (; model != NULL; model = B4SOInextModel(model))
    {    /* loop through all the instances of the model */
         for (here = B4SOIinstances(model); here != NULL ;
              here=B4SOInextInstance(here)) 
         {
              vbs = model->B4SOItype 
                  * (*(ckt->CKTrhsOld+here->B4SOIbNode) 
                  - *(ckt->CKTrhsOld+here->B4SOIsNodePrime));
              vgs = model->B4SOItype
                  * (*(ckt->CKTrhsOld+here->B4SOIgNode) 
                  - *(ckt->CKTrhsOld+here->B4SOIsNodePrime));
              vds = model->B4SOItype
                  * (*(ckt->CKTrhsOld+here->B4SOIdNodePrime) 
                  - *(ckt->CKTrhsOld+here->B4SOIsNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->B4SOIvgs) 
                   - *(ckt->CKTstate0 + here->B4SOIvds);
              delvbs = vbs - *(ckt->CKTstate0 + here->B4SOIvbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->B4SOIvbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->B4SOIvgs);
              delvds = vds - *(ckt->CKTstate0 + here->B4SOIvds);
              delvgd = vgd-vgdo;

              cd = here->B4SOIcd;
              if (here->B4SOImode >= 0)
              {   cdhat = cd - here->B4SOIgjdb * delvbd 
                        + here->B4SOIgmbs * delvbs + here->B4SOIgm * delvgs
                        + here->B4SOIgds * delvds;
              }
              else
              {   cdhat = cd - (here->B4SOIgjdb - here->B4SOIgmbs) * delvbd 
                        - here->B4SOIgm * delvgd + here->B4SOIgds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->B4SOIoff == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
              {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
                      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
                  {   ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->B4SOIcjs;
                  cbd = here->B4SOIcjd;
                  cbhat = cbs + cbd + here->B4SOIgjdb * delvbd 
                        + here->B4SOIgjsb * delvbs;
                  tol = ckt->CKTreltol * MAX(fabs(cbhat), fabs(cbs + cbd))
                      + ckt->CKTabstol;
                  if (fabs(cbhat - (cbs + cbd)) > tol) 
                  {   ckt->CKTnoncon++;
                      return(OK);
                  }
              }
         }
    }
    return(OK);
}

