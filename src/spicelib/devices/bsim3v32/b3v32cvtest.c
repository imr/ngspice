/**** BSIM3v3.2.4, Released by Xuemei Xi 12/14/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3cvtest.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001 Xuemei Xi
 * Modified by Paolo Nenzi 2002
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v32convTest (GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v32model *model = (BSIM3v32model*)inModel;
BSIM3v32instance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the BSIM3v32 device models */
    for (; model != NULL; model = BSIM3v32nextModel(model))
    {    /* loop through all the instances of the model */
         for (here = BSIM3v32instances(model); here != NULL ;
              here=BSIM3v32nextInstance(here))
         {
              vbs = model->BSIM3v32type
                  * (*(ckt->CKTrhsOld+here->BSIM3v32bNode)
                  - *(ckt->CKTrhsOld+here->BSIM3v32sNodePrime));
              vgs = model->BSIM3v32type
                  * (*(ckt->CKTrhsOld+here->BSIM3v32gNode)
                  - *(ckt->CKTrhsOld+here->BSIM3v32sNodePrime));
              vds = model->BSIM3v32type
                  * (*(ckt->CKTrhsOld+here->BSIM3v32dNodePrime)
                  - *(ckt->CKTrhsOld+here->BSIM3v32sNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->BSIM3v32vgs)
                   - *(ckt->CKTstate0 + here->BSIM3v32vds);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM3v32vbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM3v32vbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM3v32vgs);
              delvds = vds - *(ckt->CKTstate0 + here->BSIM3v32vds);
              delvgd = vgd-vgdo;

              cd = here->BSIM3v32cd - here->BSIM3v32cbd;
              if (here->BSIM3v32mode >= 0)
              {   cd += here->BSIM3v32csub;
                  cdhat = cd - here->BSIM3v32gbd * delvbd
                        + (here->BSIM3v32gmbs + here->BSIM3v32gbbs) * delvbs
                        + (here->BSIM3v32gm + here->BSIM3v32gbgs) * delvgs
                        + (here->BSIM3v32gds + here->BSIM3v32gbds) * delvds;
              }
              else
              {   cdhat = cd + (here->BSIM3v32gmbs - here->BSIM3v32gbd) * delvbd
                        + here->BSIM3v32gm * delvgd - here->BSIM3v32gds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->BSIM3v32off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
              {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
                      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
                  {   ckt->CKTnoncon++;
                      return(OK);
                  }
                  cbs = here->BSIM3v32cbs;
                  cbd = here->BSIM3v32cbd;
                  if (here->BSIM3v32mode >= 0)
                  {   cbhat = cbs + cbd - here->BSIM3v32csub
                            + here->BSIM3v32gbd * delvbd
                            + (here->BSIM3v32gbs - here->BSIM3v32gbbs) * delvbs
                            - here->BSIM3v32gbgs * delvgs
                            - here->BSIM3v32gbds * delvds;
                  }
                  else
                  {   cbhat = cbs + cbd - here->BSIM3v32csub
                            + here->BSIM3v32gbs * delvbs
                            + (here->BSIM3v32gbd - here->BSIM3v32gbbs) * delvbd
                            - here->BSIM3v32gbgs * delvgd
                            + here->BSIM3v32gbds * delvds;
                  }
                  tol = ckt->CKTreltol * MAX(fabs(cbhat),
                        fabs(cbs + cbd - here->BSIM3v32csub)) + ckt->CKTabstol;
                  if (fabs(cbhat - (cbs + cbd - here->BSIM3v32csub)) > tol)
                  {   ckt->CKTnoncon++;
                      return(OK);
                  }
              }
         }
    }
    return(OK);
}


