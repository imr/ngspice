/* $Id$  */
/* 
$Log$
Revision 1.1.1.1  2000-04-27 20:03:59  pnenzi
Imported sources

 * Revision 3.1  96/12/08  19:53:26  yuhua
 * BSIM3v3.1 release
 * 
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1cvtest.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim3v1def.h"
#include "trandefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3V1convTest(inModel,ckt)
GENmodel *inModel;
register CKTcircuit *ckt;
{
register BSIM3V1model *model = (BSIM3V1model*)inModel;
register BSIM3V1instance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the BSIM3V1 device models */
    for (; model != NULL; model = model->BSIM3V1nextModel)
    {    /* loop through all the instances of the model */
         for (here = model->BSIM3V1instances; here != NULL ;
              here=here->BSIM3V1nextInstance) 
	 {    
             if (here->BSIM3V1owner != ARCHme) continue;
             vbs = model->BSIM3V1type 
		  * (*(ckt->CKTrhsOld+here->BSIM3V1bNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3V1sNodePrime));
              vgs = model->BSIM3V1type
		  * (*(ckt->CKTrhsOld+here->BSIM3V1gNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3V1sNodePrime));
              vds = model->BSIM3V1type
		  * (*(ckt->CKTrhsOld+here->BSIM3V1dNodePrime) 
		  - *(ckt->CKTrhsOld+here->BSIM3V1sNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->BSIM3V1vgs) 
		   - *(ckt->CKTstate0 + here->BSIM3V1vds);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM3V1vbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM3V1vbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM3V1vgs);
              delvds = vds - *(ckt->CKTstate0 + here->BSIM3V1vds);
              delvgd = vgd-vgdo;

              cd = here->BSIM3V1cd;
              if (here->BSIM3V1mode >= 0)
	      {   cdhat = cd - here->BSIM3V1gbd * delvbd 
			+ here->BSIM3V1gmbs * delvbs + here->BSIM3V1gm * delvgs
			+ here->BSIM3V1gds * delvds;
              }
	      else
	      {   cdhat = cd - (here->BSIM3V1gbd - here->BSIM3V1gmbs) * delvbd 
			- here->BSIM3V1gm * delvgd + here->BSIM3V1gds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->BSIM3V1off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)  {   

#ifdef STRANGE_PATCH
/* gtri - begin - wbk - report conv prob */
                    if(ckt->enh->conv_debug.report_conv_probs) {
                        ENHreport_conv_prob(ENH_ANALOG_INSTANCE,
                                            (char *) here->BSIM3V1name,
                                            "");
                    }
/* gtri - end - wbk - report conv prob */
#endif /* STRANGE_PATCH */


		    ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->BSIM3V1cbs;
                  cbd = here->BSIM3V1cbd;
                  cbhat = cbs + cbd + here->BSIM3V1gbd * delvbd 
		        + here->BSIM3V1gbs * delvbs;
                  tol = ckt->CKTreltol * MAX(fabs(cbhat), fabs(cbs + cbd))
		      + ckt->CKTabstol;
                  if (fabs(cbhat - (cbs + cbd)) > tol) {   
#ifdef STRANGE_PATCH
/* gtri - begin - wbk - report conv prob */
                    if(ckt->enh->conv_debug.report_conv_probs) {
                        ENHreport_conv_prob(ENH_ANALOG_INSTANCE,
                                            (char *) here->BSIM3V1name,
                                            "");
                    }
/* gtri - end - wbk - report conv prob */
#endif /* STRANGE_PATCH */

		    ckt->CKTnoncon++;
                      return(OK);
                  }
              }
         }
    }
    return(OK);
}

