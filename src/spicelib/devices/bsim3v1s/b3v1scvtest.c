/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1scvtest.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3v1sdef.h"
#include "trandefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1SconvTest(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1Smodel *model = (BSIM3v1Smodel*)inModel;
BSIM3v1Sinstance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the BSIM3v1S device models */
    for (; model != NULL; model = model->BSIM3v1SnextModel)
    {    /* loop through all the instances of the model */
         for (here = model->BSIM3v1Sinstances; here != NULL ;
              here=here->BSIM3v1SnextInstance) 
	 {    
             if (here->BSIM3v1Sowner != ARCHme) continue;
             vbs = model->BSIM3v1Stype 
		  * (*(ckt->CKTrhsOld+here->BSIM3v1SbNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3v1SsNodePrime));
              vgs = model->BSIM3v1Stype
		  * (*(ckt->CKTrhsOld+here->BSIM3v1SgNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3v1SsNodePrime));
              vds = model->BSIM3v1Stype
		  * (*(ckt->CKTrhsOld+here->BSIM3v1SdNodePrime) 
		  - *(ckt->CKTrhsOld+here->BSIM3v1SsNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->BSIM3v1Svgs) 
		   - *(ckt->CKTstate0 + here->BSIM3v1Svds);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM3v1Svbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM3v1Svbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM3v1Svgs);
              delvds = vds - *(ckt->CKTstate0 + here->BSIM3v1Svds);
              delvgd = vgd-vgdo;

              cd = here->BSIM3v1Scd;
              if (here->BSIM3v1Smode >= 0)
	      {   cdhat = cd - here->BSIM3v1Sgbd * delvbd 
			+ here->BSIM3v1Sgmbs * delvbs + here->BSIM3v1Sgm * delvgs
			+ here->BSIM3v1Sgds * delvds;
              }
	      else
	      {   cdhat = cd - (here->BSIM3v1Sgbd - here->BSIM3v1Sgmbs) * delvbd 
			- here->BSIM3v1Sgm * delvgd + here->BSIM3v1Sgds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->BSIM3v1Soff == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)  {   

#ifdef STRANGE_PATCH
/* gtri - begin - wbk - report conv prob */
                    if(ckt->enh->conv_debug.report_conv_probs) {
                        ENHreport_conv_prob(ENH_ANALOG_INSTANCE,
                                            (char *) here->BSIM3v1Sname,
                                            "");
                    }
/* gtri - end - wbk - report conv prob */
#endif /* STRANGE_PATCH */


		    ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->BSIM3v1Scbs;
                  cbd = here->BSIM3v1Scbd;
                  cbhat = cbs + cbd + here->BSIM3v1Sgbd * delvbd 
		        + here->BSIM3v1Sgbs * delvbs;
                  tol = ckt->CKTreltol * MAX(fabs(cbhat), fabs(cbs + cbd))
		      + ckt->CKTabstol;
                  if (fabs(cbhat - (cbs + cbd)) > tol) {   
#ifdef STRANGE_PATCH
/* gtri - begin - wbk - report conv prob */
                    if(ckt->enh->conv_debug.report_conv_probs) {
                        ENHreport_conv_prob(ENH_ANALOG_INSTANCE,
                                            (char *) here->BSIM3v1Sname,
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

