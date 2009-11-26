/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4cvtest.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 *
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim4v2def.h"
#include "trandefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"



int
BSIM4v2convTest(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v2model *model = (BSIM4v2model*)inModel;
BSIM4v2instance *here;
double delvbd, delvbs, delvds, delvgd, delvgs;
double delvdbd, delvsbs; 
double delvbd_jct, delvbs_jct;
double vds, vgs, vgd, vgdo, vbs, vbd;
double vdbd, vdbs, vsbs;
double cbhat, cdhat, Idtot, Ibtot;
double vses, vdes, vdedo, delvses, delvded, delvdes;
double Isestot, cseshat, Idedtot, cdedhat;
double Igstot, cgshat, Igdtot, cgdhat, Igbtot, cgbhat;
double tol0, tol1, tol2, tol3, tol4, tol5, tol6;

    for (; model != NULL; model = model->BSIM4v2nextModel)
    {    for (here = model->BSIM4v2instances; here != NULL ;
              here=here->BSIM4v2nextInstance) 
	 {    
	      if (here->BSIM4v2owner != ARCHme) continue; 
	 
	      vds = model->BSIM4v2type
                  * (*(ckt->CKTrhsOld + here->BSIM4v2dNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v2sNodePrime));
              vgs = model->BSIM4v2type
                  * (*(ckt->CKTrhsOld + here->BSIM4v2gNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v2sNodePrime));
              vbs = model->BSIM4v2type
                  * (*(ckt->CKTrhsOld + here->BSIM4v2bNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v2sNodePrime));
              vdbs = model->BSIM4v2type
                   * (*(ckt->CKTrhsOld + here->BSIM4v2dbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v2sNodePrime));
              vsbs = model->BSIM4v2type
                   * (*(ckt->CKTrhsOld + here->BSIM4v2sbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v2sNodePrime));              
              vses = model->BSIM4v2type
                   * (*(ckt->CKTrhsOld + here->BSIM4v2sNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v2sNodePrime));
              vdes = model->BSIM4v2type
                   * (*(ckt->CKTrhsOld + here->BSIM4v2dNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v2sNodePrime));
              
              vgdo = *(ckt->CKTstate0 + here->BSIM4v2vgs)
                    - *(ckt->CKTstate0 + here->BSIM4v2vds);
              vbd = vbs - vds;
              vdbd = vdbs - vds;
              vgd = vgs - vds;

              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM4v2vbd);
              delvdbd = vdbd - *(ckt->CKTstate0 + here->BSIM4v2vdbd);
              delvgd = vgd - vgdo;

              delvds = vds - *(ckt->CKTstate0 + here->BSIM4v2vds);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM4v2vgs);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM4v2vbs);
              delvsbs = vsbs - *(ckt->CKTstate0 + here->BSIM4v2vsbs);

              delvses = vses - (*(ckt->CKTstate0 + here->BSIM4v2vses));
              vdedo = *(ckt->CKTstate0 + here->BSIM4v2vdes)
                    - *(ckt->CKTstate0 + here->BSIM4v2vds);
              delvdes = vdes - *(ckt->CKTstate0 + here->BSIM4v2vdes);
              delvded = vdes - vds - vdedo;

              delvbd_jct = (!here->BSIM4v2rbodyMod) ? delvbd : delvdbd;
              delvbs_jct = (!here->BSIM4v2rbodyMod) ? delvbs : delvsbs;

              if (here->BSIM4v2mode >= 0)
              {   Idtot = here->BSIM4v2cd + here->BSIM4v2csub - here->BSIM4v2cbd
			+ here->BSIM4v2Igidl;
                  cdhat = Idtot - here->BSIM4v2gbd * delvbd_jct
                        + (here->BSIM4v2gmbs + here->BSIM4v2gbbs + here->BSIM4v2ggidlb) * delvbs
                        + (here->BSIM4v2gm + here->BSIM4v2gbgs + here->BSIM4v2ggidlg) * delvgs
                        + (here->BSIM4v2gds + here->BSIM4v2gbds + here->BSIM4v2ggidld) * delvds;

                  Igstot = here->BSIM4v2Igs + here->BSIM4v2Igcs;
                  cgshat = Igstot + (here->BSIM4v2gIgsg + here->BSIM4v2gIgcsg) * delvgs
                         + here->BSIM4v2gIgcsd * delvds + here->BSIM4v2gIgcsb * delvbs;

                  Igdtot = here->BSIM4v2Igd + here->BSIM4v2Igcd;
                  cgdhat = Igdtot + here->BSIM4v2gIgdg * delvgd + here->BSIM4v2gIgcdg * delvgs
                         + here->BSIM4v2gIgcdd * delvds + here->BSIM4v2gIgcdb * delvbs;

                  Igbtot = here->BSIM4v2Igb;
                  cgbhat = here->BSIM4v2Igb + here->BSIM4v2gIgbg * delvgs + here->BSIM4v2gIgbd
                         * delvds + here->BSIM4v2gIgbb * delvbs;
              }
              else
               {   Idtot = here->BSIM4v2cd + here->BSIM4v2cbd - here->BSIM4v2Igisl;
                   cdhat = Idtot + here->BSIM4v2gbd * delvbd_jct + here->BSIM4v2gmbs 
                         * delvbd + here->BSIM4v2gm * delvgd 
                         - here->BSIM4v2gds * delvds - here->BSIM4v2ggislg * vgd 
                         - here->BSIM4v2ggislb * vbd + here->BSIM4v2ggisls * vds;

                  Igstot = here->BSIM4v2Igs + here->BSIM4v2Igcd;
                  cgshat = Igstot + here->BSIM4v2gIgsg * delvgs + here->BSIM4v2gIgcdg * delvgd
                         - here->BSIM4v2gIgcdd * delvds + here->BSIM4v2gIgcdb * delvbd;

                  Igdtot = here->BSIM4v2Igd + here->BSIM4v2Igcs;
                  cgdhat = Igdtot + (here->BSIM4v2gIgdg + here->BSIM4v2gIgcsg) * delvgd
                         - here->BSIM4v2gIgcsd * delvds + here->BSIM4v2gIgcsb * delvbd;

                  Igbtot = here->BSIM4v2Igb;
                  cgbhat = here->BSIM4v2Igb + here->BSIM4v2gIgbg * delvgd - here->BSIM4v2gIgbd
                         * delvds + here->BSIM4v2gIgbb * delvbd;
              }

              Isestot = here->BSIM4v2gstot * (*(ckt->CKTstate0 + here->BSIM4v2vses));
              cseshat = Isestot + here->BSIM4v2gstot * delvses
                      + here->BSIM4v2gstotd * delvds + here->BSIM4v2gstotg * delvgs
                      + here->BSIM4v2gstotb * delvbs;

              Idedtot = here->BSIM4v2gdtot * vdedo;
              cdedhat = Idedtot + here->BSIM4v2gdtot * delvded
                      + here->BSIM4v2gdtotd * delvds + here->BSIM4v2gdtotg * delvgs
                      + here->BSIM4v2gdtotb * delvbs;

              /*
               *  Check convergence
               */

              if ((here->BSIM4v2off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol0 = ckt->CKTreltol * MAX(fabs(cdhat), fabs(Idtot))
		       + ckt->CKTabstol;
                  tol1 = ckt->CKTreltol * MAX(fabs(cseshat), fabs(Isestot))
                       + ckt->CKTabstol;
                  tol2 = ckt->CKTreltol * MAX(fabs(cdedhat), fabs(Idedtot))
                       + ckt->CKTabstol;
                  tol3 = ckt->CKTreltol * MAX(fabs(cgshat), fabs(Igstot))
                       + ckt->CKTabstol;
                  tol4 = ckt->CKTreltol * MAX(fabs(cgdhat), fabs(Igdtot))
                       + ckt->CKTabstol;
                  tol5 = ckt->CKTreltol * MAX(fabs(cgbhat), fabs(Igbtot))
                       + ckt->CKTabstol;

                  if ((fabs(cdhat - Idtot) >= tol0) || (fabs(cseshat - Isestot) >= tol1)
                      || (fabs(cdedhat - Idedtot) >= tol2))
		  {   ckt->CKTnoncon++;
                      return(OK);
                  } 

                  if ((fabs(cgshat - Igstot) >= tol3) || (fabs(cgdhat - Igdtot) >= tol4)
                      || (fabs(cgbhat - Igbtot) >= tol5))
                  {   ckt->CKTnoncon++;
                      return(OK);
                  }

                  Ibtot = here->BSIM4v2cbs + here->BSIM4v2cbd
			- here->BSIM4v2Igidl - here->BSIM4v2Igisl - here->BSIM4v2csub;
                  if (here->BSIM4v2mode >= 0)
                  {   cbhat = Ibtot + here->BSIM4v2gbd * delvbd_jct
                            + here->BSIM4v2gbs * delvbs_jct - (here->BSIM4v2gbbs + here->BSIM4v2ggidlb)
                            * delvbs - (here->BSIM4v2gbgs + here->BSIM4v2ggidlg) * delvgs
                            - (here->BSIM4v2gbds + here->BSIM4v2ggidld) * delvds
			    - here->BSIM4v2ggislg * delvgd - here->BSIM4v2ggislb* delvbd + here->BSIM4v2ggisls * delvds ;
		  }
		  else
		  {   cbhat = Ibtot + here->BSIM4v2gbs * delvbs_jct + here->BSIM4v2gbd
                            * delvbd_jct - (here->BSIM4v2gbbs + here->BSIM4v2ggidlb) * delvbd
                            - (here->BSIM4v2gbgs + here->BSIM4v2ggidlg) * delvgd
                            + (here->BSIM4v2gbds + here->BSIM4v2ggidld) * delvds
			    - here->BSIM4v2ggislg * delvgs - here->BSIM4v2ggislb * delvbs + here->BSIM4v2ggisls * delvds;
		  }
                  tol6 = ckt->CKTreltol * MAX(fabs(cbhat), 
			fabs(Ibtot)) + ckt->CKTabstol;
                  if (fabs(cbhat - Ibtot) > tol6) 
		  {   ckt->CKTnoncon++;
                      return(OK);
                  }
              }
         }
    }
    return(OK);
}
