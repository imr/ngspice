/**** BSIM4.3.0 Released by Xuemei (Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3cvtest.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim4v3def.h"
#include "trandefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"


int
BSIM4v3convTest(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM4v3model *model = (BSIM4v3model*)inModel;
BSIM4v3instance *here;
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

    for (; model != NULL; model = model->BSIM4v3nextModel)
    {    for (here = model->BSIM4v3instances; here != NULL ;
              here=here->BSIM4v3nextInstance) 
         {        
	      if (here->BSIM4v3owner != ARCHme) continue;

	      vds = model->BSIM4v3type
                  * (*(ckt->CKTrhsOld + here->BSIM4v3dNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v3sNodePrime));
              vgs = model->BSIM4v3type
                  * (*(ckt->CKTrhsOld + here->BSIM4v3gNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v3sNodePrime));
              vbs = model->BSIM4v3type
                  * (*(ckt->CKTrhsOld + here->BSIM4v3bNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v3sNodePrime));
              vdbs = model->BSIM4v3type
                   * (*(ckt->CKTrhsOld + here->BSIM4v3dbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v3sNodePrime));
              vsbs = model->BSIM4v3type
                   * (*(ckt->CKTrhsOld + here->BSIM4v3sbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v3sNodePrime));              
              vses = model->BSIM4v3type
                   * (*(ckt->CKTrhsOld + here->BSIM4v3sNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v3sNodePrime));
              vdes = model->BSIM4v3type
                   * (*(ckt->CKTrhsOld + here->BSIM4v3dNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v3sNodePrime));
              
              vgdo = *(ckt->CKTstate0 + here->BSIM4v3vgs)
                    - *(ckt->CKTstate0 + here->BSIM4v3vds);
              vbd = vbs - vds;
              vdbd = vdbs - vds;
              vgd = vgs - vds;

              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM4v3vbd);
              delvdbd = vdbd - *(ckt->CKTstate0 + here->BSIM4v3vdbd);
              delvgd = vgd - vgdo;

              delvds = vds - *(ckt->CKTstate0 + here->BSIM4v3vds);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM4v3vgs);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM4v3vbs);
              delvsbs = vsbs - *(ckt->CKTstate0 + here->BSIM4v3vsbs);

              delvses = vses - (*(ckt->CKTstate0 + here->BSIM4v3vses));
              vdedo = *(ckt->CKTstate0 + here->BSIM4v3vdes)
                    - *(ckt->CKTstate0 + here->BSIM4v3vds);
              delvdes = vdes - *(ckt->CKTstate0 + here->BSIM4v3vdes);
              delvded = vdes - vds - vdedo;

              delvbd_jct = (!here->BSIM4v3rbodyMod) ? delvbd : delvdbd;
              delvbs_jct = (!here->BSIM4v3rbodyMod) ? delvbs : delvsbs;

              if (here->BSIM4v3mode >= 0)
              {   Idtot = here->BSIM4v3cd + here->BSIM4v3csub - here->BSIM4v3cbd
			+ here->BSIM4v3Igidl;
                  cdhat = Idtot - here->BSIM4v3gbd * delvbd_jct
                        + (here->BSIM4v3gmbs + here->BSIM4v3gbbs + here->BSIM4v3ggidlb) * delvbs
                        + (here->BSIM4v3gm + here->BSIM4v3gbgs + here->BSIM4v3ggidlg) * delvgs
                        + (here->BSIM4v3gds + here->BSIM4v3gbds + here->BSIM4v3ggidld) * delvds;

                  Igstot = here->BSIM4v3Igs + here->BSIM4v3Igcs;
                  cgshat = Igstot + (here->BSIM4v3gIgsg + here->BSIM4v3gIgcsg) * delvgs
                         + here->BSIM4v3gIgcsd * delvds + here->BSIM4v3gIgcsb * delvbs;

                  Igdtot = here->BSIM4v3Igd + here->BSIM4v3Igcd;
                  cgdhat = Igdtot + here->BSIM4v3gIgdg * delvgd + here->BSIM4v3gIgcdg * delvgs
                         + here->BSIM4v3gIgcdd * delvds + here->BSIM4v3gIgcdb * delvbs;

                  Igbtot = here->BSIM4v3Igb;
                  cgbhat = here->BSIM4v3Igb + here->BSIM4v3gIgbg * delvgs + here->BSIM4v3gIgbd
                         * delvds + here->BSIM4v3gIgbb * delvbs;
              }
              else
               {   Idtot = here->BSIM4v3cd + here->BSIM4v3cbd - here->BSIM4v3Igidl; /* bugfix */
                   cdhat = Idtot + here->BSIM4v3gbd * delvbd_jct + here->BSIM4v3gmbs 
                         * delvbd + here->BSIM4v3gm * delvgd 
                         - (here->BSIM4v3gds + here->BSIM4v3ggidls) * delvds 
                         - here->BSIM4v3ggidlg * delvgs - here->BSIM4v3ggidlb * delvbs;

                  Igstot = here->BSIM4v3Igs + here->BSIM4v3Igcd;
                  cgshat = Igstot + here->BSIM4v3gIgsg * delvgs + here->BSIM4v3gIgcdg * delvgd
                         - here->BSIM4v3gIgcdd * delvds + here->BSIM4v3gIgcdb * delvbd;

                  Igdtot = here->BSIM4v3Igd + here->BSIM4v3Igcs;
                  cgdhat = Igdtot + (here->BSIM4v3gIgdg + here->BSIM4v3gIgcsg) * delvgd
                         - here->BSIM4v3gIgcsd * delvds + here->BSIM4v3gIgcsb * delvbd;

                  Igbtot = here->BSIM4v3Igb;
                  cgbhat = here->BSIM4v3Igb + here->BSIM4v3gIgbg * delvgd - here->BSIM4v3gIgbd
                         * delvds + here->BSIM4v3gIgbb * delvbd;
              }

              Isestot = here->BSIM4v3gstot * (*(ckt->CKTstate0 + here->BSIM4v3vses));
              cseshat = Isestot + here->BSIM4v3gstot * delvses
                      + here->BSIM4v3gstotd * delvds + here->BSIM4v3gstotg * delvgs
                      + here->BSIM4v3gstotb * delvbs;

              Idedtot = here->BSIM4v3gdtot * vdedo;
              cdedhat = Idedtot + here->BSIM4v3gdtot * delvded
                      + here->BSIM4v3gdtotd * delvds + here->BSIM4v3gdtotg * delvgs
                      + here->BSIM4v3gdtotb * delvbs;

              /*
               *  Check convergence
               */

              if ((here->BSIM4v3off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
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

                  Ibtot = here->BSIM4v3cbs + here->BSIM4v3cbd
			- here->BSIM4v3Igidl - here->BSIM4v3Igisl - here->BSIM4v3csub;
                  if (here->BSIM4v3mode >= 0)
                  {   cbhat = Ibtot + here->BSIM4v3gbd * delvbd_jct
                            + here->BSIM4v3gbs * delvbs_jct - (here->BSIM4v3gbbs + here->BSIM4v3ggidlb)
                            * delvbs - (here->BSIM4v3gbgs + here->BSIM4v3ggidlg) * delvgs
                            - (here->BSIM4v3gbds + here->BSIM4v3ggidld) * delvds
			    - here->BSIM4v3ggislg * delvgd - here->BSIM4v3ggislb* delvbd + here->BSIM4v3ggisls * delvds ;
		  }
		  else
		  {   cbhat = Ibtot + here->BSIM4v3gbs * delvbs_jct + here->BSIM4v3gbd 
                         * delvbd_jct - (here->BSIM4v3gbbs + here->BSIM4v3ggislb) * delvbd
                         - (here->BSIM4v3gbgs + here->BSIM4v3ggislg) * delvgd
			 + (here->BSIM4v3gbds + here->BSIM4v3ggisld - here->BSIM4v3ggidls) * delvds
			 - here->BSIM4v3ggidlg * delvgs - here->BSIM4v3ggidlb * delvbs; 
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
