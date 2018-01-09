/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4cvtest.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v5convTest(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v5model *model = (BSIM4v5model*)inModel;
BSIM4v5instance *here;
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

    for (; model != NULL; model = BSIM4v5nextModel(model))
    {    for (here = BSIM4v5instances(model); here != NULL ;
              here=BSIM4v5nextInstance(here)) 
         {
	      vds = model->BSIM4v5type
                  * (*(ckt->CKTrhsOld + here->BSIM4v5dNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
              vgs = model->BSIM4v5type
                  * (*(ckt->CKTrhsOld + here->BSIM4v5gNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
              vbs = model->BSIM4v5type
                  * (*(ckt->CKTrhsOld + here->BSIM4v5bNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
              vdbs = model->BSIM4v5type
                   * (*(ckt->CKTrhsOld + here->BSIM4v5dbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
              vsbs = model->BSIM4v5type
                   * (*(ckt->CKTrhsOld + here->BSIM4v5sbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));              
              vses = model->BSIM4v5type
                   * (*(ckt->CKTrhsOld + here->BSIM4v5sNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
              vdes = model->BSIM4v5type
                   * (*(ckt->CKTrhsOld + here->BSIM4v5dNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v5sNodePrime));
              
              vgdo = *(ckt->CKTstate0 + here->BSIM4v5vgs)
                    - *(ckt->CKTstate0 + here->BSIM4v5vds);
              vbd = vbs - vds;
              vdbd = vdbs - vds;
              vgd = vgs - vds;

              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM4v5vbd);
              delvdbd = vdbd - *(ckt->CKTstate0 + here->BSIM4v5vdbd);
              delvgd = vgd - vgdo;

              delvds = vds - *(ckt->CKTstate0 + here->BSIM4v5vds);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM4v5vgs);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM4v5vbs);
              delvsbs = vsbs - *(ckt->CKTstate0 + here->BSIM4v5vsbs);

              delvses = vses - (*(ckt->CKTstate0 + here->BSIM4v5vses));
              vdedo = *(ckt->CKTstate0 + here->BSIM4v5vdes)
                    - *(ckt->CKTstate0 + here->BSIM4v5vds);
              delvdes = vdes - *(ckt->CKTstate0 + here->BSIM4v5vdes);
              delvded = vdes - vds - vdedo;

              delvbd_jct = (!here->BSIM4v5rbodyMod) ? delvbd : delvdbd;
              delvbs_jct = (!here->BSIM4v5rbodyMod) ? delvbs : delvsbs;

              if (here->BSIM4v5mode >= 0)
              {   Idtot = here->BSIM4v5cd + here->BSIM4v5csub - here->BSIM4v5cbd
			+ here->BSIM4v5Igidl;
                  cdhat = Idtot - here->BSIM4v5gbd * delvbd_jct
                        + (here->BSIM4v5gmbs + here->BSIM4v5gbbs + here->BSIM4v5ggidlb) * delvbs
                        + (here->BSIM4v5gm + here->BSIM4v5gbgs + here->BSIM4v5ggidlg) * delvgs
                        + (here->BSIM4v5gds + here->BSIM4v5gbds + here->BSIM4v5ggidld) * delvds;

                  Igstot = here->BSIM4v5Igs + here->BSIM4v5Igcs;
                  cgshat = Igstot + (here->BSIM4v5gIgsg + here->BSIM4v5gIgcsg) * delvgs
                         + here->BSIM4v5gIgcsd * delvds + here->BSIM4v5gIgcsb * delvbs;

                  Igdtot = here->BSIM4v5Igd + here->BSIM4v5Igcd;
                  cgdhat = Igdtot + here->BSIM4v5gIgdg * delvgd + here->BSIM4v5gIgcdg * delvgs
                         + here->BSIM4v5gIgcdd * delvds + here->BSIM4v5gIgcdb * delvbs;

                  Igbtot = here->BSIM4v5Igb;
                  cgbhat = here->BSIM4v5Igb + here->BSIM4v5gIgbg * delvgs + here->BSIM4v5gIgbd
                         * delvds + here->BSIM4v5gIgbb * delvbs;
              }
              else
               {   Idtot = here->BSIM4v5cd + here->BSIM4v5cbd - here->BSIM4v5Igidl; /* bugfix */
                   cdhat = Idtot + here->BSIM4v5gbd * delvbd_jct + here->BSIM4v5gmbs 
                         * delvbd + here->BSIM4v5gm * delvgd 
                         - (here->BSIM4v5gds + here->BSIM4v5ggidls) * delvds 
                         - here->BSIM4v5ggidlg * delvgs - here->BSIM4v5ggidlb * delvbs;

                  Igstot = here->BSIM4v5Igs + here->BSIM4v5Igcd;
                  cgshat = Igstot + here->BSIM4v5gIgsg * delvgs + here->BSIM4v5gIgcdg * delvgd
                         - here->BSIM4v5gIgcdd * delvds + here->BSIM4v5gIgcdb * delvbd;

                  Igdtot = here->BSIM4v5Igd + here->BSIM4v5Igcs;
                  cgdhat = Igdtot + (here->BSIM4v5gIgdg + here->BSIM4v5gIgcsg) * delvgd
                         - here->BSIM4v5gIgcsd * delvds + here->BSIM4v5gIgcsb * delvbd;

                  Igbtot = here->BSIM4v5Igb;
                  cgbhat = here->BSIM4v5Igb + here->BSIM4v5gIgbg * delvgd - here->BSIM4v5gIgbd
                         * delvds + here->BSIM4v5gIgbb * delvbd;
              }

              Isestot = here->BSIM4v5gstot * (*(ckt->CKTstate0 + here->BSIM4v5vses));
              cseshat = Isestot + here->BSIM4v5gstot * delvses
                      + here->BSIM4v5gstotd * delvds + here->BSIM4v5gstotg * delvgs
                      + here->BSIM4v5gstotb * delvbs;

              Idedtot = here->BSIM4v5gdtot * vdedo;
              cdedhat = Idedtot + here->BSIM4v5gdtot * delvded
                      + here->BSIM4v5gdtotd * delvds + here->BSIM4v5gdtotg * delvgs
                      + here->BSIM4v5gdtotb * delvbs;

              /*
               *  Check convergence
               */

              if ((here->BSIM4v5off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
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

                  Ibtot = here->BSIM4v5cbs + here->BSIM4v5cbd
			- here->BSIM4v5Igidl - here->BSIM4v5Igisl - here->BSIM4v5csub;
                  if (here->BSIM4v5mode >= 0)
                  {   cbhat = Ibtot + here->BSIM4v5gbd * delvbd_jct
                            + here->BSIM4v5gbs * delvbs_jct - (here->BSIM4v5gbbs + here->BSIM4v5ggidlb)
                            * delvbs - (here->BSIM4v5gbgs + here->BSIM4v5ggidlg) * delvgs
                            - (here->BSIM4v5gbds + here->BSIM4v5ggidld) * delvds
			    - here->BSIM4v5ggislg * delvgd - here->BSIM4v5ggislb* delvbd + here->BSIM4v5ggisls * delvds ;
		  }
		  else
		  {   cbhat = Ibtot + here->BSIM4v5gbs * delvbs_jct + here->BSIM4v5gbd 
                         * delvbd_jct - (here->BSIM4v5gbbs + here->BSIM4v5ggislb) * delvbd
                         - (here->BSIM4v5gbgs + here->BSIM4v5ggislg) * delvgd
			 + (here->BSIM4v5gbds + here->BSIM4v5ggisld - here->BSIM4v5ggidls) * delvds
			 - here->BSIM4v5ggidlg * delvgs - here->BSIM4v5ggidlb * delvbs; 
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
