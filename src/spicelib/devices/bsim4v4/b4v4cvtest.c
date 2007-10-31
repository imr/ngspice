/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4cvtest.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim4v4def.h"
#include "trandefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM4V4convTest(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM4V4model *model = (BSIM4V4model*)inModel;
BSIM4V4instance *here;
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

    for (; model != NULL; model = model->BSIM4V4nextModel)
    {    for (here = model->BSIM4V4instances; here != NULL ;
              here=here->BSIM4V4nextInstance) 
         {    
	            if (here->BSIM4V4owner != ARCHme) continue; 
	 	          vds = model->BSIM4V4type
                  * (*(ckt->CKTrhsOld + here->BSIM4V4dNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4V4sNodePrime));
              vgs = model->BSIM4V4type
                  * (*(ckt->CKTrhsOld + here->BSIM4V4gNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4V4sNodePrime));
              vbs = model->BSIM4V4type
                  * (*(ckt->CKTrhsOld + here->BSIM4V4bNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4V4sNodePrime));
              vdbs = model->BSIM4V4type
                   * (*(ckt->CKTrhsOld + here->BSIM4V4dbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4V4sNodePrime));
              vsbs = model->BSIM4V4type
                   * (*(ckt->CKTrhsOld + here->BSIM4V4sbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4V4sNodePrime));              
              vses = model->BSIM4V4type
                   * (*(ckt->CKTrhsOld + here->BSIM4V4sNode)
                   - *(ckt->CKTrhsOld + here->BSIM4V4sNodePrime));
              vdes = model->BSIM4V4type
                   * (*(ckt->CKTrhsOld + here->BSIM4V4dNode)
                   - *(ckt->CKTrhsOld + here->BSIM4V4sNodePrime));
              
              vgdo = *(ckt->CKTstate0 + here->BSIM4V4vgs)
                    - *(ckt->CKTstate0 + here->BSIM4V4vds);
              vbd = vbs - vds;
              vdbd = vdbs - vds;
              vgd = vgs - vds;

              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM4V4vbd);
              delvdbd = vdbd - *(ckt->CKTstate0 + here->BSIM4V4vdbd);
              delvgd = vgd - vgdo;

              delvds = vds - *(ckt->CKTstate0 + here->BSIM4V4vds);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM4V4vgs);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM4V4vbs);
              delvsbs = vsbs - *(ckt->CKTstate0 + here->BSIM4V4vsbs);

              delvses = vses - (*(ckt->CKTstate0 + here->BSIM4V4vses));
              vdedo = *(ckt->CKTstate0 + here->BSIM4V4vdes)
                    - *(ckt->CKTstate0 + here->BSIM4V4vds);
              delvdes = vdes - *(ckt->CKTstate0 + here->BSIM4V4vdes);
              delvded = vdes - vds - vdedo;

              delvbd_jct = (!here->BSIM4V4rbodyMod) ? delvbd : delvdbd;
              delvbs_jct = (!here->BSIM4V4rbodyMod) ? delvbs : delvsbs;

              if (here->BSIM4V4mode >= 0)
              {   Idtot = here->BSIM4V4cd + here->BSIM4V4csub - here->BSIM4V4cbd
			+ here->BSIM4V4Igidl;
                  cdhat = Idtot - here->BSIM4V4gbd * delvbd_jct
                        + (here->BSIM4V4gmbs + here->BSIM4V4gbbs + here->BSIM4V4ggidlb) * delvbs
                        + (here->BSIM4V4gm + here->BSIM4V4gbgs + here->BSIM4V4ggidlg) * delvgs
                        + (here->BSIM4V4gds + here->BSIM4V4gbds + here->BSIM4V4ggidld) * delvds;

                  Igstot = here->BSIM4V4Igs + here->BSIM4V4Igcs;
                  cgshat = Igstot + (here->BSIM4V4gIgsg + here->BSIM4V4gIgcsg) * delvgs
                         + here->BSIM4V4gIgcsd * delvds + here->BSIM4V4gIgcsb * delvbs;

                  Igdtot = here->BSIM4V4Igd + here->BSIM4V4Igcd;
                  cgdhat = Igdtot + here->BSIM4V4gIgdg * delvgd + here->BSIM4V4gIgcdg * delvgs
                         + here->BSIM4V4gIgcdd * delvds + here->BSIM4V4gIgcdb * delvbs;

                  Igbtot = here->BSIM4V4Igb;
                  cgbhat = here->BSIM4V4Igb + here->BSIM4V4gIgbg * delvgs + here->BSIM4V4gIgbd
                         * delvds + here->BSIM4V4gIgbb * delvbs;
              }
              else
               {   Idtot = here->BSIM4V4cd + here->BSIM4V4cbd - here->BSIM4V4Igidl; /* bugfix */
                   cdhat = Idtot + here->BSIM4V4gbd * delvbd_jct + here->BSIM4V4gmbs 
                         * delvbd + here->BSIM4V4gm * delvgd 
                         - (here->BSIM4V4gds + here->BSIM4V4ggidls) * delvds 
                         - here->BSIM4V4ggidlg * delvgs - here->BSIM4V4ggidlb * delvbs;

                  Igstot = here->BSIM4V4Igs + here->BSIM4V4Igcd;
                  cgshat = Igstot + here->BSIM4V4gIgsg * delvgs + here->BSIM4V4gIgcdg * delvgd
                         - here->BSIM4V4gIgcdd * delvds + here->BSIM4V4gIgcdb * delvbd;

                  Igdtot = here->BSIM4V4Igd + here->BSIM4V4Igcs;
                  cgdhat = Igdtot + (here->BSIM4V4gIgdg + here->BSIM4V4gIgcsg) * delvgd
                         - here->BSIM4V4gIgcsd * delvds + here->BSIM4V4gIgcsb * delvbd;

                  Igbtot = here->BSIM4V4Igb;
                  cgbhat = here->BSIM4V4Igb + here->BSIM4V4gIgbg * delvgd - here->BSIM4V4gIgbd
                         * delvds + here->BSIM4V4gIgbb * delvbd;
              }

              Isestot = here->BSIM4V4gstot * (*(ckt->CKTstate0 + here->BSIM4V4vses));
              cseshat = Isestot + here->BSIM4V4gstot * delvses
                      + here->BSIM4V4gstotd * delvds + here->BSIM4V4gstotg * delvgs
                      + here->BSIM4V4gstotb * delvbs;

              Idedtot = here->BSIM4V4gdtot * vdedo;
              cdedhat = Idedtot + here->BSIM4V4gdtot * delvded
                      + here->BSIM4V4gdtotd * delvds + here->BSIM4V4gdtotg * delvgs
                      + here->BSIM4V4gdtotb * delvbs;

              /*
               *  Check convergence
               */

              if ((here->BSIM4V4off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
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

                  Ibtot = here->BSIM4V4cbs + here->BSIM4V4cbd
			- here->BSIM4V4Igidl - here->BSIM4V4Igisl - here->BSIM4V4csub;
                  if (here->BSIM4V4mode >= 0)
                  {   cbhat = Ibtot + here->BSIM4V4gbd * delvbd_jct
                            + here->BSIM4V4gbs * delvbs_jct - (here->BSIM4V4gbbs + here->BSIM4V4ggidlb)
                            * delvbs - (here->BSIM4V4gbgs + here->BSIM4V4ggidlg) * delvgs
                            - (here->BSIM4V4gbds + here->BSIM4V4ggidld) * delvds
			    - here->BSIM4V4ggislg * delvgd - here->BSIM4V4ggislb* delvbd + here->BSIM4V4ggisls * delvds ;
		  }
		  else
		  {   cbhat = Ibtot + here->BSIM4V4gbs * delvbs_jct + here->BSIM4V4gbd 
                         * delvbd_jct - (here->BSIM4V4gbbs + here->BSIM4V4ggislb) * delvbd
                         - (here->BSIM4V4gbgs + here->BSIM4V4ggislg) * delvgd
			 + (here->BSIM4V4gbds + here->BSIM4V4ggisld - here->BSIM4V4ggidls) * delvds
			 - here->BSIM4V4ggidlg * delvgs - here->BSIM4V4ggidlb * delvbs; 
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
