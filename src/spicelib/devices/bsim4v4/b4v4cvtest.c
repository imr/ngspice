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

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v4convTest(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v4model *model = (BSIM4v4model*)inModel;
BSIM4v4instance *here;
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

    for (; model != NULL; model = model->BSIM4v4nextModel)
    {    for (here = model->BSIM4v4instances; here != NULL ;
              here=here->BSIM4v4nextInstance) 
         {    
	      vds = model->BSIM4v4type
                  * (*(ckt->CKTrhsOld + here->BSIM4v4dNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v4sNodePrime));
              vgs = model->BSIM4v4type
                  * (*(ckt->CKTrhsOld + here->BSIM4v4gNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v4sNodePrime));
              vbs = model->BSIM4v4type
                  * (*(ckt->CKTrhsOld + here->BSIM4v4bNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v4sNodePrime));
              vdbs = model->BSIM4v4type
                   * (*(ckt->CKTrhsOld + here->BSIM4v4dbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v4sNodePrime));
              vsbs = model->BSIM4v4type
                   * (*(ckt->CKTrhsOld + here->BSIM4v4sbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v4sNodePrime));              
              vses = model->BSIM4v4type
                   * (*(ckt->CKTrhsOld + here->BSIM4v4sNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v4sNodePrime));
              vdes = model->BSIM4v4type
                   * (*(ckt->CKTrhsOld + here->BSIM4v4dNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v4sNodePrime));
              
              vgdo = *(ckt->CKTstate0 + here->BSIM4v4vgs)
                    - *(ckt->CKTstate0 + here->BSIM4v4vds);
              vbd = vbs - vds;
              vdbd = vdbs - vds;
              vgd = vgs - vds;

              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM4v4vbd);
              delvdbd = vdbd - *(ckt->CKTstate0 + here->BSIM4v4vdbd);
              delvgd = vgd - vgdo;

              delvds = vds - *(ckt->CKTstate0 + here->BSIM4v4vds);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM4v4vgs);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM4v4vbs);
              delvsbs = vsbs - *(ckt->CKTstate0 + here->BSIM4v4vsbs);

              delvses = vses - (*(ckt->CKTstate0 + here->BSIM4v4vses));
              vdedo = *(ckt->CKTstate0 + here->BSIM4v4vdes)
                    - *(ckt->CKTstate0 + here->BSIM4v4vds);
              delvdes = vdes - *(ckt->CKTstate0 + here->BSIM4v4vdes);
              delvded = vdes - vds - vdedo;

              delvbd_jct = (!here->BSIM4v4rbodyMod) ? delvbd : delvdbd;
              delvbs_jct = (!here->BSIM4v4rbodyMod) ? delvbs : delvsbs;

              if (here->BSIM4v4mode >= 0)
              {   Idtot = here->BSIM4v4cd + here->BSIM4v4csub - here->BSIM4v4cbd
			+ here->BSIM4v4Igidl;
                  cdhat = Idtot - here->BSIM4v4gbd * delvbd_jct
                        + (here->BSIM4v4gmbs + here->BSIM4v4gbbs + here->BSIM4v4ggidlb) * delvbs
                        + (here->BSIM4v4gm + here->BSIM4v4gbgs + here->BSIM4v4ggidlg) * delvgs
                        + (here->BSIM4v4gds + here->BSIM4v4gbds + here->BSIM4v4ggidld) * delvds;

                  Igstot = here->BSIM4v4Igs + here->BSIM4v4Igcs;
                  cgshat = Igstot + (here->BSIM4v4gIgsg + here->BSIM4v4gIgcsg) * delvgs
                         + here->BSIM4v4gIgcsd * delvds + here->BSIM4v4gIgcsb * delvbs;

                  Igdtot = here->BSIM4v4Igd + here->BSIM4v4Igcd;
                  cgdhat = Igdtot + here->BSIM4v4gIgdg * delvgd + here->BSIM4v4gIgcdg * delvgs
                         + here->BSIM4v4gIgcdd * delvds + here->BSIM4v4gIgcdb * delvbs;

                  Igbtot = here->BSIM4v4Igb;
                  cgbhat = here->BSIM4v4Igb + here->BSIM4v4gIgbg * delvgs + here->BSIM4v4gIgbd
                         * delvds + here->BSIM4v4gIgbb * delvbs;
              }
              else
               {   Idtot = here->BSIM4v4cd + here->BSIM4v4cbd - here->BSIM4v4Igidl; /* bugfix */
                   cdhat = Idtot + here->BSIM4v4gbd * delvbd_jct + here->BSIM4v4gmbs 
                         * delvbd + here->BSIM4v4gm * delvgd 
                         - (here->BSIM4v4gds + here->BSIM4v4ggidls) * delvds 
                         - here->BSIM4v4ggidlg * delvgs - here->BSIM4v4ggidlb * delvbs;

                  Igstot = here->BSIM4v4Igs + here->BSIM4v4Igcd;
                  cgshat = Igstot + here->BSIM4v4gIgsg * delvgs + here->BSIM4v4gIgcdg * delvgd
                         - here->BSIM4v4gIgcdd * delvds + here->BSIM4v4gIgcdb * delvbd;

                  Igdtot = here->BSIM4v4Igd + here->BSIM4v4Igcs;
                  cgdhat = Igdtot + (here->BSIM4v4gIgdg + here->BSIM4v4gIgcsg) * delvgd
                         - here->BSIM4v4gIgcsd * delvds + here->BSIM4v4gIgcsb * delvbd;

                  Igbtot = here->BSIM4v4Igb;
                  cgbhat = here->BSIM4v4Igb + here->BSIM4v4gIgbg * delvgd - here->BSIM4v4gIgbd
                         * delvds + here->BSIM4v4gIgbb * delvbd;
              }

              Isestot = here->BSIM4v4gstot * (*(ckt->CKTstate0 + here->BSIM4v4vses));
              cseshat = Isestot + here->BSIM4v4gstot * delvses
                      + here->BSIM4v4gstotd * delvds + here->BSIM4v4gstotg * delvgs
                      + here->BSIM4v4gstotb * delvbs;

              Idedtot = here->BSIM4v4gdtot * vdedo;
              cdedhat = Idedtot + here->BSIM4v4gdtot * delvded
                      + here->BSIM4v4gdtotd * delvds + here->BSIM4v4gdtotg * delvgs
                      + here->BSIM4v4gdtotb * delvbs;

              /*
               *  Check convergence
               */

              if ((here->BSIM4v4off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
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

                  Ibtot = here->BSIM4v4cbs + here->BSIM4v4cbd
			- here->BSIM4v4Igidl - here->BSIM4v4Igisl - here->BSIM4v4csub;
                  if (here->BSIM4v4mode >= 0)
                  {   cbhat = Ibtot + here->BSIM4v4gbd * delvbd_jct
                            + here->BSIM4v4gbs * delvbs_jct - (here->BSIM4v4gbbs + here->BSIM4v4ggidlb)
                            * delvbs - (here->BSIM4v4gbgs + here->BSIM4v4ggidlg) * delvgs
                            - (here->BSIM4v4gbds + here->BSIM4v4ggidld) * delvds
			    - here->BSIM4v4ggislg * delvgd - here->BSIM4v4ggislb* delvbd + here->BSIM4v4ggisls * delvds ;
		  }
		  else
		  {   cbhat = Ibtot + here->BSIM4v4gbs * delvbs_jct + here->BSIM4v4gbd 
                         * delvbd_jct - (here->BSIM4v4gbbs + here->BSIM4v4ggislb) * delvbd
                         - (here->BSIM4v4gbgs + here->BSIM4v4ggislg) * delvgd
			 + (here->BSIM4v4gbds + here->BSIM4v4ggisld - here->BSIM4v4ggidls) * delvds
			 - here->BSIM4v4ggidlg * delvgs - here->BSIM4v4ggidlb * delvbs; 
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
