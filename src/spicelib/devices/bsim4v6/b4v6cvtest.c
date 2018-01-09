/**** BSIM4.6.2 Released by Wenwei Yang 04/05/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4cvtest.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v6convTest(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v6model *model = (BSIM4v6model*)inModel;
BSIM4v6instance *here;
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

    for (; model != NULL; model = BSIM4v6nextModel(model))
    {    for (here = BSIM4v6instances(model); here != NULL ;
              here=BSIM4v6nextInstance(here)) 
	 {
              vds = model->BSIM4v6type
                  * (*(ckt->CKTrhsOld + here->BSIM4v6dNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v6sNodePrime));
              vgs = model->BSIM4v6type
                  * (*(ckt->CKTrhsOld + here->BSIM4v6gNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v6sNodePrime));
              vbs = model->BSIM4v6type
                  * (*(ckt->CKTrhsOld + here->BSIM4v6bNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v6sNodePrime));
              vdbs = model->BSIM4v6type
                   * (*(ckt->CKTrhsOld + here->BSIM4v6dbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v6sNodePrime));
              vsbs = model->BSIM4v6type
                   * (*(ckt->CKTrhsOld + here->BSIM4v6sbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v6sNodePrime));              
              vses = model->BSIM4v6type
                   * (*(ckt->CKTrhsOld + here->BSIM4v6sNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v6sNodePrime));
              vdes = model->BSIM4v6type
                   * (*(ckt->CKTrhsOld + here->BSIM4v6dNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v6sNodePrime));
              
              vgdo = *(ckt->CKTstate0 + here->BSIM4v6vgs)
                    - *(ckt->CKTstate0 + here->BSIM4v6vds);
              vbd = vbs - vds;
              vdbd = vdbs - vds;
              vgd = vgs - vds;

              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM4v6vbd);
              delvdbd = vdbd - *(ckt->CKTstate0 + here->BSIM4v6vdbd);
              delvgd = vgd - vgdo;

              delvds = vds - *(ckt->CKTstate0 + here->BSIM4v6vds);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM4v6vgs);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM4v6vbs);
              delvsbs = vsbs - *(ckt->CKTstate0 + here->BSIM4v6vsbs);

              delvses = vses - (*(ckt->CKTstate0 + here->BSIM4v6vses));
              vdedo = *(ckt->CKTstate0 + here->BSIM4v6vdes)
                    - *(ckt->CKTstate0 + here->BSIM4v6vds);
              delvdes = vdes - *(ckt->CKTstate0 + here->BSIM4v6vdes);
              delvded = vdes - vds - vdedo;

              delvbd_jct = (!here->BSIM4v6rbodyMod) ? delvbd : delvdbd;
              delvbs_jct = (!here->BSIM4v6rbodyMod) ? delvbs : delvsbs;

              if (here->BSIM4v6mode >= 0)
              {   Idtot = here->BSIM4v6cd + here->BSIM4v6csub - here->BSIM4v6cbd
			+ here->BSIM4v6Igidl;
                  cdhat = Idtot - here->BSIM4v6gbd * delvbd_jct
                        + (here->BSIM4v6gmbs + here->BSIM4v6gbbs + here->BSIM4v6ggidlb) * delvbs
                        + (here->BSIM4v6gm + here->BSIM4v6gbgs + here->BSIM4v6ggidlg) * delvgs
                        + (here->BSIM4v6gds + here->BSIM4v6gbds + here->BSIM4v6ggidld) * delvds;

                  Igstot = here->BSIM4v6Igs + here->BSIM4v6Igcs;
                  cgshat = Igstot + (here->BSIM4v6gIgsg + here->BSIM4v6gIgcsg) * delvgs
                         + here->BSIM4v6gIgcsd * delvds + here->BSIM4v6gIgcsb * delvbs;

                  Igdtot = here->BSIM4v6Igd + here->BSIM4v6Igcd;
                  cgdhat = Igdtot + here->BSIM4v6gIgdg * delvgd + here->BSIM4v6gIgcdg * delvgs
                         + here->BSIM4v6gIgcdd * delvds + here->BSIM4v6gIgcdb * delvbs;

                  Igbtot = here->BSIM4v6Igb;
                  cgbhat = here->BSIM4v6Igb + here->BSIM4v6gIgbg * delvgs + here->BSIM4v6gIgbd
                         * delvds + here->BSIM4v6gIgbb * delvbs;
              }
              else
               {   Idtot = here->BSIM4v6cd + here->BSIM4v6cbd - here->BSIM4v6Igidl; /* bugfix */
                   cdhat = Idtot + here->BSIM4v6gbd * delvbd_jct + here->BSIM4v6gmbs 
                         * delvbd + here->BSIM4v6gm * delvgd 
                         - (here->BSIM4v6gds + here->BSIM4v6ggidls) * delvds 
                         - here->BSIM4v6ggidlg * delvgs - here->BSIM4v6ggidlb * delvbs;

                  Igstot = here->BSIM4v6Igs + here->BSIM4v6Igcd;
                  cgshat = Igstot + here->BSIM4v6gIgsg * delvgs + here->BSIM4v6gIgcdg * delvgd
                         - here->BSIM4v6gIgcdd * delvds + here->BSIM4v6gIgcdb * delvbd;

                  Igdtot = here->BSIM4v6Igd + here->BSIM4v6Igcs;
                  cgdhat = Igdtot + (here->BSIM4v6gIgdg + here->BSIM4v6gIgcsg) * delvgd
                         - here->BSIM4v6gIgcsd * delvds + here->BSIM4v6gIgcsb * delvbd;

                  Igbtot = here->BSIM4v6Igb;
                  cgbhat = here->BSIM4v6Igb + here->BSIM4v6gIgbg * delvgd - here->BSIM4v6gIgbd
                         * delvds + here->BSIM4v6gIgbb * delvbd;
              }

              Isestot = here->BSIM4v6gstot * (*(ckt->CKTstate0 + here->BSIM4v6vses));
              cseshat = Isestot + here->BSIM4v6gstot * delvses
                      + here->BSIM4v6gstotd * delvds + here->BSIM4v6gstotg * delvgs
                      + here->BSIM4v6gstotb * delvbs;

              Idedtot = here->BSIM4v6gdtot * vdedo;
              cdedhat = Idedtot + here->BSIM4v6gdtot * delvded
                      + here->BSIM4v6gdtotd * delvds + here->BSIM4v6gdtotg * delvgs
                      + here->BSIM4v6gdtotb * delvbs;

              /*
               *  Check convergence
               */

              if ((here->BSIM4v6off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
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

                  Ibtot = here->BSIM4v6cbs + here->BSIM4v6cbd
			- here->BSIM4v6Igidl - here->BSIM4v6Igisl - here->BSIM4v6csub;
                  if (here->BSIM4v6mode >= 0)
                  {   cbhat = Ibtot + here->BSIM4v6gbd * delvbd_jct
                            + here->BSIM4v6gbs * delvbs_jct - (here->BSIM4v6gbbs + here->BSIM4v6ggidlb)
                            * delvbs - (here->BSIM4v6gbgs + here->BSIM4v6ggidlg) * delvgs
                            - (here->BSIM4v6gbds + here->BSIM4v6ggidld) * delvds
			    - here->BSIM4v6ggislg * delvgd - here->BSIM4v6ggislb* delvbd + here->BSIM4v6ggisls * delvds ;
		  }
		  else
		  {   cbhat = Ibtot + here->BSIM4v6gbs * delvbs_jct + here->BSIM4v6gbd 
                         * delvbd_jct - (here->BSIM4v6gbbs + here->BSIM4v6ggislb) * delvbd
                         - (here->BSIM4v6gbgs + here->BSIM4v6ggislg) * delvgd
			 + (here->BSIM4v6gbds + here->BSIM4v6ggisld - here->BSIM4v6ggidls) * delvds
			 - here->BSIM4v6ggidlg * delvgs - here->BSIM4v6ggidlb * delvbs; 
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
