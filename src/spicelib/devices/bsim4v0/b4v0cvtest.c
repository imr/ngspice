/**** BSIM4v0.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4cvtest.c of BSIM4v0.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v0def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v0convTest(inModel,ckt)
GENmodel *inModel;
register CKTcircuit *ckt;
{
register BSIM4v0model *model = (BSIM4v0model*)inModel;
register BSIM4v0instance *here;
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

    for (; model != NULL; model = model->BSIM4v0nextModel)
    {    for (here = model->BSIM4v0instances; here != NULL ;
              here=here->BSIM4v0nextInstance) 
	 {    vds = model->BSIM4v0type
                  * (*(ckt->CKTrhsOld + here->BSIM4v0dNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v0sNodePrime));
              vgs = model->BSIM4v0type
                  * (*(ckt->CKTrhsOld + here->BSIM4v0gNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v0sNodePrime));
              vbs = model->BSIM4v0type
                  * (*(ckt->CKTrhsOld + here->BSIM4v0bNodePrime)
                  - *(ckt->CKTrhsOld + here->BSIM4v0sNodePrime));
              vdbs = model->BSIM4v0type
                   * (*(ckt->CKTrhsOld + here->BSIM4v0dbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v0sNodePrime));
              vsbs = model->BSIM4v0type
                   * (*(ckt->CKTrhsOld + here->BSIM4v0sbNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v0sNodePrime));              
              vses = model->BSIM4v0type
                   * (*(ckt->CKTrhsOld + here->BSIM4v0sNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v0sNodePrime));
              vdes = model->BSIM4v0type
                   * (*(ckt->CKTrhsOld + here->BSIM4v0dNode)
                   - *(ckt->CKTrhsOld + here->BSIM4v0sNodePrime));
              
              vgdo = *(ckt->CKTstate0 + here->BSIM4v0vgs)
                    - *(ckt->CKTstate0 + here->BSIM4v0vds);
              vbd = vbs - vds;
              vdbd = vdbs - vds;
              vgd = vgs - vds;

              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM4v0vbd);
              delvdbd = vdbd - *(ckt->CKTstate0 + here->BSIM4v0vdbd);
              delvgd = vgd - vgdo;

              delvds = vds - *(ckt->CKTstate0 + here->BSIM4v0vds);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM4v0vgs);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM4v0vbs);
              delvsbs = vsbs - *(ckt->CKTstate0 + here->BSIM4v0vsbs);

              delvses = vses - (*(ckt->CKTstate0 + here->BSIM4v0vses));
              vdedo = *(ckt->CKTstate0 + here->BSIM4v0vdes)
                    - *(ckt->CKTstate0 + here->BSIM4v0vds);
              delvdes = vdes - *(ckt->CKTstate0 + here->BSIM4v0vdes);
              delvded = vdes - vds - vdedo;

              delvbd_jct = (!here->BSIM4v0rbodyMod) ? delvbd : delvdbd;
              delvbs_jct = (!here->BSIM4v0rbodyMod) ? delvbs : delvsbs;

              if (here->BSIM4v0mode >= 0)
              {   Idtot = here->BSIM4v0cd + here->BSIM4v0csub - here->BSIM4v0cbd
			+ here->BSIM4v0Igidl;
                  cdhat = Idtot - here->BSIM4v0gbd * delvbd_jct
                        + (here->BSIM4v0gmbs + here->BSIM4v0gbbs + here->BSIM4v0ggidlb) * delvbs
                        + (here->BSIM4v0gm + here->BSIM4v0gbgs + here->BSIM4v0ggidlg) * delvgs
                        + (here->BSIM4v0gds + here->BSIM4v0gbds + here->BSIM4v0ggidld) * delvds;

                  Igstot = here->BSIM4v0Igs + here->BSIM4v0Igcs;
                  cgshat = Igstot + (here->BSIM4v0gIgsg + here->BSIM4v0gIgcsg) * delvgs
                         + here->BSIM4v0gIgcsd * delvds + here->BSIM4v0gIgcsb * delvbs;

                  Igdtot = here->BSIM4v0Igd + here->BSIM4v0Igcd;
                  cgdhat = Igdtot + here->BSIM4v0gIgdg * delvgd + here->BSIM4v0gIgcdg * delvgs
                         + here->BSIM4v0gIgcdd * delvds + here->BSIM4v0gIgcdb * delvbs;

                  Igbtot = here->BSIM4v0Igb;
                  cgbhat = here->BSIM4v0Igb + here->BSIM4v0gIgbg * delvgs + here->BSIM4v0gIgbd
                         * delvds + here->BSIM4v0gIgbb * delvbs;
              }
              else
              {   Idtot = here->BSIM4v0cd + here->BSIM4v0cbd;
                  cdhat = Idtot + here->BSIM4v0gbd * delvbd_jct + here->BSIM4v0gmbs
                        * delvbd + here->BSIM4v0gm * delvgd
                        - here->BSIM4v0gds * delvds;

                  Igstot = here->BSIM4v0Igs + here->BSIM4v0Igcd;
                  cgshat = Igstot + here->BSIM4v0gIgsg * delvgs + here->BSIM4v0gIgcdg * delvgd
                         - here->BSIM4v0gIgcdd * delvds + here->BSIM4v0gIgcdb * delvbd;

                  Igdtot = here->BSIM4v0Igd + here->BSIM4v0Igcs;
                  cgdhat = Igdtot + (here->BSIM4v0gIgdg + here->BSIM4v0gIgcsg) * delvgd
                         - here->BSIM4v0gIgcsd * delvds + here->BSIM4v0gIgcsb * delvbd;

                  Igbtot = here->BSIM4v0Igb;
                  cgbhat = here->BSIM4v0Igb + here->BSIM4v0gIgbg * delvgd - here->BSIM4v0gIgbd
                         * delvds + here->BSIM4v0gIgbb * delvbd;
              }

              Isestot = here->BSIM4v0gstot * (*(ckt->CKTstate0 + here->BSIM4v0vses));
              cseshat = Isestot + here->BSIM4v0gstot * delvses
                      + here->BSIM4v0gstotd * delvds + here->BSIM4v0gstotg * delvgs
                      + here->BSIM4v0gstotb * delvbs;

              Idedtot = here->BSIM4v0gdtot * vdedo;
              cdedhat = Idedtot + here->BSIM4v0gdtot * delvded
                      + here->BSIM4v0gdtotd * delvds + here->BSIM4v0gdtotg * delvgs
                      + here->BSIM4v0gdtotb * delvbs;

              /*
               *  Check convergence
               */

              if ((here->BSIM4v0off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
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

                  Ibtot = here->BSIM4v0cbs + here->BSIM4v0cbd
			- here->BSIM4v0Igidl - here->BSIM4v0csub;
                  if (here->BSIM4v0mode >= 0)
                  {   cbhat = Ibtot + here->BSIM4v0gbd * delvbd_jct
                            + here->BSIM4v0gbs * delvbs_jct - (here->BSIM4v0gbbs + here->BSIM4v0ggidlb)
                            * delvbs - (here->BSIM4v0gbgs + here->BSIM4v0ggidlg) * delvgs
                            - (here->BSIM4v0gbds + here->BSIM4v0ggidld) * delvds;
		  }
		  else
		  {   cbhat = Ibtot + here->BSIM4v0gbs * delvbs_jct + here->BSIM4v0gbd
                            * delvbd_jct - (here->BSIM4v0gbbs + here->BSIM4v0ggidlb) * delvbd
                            - (here->BSIM4v0gbgs + here->BSIM4v0ggidlg) * delvgd
                            + (here->BSIM4v0gbds + here->BSIM4v0ggidld) * delvds;
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
