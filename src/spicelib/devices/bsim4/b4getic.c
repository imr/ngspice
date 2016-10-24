/**** BSIM4.8.0 Released by Navid Paydavosi 11/01/2013 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4getic.c of BSIM4.7.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

int
BSIM4getic(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4model *model = (BSIM4model*)inModel;
BSIM4instance *here;

#ifdef USE_CUSPICE
    int i, status ;
#endif

    for (; model ; model = model->BSIM4nextModel) 
    {

#ifdef USE_CUSPICE
        i = 0 ;
#endif

        for (here = model->BSIM4instances; here; here = here->BSIM4nextInstance)
          {
              if (!here->BSIM4icVDSGiven) 
              {   here->BSIM4icVDS = *(ckt->CKTrhs + here->BSIM4dNode) 
                                   - *(ckt->CKTrhs + here->BSIM4sNode);
              }
              if (!here->BSIM4icVGSGiven) 
              {   here->BSIM4icVGS = *(ckt->CKTrhs + here->BSIM4gNodeExt) 
                                   - *(ckt->CKTrhs + here->BSIM4sNode);
              }
              if(!here->BSIM4icVBSGiven)
              {  here->BSIM4icVBS = *(ckt->CKTrhs + here->BSIM4bNode)
                                  - *(ckt->CKTrhs + here->BSIM4sNode);
              }

#ifdef USE_CUSPICE
            model->BSIM4paramCPU.BSIM4icVDSArray [i] = here->BSIM4icVDS ;
            model->BSIM4paramCPU.BSIM4icVGSArray [i] = here->BSIM4icVGS ;
            model->BSIM4paramCPU.BSIM4icVBSArray [i] = here->BSIM4icVBS ;

            i++ ;
#endif

         }

#ifdef USE_CUSPICE
        status = cuBSIM4getic ((GENmodel *)model) ;
        if (status != 0)
            return (E_NOMEM) ;
#endif

    }
    return (OK) ;
}
