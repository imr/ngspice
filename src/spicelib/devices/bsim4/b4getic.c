/* ******************************************************************************
   *  BSIM4 4.8.3 released on 05/19/2025                                        *
   *  BSIM4 Model Equations                                                     *
   ******************************************************************************

   ******************************************************************************
   *  Copyright (c) 2025 University of California                               *
   *                                                                            *
   *  Project Directors: Prof. Sayeef Salahuddin and Prof. Chenming Hu          *
   *  Developers list: https://www.bsim.berkeley.edu/models/bsim4/auth_bsim4/   * 
   ******************************************************************************/

/*
Licensed under Educational Community License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain a copy of the license at
http://opensource.org/licenses/ECL-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations
under the License.

BSIM4 model is supported by the members of Silicon Integration Initiative's Compact Model Coalition. A link to the most recent version of this
standard can be found at: http://www.si2.org/cmc 
*/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4getic(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4model *model = (BSIM4model*)inModel;
BSIM4instance *here;

    for (; model ; model = BSIM4nextModel(model)) 
    {    for (here = BSIM4instances(model); here; here = BSIM4nextInstance(here))
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
            {   here->BSIM4icVBS = *(ckt->CKTrhs + here->BSIM4bNode)
                                - *(ckt->CKTrhs + here->BSIM4sNode);
            }
        }
    }
    return(OK);
}
