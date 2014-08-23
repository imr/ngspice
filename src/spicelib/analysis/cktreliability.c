/**********
Author: Francesco Lannutti - August 2014
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

int
CKTagingAdd (CKTcircuit *ckt, int step)
{
    int i, error ;

    /* Update delvto for every instance */
    for (i = 0 ; i < DEVmaxnum ; i++)
    {
        if (i == 10)
        {
            /* BSIM4 */
            if (DEVices[i] && DEVices[i]->DEVagingAdd && ckt->CKThead[i])
            {
                error = DEVices[i]->DEVagingAdd (ckt->CKThead [i], step) ;
                if (error)
                {
                    return (error) ;
                }
            }
        }
    }

    return (OK) ;
}

int
CKTagingSetup (CKTcircuit *ckt)
{
    int i, error ;

    /* Extract Vth and calculate delvto for every instance */
    for (i = 0 ; i < DEVmaxnum ; i++)
    {
        if (i == 10)
        {
            /* BSIM4 */
            if (DEVices[i] && DEVices[i]->DEVagingSetup && ckt->CKThead[i])
            {
                error = DEVices[i]->DEVagingSetup (ckt->CKThead [i], ckt) ;
                if (error)
                {
                    return (error) ;
                }
            }
        }
    }

    return (OK) ;
}
