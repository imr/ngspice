/**********
Author: 2015 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

int
CKTreliability (CKTcircuit *ckt, unsigned int mode)
{
    int error, i ;

    for (i = 0 ; i < DEVmaxnum ; i++)
    {
        if (DEVices [i] && DEVices [i]->DEVreliability && ckt->CKThead [i])
        {
            error = DEVices [i]->DEVreliability (ckt->CKThead [i], ckt, mode) ;
        }
    }

    return (OK) ;
}
