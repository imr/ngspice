/**********
Permit to use it as your wish.
Author: 2007 Gong Ding, gdiso@ustc.edu
University of Science and Technology of China
**********/

#include "ngspice/ngspice.h"
#include "ndevdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NDEVdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
