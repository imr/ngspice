/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */
/*
 */

#include "ngspice/ngspice.h"
#include "ekvdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
EKVdelete(
GENinstance *gen_inst)
{
    EKVinstance *inst = (EKVinstance *) gen_inst;
    FREE(inst->EKVsens);
    return OK;
}
