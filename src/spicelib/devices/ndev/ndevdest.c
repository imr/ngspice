/**********
Permit to use it as your wish.
Author: 2007 Gong Ding, gdiso@ustc.edu
University of Science and Technology of China
**********/

#include "ngspice/ngspice.h"
#include "ndevdefs.h"
#include "ngspice/suffix.h"


void
NDEVdestroy(GENmodel **inModel)
{
    NDEVmodel *mod = *(NDEVmodel **) inModel;

    while (mod) {
        NDEVmodel *next_mod = NDEVnextModel(mod);
        NDEVinstance *inst = NDEVinstances(mod);
        while (inst) {
            NDEVinstance *next_inst = inst->NDEVnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        close(mod->sock);
        printf("Disconnect to remote NDEV server %s:%d\n", mod->host, mod->port);
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
