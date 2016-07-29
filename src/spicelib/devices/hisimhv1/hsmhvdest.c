/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvdest.c

 DATE : 2013.04.30

 released by
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "hsmhvdef.h"
#include "ngspice/suffix.h"


void
HSMHVdestroy(GENmodel **inModel)
{
    HSMHVmodel *mod = *(HSMHVmodel**) inModel;

    while (mod) {
        HSMHVmodel *next_mod = mod->HSMHVnextModel;
        HSMHVinstance *inst = mod->HSMHVinstances;
        while (inst) {
            HSMHVinstance *next_inst = inst->HSMHVnextInstance;
            FREE(inst);
            inst = next_inst;
        }
        FREE(mod);
        mod = next_mod;
    }

    *inModel = NULL;
}
