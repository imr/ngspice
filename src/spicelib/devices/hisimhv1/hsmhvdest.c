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
        HSMHVmodel *next_mod = HSMHVnextModel(mod);
        HSMHVinstance *inst = HSMHVinstances(mod);
        while (inst) {
            HSMHVinstance *next_inst = HSMHVnextInstance(inst);
            GENinstanceFree(GENinstanceOf(inst));
            inst = next_inst;
        }
        GENmodelFree(GENmodelOf(mod));
        mod = next_mod;
    }

    *inModel = NULL;
}
