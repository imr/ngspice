/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvdel.c

 DATE : 2013.04.30

 released by
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "hsmhvdef.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"
#include "ngspice/suffix.h"


int
HSMHVdelete(GENinstance *inst)
{
    GENinstanceFree(inst);
    return OK;
}
