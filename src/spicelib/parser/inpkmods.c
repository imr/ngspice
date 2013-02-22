/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "ngspice/ftedefs.h"
#include "inpxx.h"


extern INPmodel *modtab;

void INPkillMods(void)
{
    INPmodel *modtmp;
    INPmodel *prev = NULL;

    for (modtmp = modtab; modtmp != NULL; modtmp = modtmp->INPnextModel) {
	if (prev)
	    FREE(prev);
	prev = modtmp;
    }
    if (prev)
	FREE(prev);
    modtab = NULL;
    ft_curckt->ci_modtab = NULL;
}
