/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/inpdefs.h"
#include <string.h>

extern INPmodel *modtab;


/*-----------------------------------------------------------------
 * This fcn accepts a pointer to the model name, and returns
 * the INPmodel * if it exist in the model table.
 *----------------------------------------------------------------*/

INPmodel *
INPlookMod(const char *name)
{
    INPmodel *i;

    for (i = modtab; i; i = i->INPnextModel)
        if (strcmp(i->INPmodName, name) == 0)
            return i;

    return NULL;
}
