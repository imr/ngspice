/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include <stdio.h>
#include "inpdefs.h"
#include "iferrmsg.h"
#include "inp.h"

INPmodel *modtab;

    /* create/lookup a 'model' entry */

int INPmakeMod(char *token, int type, card * line)
{
    register INPmodel **i;

    for (i = &modtab; *i != (INPmodel *) NULL; i = &((*i)->INPnextModel)) {
	if (strcmp((*i)->INPmodName, token) == 0) {
	    return (OK);
	}
    }
    *i = (INPmodel *) MALLOC(sizeof(INPmodel));
    if (*i == NULL)
	return (E_NOMEM);
    (*i)->INPmodName = token;
    (*i)->INPmodType = type;
    (*i)->INPnextModel = (INPmodel *) NULL;
    (*i)->INPmodUsed = 0;
    (*i)->INPmodLine = line;
    (*i)->INPmodfast = NULL;
    return (OK);
}
