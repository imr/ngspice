/* $Id$  */
/*
 $Log$
 Revision 1.1  2000-04-27 20:03:59  pnenzi
 Initial revision

 * Revision 3.2 1998/6/16  18:00:00  Weidong 
 * BSIM3v3.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v2dest.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim3v2def.h"
#include "suffix.h"

void
BSIM3V2destroy(inModel)
GENmodel **inModel;
{
BSIM3V2model **model = (BSIM3V2model**)inModel;
BSIM3V2instance *here;
BSIM3V2instance *prev = NULL;
BSIM3V2model *mod = *model;
BSIM3V2model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM3V2nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (BSIM3V2instance *)NULL;
         for (here = mod->BSIM3V2instances; here; here = here->BSIM3V2nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}



