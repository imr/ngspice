/* $Id$  */
/* 
$Log$
Revision 1.1  2000-04-27 20:03:59  pnenzi
Initial revision

 * Revision 3.1  96/12/08  19:54:27  yuhua
 * BSIM3v3.1 release
 * 
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1dest.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim3v1def.h"
#include "suffix.h"

void
BSIM3V1destroy(inModel)
GENmodel **inModel;
{
BSIM3V1model **model = (BSIM3V1model**)inModel;
BSIM3V1instance *here;
BSIM3V1instance *prev = NULL;
BSIM3V1model *mod = *model;
BSIM3V1model *oldmod = NULL;

    for (; mod ; mod = mod->BSIM3V1nextModel)
    {    if(oldmod) FREE(oldmod);
         oldmod = mod;
         prev = (BSIM3V1instance *)NULL;
         for (here = mod->BSIM3V1instances; here; here = here->BSIM3V1nextInstance)
	 {    if(prev) FREE(prev);
              prev = here;
         }
         if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
    return;
}



