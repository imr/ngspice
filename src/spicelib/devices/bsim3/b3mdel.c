/* $Id$  */
/*
 $Log$
 Revision 1.1  2000-04-27 20:03:59  pnenzi
 Initial revision

 Revision 1.1.1.1  1999/11/15 10:35:08  root
 Rework imported sources

 Revision 1.2  1999/08/28 21:00:03  manu
 Big commit - merged ngspice.h, misc.h and util.h - protoized fte

 Revision 1.1.1.1  1999/07/30 09:05:13  root
 NG-Spice starting sources

 * Revision 3.2.2 1999/4/20  18:00:00  Weidong
 * BSIM3v3.2.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Author: 1997-1999 Weidong Liu.
File: b3mdel.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim3def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3mDelete(inModel,modname,kill)
GENmodel **inModel;
IFuid modname;
GENmodel *kill;
{
BSIM3model **model = (BSIM3model**)inModel;
BSIM3model *modfast = (BSIM3model*)kill;
BSIM3instance *here;
BSIM3instance *prev = NULL;
BSIM3model **oldmod;

    oldmod = model;
    for (; *model ; model = &((*model)->BSIM3nextModel)) 
    {    if ((*model)->BSIM3modName == modname || 
             (modfast && *model == modfast))
	     goto delgot;
         oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->BSIM3nextModel; /* cut deleted device out of list */
    for (here = (*model)->BSIM3instances; here; here = here->BSIM3nextInstance)
    {    if(prev) FREE(prev);
         prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);
}



