/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "ngspice/iferrmsg.h"
#include "inpxx.h"

/*  global input model table.  */
INPmodel *modtab = NULL;

/*--------------------------------------------------------------
 * This fcn takes the model name and looks to see if it is already
 * in the model table.  If it is, then just return.  Otherwise,
 * stick the model into the model table.
 * Note that the model table INPmodel
 *--------------------------------------------------------------*/

int INPmakeMod(char *token, int type, struct card *line)
{
   register INPmodel **i;

   /* First cycle through model table and see if model name
      already exists in there.  If it does, just return. */
   for (i = &modtab; *i != NULL; i = &((*i)->INPnextModel)) {
      if (strcmp((*i)->INPmodName, token) == 0) {
         return (OK);
      }
   }

   /* Model name was not already in model table.  Therefore stick
      it in the model table. Then return.  */

#ifdef TRACE
   /* debug statement */
   printf("In INPmakeMod, about to insert new model name = %s . . .\n", token);
#endif

   *i = TMALLOC(INPmodel, 1);
   if (*i == NULL)
      return (E_NOMEM);

   (*i)->INPmodName = token;                 /* model name */
   (*i)->INPmodType = type;                  /* model type */
   (*i)->INPnextModel = NULL;   /* pointer to next model (end of list) */
   (*i)->INPmodLine = line;                  /* model line */
   (*i)->INPmodfast = NULL;
   return (OK);
}

