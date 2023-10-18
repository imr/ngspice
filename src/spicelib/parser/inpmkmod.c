/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/hash.h"
#include "inpxx.h"

/*  global input model table.  */
INPmodel *modtab = NULL;
/* Global input model hash table.
   The modelname is the key, the return value is the pointer to the model. */
NGHASHPTR modtabhash = NULL;

/*--------------------------------------------------------------
 * This fcn takes the model name and looks to see if it is already
 * in the model table.  If it is, then just return.  Otherwise,
 * stick the model into the model table.
 * Note that the model table INPmodel
 *--------------------------------------------------------------*/

int INPmakeMod(char *token, int type, struct card *line)
{
   register INPmodel *newm;
   /* Initialze the hash table. The default key type is string.
      The default comparison function is strcmp.*/
   if (!modtabhash) {
       modtabhash = nghash_init(NGHASH_MIN_SIZE);
       nghash_unique(modtabhash, TRUE);
   }
   /* If the model is already there, just return. */
   else if (nghash_find(modtabhash, token))
       return (OK);

   /* Model name was not already in model table. Therefore stick
      it in the front of the model table, also into the model hash table.
      Then return.  */

#ifdef TRACE
   /* debug statement */
   printf("In INPmakeMod, about to insert new model name = %s . . .\n", token);
#endif

   newm = TMALLOC(INPmodel, 1);
   if (newm == NULL)
      return (E_NOMEM);

   newm->INPmodName = token;                 /* model name */
   newm->INPmodType = type;                  /* model type */
   newm->INPnextModel = modtab;              /* pointer to second model */
   newm->INPmodLine = line;                  /* model line */
   newm->INPmodfast = NULL;

   nghash_insert(modtabhash, token, newm);

   modtab = newm;

   return (OK);
}

