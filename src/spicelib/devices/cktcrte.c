/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTcrtElement(ckt,type,inModPtr,inInstPtr,name,subname)
 *
 * Create a device of the specified type, with the given name, using
 * the specified model in the named circuit.  */

#include <config.h>
#include <devdefs.h>
#include <sperror.h>

#include "dev.h"
#include "memory.h"

int
CKTcrtElt(void *ckt, void *inModPtr, void **inInstPtr, IFuid name)
{
  GENinstance *instPtr[1] = {NULL};             /* instPtr points to the data struct for per-instance data */
  GENmodel *modPtr = (GENmodel*)inModPtr;  /* modPtr points to the data struct for per-model data */

    SPICEdev **DEVices;
    int error;
    int type;

    DEVices = devices();

    if( (GENmodel *)modPtr == (GENmodel*)NULL ) 
	return E_NOMOD;

    type = ((GENmodel*)modPtr)->GENmodType; 

    error = CKTfndDev(ckt, &type, (void**)instPtr, name, inModPtr,
		      (char *)NULL );
    if (error == OK) { 
        if (inInstPtr)
	    *inInstPtr=(void *)instPtr[0];
        return E_EXISTS;
    } else if (error != E_NODEV)
	return error;

#ifdef TRACE
    /*------  SDB debug statement  -------*/
    printf("In CKTcrtElt, about to tmalloc new model, type = %d. . . \n", type);
#endif

    instPtr[0] = (GENinstance *) tmalloc(*DEVices[type]->DEVinstSize);
    if (instPtr[0] == (GENinstance *)NULL)
	return E_NOMEM;

    instPtr[0]->GENname = name;

    instPtr[0]->GENmodPtr = modPtr; 

    instPtr[0]->GENnextInstance = modPtr->GENinstances; 

    modPtr->GENinstances = instPtr[0];

    if(inInstPtr != NULL)
	*inInstPtr = (void *)instPtr[0];

    return OK;
}
