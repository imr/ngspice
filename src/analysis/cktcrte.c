/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTcrtElement(ckt,type,inModPtr,inInstPtr,name,subname)
     *  Create a device of the specified type, with the given name, using
     *  the specified model in the named circuit.
     */

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"



/*ARGSUSED*/
int
CKTcrtElt(void *ckt, void *inModPtr, void **inInstPtr, IFuid name)
{
    GENinstance *instPtr = NULL;
    GENmodel *modPtr=(GENmodel*)inModPtr;
    extern SPICEdev *DEVices[];
    int error;
    int type;

    if((GENmodel *)modPtr==(GENmodel*)NULL) return(E_NOMOD);
    type = ((GENmodel*)modPtr)->GENmodType;
    error =CKTfndDev(ckt,&type,(void**)&instPtr,name,inModPtr,(char *)NULL );
    if (error== OK) { 
        if(inInstPtr) *inInstPtr=(void *)instPtr;
        return(E_EXISTS);
    } else if (error != E_NODEV) return(error);
    instPtr = (GENinstance *)MALLOC(*DEVices[type]->DEVinstSize);
    if(instPtr == (GENinstance *)NULL) return(E_NOMEM);
    instPtr->GENname = name;
    instPtr->GENmodPtr = modPtr;
    instPtr->GENnextInstance = modPtr->GENinstances;
    modPtr->GENinstances = instPtr;
    if(inInstPtr != NULL) *inInstPtr = (void *)instPtr;
    return(OK);
}
