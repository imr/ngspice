/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTmodCrt(type,name,ckt,fast)
     *  Create a device model of the specified type, with the given name
     *  in the named circuit.
     */

#include "ngspice.h"
#include "devdefs.h"
#include "cktdefs.h"
#include "sperror.h"



int
CKTmodCrt(void *ckt, int type, void **modfast, IFuid name)
{
    extern SPICEdev **DEVices;
    GENmodel *mymodfast = NULL;
    int error;

    error = CKTfndMod(ckt,&type,(void**)&mymodfast,name);
    if(error == E_NOMOD) {
        mymodfast = (GENmodel *)MALLOC(*(DEVices[type]->DEVmodSize));
        if(mymodfast == (GENmodel *)NULL) return(E_NOMEM);
        mymodfast->GENmodType = type;
        mymodfast->GENmodName = name;
        mymodfast->GENnextModel =(GENmodel *)((CKTcircuit *)ckt)->CKThead[type];
        ((CKTcircuit *)ckt)->CKThead[type]=(GENmodel *)mymodfast;
        if(modfast) *modfast=(void *)mymodfast;
        return(OK);
    } else if (error==0) {
        if(modfast) *modfast=(void *)mymodfast;
        return(E_EXISTS);
    } else {
        return(error);
    }
    /*NOTREACHED*/
}
