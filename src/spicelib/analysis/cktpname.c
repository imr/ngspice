/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /*
     * CKTpName()
     *
     *  Take a parameter by Name and set it on the specified device 
     */

#include "ngspice.h"
#include "ifsim.h"
#include "devdefs.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "sperror.h"



extern SPICEdev **DEVices;


/* ARGSUSED */
int
CKTpName(char *parm, IFvalue *val, CKTcircuit *ckt, int dev, char *name, GENinstance **fast)
                /* the name of the parameter to set */
                  /* the parameter union containing the value to set */
                    /* the circuit this device is a member of */
                    /* the device type code to the device being parsed */
                    /* the name of the device being parsed */
                           /* direct pointer to device being parsed */

{
    int error;  /* int to store evaluate error return codes in */
    int i;

    for(i=0;i<(*(*DEVices[dev]).DEVpublic.numInstanceParms);i++) {
        if(strcmp(parm,
                ((*DEVices[dev]).DEVpublic.instanceParms[i].keyword))==0) {
            error = CKTparam((void*)ckt,(void *)*fast,
                    (*DEVices[dev]).DEVpublic.instanceParms[i].id,val,
                    (IFvalue *)NULL);
            if(error) return(error);
            break;
        }
    }
    if(i==(*(*DEVices[dev]).DEVpublic.numInstanceParms)) {
        return(E_BADPARM);
    }
    return(OK);
}
