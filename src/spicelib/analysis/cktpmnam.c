/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * CKTpModName()
 *
 *  Take a parameter by Name and set it on the specified model
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
CKTpModName(char *parm, IFvalue *val, CKTcircuit *ckt, int type, IFuid name, GENmodel **modfast)
                /* the name of the parameter to set */
                  /* the parameter union containing the value to set */
                    /* the circuit this model is a member of */
                     /* the device type code to the model being parsed */
                    /* the name of the model being parsed */
                           /* direct pointer to model being parsed */

{
    int error;  /* int to store evaluate error return codes in */
    int i;

    for(i=0;i<(*(*DEVices[type]).DEVpublic.numModelParms);i++) {
        if(strcmp(parm,((*DEVices[type]).DEVpublic.modelParms[i].keyword))==0){
            error = CKTmodParam((void *)ckt,(void *)*modfast,
                    (*DEVices[type]).DEVpublic.modelParms[i].id,val,
                    (IFvalue*)NULL);
            if(error) return(error);
            break;
        }
    }
    if(i==(*(*DEVices[type]).DEVpublic.numModelParms)) {
        return(E_BADPARM);
    }
    return(OK);
}
