/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "spice.h"
#include <stdio.h>
#include "ifsim.h"
#include "iferrmsg.h"
#include "sen2defs.h"
#include "cktdefs.h"
#include "util.h"
#include "suffix.h"

/* ARGSUSED */
int 
SENsetParm(ckt,anal,which,value)
    CKTcircuit *ckt;
    GENERIC *anal;
    int which;
    IFvalue *value;
{
    switch(which) {

    case SEN_DC:
        if(value->iValue) {
            ((SENstruct *)anal)->SENmode |= DCSEN;
        }
        break;
    case SEN_AC:
        if(value->iValue) {
            ((SENstruct *)anal)->SENmode |= ACSEN;
        }
        break;
    case SEN_TRAN:
        if(value->iValue) {
            ((SENstruct *)anal)->SENmode |= TRANSEN;
        }
        break;
    case SEN_DEV:
        ((SENstruct *)anal)->SENnumVal += 1;
        if( ! ((SENstruct *)anal)->SENdevices ) {
            ((SENstruct *)anal)->SENdevices = (char **) 
                    MALLOC( ((SENstruct *)anal)->SENnumVal * sizeof(char *));
            if( ((SENstruct *)anal)->SENdevices == NULL) return(E_NOMEM);
            ((SENstruct *)anal)->SENparmNames = (char **)
                    MALLOC( ((SENstruct *)anal)->SENnumVal * sizeof(char *));
            if( ((SENstruct *)anal)->SENparmNames == NULL) return(E_NOMEM);
        } else {
            ((SENstruct *)anal)->SENdevices = (char **)
                    REALLOC( ((SENstruct *)anal)->SENdevices ,
                    ((SENstruct *)anal)->SENnumVal * sizeof(char *));
            if( ((SENstruct *)anal)->SENdevices == NULL) return(E_NOMEM);
            ((SENstruct *)anal)->SENparmNames = (char **) REALLOC(
                    ((SENstruct *)anal)->SENparmNames,
                    ((SENstruct *)anal)->SENnumVal * sizeof(char *)) ;
            if( ((SENstruct *)anal)->SENparmNames == NULL) return(E_NOMEM);
        }
        *(((SENstruct *)anal)->SENdevices+((SENstruct *)anal)->SENnumVal-1)=
                value->sValue;
        break;
    case SEN_PARM:
        *(((SENstruct *)anal)->SENparmNames+((SENstruct *)anal)->SENnumVal-1)=
                value->sValue;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm SENparms[] = {
    { "dc",     SEN_DC,    IF_SET|IF_FLAG, "sensitivity in DC analysis" },
    { "op",     SEN_DC,    IF_SET|IF_FLAG, "sensitivity in DCop analysis" },
    { "ac",     SEN_AC,    IF_SET|IF_FLAG, "sensitivity in AC analysis" },
    { "tran",   SEN_TRAN,  IF_SET|IF_FLAG, "sensitivity in transient analysis"},
    { "dev",    SEN_DEV,   IF_SET|IF_INSTANCE, "instance with design param." },
    { "parm",   SEN_PARM,  IF_SET|IF_STRING, "name of design parameter" },
};

SPICEanalysis SEN2info  = {
    { 
        "SENS2",
        "Sensitivity analysis",

        sizeof(SENparms)/sizeof(IFparm),
        SENparms
    },
    sizeof(SENstruct),
    NODOMAIN,
    0,
    SENsetParm,
    SENaskQuest,
    NULL,
    SENstartup
};
