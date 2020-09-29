/*============================================================================
FILE    MIFask.c

MEMBER OF process XSPICE

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the function used by nutmeg to request the
    value of a code model static var.

INTERFACES

    MIFask()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

/* #include "prefix.h" */
#include "ngspice/ngspice.h"
#include <stdio.h>
//#include "CONST.h"
//#include "util.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#include <string.h>

#include "ngspice/mifproto.h"
#include "ngspice/mifdefs.h"

#include "ngspice/mifparse.h"

/* #include "suffix.h" */

/* forward prototypes */
int MIFaskInstVar(
    MIFinstance *inst,    /* The instance to get the value from */
    int         inst_index,  /* The inst var to get */
    IFvalue     *value     /* The value returned */
);
int MIFaskInstParam(
    MIFinstance *inst,    /* The instance to get the value from */
    int         param_index,  /* The parameter to get */
    IFvalue     *value     /* The value returned */
);


/*
MIFask

This function is called by SPICE/Nutmeg to query the value of a
parameter on an instance.  It locates the value of the parameter
in the instance structure and converts the value into the IFvalue
structure understood by Nutmeg.  For code models, this function
is provided to allow output of Instance Variables defined in the
code model's Interface Specification.
*/


int MIFask(
    CKTcircuit  *ckt,       /* The circuit structure */
    GENinstance *inInst,    /* The instance to get the value from */
    int         which,      /* The parameter to get */
    IFvalue     *value,     /* The value returned */
    IFvalue     *select)    /* Unused */
{

    MIFinstance *inst;
    MIFmodel    *model;
    int         mod_type;
    int         i;

    NG_IGNORE(ckt);
    NG_IGNORE(select);

    /* Arrange for access to MIF specific data in the instance */
    inst = (MIFinstance *) inInst;

    /* Arrange for access to MIF specific data in the model */
    model = MIFmodPtr(inst);

    /* Get model type */
    mod_type = model->MIFmodType;
    if((mod_type < 0) || (mod_type >= DEVmaxnum))
        return(E_BADPARM);

    /* see if is trying to get an instance variable or an instance parameter */
    if(which>= model->num_param)
       return MIFaskInstVar(inst, which - model->num_param, value);
    else
       return MIFaskInstParam(inst, which, value);
}


int MIFaskInstVar(
    MIFinstance *inst,    /* The instance to get the value from */
    int         inst_index,  /* The inst var to get */
    IFvalue     *value     /* The value returned */
)
{
    MIFmodel    *model;
    int         mod_type;
    int         value_type;
    int         i;
    int         size;

    Mif_Boolean_t  is_array;

    model = MIFmodPtr(inst);
    /* Get model type */
    mod_type = model->MIFmodType;
    if((mod_type < 0) || (mod_type >= DEVmaxnum))
        return(E_BADPARM);

 /* get value type to know which members of unions to access */
    for(i=0;i<*(DEVices[mod_type]->DEVpublic.numInstanceParms);i++)
    if(DEVices[mod_type]->DEVpublic.instanceParms[i].id == model->num_param+inst_index)
        break;
    if(i>=*(DEVices[mod_type]->DEVpublic.numInstanceParms))
        return(E_BADPARM);
    value_type = DEVices[mod_type]->DEVpublic.instanceParms[i].dataType;
    value_type &= IF_VARTYPES;


    /* determine if the parameter is an array or not */
    is_array = value_type & IF_VECTOR;


    /* Transfer the values to the SPICE3C1 value union from the param elements */
    /* This is analagous to what SPICE3 does with other device types */


    if(! is_array) {

        switch(value_type) {

        case  IF_FLAG:
            value->iValue = inst->inst_var[inst_index]->element[0].bvalue;
            break;

        case  IF_INTEGER:
            value->iValue = inst->inst_var[inst_index]->element[0].ivalue;
            break;

        case  IF_REAL:
            value->rValue = inst->inst_var[inst_index]->element[0].rvalue;
            break;

        case  IF_STRING:
            /* Make copy of string.  We don't trust caller to not free it */
            /* These copies could get expensive! */
            value->sValue = MIFcopy(inst->inst_var[inst_index]->element[0].svalue);
            break;

        case  IF_COMPLEX:
            /* we don't trust the caller to have a parallel complex structure */
            /* so copy the real and imaginary parts explicitly                */
            value->cValue.real = inst->inst_var[inst_index]->element[0].cvalue.real;
            value->cValue.imag = inst->inst_var[inst_index]->element[0].cvalue.imag;
            break;

        default:
            return(E_BADPARM);

        }
    }
    else {   /* it is an array */

        size = inst->inst_var[inst_index]->size;
        if(size < 0)
            size = 0;

        value->v.numValue = size;

        switch(value_type) {

        /* Note that we malloc space each time this function is called. */
        /* This is what TRAask.c does, so we do it too, even though     */
        /* we don't know if it is ever freed... */

        case  IF_FLAGVEC:
            if(size <= 0)
                break;
            value->v.vec.iVec = TMALLOC(int, size);
            for(i = 0; i < size; i++)
                value->v.vec.iVec[i] = inst->inst_var[inst_index]->element[i].bvalue;
            break;

        case  IF_INTVEC:
            if(size <= 0)
                break;
            value->v.vec.iVec = TMALLOC(int, size);
            for(i = 0; i < size; i++)
                value->v.vec.iVec[i] = inst->inst_var[inst_index]->element[i].ivalue;
            break;

        case  IF_REALVEC:
            if(size <= 0)
                break;
            value->v.vec.rVec = TMALLOC(double, size);
            for(i = 0; i < size; i++)
                value->v.vec.rVec[i] = inst->inst_var[inst_index]->element[i].rvalue;
            break;

        case  IF_STRINGVEC:
            if(size <= 0)
                break;
            value->v.vec.sVec = TMALLOC(char *, size);
            for(i = 0; i < size; i++)
                /* Make copy of string.  We don't trust caller to not free it */
                /* These copies could get expensive! */
                value->v.vec.sVec[i] = MIFcopy(inst->inst_var[inst_index]->element[i].svalue);
            break;

        case  IF_CPLXVEC:
            if(size <= 0)
                break;
            /* we don't trust the caller to have a parallel complex structure */
            /* so copy the real and imaginary parts explicitly                */
            value->v.vec.cVec = TMALLOC(IFcomplex, size);
            for(i = 0; i < size; i++) {
                value->v.vec.cVec[i].real = inst->inst_var[inst_index]->element[i].cvalue.real;
                value->v.vec.cVec[i].imag = inst->inst_var[inst_index]->element[i].cvalue.imag;
            }
            break;

        default:
            return(E_BADPARM);

        } /* end switch */

    } /* end else */

    return(OK);
}



int MIFaskInstParam(
    MIFinstance *inst,    /* The instance to get the value from */
    int         param_index,  /* The parameter to get */
    IFvalue     *value     /* The value returned */
)
{
    MIFmodel    *model;
    int         mod_type;
    int         value_type;
    int         i;
    int         size;

    int         inst_index;

    Mif_Boolean_t  is_array;

    model = MIFmodPtr(inst);
    /* Get model type */
    mod_type = model->MIFmodType;
    if((mod_type < 0) || (mod_type >= DEVmaxnum))
        return(E_BADPARM);

 /* get value type to know which members of unions to access */
    for(i=0;i<*(DEVices[mod_type]->DEVpublic.numInstanceParms);i++)
    if(DEVices[mod_type]->DEVpublic.instanceParms[i].id == param_index)
        break;
    if(i>=*(DEVices[mod_type]->DEVpublic.numInstanceParms))
        return(E_BADPARM);
    value_type = DEVices[mod_type]->DEVpublic.instanceParms[i].dataType;
    value_type &= IF_VARTYPES;
    
    /* get index into inst->iparam */
    inst_index = DEVices[mod_type]->DEVpublic.param[param_index].inst_param_index;
    if(inst_index <0)
        return(E_BADPARM);

    /* determine if the parameter is an array or not */
    is_array = value_type & IF_VECTOR;


    /* Transfer the values to the SPICE3C1 value union from the param elements */
    /* This is analagous to what SPICE3 does with other device types */


    if(! is_array) {

        switch(value_type) {

        case  IF_FLAG:
            value->iValue = inst->iparam[inst_index]->element[0].bvalue;
            break;

        case  IF_INTEGER:
            value->iValue = inst->iparam[inst_index]->element[0].ivalue;
            break;

        case  IF_REAL:
            value->rValue = inst->iparam[inst_index]->element[0].rvalue;
            break;

        case  IF_STRING:
            /* Make copy of string.  We don't trust caller to not free it */
            /* These copies could get expensive! */
            value->sValue = MIFcopy(inst->iparam[inst_index]->element[0].svalue);
            break;

        case  IF_COMPLEX:
            /* we don't trust the caller to have a parallel complex structure */
            /* so copy the real and imaginary parts explicitly                */
            value->cValue.real = inst->iparam[inst_index]->element[0].cvalue.real;
            value->cValue.imag = inst->iparam[inst_index]->element[0].cvalue.imag;
            break;

        default:
            return(E_BADPARM);

        }
    }
    else {   /* it is an array */

        size = inst->iparam[inst_index]->size;
        if(size < 0)
            size = 0;

        value->v.numValue = size;

        switch(value_type) {

        /* Note that we malloc space each time this function is called. */
        /* This is what TRAask.c does, so we do it too, even though     */
        /* we don't know if it is ever freed... */

        case  IF_FLAGVEC:
            if(size <= 0)
                break;
            value->v.vec.iVec = TMALLOC(int, size);
            for(i = 0; i < size; i++)
                value->v.vec.iVec[i] = inst->iparam[inst_index]->element[i].bvalue;
            break;

        case  IF_INTVEC:
            if(size <= 0)
                break;
            value->v.vec.iVec = TMALLOC(int, size);
            for(i = 0; i < size; i++)
                value->v.vec.iVec[i] = inst->iparam[inst_index]->element[i].ivalue;
            break;

        case  IF_REALVEC:
            if(size <= 0)
                break;
            value->v.vec.rVec = TMALLOC(double, size);
            for(i = 0; i < size; i++)
                value->v.vec.rVec[i] = inst->iparam[inst_index]->element[i].rvalue;
            break;

        case  IF_STRINGVEC:
            if(size <= 0)
                break;
            value->v.vec.sVec = TMALLOC(char *, size);
            for(i = 0; i < size; i++)
                /* Make copy of string.  We don't trust caller to not free it */
                /* These copies could get expensive! */
                value->v.vec.sVec[i] = MIFcopy(inst->iparam[inst_index]->element[i].svalue);
            break;

        case  IF_CPLXVEC:
            if(size <= 0)
                break;
            /* we don't trust the caller to have a parallel complex structure */
            /* so copy the real and imaginary parts explicitly                */
            value->v.vec.cVec = TMALLOC(IFcomplex, size);
            for(i = 0; i < size; i++) {
                value->v.vec.cVec[i].real = inst->iparam[inst_index]->element[i].cvalue.real;
                value->v.vec.cVec[i].imag = inst->iparam[inst_index]->element[i].cvalue.imag;
            }
            break;

        default:
            return(E_BADPARM);

        } /* end switch */

    } /* end else */

    return(OK);
}
