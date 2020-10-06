/*============================================================================
FILE    MIFPara.c

Derived from MIFmPara.c by Florian Ballenegger

Original copyright:

MEMBER OF process XSPICE

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

SUMMARY

    This file contains the function used to assign the value of a parameter
    read from the .model card into the appropriate structure in the model.

============================================================================*/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#include "ngspice/mifproto.h"
#include "ngspice/mifparse.h"
#include "ngspice/mifdefs.h"
#include "ngspice/mifcmdat.h"



/* ARGSUSED */
int
MIFParam(int param_index, IFvalue *value, GENinstance *inst_in, IFvalue *select)
{
    
    MIFmodel    *model;
    MIFinstance *inst;
    int         mod_type;
    int         value_type;
    int         i;
    int		iparam;
    Mif_Boolean_t  is_array;


    NG_IGNORE(select);
    
    /* Arrange for access to MIF specific data in the model */
    inst = (MIFinstance*) inst_in;
    model = MIFmodPtr(inst);

    /* Get model type */
    mod_type = model->MIFmodType;
    if((mod_type < 0) || (mod_type >= DEVmaxnum))
        return(E_BADPARM);


    /* Check parameter index for validity */
    if((param_index < 0) || (param_index >= model->num_param))
        return(E_BADPARM);

    /* search the instance parameter index */
    for(iparam=0;iparam<*(DEVices[mod_type]->DEVpublic.numInstanceParms);iparam++)
    if(DEVices[mod_type]->DEVpublic.instanceParms[iparam].id == param_index)
        break;
    
    if((iparam < 0) || (iparam >= *(DEVices[mod_type]->DEVpublic.numInstanceParms)))
        return(E_BADPARM);

    /* get value type to know which members of unions to access */
    value_type = DEVices[mod_type]->DEVpublic.modelParms[param_index].dataType;
    value_type &= IF_VARTYPES;


    /* determine if the parameter is an array or not */
    is_array = value_type & IF_VECTOR;

    /* initialize the parameter is_null and size elements and allocate elements */
    inst->iparam[iparam]->is_null = MIF_FALSE;
    /* element may exist already, if called from 'alter' */
    FREE(inst->iparam[iparam]->element);
    if(is_array) {
        inst->iparam[iparam]->size = value->v.numValue;
        inst->iparam[iparam]->element = TMALLOC(Mif_Value_t, value->v.numValue);
    }
    else {
        inst->iparam[iparam]->size = 1;
        inst->iparam[iparam]->element = TMALLOC(Mif_Value_t, 1);
    }


    /* Transfer the values from the SPICE3C1 value union to the param elements */
    /* This is analagous to what SPICE3 does with other device types */


    if(! is_array) {

        switch(value_type) {

        case  IF_FLAG:
            inst->iparam[iparam]->element[0].bvalue = value->iValue;
            inst->iparam[iparam]->eltype = IF_FLAG;
            break;

        case  IF_INTEGER:
            inst->iparam[iparam]->element[0].ivalue = value->iValue;
            inst->iparam[iparam]->eltype = IF_INTEGER;
            break;

        case  IF_REAL:
            inst->iparam[iparam]->element[0].rvalue = value->rValue;
            inst->iparam[iparam]->eltype = IF_REAL;
            break;

        case  IF_STRING:
            /* we don't trust the caller to keep the string alive, so copy it */
            inst->iparam[iparam]->element[0].svalue =
                                       TMALLOC(char, 1 + strlen(value->sValue));
            strcpy(inst->iparam[iparam]->element[0].svalue, value->sValue);
            inst->iparam[iparam]->eltype = IF_STRING;
            break;

        case  IF_COMPLEX:
            /* we don't trust the caller to have a parallel complex structure */
            /* so copy the real and imaginary parts explicitly                */
            inst->iparam[iparam]->element[0].cvalue.real = value->cValue.real;
            inst->iparam[iparam]->element[0].cvalue.imag = value->cValue.imag;
            inst->iparam[iparam]->eltype = IF_COMPLEX;
            break;

        default:
            return(E_BADPARM);

        }
    }
    else {   /* it is an array */

        for(i = 0; i < value->v.numValue; i++) {

            switch(value_type) {

            case  IF_FLAGVEC:
                inst->iparam[iparam]->element[i].bvalue = value->v.vec.iVec[i];
                inst->iparam[iparam]->eltype = IF_FLAGVEC;
                break;

            case  IF_INTVEC:
                inst->iparam[iparam]->element[i].ivalue = value->v.vec.iVec[i];
                inst->iparam[iparam]->eltype = IF_INTVEC;
                break;

            case  IF_REALVEC:
                inst->iparam[iparam]->element[i].rvalue = value->v.vec.rVec[i];
                inst->iparam[iparam]->eltype = IF_REALVEC;
                break;

            case  IF_STRINGVEC:
                /* we don't trust the caller to keep the string alive, so copy it */
                inst->iparam[iparam]->element[i].svalue =
                                           TMALLOC(char, 1 + strlen(value->v.vec.sVec[i]));
                strcpy(inst->iparam[iparam]->element[i].svalue, value->v.vec.sVec[i]);
                inst->iparam[iparam]->eltype = IF_STRINGVEC;
                break;

            case  IF_CPLXVEC:
                /* we don't trust the caller to have a parallel complex structure */
                /* so copy the real and imaginary parts explicitly                */
                inst->iparam[iparam]->element[i].cvalue.real = value->v.vec.cVec[i].real;
                inst->iparam[iparam]->element[i].cvalue.imag = value->v.vec.cVec[i].imag;
                inst->iparam[iparam]->eltype = IF_CPLXVEC;
                break;

            default:
                return(E_BADPARM);

            } /* end switch */

        } /* end for number of elements of vector */

    } /* end else */

    return(OK);
}
