/*============================================================================
FILE    MIFmAsk.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the function called by nutmeg to get the value
    of a specified code model parameter.

INTERFACES

    MIFmAsk()

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

/* #include "suffix.h" */


/*
MIFmAsk

This function is called by SPICE/Nutmeg to query the value of a
parameter on a model.  It is essentially the opposite of
MIFmParam, taking the index of the parameter, locating the value
of the parameter in the model structure, and converting that
value into the IFvalue structure understood by Nutmeg.
*/


int MIFmAsk(
    CKTcircuit *ckt,         /* The circuit structure */
    GENmodel   *inModel,     /* The model to get the value from */
    int        param_index,  /* The parameter to get */
    IFvalue    *value)       /* The value returned */
{

    MIFmodel    *model;
    int         mod_type;
    int         value_type;
    int         i;
    int         size;

    Mif_Boolean_t  is_array;

    NG_IGNORE(ckt);

    /* Arrange for access to MIF specific data in the model */
    model = (MIFmodel *) inModel;


    /* Get model type */
    mod_type = model->MIFmodType;
    if((mod_type < 0) || (mod_type >= DEVmaxnum))
        return(E_BADPARM);


    /* Check parameter index for validity */
    if((param_index < 0) || (param_index >= model->num_param))
        return(E_BADPARM);

    /* get value type to know which members of unions to access */
    value_type = DEVices[mod_type]->DEVpublic.modelParms[param_index].dataType;
    value_type &= IF_VARTYPES;


    /* determine if the parameter is an array or not */
    is_array = value_type & IF_VECTOR;


    /* Transfer the values to the SPICE3C1 value union from the param elements */
    /* This is analagous to what SPICE3 does with other device types */


    if(! is_array) {

        switch(value_type) {

        case  IF_FLAG:
            value->iValue = model->param[param_index]->element[0].bvalue;
            break;

        case  IF_INTEGER:
            value->iValue = model->param[param_index]->element[0].ivalue;
            break;

        case  IF_REAL:
            value->rValue = model->param[param_index]->element[0].rvalue;
            break;

        case  IF_STRING:
            /* Make copy of string.  We don't trust caller to not free it */
            /* These copies could get expensive! */
            value->sValue = MIFcopy(model->param[param_index]->element[0].svalue);
            break;

        case  IF_COMPLEX:
            /* we don't trust the caller to have a parallel complex structure */
            /* so copy the real and imaginary parts explicitly                */
            value->cValue.real = model->param[param_index]->element[0].cvalue.real;
            value->cValue.imag = model->param[param_index]->element[0].cvalue.imag;
            break;

        default:
            return(E_BADPARM);

        }
    }
    else {   /* it is an array */

        size = model->param[param_index]->size;
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
                value->v.vec.iVec[i] = model->param[param_index]->element[i].bvalue;
            break;

        case  IF_INTVEC:
            if(size <= 0)
                break;
            value->v.vec.iVec = TMALLOC(int, size);
            for(i = 0; i < size; i++)
                value->v.vec.iVec[i] = model->param[param_index]->element[i].ivalue;
            break;

        case  IF_REALVEC:
            if(size <= 0)
                break;
            value->v.vec.rVec = TMALLOC(double, size);
            for(i = 0; i < size; i++)
                value->v.vec.rVec[i] = model->param[param_index]->element[i].rvalue;
            break;

        case  IF_STRINGVEC:
            if(size <= 0)
                break;
            value->v.vec.sVec = TMALLOC(char *, size);
            for(i = 0; i < size; i++)
                /* Make copy of string.  We don't trust caller to not free it */
                /* These copies could get expensive! */
                value->v.vec.sVec[i] = MIFcopy(model->param[param_index]->element[i].svalue);
            break;

        case  IF_CPLXVEC:
            if(size <= 0)
                break;
            /* we don't trust the caller to have a parallel complex structure */
            /* so copy the real and imaginary parts explicitly                */
            value->v.vec.cVec = TMALLOC(IFcomplex, size);
            for(i = 0; i < size; i++) {
                value->v.vec.cVec[i].real = model->param[param_index]->element[i].cvalue.real;
                value->v.vec.cVec[i].imag = model->param[param_index]->element[i].cvalue.imag;
            }
            break;

        default:
            return(E_BADPARM);

        } /* end switch */

    } /* end else */

    return(OK);
}
