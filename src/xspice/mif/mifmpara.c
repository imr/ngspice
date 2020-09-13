/*============================================================================
FILE    MIFmParam.c

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

    This file contains the function used to assign the value of a parameter
    read from the .model card into the appropriate structure in the model.

INTERFACES

    MIFmParam()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

/*  #include "prefix.h"  */
#include "ngspice/ngspice.h"
#include <stdio.h>
//#include "CONST.h"
//#include "util.h"
#include "ngspice/ifsim.h"
//#include "resdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#include <string.h>

#include "ngspice/mifproto.h"
#include "ngspice/mifparse.h"
#include "ngspice/mifdefs.h"
#include "ngspice/mifcmdat.h"

/* #include "suffix.h"  */


/*
MIFmParam

This function is called by SPICE/Nutmeg to set the value of a
parameter on a model according to information parsed from a
.model card or information supplied interactively by a user.  It
takes the value of the parameter input in an IFvalue structure
and sets the parameter on the specified model structure.  Unlike
the procedure for SPICE 3C1 devices, MIFmParam does not use
enumerations for identifying the parameter to set.  Instead, the
parameter is identified directly by the index value of the
parameter in the SPICEdev.DEVpublic.modelParms array.
*/

int MIFmParam(
    int param_index,        /* The parameter to set */
    IFvalue *value,         /* The value of the parameter */
    GENmodel *inModel)      /* The model structure on which to set the value */
{

    MIFmodel    *model;
    int         mod_type;
    int         value_type;
    int         i;

    Mif_Boolean_t  is_array;


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

    /* initialize the parameter is_null and size elements and allocate elements */
    model->param[param_index]->is_null = MIF_FALSE;
    /* element may exist already, if called from 'altermod' */
    FREE(model->param[param_index]->element);
    if(is_array) {
        model->param[param_index]->size = value->v.numValue;
        model->param[param_index]->element = TMALLOC(Mif_Value_t, value->v.numValue);
    }
    else {
        model->param[param_index]->size = 1;
        model->param[param_index]->element = TMALLOC(Mif_Value_t, 1);
    }


    /* Transfer the values from the SPICE3C1 value union to the param elements */
    /* This is analagous to what SPICE3 does with other device types */


    if(! is_array) {

        switch(value_type) {

        case  IF_FLAG:
            model->param[param_index]->element[0].bvalue = value->iValue;
            model->param[param_index]->eltype = IF_FLAG;
            break;

        case  IF_INTEGER:
            model->param[param_index]->element[0].ivalue = value->iValue;
            model->param[param_index]->eltype = IF_INTEGER;
            break;

        case  IF_REAL:
            model->param[param_index]->element[0].rvalue = value->rValue;
            model->param[param_index]->eltype = IF_REAL;
            break;

        case  IF_STRING:
            /* we don't trust the caller to keep the string alive, so copy it */
            model->param[param_index]->element[0].svalue =
                                       TMALLOC(char, 1 + strlen(value->sValue));
            strcpy(model->param[param_index]->element[0].svalue, value->sValue);
            model->param[param_index]->eltype = IF_STRING;
            break;

        case  IF_COMPLEX:
            /* we don't trust the caller to have a parallel complex structure */
            /* so copy the real and imaginary parts explicitly                */
            model->param[param_index]->element[0].cvalue.real = value->cValue.real;
            model->param[param_index]->element[0].cvalue.imag = value->cValue.imag;
            model->param[param_index]->eltype = IF_COMPLEX;
            break;

        default:
            return(E_BADPARM);

        }
    }
    else {   /* it is an array */

        for(i = 0; i < value->v.numValue; i++) {

            switch(value_type) {

            case  IF_FLAGVEC:
                model->param[param_index]->element[i].bvalue = value->v.vec.iVec[i];
                model->param[param_index]->eltype = IF_FLAGVEC;
                break;

            case  IF_INTVEC:
                model->param[param_index]->element[i].ivalue = value->v.vec.iVec[i];
                model->param[param_index]->eltype = IF_INTVEC;
                break;

            case  IF_REALVEC:
                model->param[param_index]->element[i].rvalue = value->v.vec.rVec[i];
                model->param[param_index]->eltype = IF_REALVEC;
                break;

            case  IF_STRINGVEC:
                /* we don't trust the caller to keep the string alive, so copy it */
                model->param[param_index]->element[i].svalue =
                                           TMALLOC(char, 1 + strlen(value->v.vec.sVec[i]));
                strcpy(model->param[param_index]->element[i].svalue, value->v.vec.sVec[i]);
                model->param[param_index]->eltype = IF_STRINGVEC;
                break;

            case  IF_CPLXVEC:
                /* we don't trust the caller to have a parallel complex structure */
                /* so copy the real and imaginary parts explicitly                */
                model->param[param_index]->element[i].cvalue.real = value->v.vec.cVec[i].real;
                model->param[param_index]->element[i].cvalue.imag = value->v.vec.cVec[i].imag;
                model->param[param_index]->eltype = IF_CPLXVEC;
                break;

            default:
                return(E_BADPARM);

            } /* end switch */

        } /* end for number of elements of vector */

    } /* end else */

    return(OK);
}
