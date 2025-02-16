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

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#include <string.h>

#include "ngspice/mifproto.h"
#include "ngspice/mifparse.h"
#include "ngspice/mifdefs.h"
#include "ngspice/mifcmdat.h"

/* Check value constraints from IFS file. */

static int check_int(int v, Mif_Param_Info_t *param_info, const char *name)
{
    if (param_info->has_lower_limit && v < param_info->lower_limit.ivalue) {
        fprintf(stderr,
                "Value %d below limit %d for parameter '%s' of model '%s'.\n",
                v, param_info->lower_limit.ivalue,
                param_info->name, name);
        v = param_info->lower_limit.ivalue;
    } else if (param_info->has_upper_limit &&
               v > param_info->upper_limit.ivalue) {
        fprintf(stderr,
                "Value %d exceeds limit %d for parameter '%s' of "
                "model '%s'.\n",
                v, param_info->upper_limit.ivalue,
                param_info->name, name);
        v = param_info->upper_limit.ivalue;
    }
    return v;
}

static double check_double(double v, Mif_Param_Info_t *param_info,
                           const char *name)
{
    if (param_info->has_lower_limit && v < param_info->lower_limit.rvalue) {
        fprintf(stderr,
                "Value %g below limit %g for parameter '%s' of model '%s'.\n",
                v, param_info->lower_limit.rvalue,
                param_info->name, name);
        v = param_info->lower_limit.rvalue;
    } else if (param_info->has_upper_limit &&
               v > param_info->upper_limit.rvalue) {
        fprintf(stderr,
                "Value %g exceeds limit %g for parameter '%s' of "
                "model '%s'.\n",
                v, param_info->upper_limit.rvalue,
                param_info->name, name);
        v = param_info->upper_limit.rvalue;
    }
    return v;
}

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

    MIFmodel         *model;
    Mif_Value_t      *target;
    Mif_Param_Info_t *param_info;
    int               mod_type;
    int               value_type;
    int               size, i;

    Mif_Boolean_t     is_array;


    /* Arrange for access to MIF specific data in the model */
    model = (MIFmodel *) inModel;

    /* Check parameter index for validity */
    if ((param_index < 0) || (param_index >= model->num_param))
        return(E_BADPARM);

    /* Get model type and XSPICE parameter info. */
    mod_type = model->MIFmodType;
    if((mod_type < 0) || (mod_type >= DEVmaxnum))
        return(E_BADPARM);
    param_info = DEVices[mod_type]->DEVpublic.param + param_index;

    /* get value type to know which members of unions to access */

    value_type = DEVices[mod_type]->DEVpublic.modelParms[param_index].dataType;
    value_type &= IF_VARTYPES;

    /* determine if the parameter is an array or not */
    is_array = value_type & IF_VECTOR;

    /* Initialize the parameter is_null. */

    model->param[param_index]->is_null = MIF_FALSE;

    /* element may exist already, if called from 'altermod' */
    target = model->param[param_index]->element;

    if (is_array) {
        size = value->v.numValue;
        if (param_info->has_lower_bound && size < param_info->lower_bound) {
            fprintf(stderr,
                    "Too few values for parameter '%s' of model '%s': %s.\n",
                    param_info->name,
                    inModel->GENmodName,
                    target ? "overwriting" : "fatal");
            if (!target)
                 return E_BADPARM;
        } else if (param_info->has_upper_bound &&
                   size > param_info->upper_bound) {
            fprintf(stderr,
                    "Too many values for parameter '%s' of model '%s': "
                    "truncating.\n",
                    param_info->name,
                    inModel->GENmodName);
            size = param_info->upper_bound;
        }
    } else {
        size = 1;
    }

    if (size > model->param[param_index]->size) {
        /* Update element count and allocate. */

        model->param[param_index]->size = size;
        FREE(target);
        target = TMALLOC(Mif_Value_t, size);
        model->param[param_index]->element = target;
    }

    /* Copy the values from the SPICE3C1 value union to the param elements */
    /* This is analagous to what SPICE3 does with other device types */

    if (!is_array) {
       switch(value_type) {
       case  IF_FLAG:
           target[0].bvalue = value->iValue;
           break;

       case  IF_INTEGER:
           target[0].ivalue = check_int(value->iValue,
                                        param_info, inModel->GENmodName);
           break;

       case  IF_REAL:
           target[0].rvalue = check_double(value->rValue,
                                           param_info, inModel->GENmodName);
           break;

       case  IF_STRING:
           /* we don't trust the caller to keep the string alive, so copy it */
           target[0].svalue =
               TMALLOC(char, 1 + strlen(value->sValue));
           strcpy(target[0].svalue, value->sValue);
           break;

       case  IF_COMPLEX:
           /* we don't trust the caller to have a parallel complex structure */
           /* so copy the real and imaginary parts explicitly                */

           target[0].cvalue.real = value->cValue.real;
           target[0].cvalue.imag = value->cValue.imag;
           break;

        default:
            return E_BADPARM;

       }
    } else {
        for (i = 0; i < size; i++) {
            switch(value_type) {
            case  IF_FLAGVEC:
                target[i].ivalue = value->v.vec.iVec[i];
                break;

            case  IF_INTVEC:
                target[i].bvalue = check_int(value->v.vec.iVec[i],
                                             param_info, inModel->GENmodName);
                break;

            case  IF_REALVEC:
                target[i].rvalue = check_double(value->v.vec.rVec[i],
                                                param_info,
                                                inModel->GENmodName);
                break;

            case  IF_STRINGVEC:
                /* Don't trust the caller to keep the string alive, copy it */
                target[i].svalue =
                    TMALLOC(char, 1 + strlen(value->v.vec.sVec[i]));
                strcpy(target[i].svalue, value->v.vec.sVec[i]);
                break;

            case  IF_CPLXVEC:
                /* Don't trust the caller to have a parallel complex structure,
                 * so copy the real and imaginary parts explicitly.
                 */
                target[i].cvalue.real = value->v.vec.cVec[i].real;
                target[i].cvalue.imag = value->v.vec.cVec[i].imag;
                break;

            default:
                return E_BADPARM;
            }
        }
    }
    model->param[param_index]->eltype = value_type;
    return OK;
}
