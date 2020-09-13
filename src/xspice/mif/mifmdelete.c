/*============================================================================
FILE    MIFmDelete.c

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

    This file contains the function called by SPICE to delete a model
    structure and all instances of that model.

INTERFACES

    MIFmDelete()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

/* #include "prefix.h"  */
#include "ngspice/ngspice.h"
#include <stdio.h>
//#include "util.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"

#include "ngspice/mifproto.h"
#include "ngspice/mifdefs.h"

/* #include "suffix.h" */


/*
MIFmDelete

This function deletes a particular model defined by a .model card
from the linked list of model structures of a particular code
model type, freeing all dynamically allocated memory used by the
model structure.  It calls MIFdelete as needed to delete all
instances of the specified model.
*/

int
MIFmDelete(GENmodel *gen_model)
{
    MIFmodel *model = (MIFmodel *) gen_model;
    int i, j;

    /* Free the model params stuff allocated in MIFget_mod */
    for (i = 0; i < model->num_param; i++) {
        /* delete content of union 'element' if it contains a string */
        if (model->param[i]->element) {
            if (model->param[i]->eltype == IF_STRING)
                FREE(model->param[i]->element[0].svalue);
            else if (model->param[i]->eltype == IF_STRINGVEC)
                for (j = 0; j < model->param[i]->size; j++)
                    FREE(model->param[i]->element[j].svalue);
            FREE(model->param[i]->element);
        }
        FREE(model->param[i]);
    }
    FREE(model->param);

    return OK;
}
