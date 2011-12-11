/*============================================================================
FILE    MIFmDelete.c

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


int MIFmDelete(
    GENmodel **inModel,  /* The head of the model list */
    IFuid    modname,    /* The name of the model to delete */
    GENmodel *kill       /* The model structure to be deleted */
)
{
    MIFmodel **model;
    MIFmodel *modfast;
    MIFmodel **oldmod;
    MIFmodel *here=NULL;

    Mif_Boolean_t  found;

    int         i;


    /* Convert the generic pointers to MIF specific pointers */
    model = (MIFmodel **) inModel;
    modfast = (MIFmodel *) kill;

    /* Locate the model by name or pointer and cut it out of the list */
    oldmod = model;
    for(found = MIF_FALSE; *model; model = &((*model)->MIFnextModel)) {
        if( (*model)->MIFmodName == modname ||
                (modfast && *model == modfast) ) {
            here = *model;
            *oldmod = (*model)->MIFnextModel;
            found = MIF_TRUE;
            break;
        }
        oldmod = model;
    }

    if(! found)
        return(E_NOMOD);

    /* Free the instances under this model if any */
    /* by removing from the head of the linked list */
    /* until the head is null */
    while(here->MIFinstances) {
        MIFdelete((GENmodel *) here,
                  here->MIFinstances->MIFname,
                  (GENinstance **) &(here->MIFinstances));
    }

    /* Free the model params stuff allocated in MIFget_mod */
    for(i = 0; i < here->num_param; i++) {
        if(here->param[i]->element)
            FREE(here->param[i]->element);
        FREE(here->param[i]);
    }
    FREE(here->param);

    /* Free the model and return */
    FREE(here);
    return(OK);

}
