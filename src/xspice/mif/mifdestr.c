/*============================================================================
FILE    MIFdestroy.c

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

    This file contains a function that deletes all models of a particular
    device (code model) type from the circuit description structures.

INTERFACES

    MIFdestroy()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

/* #include "prefix.h" */
#include "ngspice/ngspice.h"
#include <stdio.h>

#include "ngspice/mifproto.h"

/* #include "suffix.h"  */



/*
MIFdestroy

This function deletes all models and all instances of a specified
device type.  It traverses the linked list of model structures
for that type and calls MIFmDelete on each model.
*/

void MIFdestroy(
    GENmodel **inModel)    /* The head of the list of models to delete */
{

    /* Free all models of this device type by removing */
    /* models from the head of the linked list until   */
    /* the head is null */

    while(*inModel) {
        MIFmDelete(inModel,
                   (*inModel)->GENmodName,
                   *inModel);
    }

}
