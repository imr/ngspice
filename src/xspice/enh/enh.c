/*============================================================================
FILE    ENH.c

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

    This file contains routines used for general enhancements made
    to the Berkeley SPICE3 core.

INTERFACES

    ENHreport_conv_prob()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

/*=== INCLUDE FILES ===*/


#include <stdio.h>
#include "ngspice/enh.h"

/*
ENHreport_conv_prob()

Report convergence problem messages from nodes, branch currents,
or instances.  This function is setup to allow providing the SI
with information identifying the type of convergence problem.
For now, it simply writes to stdout.
*/


void ENHreport_conv_prob(
    Enh_Conv_Source_t type,  /* node, branch, or instance */
    char *name,              /* the name of the node/branch/instance */
    char *msg)               /* an optional message */
{

    char *type_str;
    char *msg_str;

    /* Convert the type enum to a string for printing */
    switch(type) {

    case ENH_ANALOG_NODE:
    case ENH_EVENT_NODE:
        type_str = "node";
        break;

    case ENH_ANALOG_BRANCH:
        type_str = "branch current";
        break;

    case ENH_ANALOG_INSTANCE:
    case ENH_EVENT_INSTANCE:
    case ENH_HYBRID_INSTANCE:
        type_str = "instance";
        break;

    default:
        printf("\nERROR: Internal error in ENHreport_conv_prob - impossible type\n");
        return;
    }

    /* Check for msg == NULL and turn into null string */
    if(msg)
        msg_str = msg;
    else
        msg_str = "";

    /* Print the convergence problem report */
    printf("\nWARNING: Convergence problems at %s (%s).  %s\n",
        type_str, name, msg_str);

} /* ENHreport_conv_prob */

