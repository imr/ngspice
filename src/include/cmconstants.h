#ifndef CMCONSTANTS_DEFINED
#define CMCONSTANTS_DEFINED

/* ===========================================================================
FILE    CMconstants.h

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

    This file contains constants used by code models.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */

#include "miftypes.h"

/***** Define Constants *******************************************/

#define FALSE 0
#define TRUE  1

#define DC           MIF_DC
#define AC           MIF_AC
#define TRANSIENT    MIF_TRAN

#define ANALOG       MIF_ANALOG
#define EVENT        MIF_EVENT_DRIVEN


#endif /* CMCONSTANTS_DEFINED */
