#ifndef ngspice_CMTYPES_H
#define ngspice_CMTYPES_H

/* ===========================================================================
FILE    CMtypes.h

MEMBER OF process XSPICE

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Jeff Murray, Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains type definitions used by code models.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */

#include "ngspice/miftypes.h"

/***** Define Typedefs ********************************************/

typedef int Boolean_t;

typedef Mif_Complex_t   Complex_t;


typedef enum {
    ZERO,                /* Normally referenced as 0 */
    ONE,                 /* Normally referenced as 1 */
    UNKNOWN,             /* Unknown */
} Digital_State_t;

typedef enum {
    STRONG,              /* strong */
    RESISTIVE,           /* resistive */
    HI_IMPEDANCE,        /* high impedance */
    UNDETERMINED,        /* unknown strength */
} Digital_Strength_t;

typedef struct {
    Digital_State_t     state;
    Digital_Strength_t  strength;
} Digital_t;



#endif
