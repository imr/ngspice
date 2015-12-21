#ifndef ngspice_CM_H
#define ngspice_CM_H

/* ===========================================================================
FILE    CM.h

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

    This file is includes all include data in the CM package.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */

#include "ngspice/config.h"
#include "ngspice/cmtypes.h"
#include "ngspice/cmconstants.h"
#include "ngspice/cmproto.h"
#include "ngspice/mifcmdat.h"

#if defined (_MSC_VER)
#ifndef NAN
    static const __int64 global_nan = 0x7ff8000000000000i64;
    #define NAN (*(const double *) &global_nan)
#endif
#endif

#if !defined(NAN)
#define NAN (0.0/0.0)
#endif


#endif
