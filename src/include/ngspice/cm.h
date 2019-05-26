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

#include <math.h>

#if defined(_MSC_VER)
#include <malloc.h>
#define fileno _fileno
#else
#include <unistd.h>
#endif

#include "ngspice/const.h"

#ifndef M_PI
#define M_PI CONSTpi
#endif
#ifndef M_E
#define M_E CONSTnap
#endif
#ifndef M_LOG2E
#define M_LOG2E CONSTlog2e
#endif
#ifndef M_LOG10E
#define M_LOG10E CONSTlog10e
#endif

#if !defined(NAN)
#if defined(_MSC_VER)
    /* NAN is not defined in VS 2012 or older */
    static const __int64 global_nan = 0x7ff8000000000000i64;
    #define NAN (*(const double *) &global_nan)
#else
    #define NAN (0.0/0.0)
#endif
#endif

/*
 * type safe variants of the <ctype.h> functions for char arguments
 */

#if !defined(isalpha_c)

inline static int char_to_int(char c) { return (unsigned char) c; }

#define isalpha_c(x) isalpha(char_to_int(x))
#define islower_c(x) islower(char_to_int(x))
#define isdigit_c(x) isdigit(char_to_int(x))
#define isalnum_c(x) isalnum(char_to_int(x))
#define isprint_c(x) isprint(char_to_int(x))
#define isblank_c(x) isblank(char_to_int(x))
#define isspace_c(x) isspace(char_to_int(x))
#define isupper_c(x) isupper(char_to_int(x))

#define tolower_c(x) ((char) tolower(char_to_int(x)))
#define toupper_c(x) ((char) toupper(char_to_int(x)))

#endif


#endif
