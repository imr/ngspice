#ifndef ngspice_EVTSHARED_H
#define ngspice_EVTSHARED_H

/* ===========================================================================
FILE    EVTshared.h

MEMBER OF process XSPICE

Copyright 2018
Holger Vogt
All Rights Reserved

PROJECT A-8503

AUTHORS

    7/21/2018  Holger Vogt

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains ANSI C function prototypes for functions
    in the event-driven simulation algorithm package.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */

#include "ngspice/cktdefs.h"
#include "ngspice/cpstd.h"
#include "ngspice/evt.h"
#include "ngspice/mifdefs.h"
#include "ngspice/ipc.h"


/* ******************* */
/* Function Prototypes */
/* ******************* */

struct evt_shared_data *EVTshareddata(char *node_name);
char** EVTallnodes(void);

#endif
