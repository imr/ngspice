#ifndef ngspice_ENH_H
#define ngspice_ENH_H

/* ===========================================================================
FILE    ENH.h

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

    This file contains typedefs used by the event-driven algorithm.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */


#include "ngspice/miftypes.h"
#include "ngspice/enhtypes.h"
#include "ngspice/fteinp.h"

/*
The following data is used in implementing various enhancements made to the
simulator.  The main struct is dynamically allocated in ckt so that incremental additions
can be made more easily without the need to recompile multiple modules.
Allocation and initialization is done in CKTinit.c which should be the only
module needed to recompile after additions are made here.
*/


struct Enh_Bkpt {
   double           current;    /* The current dynamic breakpoint time */
   double           last;       /* The last used dynamic breakpoint time */
};

struct Enh_Ramp {
    double           ramptime;    /* supply ramping time specified on .options */
};

struct Enh_Conv_Debug {
    Mif_Boolean_t    last_NIiter_call;  /* True if this is the last call to NIiter() */
    Mif_Boolean_t    report_conv_probs; /* True if conv test functions should send debug info */
};


struct Enh_Conv_Limit {
    Mif_Boolean_t    enabled;           /* True if convergence limiting enabled on code models */
    double           abs_step;          /* Minimum limiting step size */
    double           step;              /* Fractional step amount */
};


struct Enh_Rshunt {
    Mif_Boolean_t    enabled;      /* True if rshunt option used */
    double           gshunt;       /* 1.0 / rshunt */
    int              num_nodes;    /* Number of nodes in matrix */
    double           **diag;       /* Pointers to matrix diagonals */
};


struct Enh_Ckt_Data {
    Enh_Bkpt_t       breakpoint;   /* Data used by dynamic breakpoints */
    Enh_Ramp_t       ramp;         /* New options added to simulator */
    Enh_Conv_Debug_t conv_debug;   /* Convergence debug info dumping data */
    Enh_Conv_Limit_t conv_limit;   /* Convergence limiting info */
    Enh_Rshunt_t     rshunt_data;  /* Shunt conductance from nodes to ground */
};



void ENHreport_conv_prob(Enh_Conv_Source_t type, char *name, char *msg);
struct card *ENHtranslate_poly(struct card *deck);


#endif
