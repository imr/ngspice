#ifndef ENH_HEADER
#define ENH_HEADER x

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


#include "miftypes.h"
#include "fteinp.h"

/*
The following data is used in implementing various enhancements made to the
simulator.  The main struct is dynamically allocated in ckt so that incremental additions
can be made more easily without the need to recompile multiple modules.
Allocation and initialization is done in CKTinit.c which should be the only
module needed to recompile after additions are made here.
*/


typedef enum {
    ENH_ANALOG_NODE,            /* An analog node */
    ENH_EVENT_NODE,             /* An event-driven node */
    ENH_ANALOG_BRANCH,          /* A branch current */
    ENH_ANALOG_INSTANCE,        /* An analog instance */
    ENH_EVENT_INSTANCE,         /* An event-driven instance */
    ENH_HYBRID_INSTANCE,        /* A hybrid (analog/event-driven) instance */
} Enh_Conv_Source_t;


typedef struct {
   double           current;    /* The current dynamic breakpoint time */
   double           last;       /* The last used dynamic breakpoint time */
} Enh_Bkpt_t;

typedef struct {
    double           ramptime;    /* supply ramping time specified on .options */
} Enh_Ramp_t;

typedef struct {
    Mif_Boolean_t    last_NIiter_call;  /* True if this is the last call to NIiter() */
    Mif_Boolean_t    report_conv_probs; /* True if conv test functions should send debug info */
} Enh_Conv_Debug_t;


typedef struct {
    Mif_Boolean_t    enabled;           /* True if convergence limiting enabled on code models */
    double           abs_step;          /* Minimum limiting step size */
    double           step;              /* Fractional step amount */
} Enh_Conv_Limit_t;


typedef struct {
    Mif_Boolean_t    enabled;      /* True if rshunt option used */
    double           gshunt;       /* 1.0 / rshunt */
    int              num_nodes;    /* Number of nodes in matrix */
    double           **diag;       /* Pointers to matrix diagonals */
} Enh_Rshunt_t;


typedef struct {
    Enh_Bkpt_t       breakpoint;   /* Data used by dynamic breakpoints */
    Enh_Ramp_t       ramp;         /* New options added to simulator */
    Enh_Conv_Debug_t conv_debug;   /* Convergence debug info dumping data */
    Enh_Conv_Limit_t conv_limit;   /* Convergence limiting info */
    Enh_Rshunt_t     rshunt_data;  /* Shunt conductance from nodes to ground */
} Enh_Ckt_Data_t;



void ENHreport_conv_prob(Enh_Conv_Source_t type, char *name, char *msg);
struct line *ENHtranslate_poly(struct line *deck);


#endif /* ENH_HEADER */
