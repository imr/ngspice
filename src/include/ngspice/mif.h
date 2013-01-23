#ifndef ngspice_MIF_H
#define ngspice_MIF_H

/* ===========================================================================
FILE    MIF.h

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

    This file structure definitions global data used with the MIF package.
    The global data structure is used to circumvent the need to modify
    argument lists in existing SPICE 3C1 functions.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */

#include  "ngspice/miftypes.h"
#include  "ngspice/mifdefs.h"
#include  "ngspice/cktdefs.h"


typedef struct {
   Mif_Boolean_t   init;        /* TRUE if first call to model */
   Mif_Boolean_t   anal_init;   /* TRUE if first call for this analysis type */
   Mif_Analysis_t  anal_type;   /* The type of analysis being performed */
   Mif_Call_Type_t call_type;   /* Type of call to code model - analog or event-driven */
   double          evt_step;    /* The current DC step or time in event analysis */
} Mif_Circuit_Info_t;


typedef struct {
   double           current;    /* The current dynamic breakpoint time */
   double           last;       /* The last used dynamic breakpoint time */
} Mif_Bkpt_Info_t;


typedef struct {
    Mif_Boolean_t   global;     /* Set by .option to force all models to use auto */
    Mif_Boolean_t   local;      /* Set by individual model to request auto partials */
} Mif_Auto_Partial_t;


typedef struct {
   Mif_Circuit_Info_t circuit;    /* Circuit data that will be needed by MIFload */
   MIFinstance        *instance;  /* Current instance struct */
   CKTcircuit         *ckt;       /* The ckt struct for the circuit */
   char               *errmsg;    /* An error msg from a cm_... function */
   Mif_Bkpt_Info_t    breakpoint; /* Data used by dynamic breakpoints */
   Mif_Auto_Partial_t auto_partial; /* Flags to enable auto partial computations */
} Mif_Info_t;


/* These are defined in mif.c */
extern int MIFiSize;
extern int MIFmSize;


extern Mif_Info_t  g_mif_info;

#endif
