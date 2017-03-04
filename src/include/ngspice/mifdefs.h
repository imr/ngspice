#ifndef ngspice_MIFDEFS_H
#define ngspice_MIFDEFS_H

/* ===========================================================================
FILE    MIFdefs.h

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

    This file contains (augmented) SPICE 3C1 compatible typedefs for use
    with code models.  These typedefs define the data structures that are
    used internally to describe instances and models in the circuit
    description linked lists.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */


#include  "ngspice/mifcmdat.h"
#include "ngspice/ifsim.h"


/* The per-instance data structure */

struct MIFinstance {

    struct MIFmodel     *MIFmodPtr;       /* backpointer to model */
    struct MIFinstance  *MIFnextInstance; /* pointer to next instance of current model */
    IFuid               MIFname;          /* pointer to character string naming this instance */

    int                 num_conn;         /* number of connections on the code model */
    Mif_Conn_Data_t     **conn;           /* array of data structures for each connection */

    int                 num_inst_var;     /* number of instance variables on the code model */
    Mif_Inst_Var_Data_t **inst_var;       /* array of structs for each instance var */

    int                 num_param;        /* number of parameters on the code model */
    Mif_Param_Data_t    **param;          /* array of structs for each parameter */

    int                 num_state;        /* Number of state tags used for this inst */
    Mif_State_t         *state;           /* Info about states */

    int                 num_intgr;        /* Number of integrals */
    Mif_Intgr_t         *intgr;           /* Info for integrals */

    int                 num_conv;         /* Number of things to be converged */
    Mif_Conv_t          *conv;            /* Info for convergence things */

    Mif_Boolean_t       initialized;      /* True if model called once already */

    Mif_Boolean_t       analog;           /* true if this inst is analog or hybrid type */
    Mif_Boolean_t       event_driven;     /* true if this inst is event-driven or hybrid type */

    int                 inst_index;       /* Index into inst_table in evt struct in ckt */
    Mif_Callback_t      callback;         /* instance callback function */

};



/* The per model data structure */

struct MIFmodel {

    int              MIFmodType;      /* type index of this device type */
    struct MIFmodel *MIFnextModel;    /* pointer to next possible model in linked list */
    struct MIFinstance *MIFinstances; /* pointer to list of instances that have this model */
    IFuid            MIFmodName;      /* pointer to character string naming this model */

    /* --- end of generic struct GENmodel --- */

    int              num_param;       /* number of parameters on the code model */
    Mif_Param_Data_t **param;         /* array of structs for each parameter */

    Mif_Boolean_t    analog;          /* true if this model is analog or hybrid type */
    Mif_Boolean_t    event_driven;    /* true if this model is event-driven or hybrid type */

};



/* NOTE:  There are no device parameter tags, since the ask, mAsk, ...    */
/* functions for code models work out of the generic code model structure */



#endif
