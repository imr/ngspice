#ifndef ngspice_MIFCMDAT_H
#define ngspice_MIFCMDAT_H

/* ===========================================================================
FILE    MIFcmdat.h

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

   This file contains the data structure definitions used by
   code model and the associated MIF package.

   A special preprocessor (cmpp) is used on models written by a
   user to turn items like INPUT(<name>) into the appropriate structure
   reference.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */


#include "ngspice/typedefs.h"
#include  "ngspice/miftypes.h"


/* ************************************************************************** */


/*
 *  Pointers into matrix for a voltage input, voltage output partial
 */

typedef struct Mif_E_Ptr_s {

    double      *branch_poscntl;   /* Branch row, positive controlling column */
    double      *branch_negcntl;   /* Branch row, negative controlling column */

} Mif_E_Ptr_t;



/*
 *  Pointers into matrix for a current input, current output partial
 */

typedef struct Mif_F_Ptr_s {

    double      *pos_ibranchcntl;    /* Positive row, controlling branch column */
    double      *neg_ibranchcntl;    /* Negative row, controlling branch column */

} Mif_F_Ptr_t;



/*
 *  Pointers into matrix for a voltage input, current output partial
 */

typedef struct Mif_G_Ptr_s {

    double      *pos_poscntl;       /* Positive row, positive controlling column */
    double      *pos_negcntl;       /* Positive row, negative controlling column */
    double      *neg_poscntl;       /* Negative row, positive controlling column */
    double      *neg_negcntl;       /* Negative row, negative controlling column */

} Mif_G_Ptr_t;


/*
 *  Pointers into matrix for a current input, voltage output partial
 */

typedef struct Mif_H_Ptr_s {

    double      *branch_ibranchcntl;  /* Branch row, controlling branch column */

} Mif_H_Ptr_t;




/*
 * Matrix pointers associated with a particular port (of a particular type)
 */


typedef union Mif_Port_Ptr_u {

    Mif_E_Ptr_t    e;          /* Pointers for voltage input, voltage output */
    Mif_F_Ptr_t    f;          /* Pointers for current input, current output */
    Mif_G_Ptr_t    g;          /* Pointers for voltage input, current output */
    Mif_H_Ptr_t    h;          /* Pointers for current input, voltage output */

} Mif_Port_Ptr_t;



/*
 * Array of matrix data pointers for particular ports in a connection
 */

typedef struct Mif_Conn_Ptr_s {

    Mif_Port_Ptr_t      *port;    /* Data for a particular port */

} Mif_Conn_Ptr_t;



/*
 * Row numbers and matrix entry pointers for loading the matrix and RHS with
 * data appropriate for the particular output port and input ports.
 */

typedef struct Mif_Smp_Ptr_s {

    /* Data at this level is for this connection.  The Mif_Conn_Ptr_t    */
    /* subtree is used only if this connection is an output. It supplies */
    /* the matrix pointers required for loading the partials from each   */
    /* input.                                                            */

    /* node connection equation numbers */
    int             pos_node;         /* Row associated with positive node */
    int             neg_node;         /* Row associated with negative node */

    /* V source branch equation numbers */
    int             branch;           /* Row associated with V output branch */
    int             ibranch;          /* Row associated with I input branch */

    /* matrix pointers for V source output */
    double          *pos_branch;      /* Positive node row, branch column */
    double          *neg_branch;      /* Negative node row, branch column */
    double          *branch_pos;      /* Branch row, positive node column */
    double          *branch_neg;      /* Branch row, negative node column */

    /* matrix pointers for the zero-valued V source associated with an I input */
    double          *pos_ibranch;      /* Positive node row, branch column */
    double          *neg_ibranch;      /* Negative node row, branch column */
    double          *ibranch_pos;      /* Branch row, positive node column */
    double          *ibranch_neg;      /* Branch row, negative node column */

    /* array of pointer info required for putting partials into the matrix */
    Mif_Conn_Ptr_t  *input;    /* Matrix pointers associated with inputs */

} Mif_Smp_Ptr_t;




/* ******************************************************************** */



/*
 * Partial derivatives wrt ports of a particular input connection
 */

typedef struct Mif_Partial_s {

    double      *port;          /* Partial wrt this port */

} Mif_Partial_t;


/*
 * AC gains wrt ports of a particular input connection
 */

typedef struct Mif_AC_Gain_s {

    Mif_Complex_t      *port;          /* AC gain wrt this port */

} Mif_AC_Gain_t;


/*
 * Data used to access information in event struct in CKTcircuit struct ckt
 */

typedef struct {
    int         node_index;      /* Index of node in event-driven structures */
    int         output_subindex; /* Subindex of output on node */
    int         port_index;      /* Index of port in event-driven structures */
    int         output_index;    /* Index of output in event-driven structures */
} Mif_Evt_Data_t;


/*
 * Information about individual port(s) of a connection.
 */

typedef struct Mif_Port_Data_s {

    Mif_Port_Type_t type;           /* Port type - e.g. MIF_VOLTAGE, ...  */
    char            *type_str;      /* Port type in string form           */
    char            *pos_node_str;  /* Positive node identifier           */
    char            *neg_node_str;  /* Negative node identifier           */
    char            *vsource_str;   /* Voltage source identifier          */

    Mif_Boolean_t   is_null;        /* Set to true if null in SPICE deck  */
    Mif_Value_t     input;          /* The input value                    */
    Mif_Value_t     output;         /* The output value                   */
    struct Evt_Output_Event *next_event;
    Mif_Partial_t   *partial;       /* Partials for this port wrt inputs  */
    Mif_AC_Gain_t   *ac_gain;       /* AC gains for this port wrt inputs  */
    int             old_input;      /* Index into CKTstate for old input  */

    Mif_Boolean_t   invert;         /* True if state should be inverted   */
    Mif_Boolean_t   changed;        /* A new output has been assigned     */
    double          load;           /* Load factor output to this port    */
    double          total_load;     /* Total load for this port           */
    double          delay;          /* Digital delay for this output port */
    char            *msg;           /* Message string output to port      */

    Mif_Smp_Ptr_t   smp_data;       /* Pointers used to load matrix/rhs   */
    Mif_Evt_Data_t  evt_data;       /* Data used to access evt struct     */

    double          nominal_output; /* Saved output when doing auto partial */

} Mif_Port_Data_t;


/* ******************************************************************** */

/*
 * Information in MIFinstance struct used by cm_.. support functions.
 */


typedef struct Mif_State_s {   /* for cm_analog_alloc() */

    int         tag;           /* Tag identifying this particular state */
    int         index;         /* Index into ckt->CKTstate[i] vector */
    int         doubles;       /* Number of doubles allocated for this state */
    int         bytes;         /* Actual number of bytes requested by cm_analog_alloc() */

} Mif_State_t;


typedef struct Mif_Intgr_s {   /* for cm_analog_integrate() */

    int         byte_index;    /* Byte offset into state array */

} Mif_Intgr_t;


typedef struct Mif_Conv_s {   /* for cm_analog_converge() */

    int         byte_index;   /* Byte offset into state array */
    double      last_value;   /* Value at last iteration */

} Mif_Conv_t;



/* ******************************************************************** */



/*
 * Information about the circuit in which this model is simulating.
 */

typedef struct Mif_Circ_Data_s {

    Mif_Boolean_t         init;         /* True if first call to model - a setup pass */
    Mif_Analysis_t        anal_type;    /* Current analysis type                      */
    Mif_Boolean_t         anal_init;    /* True if first call in this analysis type   */
    Mif_Call_Type_t       call_type;    /* Analog or event type call                  */
    double                time;         /* Current analysis time                      */
    double                frequency;    /* Current analysis frequency                 */
    double                temperature;  /* Current analysis temperature               */
    double                t[8];         /* History of last 8 analysis times t[0]=time */

} Mif_Circ_Data_t;



/*
 * The structure associated with a named "connection" on the model.
 */

typedef struct Mif_Conn_Data_s {

    char             *name;          /* Name of this connection - currently unused */
    char             *description;   /* Description of this connection - unused    */
    Mif_Boolean_t    is_null;        /* Set to true if null in SPICE deck          */
    Mif_Boolean_t    is_input;       /* Set to true if connection is an input */
    Mif_Boolean_t    is_output;      /* Set to true if connection is an output */
    int              size;           /* The size of an array (1 if scalar)         */
    Mif_Port_Data_t  **port;         /* Pointer(s) to port(s) for this connection  */

} Mif_Conn_Data_t;



/*
 * Values for model parameters
 */

typedef struct Mif_Param_Data_s {

    Mif_Boolean_t   is_null;            /* True if no value given on .model card */
    int             size;               /* Size of array (1 if scalar)           */
    Mif_Value_t     *element;           /* Value of parameter(s)                 */
    int             eltype;             /* type of the element                   */

} Mif_Param_Data_t;




/*
 * Values for instance variables
 */

typedef struct Mif_Inst_Var_Data_s {

    int         size;               /* Size of array (1 if scalar)           */
    Mif_Value_t *element;           /* Value of instance variables(s)        */

} Mif_Inst_Var_Data_t;




/* ************************************************************************* */



/*
 * HERE IT IS!!!
 * The top level data structure passed to code models.
 */

struct Mif_Private {

    Mif_Circ_Data_t        circuit;       /* Information about the circuit        */
    int                    num_conn;      /* Number of connections on this model  */
    Mif_Conn_Data_t        **conn;        /* Information about each connection    */
    int                    num_param;     /* Number of parameters on this model   */
    Mif_Param_Data_t       **param;       /* Information about each parameter     */
    int                    num_inst_var;  /* Number of instance variables         */
    Mif_Inst_Var_Data_t    **inst_var;    /* Information about each inst variable */
    Mif_Callback_t         *callback;     /* Callback function */

};



#endif
