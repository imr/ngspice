#ifndef ngspice_MIFTYPES_H
#define ngspice_MIFTYPES_H

/* ===========================================================================
FILE    MIFtypes.h

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

    This file contains typedefs shared by several header files in
    the MIF package.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */




/* ***************************************************************************** */


typedef int Mif_Boolean_t;

#define  MIF_FALSE  0
#define  MIF_TRUE   1


typedef int Mif_Status_t;

#define  MIF_OK     0
#define  MIF_ERROR  1

/*
typedef enum {
    MIF_OK,
    MIF_ERROR,
} Mif_Status_t;
*/


/* ***************************************************************************** */


/*
 * The type of call to a code model - analog or event-driven
 */

typedef enum {
    MIF_ANALOG,         /* Analog call */
    MIF_EVENT_DRIVEN,   /* Event-driven call */
} Mif_Call_Type_t;



/*
 * Analysis type enumerations
 */

typedef enum {
    MIF_DC,                /* A DC or DCOP analysis */
    MIF_AC,                /* A swept AC analysis   */
    MIF_TRAN,              /* A transient analysis  */
    MIF_NOI,               /* A noise analysis  */
} Mif_Analysis_t;



/*
 * Port type enumerations
 */

typedef enum {
    MIF_VOLTAGE,                /* v - Single-ended voltage */
    MIF_DIFF_VOLTAGE,           /* vd - Differential voltage */
    MIF_CURRENT,                /* i - Single-ended current */
    MIF_DIFF_CURRENT,           /* id - Differential current */
    MIF_VSOURCE_CURRENT,        /* vnam - Voltage source current */
    MIF_CONDUCTANCE,            /* g - Single-ended VCIS */
    MIF_DIFF_CONDUCTANCE,       /* gd - Differential VCIS */
    MIF_RESISTANCE,             /* h - Single-ended ICVS */
    MIF_DIFF_RESISTANCE,        /* hd - Differential ICVS */
    MIF_DIGITAL,                /* d - Digital */
    MIF_USER_DEFINED,           /* <identifier> - Any user defined type */
} Mif_Port_Type_t;



/*
 * The direction of a connector
 */

typedef enum {
    MIF_IN,                    /* Input only */
    MIF_OUT,                   /* Output only */
    MIF_INOUT,                 /* Input and output (e.g. g or h type) */
} Mif_Dir_t;



/*
 * The type of a parameter
 */

typedef enum {

    MIF_BOOLEAN,
    MIF_INTEGER,
    MIF_REAL,
    MIF_COMPLEX,
    MIF_STRING,

} Mif_Data_Type_t;



/*
 * The type of a token
 */

typedef enum {

    MIF_LARRAY_TOK,
    MIF_RARRAY_TOK,
    MIF_LCOMPLEX_TOK,
    MIF_RCOMPLEX_TOK,
    MIF_PERCENT_TOK,
    MIF_TILDE_TOK,
    MIF_STRING_TOK,
    MIF_NULL_TOK,
    MIF_NO_TOK,

} Mif_Token_Type_t;



/* 
 * Type of controlled source
 */

typedef enum {
    MIF_VCVS,
    MIF_VCIS,
    MIF_ICVS,
    MIF_ICIS,
    MIF_minus_one,
} Mif_Cntl_Src_Type_t;


/*
 * The "reason" for a callback invocation
 */

typedef enum {
    MIF_CB_DESTROY = 1,   /* MIFdestroy has been invoked, its time to clean up */
}  Mif_Callback_Reason_t;


/* ***************************************************************************** */


/*
 * Complex numbers
 */

typedef struct Mif_Complex {

    double  real;
    double  imag;

} Mif_Complex_t;



/*
 * Values of different types used by the load, ... routines
 */

typedef union {

    Mif_Boolean_t  bvalue;         /* For digital node value */
    int            ivalue;         /* For integer parameters */
    double         rvalue;         /* For spice node values and real parameters */  
    Mif_Complex_t  cvalue;         /* For complex parameters */
    char           *svalue;        /* For string parameters  */
    void           *pvalue;        /* For user defined nodes */

} Mif_Value_t;



/* types from mifparse.h */
typedef struct Mif_Parse_Value Mif_Parse_Value_t;
typedef struct Mif_Conn_Info Mif_Conn_Info_t;
typedef struct Mif_Param_Info Mif_Param_Info_t;
typedef struct Mif_Inst_Var_Info Mif_Inst_Var_Info_t;

/* types from mifcmdat.h */
typedef struct Mif_Private Mif_Private_t;

/* types from mifdefs.h */
typedef struct MIFinstance MIFinstance;
typedef struct MIFmodel MIFmodel;

typedef void (* Mif_Callback_t)(Mif_Private_t *, Mif_Callback_Reason_t);


#endif
