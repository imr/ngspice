#ifndef ngspice_MIFPARSE_H
#define ngspice_MIFPARSE_H

/* ===========================================================================
FILE    MIFparse.h

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

    This file contains the information structure definitions used by the
    code model parser to check for valid connections and parameters.

    Structures of these types are created by the code model preprocessor
    (cmpp) from the user created ifspec.ifs file.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */


#include  "ngspice/miftypes.h"


/*
 * Values of different types used by the parser.  Note that this is a structure
 * instead of a union because we need to do initializations in the ifspec.c files for
 * the models and unions cannot be initialized in any useful way in C
 *
 */

struct Mif_Parse_Value {

    Mif_Boolean_t     bvalue;         /* For boolean values */
    int               ivalue;         /* For integer values */
    double            rvalue;         /* For real values */
    Mif_Complex_t     cvalue;         /* For complex values */
    char              *svalue;        /* For string values  */

};


/*
 * Information about a connection used by the parser to error check input
 */


struct Mif_Conn_Info {

    char            *name;             /* Name of this connection */
    char            *description;      /* Description of this connection */
    Mif_Dir_t       direction;         /* Is this connection an input, output, or both? */
    Mif_Port_Type_t default_port_type; /* The default port type */
    char            *default_type;     /* The default type in string form */
    int             num_allowed_types; /* The size of the allowed type arrays */
    Mif_Port_Type_t *allowed_type;     /* The allowed types */
    char            **allowed_type_str; /* The allowed types in string form */
    Mif_Boolean_t   is_array;          /* True if connection is an array       */
    Mif_Boolean_t   has_lower_bound;   /* True if there is an array size lower bound */
    int             lower_bound;       /* Array size lower bound */
    Mif_Boolean_t   has_upper_bound;   /* True if there is an array size upper bound */
    int             upper_bound;       /* Array size upper bound */
    Mif_Boolean_t   null_allowed;      /* True if null is allowed for this connection */

};




/*
 * Information about a parameter used by the parser to error check input
 */

struct Mif_Param_Info {

    char                *name;            /* Name of this parameter */
    char                *description;     /* Description of this parameter */
    Mif_Data_Type_t     type;             /* Is this a real, boolean, string, ... */
    Mif_Boolean_t       has_default;      /* True if there is a default value */
    Mif_Parse_Value_t   default_value;    /* The default value */
    Mif_Boolean_t       has_lower_limit;  /* True if there is a lower limit */
    Mif_Parse_Value_t   lower_limit;      /* The lower limit for this parameter */
    Mif_Boolean_t       has_upper_limit;  /* True if there is a upper limit */
    Mif_Parse_Value_t   upper_limit;      /* The upper limit for this parameter */
    Mif_Boolean_t       is_array;         /* True if parameter is an array       */
    Mif_Boolean_t       has_conn_ref;     /* True if parameter is associated with a connector */
    int                 conn_ref;         /* The subscript of the associated connector */
    Mif_Boolean_t       has_lower_bound;  /* True if there is an array size lower bound */
    int                 lower_bound;      /* Array size lower bound */
    Mif_Boolean_t       has_upper_bound;  /* True if there is an array size upper bound */
    int                 upper_bound;      /* Array size upper bound */
    Mif_Boolean_t       null_allowed;     /* True if null is allowed for this parameter */

};




/*
 * Information about an instance parameter used by the parser to error check input
 */

struct Mif_Inst_Var_Info {

    char                *name;            /* Name of this instance var */
    char                *description;     /* Description of this instance var */
    Mif_Data_Type_t     type;             /* Is this a real, boolean, string, ... */
    Mif_Boolean_t       is_array;         /* True if instance var is an array       */

};


#endif
