/*============================================================================
FILE  cmpp.h

MEMBER OF process cmpp

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

    This file contains shared constants, type definitions,
    and function prototypes used in the cmpp process.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/


#include <stdio.h>

#define IFSPEC_FILENAME  "ifspec.ifs"
#define UDNFUNC_FILENAME "udnfunc.c"
#define MODPATH_FILENAME "modpath.lst"
#define UDNPATH_FILENAME "udnpath.lst"

/* *********************************************************************** */

typedef enum {

   OK,       /* Returned with no error */
   ERROR,    /* Returned with error */

} Status_t;



#define  GET_IFS_TABLE    0  /* Read the entire ifs table */
#define  GET_IFS_NAME     1  /* Get the C function name out of the table only */

#define  MAX_PATH_LEN  1024  /* Maximum pathname length */
#define  MAX_NAME_LEN  1024  /* Maximum SPICE name length */
#define  MAX_FN_LEN      31  /* Maximum filename length */


/* ******************************************************************** */
/* Structures used by parser to check for valid connections/parameters  */
/* ******************************************************************** */

/*
 * The boolean type
 */

typedef enum {
    FALSE,
    TRUE,
} Boolean_t;


/*
 * The direction of a connector
 */

typedef enum {
    IN,
    OUT,
    INOUT,
} Dir_t;


/*
 * The type of a port
 */

typedef enum {
    VOLTAGE,                /* v - Single-ended voltage */
    DIFF_VOLTAGE,           /* vd - Differential voltage */
    CURRENT,                /* i - Single-ended current */
    DIFF_CURRENT,           /* id - Differential current */
    VSOURCE_CURRENT,        /* vnam - Vsource name for input current */
    CONDUCTANCE,            /* g - Single-ended VCIS */
    DIFF_CONDUCTANCE,       /* gd - Differential VCIS */
    RESISTANCE,             /* h - Single-ended ICVS */
    DIFF_RESISTANCE,        /* hd - Differential ICVS */
    DIGITAL,                /* d - Digital */
    USER_DEFINED,           /* <identifier> - Any user defined type */
} Port_Type_t;



/*
 * The type of a parameter or Static_Var
 */

typedef enum {
    BOOLEAN,
    INTEGER,
    REAL,
    COMPLEX,
    STRING,
    POINTER,   /* NOTE: POINTER should not be used for Parameters - only
		* Static_Vars  - this is enforced by the cmpp.
		*/
 } Data_Type_t;



/*
 * The complex type
 */

typedef struct {
    double  real;
    double  imag;
} Complex_t;



/*
 * Values of different types.
 *
 * Note that a struct is used instead of a union for conformity
 * with the use of the Mif_Value_t type in the simulator where
 * the type must be statically initialized.  ANSI C does not
 * support useful initialization of unions.
 *
 */

typedef struct {

    Boolean_t   bvalue;         /* For BOOLEAN parameters */
    int         ivalue;         /* For INTEGER parameters */
    double      rvalue;         /* For REAL parameters    */
    Complex_t   cvalue;         /* For COMPLEX parameters */
    char        *svalue;        /* For STRING parameters  */

} Value_t;



/*
 * Information about the model as a whole
 */


typedef struct {

    char        *c_fcn_name;        /* Name used in the C function */
    char        *model_name;        /* Name used in a spice deck   */
    char        *description;       /* Description of the model    */

} Name_Info_t;




/*
 * Information about a connection
 */


typedef struct {

    char            *name;              /* Name of this connection */
    char            *description;       /* Description of this connection */
    Dir_t           direction;          /* IN, OUT, or INOUT             */
    Port_Type_t     default_port_type;  /* The default port type */
    char            *default_type;      /* The default type in string form */
    int             num_allowed_types;  /* The size of the allowed type arrays */
    Port_Type_t     *allowed_port_type; /* Array of allowed types */
    char            **allowed_type;     /* Array of allowed types in string form */
    Boolean_t       is_array;           /* True if connection is an array       */
    Boolean_t       has_conn_ref;     /* True if there is associated with an array conn */
    int             conn_ref;         /* Subscript of the associated array conn */
    Boolean_t       has_lower_bound;    /* True if there is an array size lower bound */
    int             lower_bound;        /* Array size lower bound */
    Boolean_t       has_upper_bound;    /* True if there is an array size upper bound */
    int             upper_bound;        /* Array size upper bound */
    Boolean_t       null_allowed;       /* True if null is allowed for this connection */

} Conn_Info_t;




/*
 * Information about a parameter
 */

typedef struct {

    char            *name;            /* Name of this parameter */
    char            *description;     /* Description of this parameter */
    Data_Type_t     type;             /* Data type, e.g. REAL, INTEGER, ... */
    Boolean_t       has_default;      /* True if there is a default value */
    Value_t         default_value;    /* The default value */
    Boolean_t       has_lower_limit;  /* True if there is a lower limit */
    Value_t         lower_limit;      /* The lower limit for this parameter */
    Boolean_t       has_upper_limit;  /* True if there is a upper limit */
    Value_t         upper_limit;      /* The upper limit for this parameter */
    Boolean_t       is_array;         /* True if parameter is an array       */
    Boolean_t       has_conn_ref;     /* True if there is associated with an array conn */
    int             conn_ref;         /* Subscript of the associated array conn */
    Boolean_t       has_lower_bound;  /* True if there is an array size lower bound */
    int             lower_bound;      /* Array size lower bound */
    Boolean_t       has_upper_bound;  /* True if there is an array size upper bound */
    int             upper_bound;      /* Array size upper bound */
    Boolean_t       null_allowed;     /* True if null is allowed for this parameter */

} Param_Info_t;


/*
 * Information about an instance variable
 */

typedef struct {

    char            *name;            /* Name of this parameter */
    char            *description;     /* Description of this parameter */
    Data_Type_t     type;             /* Data type, e.g. REAL, INTEGER, ... */
    Boolean_t       is_array;         /* True if parameter is an array       */

} Inst_Var_Info_t;



/*
 * The all encompassing structure for the ifs table information
 */

typedef struct {

    Name_Info_t     name;             /* The name table entries */
    int             num_conn;         /* Number of entries in the connection table(s) */
    Conn_Info_t     *conn;            /* Array of connection info structs */
    int             num_param;        /* Number of entries in the parameter table(s) */
    Param_Info_t    *param;           /* Array of parameter info structs */
    int             num_inst_var;     /* Number of entries in the instance var table(s) */
    Inst_Var_Info_t *inst_var;        /* Array of instance variable info structs */

} Ifs_Table_t;



/* *********************************************************************** */



void preprocess_ifs_file(void);

void preprocess_lst_files(void);

void preprocess_mod_file(char *filename);


void init_error (char *program_name);

#ifdef __GNUC__
void print_error(const char *fmt, ...) __attribute__ ((format (__printf__, 1, 2)));
#else
void print_error(const char *fmt, ...);
#endif

void str_to_lower(char *s);


Status_t read_ifs_file(char *filename, int mode, Ifs_Table_t *ifs_table);

Status_t write_ifs_c_file(char *filename, Ifs_Table_t *ifs_table);


FILE *fopen_with_path(const char *path, const char *mode);
