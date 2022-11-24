/*============================================================================
FILE  writ_ifs.c

MEMBER OF process cmpp

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains functions used to write out the file "ifspec.c"
    based on information read from "ifspec.ifs", which is now held in
    structure 'ifs_table' passed to write_ifs_c_file().

INTERFACES

    write_ifs_c_file()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/


#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include  "cmpp.h"

/* Local function prototypes */
static int write_comment(FILE *fp, Ifs_Table_t *ifs_table);
static int write_includes(FILE *fp);
static int write_mPTable(FILE *fp, Ifs_Table_t *ifs_table);
static int write_pTable(FILE *fp, Ifs_Table_t *ifs_table);
static int write_conn_info(FILE *fp, Ifs_Table_t *ifs_table);
static int write_param_info(FILE *fp, Ifs_Table_t *ifs_table);
static int write_inst_var_info(FILE *fp, Ifs_Table_t *ifs_table);
static int write_SPICEdev(FILE *fp, Ifs_Table_t *ifs_table);



static const char *data_type_to_str(Data_Type_t type);
static const char *port_type_to_str(Port_Type_t port);
static const char  *dir_to_str(Dir_t dir);
static char  *value_to_str(Data_Type_t type, Value_t value);
static const char *no_value_to_str(void);
static const char  *boolean_to_str(bool value);
static char  *integer_to_str(int value);
static const char *gen_port_type_str(Port_Type_t port);



/* *********************************************************************** */



/*
write_ifs_c_file

Function write_ifs_c_file is a top-level driver function for
creating a C file (ifspec.c) that defines the model Interface
Specification in a form usable by the simulator.  The ifspec.c
output file is opened for writing, and then each of the following
functions is called in order to write the necessary statements
and data structures into the file:

     write_comment
     write_includes
     write_mPTable
     write_conn_info
     write_param_info
     write_inst_var_info
     write_SPICEdev

The output file is then closed.
*/



int write_ifs_c_file(
    const char  *filename_in,    /* File to write to */
    Ifs_Table_t *ifs_table)   /* Table of Interface Specification data */
{
    FILE     *fp = (FILE *) NULL; /* File pointer */
    char *filename = (char *) NULL;
    int xrc = 0;


    /* Open the ifspec.c file for write access */
    if ((filename = gen_filename(filename_in, "w")) == (char *) NULL) {
        print_error("ERROR - Unable to build path to \"%s\".", filename_in);
        xrc = -1;
        goto EXITPOINT;
    }
    fp = fopen(filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening \"%s\" for write", filename);
        xrc = -1;
        goto EXITPOINT;
    }

    /* Write out a comment section at the top of the file */
    if (write_comment(fp, ifs_table) != 0) {
        print_error("ERROR - Problems writing comment to \"%s\"", filename);
        xrc = -1;
        goto EXITPOINT;
    }

    /* Put in the # includes */
    if (write_includes(fp) != 0) {
        print_error("ERROR - Problems writing includes to \"%s\"", filename);
        xrc = -1;
        goto EXITPOINT;
    }

    /* Write the SPICE3 required XXXmPTable structure */
    if (write_mPTable(fp, ifs_table) != 0) {
        print_error("ERROR - Problems writing mPTable to \"%s\"", filename);
        xrc = -1;
        goto EXITPOINT;
    }

    /* Write the SPICE3 required XXXpTable structure */
    if (write_pTable(fp, ifs_table) != 0) {
        print_error("ERROR - Problems writing pTable to \"%s\"", filename);
        xrc = -1;
        goto EXITPOINT;
    }

    /* Write out the connector table required for the code model element parser */
    if (write_conn_info(fp, ifs_table) != 0) {
        print_error("ERROR - Problems writing includes to \"%s\"", filename);
        xrc = -1;
        goto EXITPOINT;
    }

    /* Write out the parameter table required for the code model element parser */
    if (write_param_info(fp, ifs_table) != 0) {
        print_error("ERROR - Problems writing params to \"%s\"", filename);
        xrc = -1;
        goto EXITPOINT;
    }

    /* Write out the instance variable table required for the code model element parser */
    if (write_inst_var_info(fp, ifs_table) != 0) {
        print_error("ERROR - Problems writing instaice variables to \"%s\"",
                filename);
        xrc = -1;
        goto EXITPOINT;
    }

    /* Write out the externally visible structure for this model */
    if (write_SPICEdev(fp, ifs_table) != 0) {
        print_error("ERROR - Problems writing model structureto  \"%s\"",
                filename);
        xrc = -1;
        goto EXITPOINT;
    }


EXITPOINT:
    /* Close the ifspec.c file, free allocation, and return */
    if (fp != (FILE *) NULL) {
        if (fclose(fp) != 0) {
            print_error("ERROR - Problems closing \"%s\": %s.",
                    filename, strerror(errno));
            xrc = -1;
        }
    }

    if (filename != (char *) NULL) {
        free(filename);
    }

    return xrc;
} /* end of function write_ifs_c_file */



/* *********************************************************************** */


/*
write_comment

Function write_comment places a comment at the top of the
ifspec.c file warning the user that this file is automatically
generated and should not be edited.
*/


static int write_comment(
    FILE        *fp,          /* File to write to */
    Ifs_Table_t *ifs_table)   /* Table of Interface Specification data */
{
    const int rc = fprintf(fp,
            "\n"
            "/*\n"
            " * Structures for model: %s\n"
            " *\n"
            " * Automatically generated by cmpp preprocessor\n"
            " *\n"
            " * !!! DO NOT EDIT !!!\n"
            " *\n"
            " */\n"
            "\n",
            ifs_table->name.model_name);
    if (rc < 0) {
        print_error("Comment writing failed.");
        return -1;
    }
    return 0;
} /* end of function write_comment */



/* *********************************************************************** */

/*
write_includes

Function write_includes writes the C header files required in
ifspec.c.
*/


static int write_includes(
    FILE *fp)                /* File to write to */
{
    const int rc = fprintf(fp,
            "\n"
            "#include \"ngspice/ngspice.h\"\n"
/*          "#include \"ngspice/prefix.h\"\n"*/
            "#include <stdio.h>\n"
            "#include \"ngspice/devdefs.h\"\n"
            "#include \"ngspice/ifsim.h\"\n"
            "#include \"ngspice/mifdefs.h\"\n"
            "#include \"ngspice/mifproto.h\"\n"
            "#include \"ngspice/mifparse.h\"\n"
/*          "#include \"ngspice/suffix.h\"\n"*/
            "\n");
    if (rc < 0) {
        print_error("Include writing failed.");
        return -1;
    }
    return 0;
} /* end of function write_includes */



/* *********************************************************************** */



/*
write_pTable

Function write_pTable writes the instance parameter information
using SPICE's IFparm structure type.  This table defines the
parameters as output only variables using SPICE's ``OP'' macro. 
These instance parameters are derived from the Interface
Specification file's STATIC_VAR table and define the parameters
that can be queried using the SPICE 3C1 .save feature.
*/


static int write_pTable(
    FILE        *fp,         /* File to write to */
    Ifs_Table_t *ifs_table)  /* Table of Interface Specification data */
{
    int xrc = 0;
    int             i;
    bool       is_array;
    Data_Type_t     type;


    /* Only write the pTable if there is something to put in it.         */
    /* Otherwise, we will put NULL in the SPICEdev structure in its slot */
    if (ifs_table->num_inst_var == 0) {
        return 0;
    }

    int rc = 0;

    /* Write the structure beginning */
    rc |= fprintf(fp,
            "\n"
            "static IFparm MIFpTable[] = {\n");


    /* Write out an entry for each instance variable in the table       */

    /* Use the index of the element in the instance variable info array */
    /* ADDED TO the number of parameters as the SPICE3 integer tag.  */

    for (i = 0; i < ifs_table->num_inst_var; i++) {

        /* Use the SPICE3 OP macro since instance vars are output-only */

        rc |= fprintf(fp, "    OP(");

        /* Put in the name of the parameter and the integer tag */

        rc |= fprintf(fp, "\"%s\", ", ifs_table->inst_var[i].name);
        rc |= fprintf(fp, "%d, ", i + ifs_table->num_param);

        /* Format SPICE3 type according to parameter type field */

        type = ifs_table->inst_var[i].type;
        is_array = ifs_table->inst_var[i].is_array;

        if(is_array == true) {
            rc |= fprintf(fp, "(");
        }

        if(type == CMPP_BOOLEAN) {
            rc |= fprintf(fp, "IF_FLAG");   /* no BOOLEAN in SPICE3 */
        }
        else if(type == CMPP_INTEGER) {
            rc |= fprintf(fp, "IF_INTEGER");
        }
        else if(type == CMPP_REAL) {
            rc |= fprintf(fp, "IF_REAL");
        }
        else if(type == CMPP_COMPLEX) {
            rc |= fprintf(fp, "IF_COMPLEX");
        }
        else if(type == CMPP_STRING) {
            rc |= fprintf(fp, "IF_STRING");
        }
        else if(type == CMPP_POINTER) {
            rc |= fprintf(fp, "IF_STRING");
        }
        else {
            print_error("INTERNAL ERROR - write_pTable() - Impossible data type.");
            xrc = -1;
            rc |= fprintf(fp, "INVALID DATA TYPE");
        }

        if(is_array == true) {
            rc |= fprintf(fp, "|IF_VECTOR)");
        }


        /* Put in the description string and finish this line off */
        rc |= fprintf(fp, ", \"%s\"),\n", ifs_table->inst_var[i].description);
    } /* end of loop over instance variables */

    /* Finish off the structure */
    rc |= fprintf(fp, "};\n\n");

    /* Check outputs */
    if (rc < 0) {
        print_error("pTable writing failed.");
        xrc = -1;
    }

    return xrc;
} /* end of function write_pTable */



/* *********************************************************************** */


/*
write_mPTable

Function write_mPTable writes the model parameter information
using SPICE's IFparm structure type.  This table defines the
parameters to be input/output variables using SPICE's ``IOP'' macro
so that these variables can be set or queried from SPICE.  These
model parameters are derived from the Interface Specification's
PARAMETER table.
*/

static int write_mPTable(
    FILE        *fp,         /* File to write to */
    Ifs_Table_t *ifs_table)  /* Table of Interface Specification data */
{
    int xrc = 0;
    int             i;


    /* Only write the mPTable if there is something to put in it.         */
    /* Otherwise, we will put NULL in the SPICEdev structure in its slot */

    if (ifs_table->num_param == 0) {
        return 0;
    }

    int rc = 0;

    /* Write the structure beginning */
    rc |= fprintf(fp,
            "\n"
            "static IFparm MIFmPTable[] = {\n");


    /* Write out an entry for each parameter in the table               */

    /* Use the index of the element in the parameter info array */
    /* as the SPICE3 integer tag.                                       */
    Param_Info_t *param = ifs_table->param;
    for(i = 0; i < ifs_table->num_param; i++) {
        Param_Info_t *param_cur = param + i;

        /* Use the SPICE3 IOP macro since model parameters are input/output */

        rc |= fprintf(fp, "    IOP(");

        /* Put in the name of the parameter and the integer tag */
        rc |= fprintf(fp, "\"%s\", ", param_cur->name);
        rc |= fprintf(fp, "%d, ", i);

        /* Format SPICE3 type according to parameter type field */

        const bool is_array = param_cur->is_array;
        const Data_Type_t type = param_cur->type;

        if (is_array) {
            rc |= fprintf(fp, "(");
        }

        if (type == CMPP_BOOLEAN) {
            rc |= fprintf(fp, "IF_FLAG");   /* no BOOLEAN in SPICE3 */
        }
        else if (type == CMPP_INTEGER) {
            rc |= fprintf(fp, "IF_INTEGER");
        }
        else if (type == CMPP_REAL) {
            rc |= fprintf(fp, "IF_REAL");
        }
        else if (type == CMPP_COMPLEX) {
            rc |= fprintf(fp, "IF_COMPLEX");
        }
        else if (type == CMPP_STRING) {
            rc |= fprintf(fp, "IF_STRING");
        }
        else {
            print_error("INTERNAL ERROR - write_mPTable() - Impossible data type.");
            xrc = -1;
        }

        if (is_array) {
            rc |= fprintf(fp, "|IF_VECTOR)");
        }


        /* Put in the description string and finish this line off */
        rc |= fprintf(fp, ", \"%s\"),\n", ifs_table->param[i].description);
    } /* end of loop over parameters */

    /* Finish off the structure */
    rc |= fprintf(fp, "};\n\n");

    /* Check outputs */
    if (rc < 0) {
        print_error("mPTable writing failed.");
        xrc = -1;
    }

    return xrc;
} /* end of function write_mPTable */



/* *********************************************************************** */


/*
write_conn_info

Function write_conn_info writes information used by the
Simulator's new MIF package to interpret and error check a
model's connection list in a SPICE deck.  This information is
derived from the Interface Specification file's PORT table.
*/



static int write_conn_info(
    FILE        *fp,          /* File to write to */
    Ifs_Table_t *ifs_table)   /* Table of Interface Specification data */
{
    int xrc = 0;
    int             i;
    int             j;
    const char *str;
    int rc = 0;

    /* Only write the connTable if there is something to put in it.      */
    /* Otherwise, we will put NULL in the SPICEdev structure in its slot */

    if (ifs_table->num_conn == 0) { /* An unlikely condition for sure ... */
        return 0;
    }

    /* First, we must define arrays of port types */

    /* Note that there should be always at least one allowed port type */
    /* so we don't have to worry about arrays with no elements         */

    const Conn_Info_t * const conn = ifs_table->conn;
    const int num_conn = ifs_table->num_conn;
    for (i = 0; i < num_conn; i++) {
        const Conn_Info_t * const p_conn_cur = conn + i;
        rc |= fprintf(fp,
                "\n"
                "static Mif_Port_Type_t MIFportEnum%d[] = {\n", i);

        if (p_conn_cur->num_allowed_types < 1) {
            print_error("ERROR - write_conn_info() - "
                    "Number of allowed types cannot be zero");
            xrc = -1;
        }

        const int num_allowed_types = p_conn_cur->num_allowed_types;
        for (j = 0; j < num_allowed_types; j++) {
            rc |= fprintf(fp, "    %s,\n",
                    port_type_to_str(p_conn_cur->allowed_port_type[j]));
        }  /* for number of allowed types */

        rc |= fprintf(fp,
                "};\n"
                "\n"
                "\n"
                "static char *MIFportStr%d[] = {\n", i);

        for (j = 0; j < num_allowed_types; j++) {
            if (p_conn_cur->allowed_port_type[j] == USER_DEFINED) {
                rc |= fprintf(fp, "    \"%s\",\n",
                        p_conn_cur->allowed_type[j]);
            }
            else {
                str = gen_port_type_str(p_conn_cur->allowed_port_type[j]);
                rc |= fprintf(fp, "    \"%s\",\n", str);
            }
        }  /* for number of allowed types */

        rc |= fprintf(fp,
                "};\n"
                "\n");
    }  /* for number of connections */



    /* Now write the structure */
    rc |= fprintf(fp,
            "\n"
            "static Mif_Conn_Info_t MIFconnTable[] = {\n");


    /* Write out an entry for each parameter in the table               */

    for (i = 0; i < num_conn; i++) {
        const Conn_Info_t * const p_conn_cur = conn + i;

        rc |= fprintf(fp, "  {\n");
        rc |= fprintf(fp, "    \"%s\",\n", p_conn_cur->name);
        rc |= fprintf(fp, "    \"%s\",\n", p_conn_cur->description);

        str = dir_to_str(p_conn_cur->direction);
        rc |= fprintf(fp, "    %s,\n", str);

        rc |= fprintf(fp, "    %s,\n",
                port_type_to_str(p_conn_cur->default_port_type));

        rc |= fprintf(fp, "    \"%s\",\n",
                (p_conn_cur->default_port_type == USER_DEFINED)
                ? p_conn_cur->default_type
                : gen_port_type_str(p_conn_cur->default_port_type));

        rc |= fprintf(fp,"    %d,\n", p_conn_cur->num_allowed_types);
        rc |= fprintf(fp, "    MIFportEnum%d,\n", i);
        rc |= fprintf(fp, "    MIFportStr%d,\n", i);


        str = boolean_to_str(p_conn_cur->is_array);
        rc |= fprintf(fp, "    %s,\n", str);

        if (p_conn_cur->is_array == false) {

            str = boolean_to_str(false);    /* has_lower_bound */
            rc |= fprintf(fp, "    %s,\n", str);

            str = integer_to_str(0);        /* lower_bound */
            rc |= fprintf(fp, "    %s,\n", str);

            str = boolean_to_str(false);    /* has_upper_bound */
            rc |= fprintf(fp, "    %s,\n", str);

            str = integer_to_str(0);        /* upper_bound */
            rc |= fprintf(fp, "    %s,\n", str);
        }
        else {  /* is_array == true */

            str = boolean_to_str(p_conn_cur->has_lower_bound);
            rc |= fprintf(fp, "    %s,\n", str);

            if (p_conn_cur->has_lower_bound == true)
                str = integer_to_str(p_conn_cur->lower_bound);
            else
                str = integer_to_str(0);
            rc |= fprintf(fp, "    %s,\n", str);

            str = boolean_to_str(p_conn_cur->has_upper_bound);
            rc |= fprintf(fp, "    %s,\n", str);

            if (p_conn_cur->has_upper_bound == true)
                str = integer_to_str(p_conn_cur->upper_bound);
            else
                str = integer_to_str(0);
            rc |= fprintf(fp, "    %s,\n", str);

        }  /* if is_array */
    
        str = boolean_to_str(p_conn_cur->null_allowed);
        rc |= fprintf(fp, "    %s,\n", str);

        rc |= fprintf(fp, "  },\n");

    } /* for number of parameters */


    /* Finish off the structure */
    rc |= fprintf(fp,
            "};\n"
            "\n");

    /* Check outputs */
    if (rc < 0) {
        print_error("Writing of connection information failed.");
        xrc = -1;
    }

    return xrc;
} /* end of function write_conn_info */



/* *********************************************************************** */


/*
write_param_info

Function write_param_info writes information used by the
Simulator's new MIF package to interpret and error check a
model's parameter list in an XSPICE deck.  This information is
derived from the Interface Specification file's PARAMETER table. 
It is essentially a superset of the IFparm information written by
write_mPTable().  The IFparm information written by
write_mPTable() is required to work with SPICE's device set and
query functions.  The information written by write_param_info is
more extensive and is required to parse and error check the XSPICE
input deck.
*/



static int write_param_info(
    FILE        *fp,           /* File to write to */
    Ifs_Table_t *ifs_table)    /* Table of Interface Specification data */
{
    int xrc = 0;
    int             i;
    const char *str;


    /* Only write the paramTable if there is something to put in it.      */
    /* Otherwise, we will put NULL in the SPICEdev structure in its slot */

    if (ifs_table->num_param == 0) {
        return 0;
    }


    /* Write the structure beginning */
    int rc = 0;
    rc |= fprintf(fp,
            "\n"
            "static Mif_Param_Info_t MIFparamTable[] = {\n");


    /* Write out an entry for each parameter in the table               */
    const Param_Info_t * const param = ifs_table->param;
    const int num_param = ifs_table->num_param;
    for (i = 0; i < num_param; i++) {
        const Param_Info_t * const p_param_cur = param + i;

        rc |= fprintf(fp, "  {\n");
        rc |= fprintf(fp, "    \"%s\",\n", p_param_cur->name);
        rc |= fprintf(fp, "    \"%s\",\n", p_param_cur->description);

        rc |= fprintf(fp, "    %s,\n",
                data_type_to_str(p_param_cur->type));

        str = boolean_to_str(p_param_cur->has_default);
        rc |= fprintf(fp, "    %s,\n", str);

        if (p_param_cur->has_default == true)
            str = value_to_str(p_param_cur->type, p_param_cur->default_value);
        else
            str = no_value_to_str();
        rc |= fprintf(fp, "    %s,\n", str);

        str = boolean_to_str(p_param_cur->has_lower_limit);
        rc |= fprintf(fp, "    %s,\n", str);

        if (p_param_cur->has_lower_limit == true)
            str = value_to_str(p_param_cur->type, p_param_cur->lower_limit);
        else
            str = no_value_to_str();
        rc |= fprintf(fp, "    %s,\n", str);

        str = boolean_to_str(p_param_cur->has_upper_limit);
        rc |= fprintf(fp, "    %s,\n", str);

        if (p_param_cur->has_upper_limit == true)
            str = value_to_str(p_param_cur->type, p_param_cur->upper_limit);
        else
            str = no_value_to_str();
        rc |= fprintf(fp, "    %s,\n", str);

        str = boolean_to_str(p_param_cur->is_array);
        rc |= fprintf(fp, "    %s,\n", str);

        if (!p_param_cur->is_array) {

            str = boolean_to_str(false);    /* has_conn_ref */
            rc |= fprintf(fp, "    %s,\n", str);

            str = integer_to_str(0);        /* conn_ref */
            rc |= fprintf(fp, "    %s,\n", str);

            str = boolean_to_str(false);    /* has_lower_bound */
            rc |= fprintf(fp, "    %s,\n", str);

            str = integer_to_str(0);        /* lower_bound */
            rc |= fprintf(fp, "    %s,\n", str);

            str = boolean_to_str(false);    /* has_upper_bound */
            rc |= fprintf(fp, "    %s,\n", str);

            str = integer_to_str(0);        /* upper_bound */
            rc |= fprintf(fp, "    %s,\n", str);
        }
        else {  /* is_array == true */

            str = boolean_to_str(p_param_cur->has_conn_ref);
            rc |= fprintf(fp, "    %s,\n", str);

            if (p_param_cur->has_conn_ref) {

                str = integer_to_str(p_param_cur->conn_ref);
                rc |= fprintf(fp, "    %s,\n", str);

                str = boolean_to_str(false);    /* has_lower_bound */
                rc |= fprintf(fp, "    %s,\n", str);

                str = integer_to_str(0);        /* lower_bound */
                rc |= fprintf(fp, "    %s,\n", str);

                str = boolean_to_str(false);    /* has_upper_bound */
                rc |= fprintf(fp, "    %s,\n", str);

                str = integer_to_str(0);        /* upper_bound */
                rc |= fprintf(fp, "    %s,\n", str);
            }
            else {  /* has_conn_ref == false */

                str = integer_to_str(0);        /* conn_ref */
                rc |= fprintf(fp, "    %s,\n", str);

                str = boolean_to_str(p_param_cur->has_lower_bound);
                rc |= fprintf(fp, "    %s,\n", str);

                if (p_param_cur->has_lower_bound)
                    str = integer_to_str(p_param_cur->lower_bound);
                else
                    str = integer_to_str(0);
                rc |= fprintf(fp, "    %s,\n", str);

                str = boolean_to_str(p_param_cur->has_upper_bound);
                rc |= fprintf(fp, "    %s,\n", str);

                if (p_param_cur->has_upper_bound)
                    str = integer_to_str(p_param_cur->upper_bound);
                else
                    str = integer_to_str(0);
                rc |= fprintf(fp, "    %s,\n", str);

            }  /* if has_conn_ref */

        }  /* if is_array */
    
        str = boolean_to_str(p_param_cur->null_allowed);
        rc |= fprintf(fp, "    %s,\n", str);

        rc |= fprintf(fp, "  },\n");

    } /* for number of parameters */


    /* Finish off the structure */

    rc |= fprintf(fp, "};\n\n");

    /* Check outputs */
    if (rc < 0) {
        print_error("Writing of param information failed.");
        xrc = -1;
    }

    return xrc;
} /* end of function write_param_info */



/* *********************************************************************** */


/*
write_inst_var_info

Function write_inst_var_info writes information used by the
Simulator's new MIF package to allocate space for and to output
(using SPICE's .save feature) variables defined in the Interface
Specification file's STATIC_VAR table.  It is essentially a
superset of the IFparm information written by write_mPTable(). 
The IFparm information written by write_pTable() is required to
work with SPICE's device query functions.  The information
written by write_inst_var_info is more extensive.
*/



static int write_inst_var_info(
    FILE        *fp,         /* File to write to */
    Ifs_Table_t *ifs_table)  /* Table of Interface Specification data */
{
    int xrc = 0;
    int             i;
    const char *str;

    /* Only write the inst_varTable if there is something to put in it.  */
    /* Otherwise, we will put NULL in the SPICEdev structure in its slot */

    if (ifs_table->num_inst_var == 0)
        return 0;


    /* Write the structure beginning */
    int rc = 0;
    rc |= fprintf(fp,
            "\n"
            "static Mif_Inst_Var_Info_t MIFinst_varTable[] = {\n");


    /* Write out an entry for each parameter in the table               */

    for(i = 0; i < ifs_table->num_inst_var; i++) {

        rc |= fprintf(fp, "  {\n");
        rc |= fprintf(fp, "    \"%s\",\n",ifs_table->inst_var[i].name);
        rc |= fprintf(fp, "    \"%s\",\n",ifs_table->inst_var[i].description);

        rc |= fprintf(fp, "    %s,\n",
                data_type_to_str(ifs_table->inst_var[i].type));

        str = boolean_to_str(ifs_table->inst_var[i].is_array);
        rc |= fprintf(fp, "    %s,\n", str);

        rc |= fprintf(fp, "  },\n");

    } /* for number of parameters */


    /* Finish off the structure */

    rc |= fprintf(fp, "};\n\n");

    /* Check outputs */
    if (rc < 0) {
        print_error("Writing of instance variable information failed.");
        xrc = -1;
    }

    return xrc;
} /* end of function write_inst_var_info */



/* *********************************************************************** */


/*
write_SPICEdev

Function write_SPICEdev writes the global XXX_info structure used
by SPICE to define a model.  Here ``XXX'' is the name of the code
model.  This structure contains the name of the
model, a pointer to the C function that implements the model, and
pointers to all of the above data structures.
*/



static int write_SPICEdev(
    FILE        *fp,         /* File to write to */
    Ifs_Table_t *ifs_table)  /* Table of Interface Specification data */
{
    int rc = 0; /* init print rc to nonnegative (no error) */
    int xrc = 0;

    /* Extern the code model function name */
    rc |= fprintf(fp,
            "\n"
            "extern void %s(Mif_Private_t *);\n",
            ifs_table->name.c_fcn_name);

    /* SPICE now needs these static integers */
    rc |= fprintf(fp,
            "\n"
            "static int val_terms             = 0;\n"
            "static int val_numNames          = 0;\n"
            "static int val_numInstanceParms  = %d;\n"
            "static int val_numModelParms     = %d;\n"
            "static int val_sizeofMIFinstance = sizeof(MIFinstance);\n"
            "static int val_sizeofMIFmodel    = sizeof(MIFmodel);\n",
            ifs_table->num_inst_var, ifs_table->num_param);

    /* Write out the structure beginning */

    /* Use the c function external identifier appended with _info as the */
    /* external identifier for the structure.                            */

    rc |= fprintf(fp,
            "\n"
            "SPICEdev %s_info = {\n", ifs_table->name.c_fcn_name);

    /* Write the IFdevice structure */

    rc |= fprintf(fp, "    .DEVpublic = {\n");
    rc |= fprintf(fp, "        .name = \"%s\",\n", ifs_table->name.model_name);
    rc |= fprintf(fp, "        .description = \"%s\",\n", ifs_table->name.description);
    rc |= fprintf(fp,
            "        .terms = &val_terms,\n"
            "        .numNames = &val_numNames,\n"
            "        .termNames = NULL,\n"
            "        .numInstanceParms = &val_numInstanceParms,\n");

    if(ifs_table->num_inst_var > 0)
        rc |= fprintf(fp, "        .instanceParms = MIFpTable,\n");
    else
        rc |= fprintf(fp, "        .instanceParms = NULL,\n");

    rc |= fprintf(fp, "        .numModelParms = &val_numModelParms,\n");
    if(ifs_table->num_param > 0)
        rc |= fprintf(fp, "        .modelParms = MIFmPTable,\n");
    else
        rc |= fprintf(fp, "        .modelParms = NULL,\n");
    rc |= fprintf(fp, "        .flags = 0,\n\n");

    rc |= fprintf(fp, "        .cm_func = %s,\n", ifs_table->name.c_fcn_name);

    rc |= fprintf(fp, "        .num_conn = %d,\n", ifs_table->num_conn);
    if(ifs_table->num_conn > 0)
        rc |= fprintf(fp, "        .conn = MIFconnTable,\n");
    else
        rc |= fprintf(fp, "        .conn = NULL,\n");

    rc |= fprintf(fp, "        .num_param = %d,\n", ifs_table->num_param);
    if(ifs_table->num_param > 0)
        rc |= fprintf(fp, "        .param = MIFparamTable,\n");
    else
        rc |= fprintf(fp, "        .param = NULL,\n");

    rc |= fprintf(fp, "        .num_inst_var = %d,\n", ifs_table->num_inst_var);
    if(ifs_table->num_inst_var > 0)
        rc |= fprintf(fp, "        .inst_var = MIFinst_varTable,\n");
    else
        rc |= fprintf(fp, "        .inst_var = NULL,\n");

    rc |= fprintf(fp, "    },\n\n");

    /* Write the names of the generic code model functions */

    rc |= fprintf(fp,
            "    .DEVparam = NULL,\n"
            "    .DEVmodParam = MIFmParam,\n"
            "    .DEVload = MIFload,\n"
            "    .DEVsetup = MIFsetup,\n"
            "    .DEVunsetup = MIFunsetup,\n"
            "    .DEVpzSetup = NULL,\n"
            "    .DEVtemperature = NULL,\n"
            "    .DEVtrunc = MIFtrunc,\n"
            "    .DEVfindBranch = NULL,\n"
            "    .DEVacLoad = MIFload,\n"
            "    .DEVaccept = NULL,\n"
            "    .DEVdestroy = NULL,\n"
            "    .DEVmodDelete = MIFmDelete,\n"
            "    .DEVdelete = MIFdelete,\n"
            "    .DEVsetic = NULL,\n"
            "    .DEVask = MIFask,\n"
            "    .DEVmodAsk = MIFmAsk,\n"
            "    .DEVpzLoad = NULL,\n"
            "    .DEVconvTest = MIFconvTest,\n"
            "    .DEVsenSetup = NULL,\n"
            "    .DEVsenLoad = NULL,\n"
            "    .DEVsenUpdate = NULL,\n"
            "    .DEVsenAcLoad = NULL,\n"
            "    .DEVsenPrint = NULL,\n"
            "    .DEVsenTrunc = NULL,\n"
            "    .DEVdisto = NULL,\n"
            "    .DEVnoise = NULL,\n"
            "    .DEVsoaCheck = NULL,\n"
            "    .DEVinstSize = &val_sizeofMIFinstance,\n"
            "    .DEVmodSize = &val_sizeofMIFmodel,\n"
            "\n"
            "#ifdef CIDER\n"
            "    .DEVdump = NULL,\n"
            "    .DEVacct = NULL,\n"
            "#endif\n"
            "};\n\n"
            );

    /* Check outputs */
    if (rc < 0) {
        print_error("Writing of SPICE device information failed.");
        xrc = -1;
    }

    return xrc;
} /* end of function write_SPICEdev */



/* *********************************************************************** */


/*
The following functions are utility routines used to convert internal
enums and data to ASCII form for placing into the .c file
being created.
*/


#define BASE_STR_LEN  80


static const char *data_type_to_str(Data_Type_t type)
{
    switch (type) {
        case CMPP_BOOLEAN:
            return "MIF_BOOLEAN";
        case CMPP_INTEGER:
            return "MIF_INTEGER";
        case CMPP_REAL:
            return "MIF_REAL";
        case CMPP_COMPLEX:
            return "MIF_COMPLEX";
        case CMPP_STRING:
        case CMPP_POINTER:
            return "MIF_STRING";
        default:
            print_error("INTERNAL ERROR - data_type_to_str() - Impossible data type.");
            return "INVALID DATA TYPE";
    }
}


/* *********************************************************************** */

static const char *port_type_to_str(Port_Type_t port)
{
    switch (port) {
        case VOLTAGE:
            return "MIF_VOLTAGE";
        case DIFF_VOLTAGE:
            return "MIF_DIFF_VOLTAGE";
        case CURRENT:
            return "MIF_CURRENT";
        case DIFF_CURRENT:
            return "MIF_DIFF_CURRENT";
        case VSOURCE_CURRENT:
            return "MIF_VSOURCE_CURRENT";
        case CONDUCTANCE:
            return "MIF_CONDUCTANCE";
        case DIFF_CONDUCTANCE:
            return "MIF_DIFF_CONDUCTANCE";
        case RESISTANCE:
            return "MIF_RESISTANCE";
        case DIFF_RESISTANCE:
            return "MIF_DIFF_RESISTANCE";
        case DIGITAL:
            return "MIF_DIGITAL";
        case USER_DEFINED:
            return "MIF_USER_DEFINED";
        default:
            print_error("INTERNAL ERROR - port_type_to_str() - Impossible port type.");
            return "INVALID PORT TYPE";
    }
}

/* *********************************************************************** */

static const char *gen_port_type_str(Port_Type_t port)
{
    switch (port) {
        case VOLTAGE:
            return "v";
        case DIFF_VOLTAGE:
            return "vd";
        case CURRENT:
            return "i";
        case DIFF_CURRENT:
            return "id";
        case VSOURCE_CURRENT:
            return "vnam";
        case CONDUCTANCE:
            return "g";
        case DIFF_CONDUCTANCE:
            return "gd";
        case RESISTANCE:
            return "h";
        case DIFF_RESISTANCE:
            return "hd";
        case DIGITAL:
            return "d";
        case USER_DEFINED:
            return "";
        default:
            print_error("INTERNAL ERROR - gen_port_type_str() - "
                    "Impossible port type.");
            return "INVALID PORT TYPE";
    }
} /* end of function gen_port_type_str */



/* *********************************************************************** */

static const char *dir_to_str(Dir_t dir)
{
    switch(dir) {
        case CMPP_IN:
            return "MIF_IN";
        case CMPP_OUT:
            return "MIF_OUT";
        case CMPP_INOUT:
            return "MIF_INOUT";
        default:
            print_error("INTERNAL ERROR - dir_to_str() - Impossible direction type.");
            return "MIF_DIRECTION_INVALID";
    }
}



/* *********************************************************************** */

static char  *value_to_str(Data_Type_t type, Value_t value)
{
    static char *str = NULL;
    static int  max_len = 0;

    const char *bool_str;
    int  str_len;


    if(str == NULL) {
        if ((str = (char *) malloc(2 * BASE_STR_LEN + 1)) == (char *) NULL) {
            (void) fprintf(stderr, "Unable to allocate string buffer.\n");
            return (char *) NULL;
        }
        max_len = 2 * BASE_STR_LEN;
    }

    switch(type) {

        case CMPP_BOOLEAN:
            bool_str = boolean_to_str(value.bvalue);
            sprintf(str, "{%s, 0, 0.0, {0.0, 0.0}, NULL}", bool_str);
            break;

        case CMPP_INTEGER:
            sprintf(str, "{MIF_FALSE, %d, 0.0, {0.0, 0.0}, NULL}", value.ivalue);
            break;

        case CMPP_REAL:
            sprintf(str, "{MIF_FALSE, 0, %e, {0.0, 0.0}, NULL}", value.rvalue);
            break;

        case CMPP_COMPLEX:
            sprintf(str, "{MIF_FALSE, 0, 0.0, {%e, %e}, NULL}",
                                      value.cvalue.real, value.cvalue.imag);
            break;

        case CMPP_STRING:
            /* be careful, the string could conceivably be very long... */
            str_len = (int) strlen(value.svalue);
            if ((str_len + BASE_STR_LEN) > max_len) {
                int n_byte_alloc = max_len + str_len + 1;
                void * const p = realloc(str, (size_t)n_byte_alloc);
                if (p == NULL) {
                    (void) fprintf(stderr,
                            "Unable to resize string buffer to size %d.\n",
                            n_byte_alloc);
                    free(str);
                    return (char *) NULL;
                }
                str = (char *) p;
                max_len += str_len;
            } /* end of resize */

            sprintf(str, "{MIF_FALSE, 0, 0.0, {0.0, 0.0}, \"%s\"}", value.svalue);
            break;

        default:
            print_error("INTERNAL ERROR - value_to_str() - Impossible data type.");

    } /* end of switch */

    return str;
} /* end of function value_to_string */



/* *********************************************************************** */

static const char *boolean_to_str(bool value)
{
    return value ? "MIF_TRUE" : "MIF_FALSE";
}


/* *********************************************************************** */

static char  *integer_to_str(int value)
{
    static char str[3 * sizeof(int) + 1];
    sprintf(str, "%d", value);
    return str;
}


/* *********************************************************************** */

static const char *no_value_to_str(void)
{
    return "{MIF_FALSE, 0, 0.0, {0.0, 0.0}, NULL}";
}



