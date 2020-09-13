/*============================================================================
FILE  pp_ifs.c

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

    This file contains the main function for processing an Interface Spec
    File (ifspec.ifs).

INTERFACES

    preprocess_ifs_file()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include <stdlib.h>
#include  "cmpp.h"



/*
preprocess_ifs_file

Function preprocess_ifs_file is the top-level driver function for
preprocessing an Interface Specification file (ifspec.ifs).  This
function calls read_ifs_file() requesting it to read and parse
the Interface Specification file and place the information
contained in it into an internal data structure.  Then
write_ifs_c_file() is called to write the information out in a C
file that will be compiled and linked with the simulator.
*/


void preprocess_ifs_file(void)
{

    Ifs_Table_t     ifs_table;   /* Repository for info read from ifspec.ifs file */

    int status;      /* Return status */

    /* Read the entire ifspec.ifs file and load the data into ifs_table */

    status = read_ifs_file(IFSPEC_FILENAME,GET_IFS_TABLE,&ifs_table);

    if(status != 0) {
        exit(1);
    }


    /* Write the ifspec.c file required by the spice simulator */

    status = write_ifs_c_file("ifspec.c",&ifs_table);

    if(status != 0) {
        exit(1);
    }
    rem_ifs_table(&ifs_table);
}

void rem_ifs_table(Ifs_Table_t *ifs_table)
{
    /* Remove the ifs_table */
    free(ifs_table->name.c_fcn_name);
    free(ifs_table->name.description);
    free(ifs_table->name.model_name);

    {
        Conn_Info_t * const p_conn = ifs_table->conn;
        const int num_conn = ifs_table->num_conn;
        int i;
        for (i = 0; i < num_conn; ++i) {
            Conn_Info_t *p_conn_cur = p_conn + i;
            free(p_conn_cur->name);
            free(p_conn_cur->description);
            free(p_conn_cur->default_type);
        }
    }

    {
        Param_Info_t * const p_param = ifs_table->param;
        const int num_param = ifs_table->num_param;
        int i;
        for (i = 0; i < num_param; ++i) {
            Param_Info_t *p_param_cur = p_param + i;
            free(p_param_cur->name);
            free(p_param_cur->description);
        }
    }

    {
        Inst_Var_Info_t * const p_inst_var = ifs_table->inst_var;
        const int num_inst_var = ifs_table->num_inst_var;
        int i;
        for(i = 0; i < num_inst_var; ++i) {
            Inst_Var_Info_t *p_inst_var_cur = p_inst_var + i;
            free(p_inst_var_cur->name);
            free(p_inst_var_cur->description);
        }
    }

    free(ifs_table->conn);
    free(ifs_table->param);
    free(ifs_table->inst_var);
} /* end of function rem_ifs_table */





