/*============================================================================
FILE  pp_lst.c

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

    This file contains functions used in processing the files:

        modpath.lst
        udnpath.lst

    The files 'modpath.lst' and 'udnpath.lst' are read to get
    the pathnames to directories containing the models and node types
    desired in the simulator to be built.  Files in each of these
    directories are then examined to get the names of functions and/or
    data structures that the simulator must know about.  The names
    are then checked for uniqueness, and finally, a collection of
    files needed in the 'make' for the simulator are written.

INTERFACES

    preprocess_lst_files()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include  "cmpp.h"
#include  <ctype.h>
#include  <stdlib.h>
#include  <string.h>


/*
void    *malloc(unsigned size);
void    *realloc(void *ptr, unsigned size);
*/

/* *********************************************************************** */

/*
 * Information for processing the pathname files
 */

typedef struct {
    char        *path_name;     /* Pathname read from model path file */
    char        *spice_name;    /* Name of model from ifspec.ifs  */
    char        *cfunc_name;    /* Name of C fcn from ifspec.ifs  */
    Boolean_t   spice_unique;   /* True if spice name unique */
    Boolean_t   cfunc_unique;   /* True if C fcn name unique */
} Model_Info_t;


typedef struct {
    char        *path_name;     /* Pathname read from udn path file */
    char        *node_name;     /* Name of node type  */
    Boolean_t   unique;         /* True if node type name unique */
} Node_Info_t;



/* *********************************************************************** */


static Status_t read_modpath(int *num_models, Model_Info_t **model_info);
static Status_t read_udnpath(int *num_nodes, Node_Info_t **node_info);
static Status_t read_model_names(int num_models, Model_Info_t *model_info);
static Status_t read_node_names(int num_nodes, Node_Info_t *node_info);
static Status_t check_uniqueness(int num_models, Model_Info_t *model_info,
                                 int num_nodes, Node_Info_t *node_info);
static Status_t write_CMextrn(int num_models, Model_Info_t *model_info);
static Status_t write_CMinfo(int num_models, Model_Info_t *model_info);
static Status_t write_UDNextrn(int num_nodes, Node_Info_t *node_info);
static Status_t write_UDNinfo(int num_nodes, Node_Info_t *node_info);
static Status_t write_objects_inc(int num_models, Model_Info_t *model_info,
                                  int num_nodes, Node_Info_t *node_info);
static Status_t read_udn_type_name(const char *path, char **node_name);


/* *********************************************************************** */


/*
preprocess_lst_files

Function preprocess_lst_files is the top-level driver function for
preprocessing a simulator model path list file (modpath.lst). 
This function calls read_ifs_file() requesting it to read and
parse the Interface Specification file (ifspec.ifs) to extract
the model name and associated C function name and place this
information into an internal data structure.  It then calls
check_uniqueness() to verify that the model names and function
names are unique with respect to each other and to the models and
functions internal to SPICE 3C1.  Following this check, it calls
write_CMextrn(), write_CMinfo(), and write_make_include() to
write out the C include files CMextrn.h and CMinfo.h required by
SPICE function SPIinit.c to define the models known to the
simulator, and to write out a make include file used by the make
utility to locate the object modules necessary to link the models
into the simulator.
*/

void preprocess_lst_files(void)
{
    Status_t        status;      /* Return status */
    Model_Info_t    *model_info; /* Info about each model */
    Node_Info_t     *node_info;  /* Info about each user-defined node type */
    int             num_models;  /* The number of models */
    int             num_nodes;   /* The number of user-defined nodes */


    /* Get list of models from model pathname file */
    status = read_modpath(&num_models, &model_info);
    if(status != OK) {
        exit(1);
    }

    /* Get list of node types from udn pathname file */
    status = read_udnpath(&num_nodes, &node_info);
    if(status != OK) {
        exit(1);
    }


    /* Get the spice and C function names from the ifspec.ifs files */
    status = read_model_names(num_models, model_info);
    if(status != OK) {
        exit(1);
    }

    /* Get the user-defined node type names */
    status = read_node_names(num_nodes, node_info);
    if(status != OK) {
        exit(1);
    }


    /* Check to be sure the names are unique */
    status = check_uniqueness(num_models, model_info,
                              num_nodes, node_info);
    if(status != OK) {
        exit(1);
    }


    /* Write out the CMextrn.h file used to compile SPIinit.c */
    status = write_CMextrn(num_models, model_info);
    if(status != OK) {
        exit(1);
    }

    /* Write out the CMinfo.h file used to compile SPIinit.c */
    status = write_CMinfo(num_models, model_info);
    if(status != OK) {
        exit(1);
    }


    /* Write out the UDNextrn.h file used to compile SPIinit.c */
    status = write_UDNextrn(num_nodes, node_info);
    if(status != OK) {
        exit(1);
    }

    /* Write out the UDNinfo.h file used to compile SPIinit.c */
    status = write_UDNinfo(num_nodes, node_info);
    if(status != OK) {
        exit(1);
    }


    /* Write the make_include file used to link the models and */
    /* user-defined node functions with the simulator */
    status = write_objects_inc(num_models, model_info,
                              num_nodes, node_info);
    if(status != OK) {
        exit(1);
    }

}



/* *********************************************************************** */


/*
read_modpath

This function opens the modpath.lst file, reads the pathnames from the
file, and puts them into an internal data structure for future
processing.
*/


static Status_t read_modpath(
    int            *num_models,       /* Number of model pathnames found */
    Model_Info_t   **model_info       /* Info about each model */
)
{
    FILE     *fp;                     /* Model pathname file pointer */
    char     path[MAX_PATH_LEN+2];    /* space to read pathnames into */
    Model_Info_t  *model = NULL;      /* temporary pointer to model info */

    int     n;
    int     i;
    int     j;
    int     len;
    int     line_num;

    const char *filename = MODPATH_FILENAME;


    /* Initialize number of models to zero in case of error */
    *num_models = 0;

    /* Open the model pathname file */
    fp = fopen_cmpp(&filename, "r");

    if(fp == NULL) {
        print_error("ERROR - File not found: %s", filename);
        return(ERROR);
    }

    /* Read the pathnames from the file, one line at a time until EOF */
    n = 0;
    line_num = 0;
    while( fgets(path, sizeof(path), fp) ) {

        line_num++;
        len = (int) strlen(path);

        /* If line was too long for buffer, exit with error */
        if(len > MAX_PATH_LEN) {
            print_error("ERROR - Line %d of %s exceeds %d characters",
                        line_num, filename, MAX_PATH_LEN);
            return(ERROR);
        }

        /* Strip white space including newline */
        for(i = 0, j = 0; i < len; ) {
            if(isspace(path[i])) {
                i++;
            }
            else {
                path[j] = path[i];
                i++;
                j++;
            }
        }
        path[j] = '\0';
        len = j;

        /* Strip trailing '/' if any */
        if(path[len - 1] == '/')
            path[--len] = '\0';

        /* If blank line, continue */
        if(len == 0)
            continue;

        /* Make sure pathname is short enough to add a filename at the end */
        if(len > (MAX_PATH_LEN - (MAX_FN_LEN + 1)) ) {
            print_error("ERROR - Pathname on line %d of %s exceeds %d characters",
                        line_num, filename, (MAX_PATH_LEN - (MAX_FN_LEN + 1)));
            return(ERROR);
        }

        /* Allocate and initialize a new model info structure */
        if(n == 0)
            model = (Model_Info_t *) malloc(sizeof(Model_Info_t));
        else
            model = (Model_Info_t *) realloc(model, (size_t) (n + 1) * sizeof(Model_Info_t));
        model[n].path_name = NULL;
        model[n].spice_name = NULL;
        model[n].cfunc_name = NULL;
        model[n].spice_unique = TRUE;
        model[n].cfunc_unique = TRUE;

        /* Put pathname into info structure */
        model[n].path_name = (char *) malloc((size_t) (len+1));
        strcpy(model[n].path_name, path);

        /* Increment count of paths read */
        n++;
    }


    /* Close model pathname file and return data read */
    fclose(fp);

    *num_models = n;
    *model_info = model;

    return(OK);

}


/* *********************************************************************** */


/*
read_udnpath

This function opens the udnpath.lst file, reads the pathnames from the
file, and puts them into an internal data structure for future
processing.
*/


static Status_t read_udnpath(
    int            *num_nodes,       /* Number of node pathnames found */
    Node_Info_t    **node_info       /* Info about each node */
)
{
    FILE     *fp;                     /* Udn pathname file pointer */
    char     path[MAX_PATH_LEN+2];    /* space to read pathnames into */
    Node_Info_t  *node = NULL;        /* temporary pointer to node info */

    int     n;
    int     i;
    int     j;
    int     len;
    int     line_num;

    const char *filename = UDNPATH_FILENAME;


    /* Initialize number of nodes to zero in case of error */
    *num_nodes = 0;

    /* Open the node pathname file */
    fp = fopen_cmpp(&filename, "r");

    /* For backward compatibility, return with WARNING only if file not found */
    if(fp == NULL) {
        print_error("WARNING - File not found: %s", filename);
        return(OK);
    }

    /* Read the pathnames from the file, one line at a time until EOF */
    n = 0;
    line_num = 0;
    while( fgets(path, sizeof(path), fp) ) {

        line_num++;
        len = (int) strlen(path);

        /* If line was too long for buffer, exit with error */
        if(len > MAX_PATH_LEN) {
            print_error("ERROR - Line %d of %s exceeds %d characters",
                        line_num, filename, MAX_PATH_LEN);
            return(ERROR);
        }

        /* Strip white space including newline */
        for(i = 0, j = 0; i < len; ) {
            if(isspace(path[i])) {
                i++;
            }
            else {
                path[j] = path[i];
                i++;
                j++;
            }
        }
        path[j] = '\0';
        len = j;

        /* Strip trailing '/' if any */
        if(path[len - 1] == '/')
            path[--len] = '\0';

        /* If blank line, continue */
        if(len == 0)
            continue;

        /* Make sure pathname is short enough to add a filename at the end */
        if(len > (MAX_PATH_LEN - (MAX_FN_LEN + 1)) ) {
            print_error("ERROR - Pathname on line %d of %s exceeds %d characters",
                        line_num, filename, (MAX_PATH_LEN - (MAX_FN_LEN + 1)));
            return(ERROR);
        }

        /* Allocate and initialize a new node info structure */
        if(n == 0)
            node = (Node_Info_t *) malloc(sizeof(Node_Info_t));
        else
            node = (Node_Info_t *) realloc(node, (size_t) (n + 1) * sizeof(Node_Info_t));
        node[n].path_name = NULL;
        node[n].node_name = NULL;
        node[n].unique = TRUE;

        /* Put pathname into info structure */
        node[n].path_name = (char *) malloc((size_t) (len+1));
        strcpy(node[n].path_name, path);

        /* Increment count of paths read */
        n++;
    }


    /* Close node pathname file and return data read */
    fclose(fp);

    *num_nodes = n;
    *node_info = node;

    return(OK);

}



/* *********************************************************************** */


/*
read_model_names

This function opens each of the models and gets the names of the model
and the C function into the internal model information structure.
*/

static Status_t read_model_names(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info  /* Info about each model */
)
{
    Boolean_t   all_found;               /* True if all ifspec files read */
    int         i;                       /* A temporary counter */
    char        path[MAX_PATH_LEN+1];    /* full pathname to ifspec file */
    Status_t    status;                  /* Return status */
    Ifs_Table_t ifs_table;   /* Repository for info read from ifspec file */


    /* For each model found in model pathname file, read the interface */
    /* spec file to get the SPICE and C function names into model_info */

    for(i = 0, all_found = TRUE; i < num_models; i++) {

        /* Form the full pathname to the interface spec file */
        strcpy(path, model_info[i].path_name);
        strcat(path, "/");
        strcat(path, IFSPEC_FILENAME);

        /* Read the SPICE and C function names from the interface spec file */
        status = read_ifs_file(path, GET_IFS_NAME, &ifs_table);

        /* Transfer the names into the model_info structure */
        if(status == OK) {
            model_info[i].spice_name = ifs_table.name.model_name;
            model_info[i].cfunc_name = ifs_table.name.c_fcn_name;
        }
        else {
            all_found = FALSE;
            print_error("ERROR - Problems reading %s in directory %s",
                        IFSPEC_FILENAME, model_info[i].path_name);
        }
    }

    if(all_found)
        return(OK);
    else
        return(ERROR);

}


/* *********************************************************************** */


/*
read_node_names

This function opens each of the user-defined node definition files
and gets the names of the node into the internal information structure.
*/


static Status_t read_node_names(
    int           num_nodes,  /* Number of node pathnames */
    Node_Info_t   *node_info  /* Info about each node */
)
{
    Boolean_t   all_found;               /* True if all files read OK */
    int         i;                       /* A temporary counter */
    char        path[MAX_PATH_LEN+1];    /* full pathname to file */
    Status_t    status;                  /* Return status */
    char        *node_name;              /* Name of node type read from file */

    /* For each node found in node pathname file, read the udnfunc.c */
    /* file to get the node type names into node_info */

    for(i = 0, all_found = TRUE; i < num_nodes; i++) {

        /* Form the full pathname to the interface spec file */
        strcpy(path, node_info[i].path_name);
        strcat(path, "/");
        strcat(path, UDNFUNC_FILENAME);

        /* Read the udn node type name from the file */
        status = read_udn_type_name(path, &node_name);

        /* Transfer the names into the node_info structure */
        if(status == OK) {
            node_info[i].node_name = node_name;
        }
        else {
            all_found = FALSE;
            print_error("ERROR - Problems reading %s in directory %s",
                        UDNFUNC_FILENAME, node_info[i].path_name);
        }
    }

    if(all_found)
        return(OK);
    else
        return(ERROR);

}



/* *********************************************************************** */


/*
check_uniqueness

Function check_uniqueness determines if model names and function
names are unique with respect to each other and to the models and
functions internal to SPICE 3C1.
*/

static Status_t check_uniqueness(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info, /* Info about each model */
    int            num_nodes,   /* Number of node type pathnames */
    Node_Info_t    *node_info   /* Info about each node type */
)
{
    int         i;                       /* A temporary counter */
    int         j;                       /* A temporary counter */
    Boolean_t   all_unique;              /* true if names are unique */

    /* Define a list of model names used internally by XSPICE */
    /* These names (except 'poly') are defined in src/sim/INP/INPdomodel.c and */
    /* are case insensitive */
    static char *SPICEmodel[] = { "npn",
                                  "pnp",
                                  "d",
                                  "njf",
                                  "pjf",
                                  "nmf",
                                  "pmf",
                                  "urc",
                                  "nmos",
                                  "pmos",
                                  "r",
                                  "c",
                                  "sw",
                                  "csw",
                                  "poly" };
    static int  numSPICEmodels = sizeof(SPICEmodel) / sizeof(char *);


    /* Define a list of node type names used internally by the simulator */
    /* These names are defined in src/sim/include/MIFtypes.h and are case */
    /* insensitive */
    static char *UDNidentifier[] = { "v",
                                     "vd",
                                     "i",
                                     "id",
                                     "vnam",
                                     "g",
                                     "gd",
                                     "h",
                                     "hd",
                                     "d" };
    static int  numUDNidentifiers = sizeof(UDNidentifier) / sizeof(char *);



    /* First, normalize case of all model and node names to lower since */
    /* SPICE is case insensitive in parsing decks */
    for(i = 0; i < num_models; i++)
        str_to_lower(model_info[i].spice_name);
    for(i = 0; i < num_nodes; i++)
        str_to_lower(node_info[i].node_name);

    /* Then, loop through all model names and report errors if same as SPICE */
    /* model name or same as another model name in list */
    for(i = 0, all_unique = TRUE; i < num_models; i++) {

        /* First check against list of SPICE internal names */
        for(j = 0; j < numSPICEmodels; j++) {
            if(strcmp(model_info[i].spice_name, SPICEmodel[j]) == 0) {
                all_unique = FALSE;
                print_error("ERROR - Model name '%s' is same as internal SPICE model name\n",
                            model_info[i].spice_name);
            }
        }

        /* Skip if already seen once */
        if(model_info[i].spice_unique == FALSE)
            continue;

        /* Then check against other names in list */
        for(j = 0; j < num_models; j++) {

            /* Skip checking against itself */
            if(i == j)
                continue;

            /* Compare the names */
            if(strcmp(model_info[i].spice_name, model_info[j].spice_name) == 0) {
                all_unique = FALSE;
                model_info[i].spice_unique = FALSE;
                model_info[j].spice_unique = FALSE;
                print_error("ERROR - Model name '%s' in directory: %s",
                            model_info[i].spice_name,
                            model_info[i].path_name);
                print_error("        is same as");
                print_error("        model name '%s' in directory: %s\n",
                            model_info[j].spice_name,
                            model_info[j].path_name);
            }
        }

    }

    /* Loop through all C function names and report errors if duplicates found */
    for(i = 0; i < num_models; i++) {

        /* Skip if already seen once */
        if(model_info[i].cfunc_unique == FALSE)
            continue;

        /* Check against other names in list only, not against SPICE identifiers */
        /* Let linker catch collisions with SPICE identifiers for now */
        for(j = 0; j < num_models; j++) {

            /* Skip checking against itself */
            if(i == j)
                continue;

            /* Compare the names */
            if(strcmp(model_info[i].cfunc_name, model_info[j].cfunc_name) == 0) {
                all_unique = FALSE;
                model_info[i].cfunc_unique = FALSE;
                model_info[j].cfunc_unique = FALSE;
                print_error("ERROR - C function name '%s' in directory: %s",
                            model_info[i].cfunc_name,
                            model_info[i].path_name);
                print_error("        is same as");
                print_error("        C function name '%s' in directory: %s\n",
                            model_info[j].cfunc_name,
                            model_info[j].path_name);
            }
        }

    }

    /* Loop through all node type names and report errors if same as internal */
    /* name or same as another name in list */
    for(i = 0; i < num_nodes; i++) {

        /* First check against list of internal names */
        for(j = 0; j < numUDNidentifiers; j++) {
            if(strcmp(node_info[i].node_name, UDNidentifier[j]) == 0) {
                all_unique = FALSE;
                print_error("ERROR - Node type '%s' is same as internal node type\n",
                            node_info[i].node_name);
            }
        }

        /* Skip if already seen once */
        if(node_info[i].unique == FALSE)
            continue;

        /* Then check against other names in list */
        for(j = 0; j < num_nodes; j++) {

            /* Skip checking against itself */
            if(i == j)
                continue;

            /* Compare the names */
            if(strcmp(node_info[i].node_name, node_info[j].node_name) == 0) {
                all_unique = FALSE;
                node_info[i].unique = FALSE;
                node_info[j].unique = FALSE;
                print_error("ERROR - Node type '%s' in directory: %s",
                            node_info[i].node_name,
                            node_info[i].path_name);
                print_error("        is same as");
                print_error("        node type '%s' in directory: %s\n",
                            node_info[j].node_name,
                            node_info[j].path_name);
            }
        }
    }


    /* Return error status */
    if(all_unique)
        return(OK);
    else
        return(ERROR);

}


/* *********************************************************************** */


/*
write_CMextrn

Function write_CMextrn writes the CMextrn.h file used in
compiling SPIinit.c immediately prior to linking the simulator
and code models.  This SPICE source file uses the structures
mentioned in the include file to define the models known to the
simulator.
*/

static Status_t write_CMextrn(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info  /* Info about each model */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing CMextrn.h */

    const char *filename = "cmextrn.h";

    /* Open the file to be written */
    fp = fopen_cmpp(&filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        return(ERROR);
    }

    /* Write out the data */
    for(i = 0; i < num_models; i++) {
        fprintf(fp, "extern  SPICEdev  %s_info;\n", model_info[i].cfunc_name);
    }

    /* Close the file and return */
    fclose(fp);
    return(OK);
}


/* *********************************************************************** */


/*
write_CMinfo

Function write_CMinfo writes the CMinfo.h file used in compiling
SPIinit.c immediately prior to linking the simulator and code
models.  This SPICE source file uses the structures mentioned in
the include file to define the models known to the simulator.
*/


static Status_t write_CMinfo(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info  /* Info about each model */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing CMinfo.h */

    const char *filename = "cminfo.h";

    /* Open the file to be written */
    fp = fopen_cmpp(&filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        return(ERROR);
    }

    /* Write out the data */
    for(i = 0; i < num_models; i++) {
        fprintf(fp, "&%s_info,\n", model_info[i].cfunc_name);
    }

    /* Close the file and return */
    fclose(fp);
    return(OK);
}



/* *********************************************************************** */



/*
write_UDNextrn

Function write_UDNextrn writes the UDNextrn.h file used in
compiling SPIinit.c immediately prior to linking the simulator
and user-defined nodes.  This SPICE source file uses the structures
mentioned in the include file to define the node types known to the
simulator.
*/



static Status_t write_UDNextrn(
    int            num_nodes,  /* Number of node pathnames */
    Node_Info_t    *node_info  /* Info about each node */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing UDNextrn.h */

    const char *filename = "udnextrn.h";

    /* Open the file to be written */
    fp = fopen_cmpp(&filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        return(ERROR);
    }

    /* Write out the data */
    for(i = 0; i < num_nodes; i++) {
        fprintf(fp, "extern Evt_Udn_Info_t  udn_%s_info;\n", node_info[i].node_name);
    }

    /* Close the file and return */
    fclose(fp);
    return(OK);
}


/* *********************************************************************** */


/*
write_UDNinfo

Function write_UDNinfo writes the UDNinfo.h file used in
compiling SPIinit.c immediately prior to linking the simulator
and user-defined nodes.  This SPICE source file uses the structures
mentioned in the include file to define the node types known to the
simulator.
*/



static Status_t write_UDNinfo(
    int            num_nodes,  /* Number of node pathnames */
    Node_Info_t    *node_info  /* Info about each node */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing UDNinfo.h */

    const char *filename = "udninfo.h";

    /* Open the file to be written */
    fp = fopen_cmpp(&filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        return(ERROR);
    }

    /* Write out the data */
    for(i = 0; i < num_nodes; i++) {
        fprintf(fp, "&udn_%s_info,\n", node_info[i].node_name);
    }

    /* Close the file and return */
    fclose(fp);
    return(OK);
}


/* *********************************************************************** */

/*
write_objects_inc

Function write_objects_inc writes a make include file used by
the make utility to locate the object modules needed to link the
simulator with the code models and user-defined node types.
*/

static Status_t write_objects_inc(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info, /* Info about each model */
    int            num_nodes,   /* Number of udn pathnames */
    Node_Info_t    *node_info   /* Info about each node type */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing make_include */

    const char *filename = "objects.inc";

    /* Open the file to be written */
    fp = fopen_cmpp(&filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        return(ERROR);
    }

    /* Write out the data */
    for(i = 0; i < num_models; i++) {
        fprintf(fp, "%s/*.o", model_info[i].path_name);
        if((i < (num_models - 1)) || (num_nodes > 0))
            fprintf(fp, " \\\n");
        else
            fprintf(fp, "\n");
    }
    for(i = 0; i < num_nodes; i++) {
        fprintf(fp, "%s/*.o", node_info[i].path_name);
        if(i < (num_nodes - 1))
            fprintf(fp, " \\\n");
        else
            fprintf(fp, "\n");
    }

    /* Close the file and return */
    fclose(fp);
    return(OK);
}


/*
read_udn_type_name

This function reads a User-Defined Node Definition File until
the definition of the Evt_Udn_Info_t struct
is found, and then gets the name of the node type from the first
member of the structure.
*/

static Status_t read_udn_type_name(
    const char *path,           /* the path to the node definition file */
    char **node_name            /* the node type name found in the file */
)
{
    FILE        *fp;                    /* file pointer for opened file */
    Boolean_t   found;                  /* true if name found successfully */
    Boolean_t   in_struct;              /* true if found struct with name */
    char        name[MAX_NAME_LEN + 1]; /* temporary storage for name read */
    int         c;                      /* a character read from the file */
    int         i;                      /* a counter */

    static char *struct_type = "Evt_Udn_Info_t";

    /* Open the file from which the node type name will be read */
    fp = fopen_cmpp(&path, "r");
    if(fp == NULL)
        return(ERROR);

    /* Read the file until the definition of the Evt_Udn_Info_t struct */
    /* is found, then get the name of the node type from the first */
    /* member of the structure */
    found = FALSE;
    do {
        /* read the next character */
        c = fgetc(fp);

        /* check for and gobble up comments */
        if(c == '/') {
            c = fgetc(fp);
            if(c == '*') {
                do {
                    c = fgetc(fp);
                    if(c == '*') {
                        c = fgetc(fp);
                        if((c == '/') || (c == EOF))
                            break;
                        else
                            ungetc(c, fp);
                    }
                } while(c != EOF);
            }
        }
        if(c == EOF)
            break;

        /* read until "Evt_Udn_Info_t" is encountered */
        for(i = 0, in_struct = FALSE; ; i++) {
            if(c != struct_type[i])
                break;
            else if(i == (sizeof(struct_type) - 2)) {
                in_struct = TRUE;
                break;
            }
            else
                c = fgetc(fp);
        }
        /* if found it, read until open quote found */
        /* and then read the name until the closing quote is found */
        if(in_struct) {
            do {
                c = fgetc(fp);
                if(c == '"') {
                    i = 0;
                    do {
                        c = fgetc(fp);
                        if(c == '"') {
                            found = TRUE;
                            name[i] = '\0';
                        }
                        else if(c != EOF)
                            name[i++] = (char) c;
                    } while((c != EOF) && (! found));
                }
            } while((c != EOF) && (! found));
        }

    } while((c != EOF) && (! found));

    /* Close the file and return */
    fclose(fp);

    if(found) {
        *node_name = (char *) malloc(strlen(name) + 1);
        strcpy(*node_name, name);
        return(OK);
    }
    else {
        *node_name = NULL;
        return(ERROR);
    }
}

