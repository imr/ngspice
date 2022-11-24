/*============================================================================
FILE  pp_lst.c

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

#include  <ctype.h>
#include  <errno.h>
#include <limits.h>
#include  <stdbool.h>
#include  <stdlib.h>
#include  <string.h>

#include "cmpp.h"
#include "file_buffer.h"

/* *********************************************************************** */

/*
 * Information for processing the pathname files
 */

typedef struct {
    char        *path_name;     /* Pathname read from model path file */
    char        *spice_name;    /* Name of model from ifspec.ifs  */
    char        *cfunc_name;    /* Name of C fcn from ifspec.ifs  */
    unsigned int version; /* version of the code model */
} Model_Info_t;


typedef struct {
    char        *path_name;     /* Pathname read from udn path file */
    char        *node_name;     /* Name of node type  */
    unsigned int version; /* version of the code model */
} Node_Info_t;


/* Structure for uniqueness testing */
struct Key_src {
    const char *key; /* value to compare */
    const char *src; /* source of value */
};





/************************************************************************ */

static int cmpr_ks(const struct Key_src *ks1, const struct Key_src *ks2);
static void free_model_info(int num_models, Model_Info_t *p_model_info);
static void free_node_info(int num_nodes, Node_Info_t *p_node_info);
static int read_modpath(int *num_models, Model_Info_t **model_info);
static int read_udnpath(int *num_nodes, Node_Info_t **node_info);
static int read_model_names(int num_models, Model_Info_t *model_info);
static int read_node_names(int num_nodes, Node_Info_t *node_info);
static int check_uniqueness(int num_models, Model_Info_t *model_info,
        int num_nodes, Node_Info_t *node_info);
static void report_error_spice_name(const struct Key_src *p_ks1,
        const struct Key_src *p_ks2);
static void report_error_function_name (const struct Key_src *p_ks1,
        const struct Key_src *p_ks2);
static void report_error_udn_name(const struct Key_src *p_ks1,
        const struct Key_src *p_ks2);
static bool test_for_duplicates(unsigned int n, struct Key_src *p_ks,
        void (*p_error_reporter)(const struct Key_src *p_ks1,
                const struct Key_src *p_ks2));
static inline void trim_slash(size_t *p_n, char *p);
static int write_CMextrn(int num_models, Model_Info_t *model_info);
static int write_CMinfo(int num_models, Model_Info_t *model_info);
static int write_CMinfo2(int num_models, Model_Info_t *model_info);
static int write_UDNextrn(int num_nodes, Node_Info_t *node_info);
static int write_UDNinfo(int num_nodes, Node_Info_t *node_info);
static int write_UDNinfo2(int num_nodes, Node_Info_t *node_info);
static int write_objects_inc(int num_models, Model_Info_t *model_info,
        int num_nodes, Node_Info_t *node_info);
static int read_udn_type_name(const char *path, char **node_name);


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
    int        status;      /* Return status */
    Model_Info_t    *model_info; /* Info about each model */
    Node_Info_t     *node_info;  /* Info about each user-defined node type */
    int             num_models;  /* The number of models */
    int             num_nodes;   /* The number of user-defined nodes */

    /* Get list of models from model pathname file */
    status = read_modpath(&num_models, &model_info);
    if (status != 0) {
        exit(1);
    }

    /* Get list of node types from udn pathname file */
    status = read_udnpath(&num_nodes, &node_info);
    if (status < 0) {
        exit(1);
    }

    /* Get the spice and C function names from the ifspec.ifs files */
    status = read_model_names(num_models, model_info);
    if(status != 0) {
        exit(1);
    }

    /* Get the user-defined node type names */
    status = read_node_names(num_nodes, node_info);
    if(status != 0) {
        exit(1);
    }

    /* Check to be sure the names are unique */
    status = check_uniqueness(num_models, model_info, num_nodes, node_info);
    if(status != 0) {
        exit(1);
    }

    /* Write out the CMextrn.h file used to compile SPIinit.c */
    status = write_CMextrn(num_models, model_info);
    if(status != 0) {
        exit(1);
    }

    /* Write out the CMinfo.h file used to compile SPIinit.c */
    status = write_CMinfo(num_models, model_info);
    if(status != 0) {
        exit(1);
    }
    status = write_CMinfo2(num_models, model_info);
    if(status != 0) {
        exit(1);
    }

    /* Write out the UDNextrn.h file used to compile SPIinit.c */
    status = write_UDNextrn(num_nodes, node_info);
    if(status != 0) {
        exit(1);
    }

    /* Write out the UDNinfo.h file used to compile SPIinit.c */
    status = write_UDNinfo(num_nodes, node_info);
    if(status != 0) {
        exit(1);
    }
    status = write_UDNinfo2(num_nodes, node_info);
    if(status != 0) {
        exit(1);
    }

    /* Write the make_include file used to link the models and */
    /* user-defined node functions with the simulator */
    status = write_objects_inc(num_models, model_info,
                              num_nodes, node_info);
    if(status != 0) {
        exit(1);
    }

    /* Free allocations */
    if (model_info != (Model_Info_t *) NULL) {
        free_model_info(num_models, model_info);
    }
    if (node_info != (Node_Info_t *) NULL) {
        free_node_info(num_nodes, node_info);
    }
} /* end of function preprocess_lst_files */



/* This function parses the supplied .lst file and outputs the paths to
 * stdout */
int output_paths_from_lst_file(const char *filename)
{
    int xrc = 0;
    FILEBUF *fbp = (FILEBUF *) NULL; /* for reading MODPATH_FILENAME */
    unsigned int n_path = 0;

    /* Open the file */
    if ((fbp = fbopen(filename, 0)) == (FILEBUF *) NULL) {
        print_error("ERROR - Unable to open file \"%s\" to obtain "
                "paths: %s",
                filename, strerror(errno));
        xrc = -1;
        goto EXITPOINT;
    }

    bool f_have_path = false; /* do not have a path to store */
    FBTYPE fbtype;
    FBOBJ fbobj;
    for ( ; ; ) { /* Read items until end of file */
        /* Get the next path if not found yet */
        if (!f_have_path) {
            const int rc = fbget(fbp, 0, (FBTYPE *) NULL, &fbtype, &fbobj);
            if (rc != 0) {
                if (rc == -1) { /* Error */
                    print_error("ERROR - Unable to read item to buffer");
                    xrc = -1;
                    goto EXITPOINT;
                }
                else { /* +1 -- EOF */
                    break;
                }
            } /* end of abnormal case */

            /* Remove trailing slash if appended to path */
            trim_slash(&fbobj.str_value.n_char, fbobj.str_value.sz);
        }

        /* Output the file that was found */
        if (fprintf(stdout, "%s\n", fbobj.str_value.sz) < 0) {
            print_error("ERROR - Unable to output path name to stdout: %s",
                    strerror(errno));
            xrc = -1;
            goto EXITPOINT;
        }
        ++n_path; /* 1 more path printed OK */

        /* Try getting a version. If found, it will be discarded */
        FBTYPE type_wanted = BUF_TYPE_ULONG;
        const int rc = fbget(fbp, 1, &type_wanted, &fbtype, &fbobj);
        if (rc != 0) {
            if (rc == -1) { /* Error */
                print_error("ERROR - Unable to read item to buffer");
                xrc = -1;
                goto EXITPOINT;
            }
            else { /* +1 -- EOF */
                break;
            }
        } /* end of abnormal case */

        if (fbtype == BUF_TYPE_ULONG) { /* found version number */
            f_have_path = false;
        }
        else { /* it was a string, so it is the next path */
            f_have_path = true;

            /* Remove trailing slash if appended to path */
            trim_slash(&fbobj.str_value.n_char, fbobj.str_value.sz);
        }
    } /* end of loop reading items into buffer */

EXITPOINT:
    /* Close model pathname file and return data read */
    if (fbclose(fbp) != 0) {
        print_error("ERROR - Unable to close file with path names: %s",
                strerror(errno));
        xrc = -1;
    }

    /* Return the path count if no errors */
    if (xrc == 0) {
        xrc = (int) n_path;
    }

    return xrc;
} /* end of function output_paths_from_lst_file */




/* *********************************************************************** */


/*
read_modpath

This function opens the modpath.lst file, reads the pathnames from the
file, and puts them into an internal data structure for future
processing.
*/

/* Structure for retrieving data items from the file. These will be either
 * names, such as directory names or unsigned integers that are version
 * numbers */
#define N_MODEL_INIT    10

static int read_modpath(
        int *p_num_model_info, /* Number of model pathnames found */
        Model_Info_t **pp_model_info) /* Info about each model */
{
    int xrc = 0;
    FILEBUF *fbp = (FILEBUF *) NULL; /* for reading MODPATH_FILENAME */
    Model_Info_t *p_model_info = (Model_Info_t *) NULL;
    unsigned int n_model_info = 0;
    unsigned int n_model_info_alloc = 0;
    char *filename = (char *) NULL;

    /* Open the model pathname file */
    if ((filename = gen_filename(MODPATH_FILENAME, "r")) == (char *) NULL) {
        print_error("ERROR - Unable to build mod path file name");
        xrc = -1;
        goto EXITPOINT;
    }
    /* Open the file using the default buffer size. For debugging, a very
     * small value can be used, for example by giving the size as 1, to
     * force the function do actions that otherwise would be done rarely
     * if at all. */
    if ((fbp = fbopen(filename, 0)) == (FILEBUF *) NULL) {
        print_error("ERROR - Unable to open mod path file \"%s\": %s",
                filename, strerror(errno));
        xrc = -1;
        goto EXITPOINT;
    }

    /* Allocate initial model info array */
    if ((p_model_info = (Model_Info_t *) malloc(N_MODEL_INIT *
            sizeof(Model_Info_t))) == (Model_Info_t *) NULL) {
        print_error("ERROR - Unable to allocate initial model array");
        xrc = -1;
        goto EXITPOINT;
    }
    n_model_info_alloc = N_MODEL_INIT;

    bool f_have_path = false; /* do not have a path to store */
    FBTYPE fbtype;
    FBOBJ fbobj;
    for ( ; ; ) { /* Read items until end of file */
        /* Get the next path if not found yet */
        if (!f_have_path) {
            const int rc = fbget(fbp, 0, (FBTYPE *) NULL, &fbtype, &fbobj);
            if (rc != 0) {
                if (rc == -1) { /* Error */
                    print_error("ERROR - Unable to read item to buffer");
                    xrc = -1;
                    goto EXITPOINT;
                }
                else { /* +1 -- EOF */
                    break;
                }
            } /* end of abnormal case */

            /* Remove trailing slash if appended to path */
            trim_slash(&fbobj.str_value.n_char, fbobj.str_value.sz);
        }

        /* Enlarge model array if full */
        if (n_model_info_alloc == n_model_info) {
            n_model_info_alloc *= 2;
            void * const p = realloc(p_model_info,
                    n_model_info_alloc * sizeof(Model_Info_t));
            if (p == NULL) {
                print_error("ERROR - Unable to enlarge model array");
                xrc = -1;
                goto EXITPOINT;
            }
            p_model_info = (Model_Info_t *) p;
        }

        /* Put pathname into info structure */
        Model_Info_t * const p_model_info_cur = p_model_info +
                n_model_info;

        if ((p_model_info_cur->path_name = (char *) malloc(
                fbobj.str_value.n_char + 1)) == (char *) NULL) {
            print_error("ERROR - Unable to allocate path name");
            xrc = -1;
            goto EXITPOINT;
        }

        strcpy(p_model_info_cur->path_name, fbobj.str_value.sz);
        p_model_info_cur->spice_name = (char *) NULL;
        p_model_info_cur->cfunc_name = (char *) NULL;

        /* Must set before returning due to EOF. */
        p_model_info_cur->version = 1;
        ++n_model_info;

        /* Try getting a version */
        FBTYPE type_wanted = BUF_TYPE_ULONG;
        const int rc = fbget(fbp, 1, &type_wanted, &fbtype, &fbobj);
        if (rc != 0) {
            if (rc == -1) { /* Error */
                print_error("ERROR - Unable to read item to buffer");
                xrc = -1;
                goto EXITPOINT;
            }
            else { /* +1 -- EOF */
                break;
            }
        } /* end of abnormal case */

        if (fbtype == BUF_TYPE_ULONG) { /* found version number */
            f_have_path = false;
            p_model_info_cur->version = (unsigned int) fbobj.ulong_value;
        }
        else { /* it was a string, so it is the next path */
            f_have_path = true;

            /* Remove trailing slash if appended to path */
            trim_slash(&fbobj.str_value.n_char, fbobj.str_value.sz);
        }
    } /* end of loop reading items into buffer */

EXITPOINT:
    /* Close model pathname file and return data read */
    if (fbclose(fbp) != 0) {
        print_error("ERROR - Unable to close file with model info: %s",
                strerror(errno));
        xrc = -1;
    }

    /* Free name of file being used */
    if (filename) {
        free((void *) filename);
    }

    /* If error, free model info */
    if (xrc != 0) {
        free_model_info((int)n_model_info, p_model_info);
        n_model_info = 0;
        p_model_info = (Model_Info_t *) NULL;
    }

    *p_num_model_info = (int)n_model_info;
    *pp_model_info = p_model_info;

    return xrc;
} /* end of function read_modpath */



/* Remove slash at end of path name if present */
static inline void trim_slash(size_t *p_n, char *p)
{
    size_t n = *p_n;
    if (n > 1) {
        char * const p_last = p + n - 1;
        if (*p_last == '/') {
            *p_last = '\0';
            --*p_n;
        }
    }
} /* end of function trim_slash */



/* *********************************************************************** */


/*
read_udnpath

This function opens the udnpath.lst file, reads the pathnames from the
file, and puts them into an internal data structure for future
processing.
*/


#define N_NODE_INIT     5

static int read_udnpath(
    int            *p_num_node_info, /* Addr to receive # node pathnames */
    Node_Info_t    **pp_node_info /* Addr to receive node info */
)
{
    int xrc = 0;
    FILEBUF *fbp = (FILEBUF *) NULL; /* For reading Udn pathname */
    Node_Info_t  *p_node_info = (Node_Info_t *) NULL;
                                            /* array of node info */
    unsigned int n_node_info = 0;
    unsigned int n_node_info_alloc = 0;
    char *filename = (char *) NULL;


    /* Open the node pathname file */
    if ((filename = gen_filename(UDNPATH_FILENAME, "r")) == (char *) NULL) {
        print_error("ERROR - Unable to build udn path file name");
        xrc = -1;
        goto EXITPOINT;
    }

    /* For backward compatibility, return with WARNING only if file
     * not found */
    if ((fbp = fbopen(filename, 0)) == (FILEBUF *) NULL) {
        print_error("WARNING - File not found: %s", filename);
        xrc = +1;
        goto EXITPOINT;
    }

    /* Allocate initial node info array */
    if ((p_node_info = (Node_Info_t *) malloc(N_NODE_INIT *
            sizeof(Node_Info_t))) == (Node_Info_t *) NULL) {
        print_error("ERROR - Unable to allocate initial node array");
        xrc = -1;
        goto EXITPOINT;
    }
    n_node_info_alloc = N_NODE_INIT;

    bool f_have_path = false; /* do not have a path to store */
    FBTYPE fbtype;
    FBOBJ fbobj;

    for ( ; ; ) { /* Read items until end of file */
        /* Get the next path if not found yet */
        if (!f_have_path) {
            const int rc = fbget(fbp, 0, (FBTYPE *) NULL, &fbtype, &fbobj);
            if (rc != 0) {
                if (rc == -1) { /* Error */
                    print_error("ERROR - Unable to read item to buffer");
                    xrc = -1;
                    goto EXITPOINT;
                }
                else { /* +1 -- EOF */
                    break;
                }
            } /* end of abnormal case */

            /* Remove trailing slash if appended to path */
            trim_slash(&fbobj.str_value.n_char, fbobj.str_value.sz);
        }


        /* Enlarge node array if full */
        if (n_node_info_alloc == n_node_info) {
            n_node_info_alloc *= 2;
            void * const p = realloc(p_node_info,
                    n_node_info_alloc * sizeof(Node_Info_t));
            if (p == NULL) {
                print_error("ERROR - Unable to enlarge node array");
                xrc = -1;
                goto EXITPOINT;
            }
            p_node_info = (Node_Info_t *) p;
        }

        /* Put pathname into info structure */
        Node_Info_t * const p_node_info_cur = p_node_info +
                n_node_info;

        if ((p_node_info_cur->path_name = (char *) malloc(
                fbobj.str_value.n_char + 1)) == (char *) NULL) {
            print_error("ERROR - Unable to allocate path name");
            xrc = -1;
            goto EXITPOINT;
        }

        strcpy(p_node_info_cur->path_name, fbobj.str_value.sz);
        p_node_info_cur->node_name = NULL;

        /* Must set before returning due to EOF. */
        p_node_info_cur->version = 1;
        ++n_node_info;

        /* Try getting a version */
        FBTYPE type_wanted = BUF_TYPE_ULONG;
        const int rc = fbget(fbp, 1, &type_wanted, &fbtype, &fbobj);
        if (rc != 0) {
            if (rc == -1) { /* Error */
                print_error("ERROR - Unable to read item to buffer");
                xrc = -1;
                goto EXITPOINT;
            }
            else { /* +1 -- EOF */
                break;
            }
        } /* end of abnormal case */

        if (fbtype == BUF_TYPE_ULONG) { /* found version number */
            f_have_path = false;
            p_node_info_cur->version = (unsigned int) fbobj.ulong_value;
        }
        else { /* it was a string, so it is the next path */
            f_have_path = true;

            /* Remove trailing slash if appended to path */
            trim_slash(&fbobj.str_value.n_char, fbobj.str_value.sz);
        }

    } /* end of loop reading items into buffer */

EXITPOINT:
    /* Close model pathname file and return data read */
    if (fbclose(fbp) != 0) {
        print_error("ERROR - Unable to close file with node info: %s",
                strerror(errno));
        xrc = -1;
    }

    /* Free name of file being used */
    if (filename) {
        free((void *) filename);
    }

    /* If error, free node info */
    if (xrc != 0) {
        free_node_info((int)n_node_info, p_node_info);
        n_node_info = 0;
        p_node_info = (Node_Info_t *) NULL;
    }

    *p_num_node_info = (int)n_node_info;
    *pp_node_info = p_node_info;

    return xrc;
} /* end of function read_udnpath */



/* *********************************************************************** */


/*
read_model_names

This function opens each of the models and gets the names of the model
and the C function into the internal model information structure.
*/

static int read_model_names(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info  /* Info about each model */
)
{
    bool all_found = true; /* True if all ifspec files read */
    int i; /* A temporary counter */
    char path_stack[100]; /* full pathname to ifspec file if from stack */
    char *path = path_stack; /* actual path buffer */
    int status; /* Return status */

    /* Repository for info read from ifspec file.*/
    Ifs_Table_t ifs_table;

    /* Find the required buffer size and allocate if the default buffer
     * on the stack is too small */
    {
        int j;
        size_t n_byte_needed = 0;
        for (j = 0; j < num_models; j++) { /* max(model path lengths) */
            const size_t n = strlen(model_info[j].path_name);
            if (n_byte_needed < n) {
                n_byte_needed = n;
            }
        }
        n_byte_needed += 1 + strlen(IFSPEC_FILENAME) + 1;
        if (n_byte_needed > sizeof path_stack) {
            if ((path = (char *) malloc(n_byte_needed)) == (char *) NULL) {
                print_error("ERROR - Unable to allocate a buffer "
                        "for model paths");
                return -1;
            }
        }
    }


    /* For each model found in model pathname file, read the interface */
    /* spec file to get the SPICE and C function names into model_info */
    for(i = 0; i < num_models; i++) {
        Model_Info_t *p_model_info_cur = model_info + i;

        /* 0 for error recovery */
        (void) memset(&ifs_table, 0, sizeof ifs_table);

        /* Form the full pathname to the interface spec file. Size has
         * been checked to ensure that all strings will fit. */
        {
            char *p_dst = path;

            /* Copy path name */
            const char *p_src = model_info[i].path_name;
            for ( ; ; ) {
                const char ch_cur = *p_src;
                if (ch_cur == '\0') {
                    break;
                }
                *p_dst++ = ch_cur;
                ++p_src;
            }
            *p_dst++ = '/'; /* add directory separator */
            strcpy(p_dst, IFSPEC_FILENAME);
        }

        /* Read the SPICE and C function names from the interface spec file */
        status = read_ifs_file(path, GET_IFS_NAME, &ifs_table);

        /* Transfer the names into the model_info structure */
        if (status == 0) {
            if ((p_model_info_cur->spice_name = strdup(
                    ifs_table.name.model_name)) == (char *) NULL) {
                print_error("ERROR - Unable to copy code model name");
                all_found = false;
                break;
            }
            if ((p_model_info_cur->cfunc_name = strdup(
                    ifs_table.name.c_fcn_name)) == (char *) NULL) {
                print_error("ERROR - Unable to copy code model function name");
                all_found = false;
                break;
            }
        }
        else {
            print_error(
                    "ERROR - Problems reading \"%s\" in directory \"%s\"",
                    IFSPEC_FILENAME, p_model_info_cur->path_name);
            all_found = false;
            break;
        }

        /* Remove the ifs_table */
        rem_ifs_table(&ifs_table);
    } /* end of loop over models */

    /* Free buffer if allocated */
    if (path != path_stack) {
        free(path);
    }

    if (all_found) {
        return 0;
    }
    else {
        /* Free allocations of model when failure occurred */
        rem_ifs_table(&ifs_table);
        return -1;
    }
} /* end of function read_model_names */



/* *********************************************************************** */


/*
read_node_names

This function opens each of the user-defined node definition files
and gets the names of the node into the internal information structure.
*/


static int read_node_names(
    int           num_nodes,  /* Number of node pathnames */
    Node_Info_t   *node_info  /* Info about each node */
)
{
    bool all_found = true; /* True if all ifspec files read */
    int i; /* A temporary counter */
    char path_stack[100]; /* full pathname to ifspec file if from stack */
    char *path = path_stack; /* actual path buffer */
    int status; /* Return status */
    char        *node_name;              /* Name of node type read from file */

    /* Find the required buffer size and allocate if the default buffer
     * on the stack is too small */
    {
        int j;
        size_t n_byte_needed = 0;
        for (j = 0; j < num_nodes; j++) { /* max(model path lengths) */
            const size_t n = strlen(node_info[j].path_name);
            if (n_byte_needed < n) {
                n_byte_needed = n;
            }
        }
        n_byte_needed += 1 + strlen(UDNFUNC_FILENAME) + 1;
        if (n_byte_needed > sizeof path_stack) {
            if ((path = (char *) malloc(n_byte_needed)) == (char *) NULL) {
                print_error("ERROR - Unable to allocate a buffer "
                        "for user types");
                return -1;
            }
        }
    }


    /* For each node found in node pathname file, read the udnfunc.c */
    /* file to get the node type names into node_info */

    for(i = 0, all_found = true; i < num_nodes; i++) {
        /* Form the full pathname to the user-defined type file. Size has
         * been checked to ensure that all strings will fit. */
        {
            char *p_dst = path;

            /* Copy path name */
            const char *p_src = node_info[i].path_name;
            for ( ; ; ) {
                const char ch_cur = *p_src;
                if (ch_cur == '\0') {
                    break;
                }
                *p_dst++ = ch_cur;
                ++p_src;
            }
            *p_dst++ = '/'; /* add directory separator */
            strcpy(p_dst, UDNFUNC_FILENAME);
        }

        /* Read the udn node type name from the file */
        status = read_udn_type_name(path, &node_name);

        /* Transfer the names into the node_info structure */
        if(status == 0) {
            node_info[i].node_name = node_name;
        }
        else {
            all_found = false;
            print_error("ERROR - Problems reading %s in directory %s",
                        UDNFUNC_FILENAME, node_info[i].path_name);
        }
    }

    if(all_found)
        return 0;
    else
        return -1;
} /* end of function read_node_names */



/* *********************************************************************** */


/*
check_uniqueness

Function check_uniqueness determines if model names and function
names are unique with respect to each other and to the models and
functions internal to SPICE 3C1.
*/

static int check_uniqueness(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info, /* Info about each model */
    int            num_nodes,   /* Number of node type pathnames */
    Node_Info_t    *node_info   /* Info about each node type */
)
{
    /* Define a list of model names used internally by XSPICE */
    /* These names (except 'poly') are defined in src/sim/INP/INPdomodel.c and */
    /* are case insensitive */
    static const char *SPICEmodel[] = {
                                  "npn",
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
    static const int numSPICEmodels = sizeof(SPICEmodel) / sizeof(char *);


    /* Define a list of node type names used internally by the simulator */
    /* These names are defined in src/sim/include/MIFtypes.h and are case */
    /* insensitive */
    static const char *UDNidentifier[] = {
                                     "v",
                                     "vd",
                                     "i",
                                     "id",
                                     "vnam",
                                     "g",
                                     "gd",
                                     "h",
                                     "hd",
                                     "d" };
    static const int numUDNidentifiers = sizeof(UDNidentifier) / sizeof(char *);

    bool f_have_duplicate = false; /* no duplicates found yet */

    /* First, normalize case of all model and node names to lower since */
    /* SPICE is case insensitive in parsing decks */
    {
        int i;
        for(i = 0; i < num_models; i++) {
            str_to_lower(model_info[i].spice_name);
        }
    }
    {
        int i;
        for(i = 0; i < num_nodes; i++) {
            str_to_lower(node_info[i].node_name);
        }
    }

    /* Sizes of models and nodes */
    const unsigned int n_model = (unsigned int)(numSPICEmodels + num_models);
    const unsigned int n_node = (unsigned int)(numUDNidentifiers + num_nodes);
    const unsigned int n_ks = n_model > n_node ? n_model : n_node;

    /* Allocate structure to compare */
    struct Key_src * const p_ks = (struct Key_src *) malloc(
            n_ks * sizeof *p_ks);
    if (p_ks == (struct Key_src *) NULL) {
        print_error("ERROR - Unable to allocate array to check uniqueness.");
        return -1;
    }

    /* Set up for test of SPICE name of models and test */
    if (num_models > 0) { /* Have user-defined models */
        {
            int i, j;

            /* Fill up with SPICE models */
            for (i = 0; i < numSPICEmodels; ++i) {
                struct Key_src *p_ks_cur = p_ks + i;
                p_ks_cur->key = SPICEmodel[i];
                p_ks_cur->src = (char *) NULL; /* denotes internal SPICE */
            }
            /* Add SPICE model names for code models */
            for (j = 0; j < num_models; ++i, ++j) {
                struct Key_src *p_ks_cur = p_ks + i;
                Model_Info_t *p_mi_cur = model_info + j;
                p_ks_cur->key = p_mi_cur->spice_name;
                p_ks_cur->src = p_mi_cur->path_name;
            }

            /* Test for duplicates */
            f_have_duplicate |= test_for_duplicates(n_model, p_ks,
                    &report_error_spice_name);
        }

        /* Set up for test of function names */
        {
            int i;

            /* Fill up with C function names from code models */
            for (i = 0; i < num_models; ++i) {
                struct Key_src *p_ks_cur = p_ks + i;
                Model_Info_t *p_mi_cur = model_info + i;
                p_ks_cur->key = p_mi_cur->cfunc_name;
                p_ks_cur->src = p_mi_cur->path_name;
            }

            /* Test for duplicates */
            f_have_duplicate |= test_for_duplicates((unsigned int)num_models,
                    p_ks, &report_error_function_name);
        }
    }

    /* Set up for test of node types and test */
    if (num_nodes > 0) { /* Have user-defined types */
        int i, j;

        /* Fill up with SPICE node types */
        for (i = 0; i < numUDNidentifiers; ++i) {
            struct Key_src *p_ks_cur = p_ks + i;
            p_ks_cur->key = UDNidentifier[i];
            p_ks_cur->src = (char *) NULL; /* denotes internal SPICE */
        }
        /* Add user-defined nodes */
        for (j = 0; j < num_nodes; ++i, ++j) {
            struct Key_src *p_ks_cur = p_ks + i;
            Node_Info_t *p_ni_cur = node_info + j;
            p_ks_cur->key = p_ni_cur->node_name;
            p_ks_cur->src = p_ni_cur->path_name;
        }

        /* Test for duplicates */
        f_have_duplicate |= test_for_duplicates(n_node, p_ks,
                &report_error_udn_name);
    }

    /* Free allocation for compares */
    free(p_ks);

    /* Return error status */
    return f_have_duplicate ? -1 : 0;
} /* end of function check_uniqueness */



/* Test for duplicate key values and report using the supplied function
 * if found */
static bool test_for_duplicates(unsigned int n, struct Key_src *p_ks,
        void (*p_error_reporter)(const struct Key_src *p_ks1,
                const struct Key_src *p_ks2))
{
    bool f_have_duplicate = false;

    /* Sort to put duplicates together */
    qsort(p_ks, n, sizeof(struct Key_src),
            (int (*)(const void *, const void *)) &cmpr_ks);

    unsigned int i;
    for (i = 0; i != n; ) {
        const struct Key_src * const p_ks_i = p_ks + i;
        const char * const p_key_i_val = p_ks_i->key;
        unsigned int j;
        for (j = i + 1; j != n; ++j) {
            const struct Key_src * const p_ks_j = p_ks + j;
            const char * const p_key_j_val = p_ks_j->key;
            if (strcmp(p_key_i_val, p_key_j_val) != 0) {
                break;
            }

            /* Duplicate found. Indicate a duplicate was found and report
             * the error */
            f_have_duplicate = true;
            (*p_error_reporter)(p_ks_i, p_ks_j);
        } /* end of loop testing for duplicates */
        i = j; /* advance to next unique value or end of list */
    } /* end of loop over items to test */
    return f_have_duplicate;
} /* end of function test_for_duplicates */



/* Compare function for struct Key_src.
 *
 * Remarks
 * the src field may be NULL to indicate internal values. These should
 * be ordered before values that are not NULL for nicer error messages.
 * Note that for a given key, only one src field can be NULL. */
static int cmpr_ks(const struct Key_src *ks1, const struct Key_src *ks2)
{
    /* First order by the value of the key */
    const int rc = strcmp(ks1->key, ks2->key);
    if (rc != 0) {
        return rc;
    }

    /* Test for NULL src fields. Only one can be NULL for the same key */
    if (ks1->src == (char *) NULL) {
        return -1;
    }
    if (ks2->src == (char *) NULL) {
        return +1;
    }

    /* Both keys not NULL, so compare */
    return strcmp(ks1->src, ks2->src);
} /* end of function cmpr_ks */



static void report_error_spice_name(const struct Key_src *p_ks1,
        const struct Key_src *p_ks2)
{
    const char * const p_ks1_src = p_ks1->src;
    if (p_ks1_src == (char *) NULL) { /* internal SPICE name */
        print_error("ERROR: Model name \"%s\" from directory \"%s\" "
                "is the name of an internal SPICE model",
                p_ks1->key, p_ks2->src);
    }
    else {
        print_error("ERROR: Model name \"%s\" in directory \"%s\" "
                "is also in directory \"%s\".",
                p_ks1->key, p_ks1_src, p_ks2->src);
    }
} /* end of function report_error_spice_name */



static void report_error_function_name (const struct Key_src *p_ks1,
        const struct Key_src *p_ks2)
{
    print_error("ERROR: C function name \"%s\" in directory \"%s\" "
            "is also in directory \"%s\".",
            p_ks1->key, p_ks1->src, p_ks2->src);
} /* end of function report_error_spice_name */



static void report_error_udn_name(const struct Key_src *p_ks1,
        const struct Key_src *p_ks2)
{
    const char * const p_ks1_src = p_ks1->src;
    if (p_ks1_src == (char *) NULL) { /* internal SPICE name */
        print_error("ERROR: Node type \"%s\" from directory \"%s\" "
                "is the name of an internal SPICE node type",
                p_ks1->key, p_ks2->src);
    }
    else {
        print_error("ERROR: Node type \"%s\" in directory \"%s\" "
                "is also in directory \"%s\".",
                p_ks1->key, p_ks1_src, p_ks2->src);
    }
} /* end of function report_error_udn_name */



/* *********************************************************************** */


/*
write_CMextrn

Function write_CMextrn writes the CMextrn.h file used in
compiling SPIinit.c immediately prior to linking the simulator
and code models.  This SPICE source file uses the structures
mentioned in the include file to define the models known to the
simulator.
*/

static int write_CMextrn(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info  /* Info about each model */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing CMextrn.h */

    char *filename = (char *) NULL;

    /* Open the file to be written */
    if ((filename = gen_filename("cmextrn.h", "w")) == (char *) NULL) {
        print_error("ERROR - Unable to build path to cmextrn.h");
        return -1;
    }
    fp = fopen(filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        free(filename);
        return -1;
    }

    /* Write out the data */
    for(i = 0; i < num_models; i++) {
        fprintf(fp, "extern  SPICEdev  %s_info;\n", model_info[i].cfunc_name);
    }

    /* Close the file and return */
    if (fclose(fp) != 0) {
        print_error("ERROR - Problems closing %s", filename);
        free(filename);
        return -1;
    }
    free(filename);
    return 0;
} /* end of function write_CMextrn */


/* *********************************************************************** */


/*
write_CMinfo

Function write_CMinfo writes the CMinfo.h file used in compiling
SPIinit.c immediately prior to linking the simulator and code
models.  This SPICE source file uses the structures mentioned in
the include file to define the models known to the simulator.
*/


static int write_CMinfo(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info  /* Info about each model */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing CMinfo.h */

    char *filename = (char *) NULL;

    /* Open the file to be written */
    if ((filename = gen_filename("cminfo.h", "w")) == (char *) NULL) {
        print_error("ERROR - Unable to build path to cminfo.h");
        return -1;
    }
    fp = fopen(filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        free(filename);
        return -1;
    }

    /* Write out the data */
    for(i = 0; i < num_models; i++) {
        Model_Info_t *p_mi_cur = model_info + i;
        if (p_mi_cur->version == 1) {
            fprintf(fp, "&%s_info,\n", model_info[i].cfunc_name);
        }
    }

    /* Close the file and return */
    if (fclose(fp) != 0) {
        print_error("ERROR - Problems closing %s", filename);
        free(filename);
        return -1;
    }

    free(filename);
    return 0;
} /* end of function write_CMinfo */



static int write_CMinfo2(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info  /* Info about each model */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing CMinfo.h */

    char *filename = (char *) NULL;

    /* Open the file to be written */
    if ((filename = gen_filename("cminfo2.h", "w")) == (char *) NULL) {
        print_error("ERROR - Unable to build path to cminfo2.h");
        return -1;
    }
    fp = fopen(filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        free(filename);
        return -1;
    }

    /* Write out the data */
    for (i = 0; i < num_models; i++) {
        Model_Info_t *p_mi_cur = model_info + i;
        if (p_mi_cur->version <= 2) {
            fprintf(fp, "&%s_info,\n", model_info[i].cfunc_name);
        }
    }

    /* Close the file and return */
    if (fclose(fp) != 0) {
        print_error("ERROR - Problems closing %s", filename);
        free(filename);
        return -1;
    }

    free(filename);
    return 0;
} /* end of function write_CMinfo2 */



/* *********************************************************************** */



/*
write_UDNextrn

Function write_UDNextrn writes the UDNextrn.h file used in
compiling SPIinit.c immediately prior to linking the simulator
and user-defined nodes.  This SPICE source file uses the structures
mentioned in the include file to define the node types known to the
simulator.
*/



static int write_UDNextrn(
    int            num_nodes,  /* Number of node pathnames */
    Node_Info_t    *node_info  /* Info about each node */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing UDNextrn.h */

    char *filename = (char *) NULL;

    /* Open the file to be written */
    if ((filename = gen_filename("udnextrn.h", "w")) == (char *) NULL) {
        print_error("ERROR - Unable to build path to udnextrn.h");
        return -1;
    }
    fp = fopen(filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        return -1;
    }

    /* Write out the data */
    for(i = 0; i < num_nodes; i++) {
        fprintf(fp, "extern Evt_Udn_Info_t  udn_%s_info;\n", node_info[i].node_name);
    }

    /* Close the file and return */
    if (fclose(fp) != 0) {
        print_error("ERROR - Problems closing %s", filename);
        free(filename);
        return -1;
    }

    free(filename);
    return 0;
} /* end of function write_UDNextrn */



/* *********************************************************************** */


/*
write_UDNinfo

Function write_UDNinfo writes the UDNinfo.h file used in
compiling SPIinit.c immediately prior to linking the simulator
and user-defined nodes.  This SPICE source file uses the structures
mentioned in the include file to define the node types known to the
simulator.
*/
static int write_UDNinfo(
    int            num_nodes,  /* Number of node pathnames */
    Node_Info_t    *node_info  /* Info about each node */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing UDNinfo.h */

    char *filename = (char *) NULL;

    /* Open the file to be written */
    if ((filename = gen_filename("udninfo.h", "w")) == (char *) NULL) {
        print_error("ERROR - Unable to build path to udninfo.h");
        return -1;
    }
    fp = fopen(filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        return -1;
    }

    /* Write out the data */
    for(i = 0; i < num_nodes; i++) {
        Node_Info_t *p_ni_cur = node_info + i;
        if (p_ni_cur->version == 1) {
            fprintf(fp, "&udn_%s_info,\n", p_ni_cur->node_name);
        }
    }

    /* Close the file and return */
    if (fclose(fp) != 0) {
        print_error("ERROR - Problems closing %s", filename);
        free(filename);
        return -1;
    }

    free(filename);
    return 0;
} /* end of function write_UDNinfo */



static int write_UDNinfo2(
    int            num_nodes,  /* Number of node pathnames */
    Node_Info_t    *node_info  /* Info about each node */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing UDNinfo.h */

    char *filename = (char *) NULL;

    /* Open the file to be written */
    if ((filename = gen_filename("udninfo2.h", "w")) == (char *) NULL) {
        print_error("ERROR - Unable to build path to udninfo2.h");
        return -1;
    }
    fp = fopen(filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        return -1;
    }

    /* Write out the data */
    for(i = 0; i < num_nodes; i++) {
        Node_Info_t *p_ni_cur = node_info + i;
        if (p_ni_cur->version <= 2) {
            fprintf(fp, "&udn_%s_info,\n", p_ni_cur->node_name);
        }
    }

    /* Close the file and return */
    if (fclose(fp) != 0) {
        print_error("ERROR - Problems closing %s", filename);
        free(filename);
        return -1;
    }

    free(filename);
    return 0;
} /* end of function write_UDNinfo */



/* *********************************************************************** */

/*
write_objects_inc

Function write_objects_inc writes a make include file used by
the make utility to locate the object modules needed to link the
simulator with the code models and user-defined node types.
*/

static int write_objects_inc(
    int            num_models,  /* Number of model pathnames */
    Model_Info_t   *model_info, /* Info about each model */
    int            num_nodes,   /* Number of udn pathnames */
    Node_Info_t    *node_info   /* Info about each node type */
)
{
    int         i;                       /* A temporary counter */
    FILE        *fp;   /* File pointer for writing make_include */

    char *filename = (char *) NULL;

    /* Open the file to be written */
    if ((filename = gen_filename("objects.inc", "w")) == (char *) NULL) {
        print_error("ERROR - Unable to build path to objects.inc");
        return -1;
    }
    fp = fopen(filename, "w");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for write", filename);
        return -1;
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
    if (fclose(fp) != 0) {
        print_error("ERROR - Problems closing %s", filename);
        free(filename);
        return -1;
    }

    free(filename);
    return 0;
} /* end of function write_objects_inc */



/*
read_udn_type_name

This function reads a User-Defined Node Definition File until
the definition of the Evt_Udn_Info_t struct
is found, and then gets the name of the node type from the first
member of the structure.
*/

static int read_udn_type_name(
    const char *path,           /* the path to the node definition file */
    char **node_name            /* the node type name found in the file */
)
{
    FILE        *fp;                    /* file pointer for opened file */
    bool   found;                  /* true if name found successfully */
    bool   in_struct;              /* true if found struct with name */
    char        name[MAX_NAME_LEN + 1]; /* temporary storage for name read */
    int         c;                      /* a character read from the file */
    int         i;                      /* a counter */

    static char *struct_type = "Evt_Udn_Info_t";
    char *filename = (char *) NULL;

    /* Open the file from which the node type name will be read */
    if ((filename = gen_filename(path, "r")) == (char *) NULL) {
        print_error("ERROR - Unable to build path to Evt_Udn_Info_t");
        return -1;
    }
    fp = fopen(filename, "r");
    if(fp == NULL) {
        print_error("ERROR - Problems opening %s for reading", filename);
        return -1;
    }

    /* Read the file until the definition of the Evt_Udn_Info_t struct */
    /* is found, then get the name of the node type from the first */
    /* member of the structure */
    found = false;
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
        for(i = 0, in_struct = false; ; i++) {
            if(c != struct_type[i])
                break;
            else if(i == (sizeof(struct_type) - 2)) {
                in_struct = true;
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
                            found = true;
                            if (i >= (int)sizeof name) {
                                print_error("name too long");
                                exit(1);
                            }
                            name[i] = '\0';
                        }
                        else if(c != EOF) {
                            if (i > (int)sizeof name) {
                                print_error("name too long");
                                exit(1);
                            }
                            name[i++] = (char) c;
                        }
                    } while((c != EOF) && (! found));
                }
            } while((c != EOF) && (! found));
        }

    } while((c != EOF) && (! found));

    /* Close the file and return */
    fclose(fp);

    if(found) {
        if ((*node_name = (char *) malloc(
                strlen(name) + 1)) == (char *) NULL) {
            print_error("ERROR - Unable to allocate node name");
            return -1;
        }
        strcpy(*node_name, name);
        return 0;
    }
    else {
        *node_name = NULL;
        return -1;
    }
}



/* Free allocations in p_model_info array */
static void free_model_info(int num_models, Model_Info_t *p_model_info)
{
    /* Return if no structure */
    if (p_model_info == (Model_Info_t *) NULL) {
        return;
    }

    int i;
    for (i = 0; i < num_models; ++i) {
        Model_Info_t *p_cur = p_model_info + i;
        void *p;
        if ((p = p_cur->cfunc_name) != NULL) {
            free(p);
        }
        if ((p = p_cur->path_name) != NULL) {
            free(p);
        }
        if ((p = p_cur->spice_name) != NULL) {
            free(p);
        }
    }

    free(p_model_info);
} /* end of function free_model_info */



/* Free allocations in p_nod_info array */
static void free_node_info(int num_nodes, Node_Info_t *p_node_info)
{
    /* Return if no structure */
    if (p_node_info == (Node_Info_t *) NULL) {
        return;
    }

    int i;
    for (i = 0; i < num_nodes; ++i) {
        Node_Info_t *p_cur = p_node_info + i;
        void *p;
        if ((p = p_cur->node_name) != NULL) {
            free(p);
        }
        if ((p = p_cur->path_name) != NULL) {
            free(p);
        }
    }

    free(p_node_info);
} /* end of function free_node_info */



