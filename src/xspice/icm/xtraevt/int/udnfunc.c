/*============================================================================
FILE    int/udnfunc.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the definition of the 'int' node type
    used by event-driven models that simulate with integer type data.
    These functions are called exclusively through function
    pointers in an Evt_Udn_Info_t data structure.

INTERFACES

    Evt_Udn_Info_t udn_int_info

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include <stdio.h>


#include "ngspice/cm.h"
#include "ngspice/evtudn.h"
#include "ngspice/memory.h"


/* macro to ignore unused variables and parameters */
#define NG_IGNORE(x)  (void)x

/* ************************************************************************ */

static void udn_int_create(CREATE_ARGS)
{
    /* Malloc space for an int */
    MALLOCED_PTR = TMALLOC(int, 1);
}


/* ************************************************************************ */

static void udn_int_dismantle(DISMANTLE_ARGS)
{
    NG_IGNORE(STRUCT_PTR);

    /* Do nothing.  There are no internally malloc'ed things to dismantle */
}


/* ************************************************************************ */

static void udn_int_initialize(INITIALIZE_ARGS)
{
    int  *int_struct = (int *) STRUCT_PTR;


    /* Initialize to zero */
    *int_struct = 0;
}


/* ************************************************************************ */

static void udn_int_invert(INVERT_ARGS)
{
    int      *int_struct = (int *) STRUCT_PTR;


    /* Invert the state */
    *int_struct = -(*int_struct);
}


/* ************************************************************************ */

static void udn_int_copy(COPY_ARGS)
{
    int  *int_from_struct = (int *) INPUT_STRUCT_PTR;
    int  *int_to_struct   = (int *) OUTPUT_STRUCT_PTR;

    /* Copy the structure */
    *int_to_struct = *int_from_struct;
}

/* ************************************************************************ */

static void udn_int_resolve(RESOLVE_ARGS)
{
    int **array    = (int**)INPUT_STRUCT_PTR_ARRAY;
    int *out       = (int *) OUTPUT_STRUCT_PTR;
    int num_struct = INPUT_STRUCT_PTR_ARRAY_SIZE;

    int         sum;
    int         i;

    /* Sum the values */
    for(i = 0, sum = 0; i < num_struct; i++)
        sum += *(array[i]);

    /* Assign the result */
    *out = sum;
}

/* ************************************************************************ */

static void udn_int_compare(COMPARE_ARGS)
{
    int  *int_struct1 = (int *) STRUCT_PTR_1;
    int  *int_struct2 = (int *) STRUCT_PTR_2;

    /* Compare the structures */
    if((*int_struct1) == (*int_struct2))
        EQUAL = TRUE;
    else
        EQUAL = FALSE;
}


/* ************************************************************************ */

static void udn_int_plot_val(PLOT_VAL_ARGS)
{
    int   *int_struct = (int *) STRUCT_PTR;

    NG_IGNORE(STRUCT_MEMBER_ID);

    /* Output a value for the int struct */
    PLOT_VAL = *int_struct;
}


/* ************************************************************************ */

static void udn_int_print_val(PRINT_VAL_ARGS)
{
    int   *int_struct = (int *) STRUCT_PTR;

    NG_IGNORE(STRUCT_MEMBER_ID);

    /* Allocate space for the printed value */
    PRINT_VAL = TMALLOC(char, 30);

    /* Print the value into the string */
    sprintf(PRINT_VAL, "%8d", *int_struct);
}



/* ************************************************************************ */

static void udn_int_ipc_val(IPC_VAL_ARGS)
{
    /* Simply return the structure and its size */
    IPC_VAL = STRUCT_PTR;
    IPC_VAL_SIZE = sizeof(int);
}



Evt_Udn_Info_t udn_int_info = {

    "int",
    "integer valued data",
    NULL,

    udn_int_create,
    udn_int_dismantle,
    udn_int_initialize,
    udn_int_invert,
    udn_int_copy,
    udn_int_resolve,
    udn_int_compare,
    udn_int_plot_val,
    udn_int_print_val,
    udn_int_ipc_val

};
