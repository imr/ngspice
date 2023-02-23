#ifndef ngspice_EVTUDN_H
#define ngspice_EVTUDN_H

/* ===========================================================================
FILE    EVTudn.h

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

    This file contains the definition of "User-Defined Nodes".
    These nodes are integrated into the simulator similar to the
    way models are tied into SPICE 3C1, so that new node types
    can be relatively easily added.  The functions (required and
    optional) are listed below.  For optional functions, the
    function can be left undefined and the pointer placed into the
    Evt_Udn_Info_t structure can be specified as NULL.

    Required functions:
        create  -  allocate data structure used as inputs and outputs to code models
        initialize  -  set structure to appropriate initial value for first use as model input
        copy  -  make a copy of the contents into created but possibly uninitialized structure
        compare  -  determine if two structures are equal in value

    Optional functions:
        dismantle  -  free allocations _inside_ structure (but not structure itself)
        invert  -  invert logical value of structure
        resolve  -  determine the resultant when multiple outputs are connected to a node
        plot_val  -  output a real value for specified structure component for plotting purposes
        print_val  -  output a string value for specified structure component for printing
        ipc_val  -  output a binary data structure and size of the structure for IPC

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */


#include "ngspice/miftypes.h"  /* for Mif_Boolean_t used in udn_..._compare */

#define MALLOCED_PTR                 (*evt_struct_ptr)
#define STRUCT_PTR                   evt_struct_ptr
#define STRUCT_PTR_1                 evt_struct_ptr_1
#define STRUCT_PTR_2                 evt_struct_ptr_2
#define EQUAL                        (*evt_equal)
#define INPUT_STRUCT_PTR             evt_input_struct_ptr
#define OUTPUT_STRUCT_PTR            evt_output_struct_ptr
#define INPUT_STRUCT_PTR_ARRAY       evt_input_struct_ptr_array
#define INPUT_STRUCT_PTR_ARRAY_SIZE  evt_input_struct_ptr_array_size
#define STRUCT_MEMBER_ID             evt_struct_member_id
#define PLOT_VAL                     (*evt_plot_val)
#define PRINT_VAL                    (*evt_print_val)
#define IPC_VAL                      (*evt_ipc_val)
#define IPC_VAL_SIZE                 (*evt_ipc_val_size)

#define CREATE_ARGS      void **evt_struct_ptr
#define INITIALIZE_ARGS  void *evt_struct_ptr
#define COMPARE_ARGS     void *evt_struct_ptr_1, \
                         void *evt_struct_ptr_2, \
                         Mif_Boolean_t *evt_equal
#define COPY_ARGS        void *evt_input_struct_ptr, \
                         void *evt_output_struct_ptr
#define DISMANTLE_ARGS   void *evt_struct_ptr
#define INVERT_ARGS      void *evt_struct_ptr
#define RESOLVE_ARGS     int evt_input_struct_ptr_array_size, \
                         void **evt_input_struct_ptr_array, \
                         void *evt_output_struct_ptr
#define PLOT_VAL_ARGS    void *evt_struct_ptr, \
                         char *evt_struct_member_id, \
                         double *evt_plot_val
#define PRINT_VAL_ARGS   void *evt_struct_ptr, \
                         char *evt_struct_member_id, \
                         char **evt_print_val
#define IPC_VAL_ARGS     void *evt_struct_ptr, \
                         void **evt_ipc_val, \
                         int  *evt_ipc_val_size

/* Information about each Digital/User-defined type of node value. */

typedef struct {
    char                 *name;
    char                 *description;
    Evt_Output_Event_t   *free_list;  /* Event structs for these values. */
    void               ((*create)(CREATE_ARGS));
    void               ((*dismantle)(DISMANTLE_ARGS));
    void               ((*initialize)(INITIALIZE_ARGS));
    void               ((*invert)(INVERT_ARGS));
    void               ((*copy)(COPY_ARGS));
    void               ((*resolve)(RESOLVE_ARGS));
    void               ((*compare)(COMPARE_ARGS));
    void               ((*plot_val)(PLOT_VAL_ARGS));
    void               ((*print_val)(PRINT_VAL_ARGS));
    void               ((*ipc_val)(IPC_VAL_ARGS));
} Evt_Udn_Info_t;


extern int             g_evt_num_udn_types;
extern Evt_Udn_Info_t  **g_evt_udn_info;


#endif
