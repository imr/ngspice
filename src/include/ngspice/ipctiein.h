/*============================================================================
FILE    IPCtiein.h

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

    Provides a protocol independent interface between the simulator
    and the IPC method used to interface to CAE packages.

INTERFACES


REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/


#ifndef ngspice_IPCTIEIN_H
#define ngspice_IPCTIEIN_H


#include "ngspice/ipc.h"
#include "ngspice/ipcproto.h"


#define  IPC_STDOUT_FILE_NAME  "/usr/tmp/atesse_xspice.out"
#define  IPC_STDERR_FILE_NAME  "/usr/tmp/atesse_xspice.err"


/*
Ipc_Vtrans_t is used by functions that return results to translate
voltage source names to the names of the devices they monitor.
This table is built from #VTRANS cards in the incoming deck and
is provided for ATESSE 1.0 compatibility.
*/

typedef struct {
    int     size;               /* Size of arrays */
    char    **vsrc_name;        /* Array of voltage source name prefixes */
    char    **device_name;      /* Array of device names the vsources map to */
} Ipc_Vtrans_t;


/*
Ipc_Tiein_t is used by the SPICE mods that take care of interprocess communications
activities.
*/

typedef struct {

    Ipc_Boolean_t   enabled;      /* True if we are using IPC */
    Ipc_Mode_t      mode;         /* INTERACTIVE or BATCH mode */
    Ipc_Anal_t      anal_type;    /* DCOP, AC, ... mode */
    Ipc_Boolean_t   syntax_error; /* True if error occurred during parsing */
    Ipc_Boolean_t   run_error;    /* True if error occurred during simulation */
    Ipc_Boolean_t   errchk_sent;  /* True if #ERRCHK has been sent */
    Ipc_Boolean_t   returni;      /* True if simulator should return currents */
    double          mintime;      /* Minimum time between timepoints returned */
    double          last_time;    /* Last timepoint returned */
    double          cpu_time;     /* CPU time used during simulation */
    Ipc_Boolean_t   *send;        /* Used by OUTinterface to determine what to send */
    char            *log_file;    /* Path to write log file */
    Ipc_Vtrans_t    vtrans;       /* Used by OUTinterface to translate v sources */
    Ipc_Boolean_t   stop_analysis; /* True if analysis should be terminated */

} Ipc_Tiein_t;



extern  Ipc_Tiein_t  g_ipc;
extern bool wantevtdata;


#endif

