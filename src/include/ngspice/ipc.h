/*============================================================================
FILE    IPC.h

MEMBER OF process XSPICE

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Steve Tynor

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    Provides compatibility for the new SPICE simulator to both the MSPICE user
    interface and BCP (via ATESSE v.1 style AEGIS mailboxes) and the new ATESSE
    v.2 Simulator Interface and BCP (via Bsd Sockets).

INTERFACES


REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#ifndef ngspice_IPC_H
#define ngspice_IPC_H


#define  IPC_MAX_LINE_LEN       80
#define  IPC_MAX_PATH_LEN      2048

/* Known socket port for server and client to communicate: */
#define SOCKET_PORT   1064

/* Recognition character for Beginning Of Line of message: */
#define BOL_CHAR  '\\'

/* Length (in bytes) of a socket message header:                             */
#define SOCK_MSG_HDR_LEN 5


typedef int Ipc_Boolean_t;

#define IPC_FALSE 0
#define IPC_TRUE  1
   
typedef struct {   /* Don't change this type!  It is cast elsewhere */
   double real;
   double imag;
} Ipc_Complex_t;

/*---------------------------------------------------------------------------*/
typedef enum {
   IPC_STATUS_OK,
   IPC_STATUS_NO_DATA,
   IPC_STATUS_END_OF_DECK,
   IPC_STATUS_EOF,
   IPC_STATUS_ERROR,
} Ipc_Status_t;

#if 0
/*---------------------------------------------------------------------------*/
typedef void* Ipc_Connection_t;
/*
 * A connection is an `opaque' type - the user has no access to the details of
 * the implementation. Indeed the details are different depending on whether
 * underlying transport mechanism is AEGIS Mailboxes or Bsd Sockets (or
 * something else...)
 */
#endif

/*---------------------------------------------------------------------------*/
typedef enum {
   IPC_WAIT,
   IPC_NO_WAIT,
} Ipc_Wait_t;

/*---------------------------------------------------------------------------*/
typedef enum {
   IPC_PROTOCOL_V1,     /* >DATAB records in ATESSE v.1 format
                         * Handles v.1 style logfile name passing protocol
                         */
   IPC_PROTOCOL_V2,     /* >DATAB records in ATESSE v.2 format
                         */
} Ipc_Protocol_t;

/*---------------------------------------------------------------------------*/
typedef enum {
   IPC_MODE_BATCH,
   IPC_MODE_INTERACTIVE,
} Ipc_Mode_t;


/*---------------------------------------------------------------------------*/
typedef enum {
   IPC_ANAL_DCOP,
   IPC_ANAL_DCTRCURVE,
   IPC_ANAL_AC,
   IPC_ANAL_TRAN,
#ifdef RFSPICE
   IPC_ANAL_SP,
#endif
   IPC_ANAL_NOI
} Ipc_Anal_t;



#endif
