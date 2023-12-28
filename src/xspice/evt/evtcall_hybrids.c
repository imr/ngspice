/*============================================================================
FILE    EVTcall_hybrids.c

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

    This file contains function EVTcall_hybrids which calls all models
    which have both analog and event-driven ports or have declared
    themselves to be irreversible (no back-out).  It is called following
    successful evaluation of an analog iteration attempt to allow
    events to be scheduled by the hybrid models.  The 'CALL_TYPE' is set
    to 'EVENT_DRIVEN' or 'STEP_PENDING' when the model is called
    from this function.

INTERFACES

    void EVTcall_hybrids(CKTcircuit *ckt)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"

#include "ngspice/evtproto.h"


/*
EVTcall_hybrids

This function calls all the hybrid instances.  It is called following
the successful evaluation of an analog iteration.
*/


void EVTcall_hybrids(
    CKTcircuit  *ckt)    /* the main circuit structure */
{

    int           i;
    int           num_hybrids;
    MIFinstance **hybrids;


    /* Get needed data for fast access */

    num_hybrids = ckt->evt->counts.num_hybrids;
    hybrids = ckt->evt->info.hybrids;

    /* Call EVTload for all hybrids */

    for(i = 0; i < num_hybrids; i++) {
        EVTload_with_event(ckt, hybrids[i], MIF_STEP_PENDING);
        if (g_mif_info.breakpoint.current < ckt->CKTtime) {
            /* An XSPICE instance rejected the time-step. */

            break;
        }
    }
}
