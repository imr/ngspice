/*============================================================================
FILE    EVTnext_time.c

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

    This file contains function EVTnext_time which determines and
    returns the time of the next scheduled event on the inst and output
    queues.

INTERFACES

    double EVTnext_time(CKTcircuit *ckt)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
//#include "util.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"

#include "ngspice/evtproto.h"



/*
EVTnext_time

Get the next event time as the minimum of the next times
in the inst and output queues.  If no next time in either,
return machine infinity.
*/


double EVTnext_time(
    CKTcircuit *ckt)   /* The circuit structure */
{

    double  next_time;

    Evt_Inst_Queue_t    *inst_queue;
    Evt_Output_Queue_t  *output_queue;


    /* Initialize next time to machine infinity */
    next_time = 1e30;

    /* Get pointers for fast access */
    inst_queue = &(ckt->evt->queue.inst);
    output_queue = &(ckt->evt->queue.output);

    /* If anything pending in inst queue, set next time */
    /* to minimum of itself and the inst queue next time */
    if(inst_queue->num_pending)
        if(inst_queue->next_time < next_time)
            next_time = inst_queue->next_time;

    /* If anything pending in output queue, set next time */
    /* to minimum of itself and the output queue next time */
    if(output_queue->num_pending)
        if(output_queue->next_time < next_time)
            next_time = output_queue->next_time;

    return(next_time);
}
