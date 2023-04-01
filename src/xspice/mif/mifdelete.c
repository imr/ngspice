/*============================================================================
FILE    MIFdelete.c

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

    This file contains the function used by SPICE to delete an
    instance and its allocated data structures from the internal
    circuit description data structures.

INTERFACES

    MIFdelete()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"
#include "ngspice/sperror.h"
#include "ngspice/gendefs.h"

#include "ngspice/mifproto.h"
#include "ngspice/mifdefs.h"
#include "ngspice/evt.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include "ngspice/mifparse.h"
#endif

#include "ngspice/suffix.h"
#include "ngspice/devdefs.h"


/*
MIFdelete

This function deletes a particular instance from the linked list
of instance structures, freeing all dynamically allocated memory
used by the instance structure.
*/

int
MIFdelete(GENinstance *gen_inst)
{
    int         i;
    int         j;
    int         k;

    int         num_conn;
    int         num_port;
    int         num_inst_var;

    MIFinstance *here = (MIFinstance *) gen_inst;

    /*******************************************/
    /* instance->callback(..., MIF_CB_DESTROY) */
    /*******************************************/

    if (here->callback) {
        Mif_Private_t cm_data;

        /* Prepare the structure to be passed to the code model */
        cm_data.num_conn = here->num_conn;
        cm_data.conn = here->conn;
        cm_data.num_param = here->num_param;
        cm_data.param = here->param;
        cm_data.num_inst_var = here->num_inst_var;
        cm_data.inst_var = here->inst_var;
        cm_data.callback = &(here->callback);

        here->callback(&cm_data, MIF_CB_DESTROY);
    }

    /*******************************/
    /* Free the instance structure */
    /*******************************/

    /* Loop through all connections on the instance */
    /* and dismantle the stuff allocated during readin/setup */
    /* in MIFinit_inst, MIFget_port, and MIFsetup   */

    num_conn = here->num_conn;
    for (i = 0; i < num_conn; i++) {

        /* If connection never used, skip it */
        if (here->conn[i]->is_null)
            continue;

        /* If analog output, lots to free... */
        if (here->conn[i]->is_output && here->analog) {
            num_port = here->conn[i]->size;
            /* For each port on the connector */
            for (j = 0; j < num_port; j++) {
                /* Free the partial/ac_gain/smp stuff allocated in MIFsetup */
                for (k = 0; k < num_conn; k++) {
                    if ((here->conn[k]->is_null) || (! here->conn[k]->is_input))
                        continue;
                    if (here->conn[i]->port[j]->partial)
                        FREE(here->conn[i]->port[j]->partial[k].port);
                    if (here->conn[i]->port[j]->ac_gain)
                        FREE(here->conn[i]->port[j]->ac_gain[k].port);
                    if (here->conn[i]->port[j]->smp_data.input)
                        FREE(here->conn[i]->port[j]->smp_data.input[k].port);
                }
                FREE(here->conn[i]->port[j]->partial);
                FREE(here->conn[i]->port[j]->ac_gain);
                FREE(here->conn[i]->port[j]->smp_data.input);
                /* but don't free strings.  They are either not owned */
                /* by the inst or are part of tokens.  SPICE3C1 never */
                /* frees tokens, so we don't either... */
            }
        }
        /* Free the basic port structure allocated in MIFget_port */
        num_port = here->conn[i]->size;
        for (j = 0; j < num_port; j++) {
            Evt_Output_Event_t *evt;

            /* Memory allocated in mif_inp2.c and evtload.c. */

            FREE(here->conn[i]->port[j]->type_str);
            evt = here->conn[i]->port[j]->next_event;
            if (evt) {
                FREE(evt->value);
                FREE(evt);
            }
            FREE(here->conn[i]->port[j]);
        }
        FREE(here->conn[i]->port);
    }

    /* Free the connector stuff allocated in MIFinit_inst */
    /* Don't free name/description!  They are not owned */
    /* by the instance */
    for (i = 0; i < num_conn; i++) {
        FREE(here->conn[i]);
    }
    FREE(here->conn);

    /* Loop through all instance variables on the instance */
    /* and free stuff */

    num_inst_var = here->num_inst_var;
    for (i = 0; i < num_inst_var; i++) {
        if (here->inst_var[i]->element != NULL) {
    /* Do not delete inst_var[i]->element if MS Windows and is_array==1.
       Memory is then allocated in the code model dll, and it cannot be
       guaranteed that it can be freed safely here! A small memory leak is created.
       FIXME
       Finally one has to free the memory in the same module where allocated. */
#if defined(_MSC_VER) || defined(__MINGW32__)
            if (!DEVices[MIFmodPtr(here)->MIFmodType]->DEVpublic.inst_var[i].is_array)
#endif
                FREE(here->inst_var[i]->element);
        }
        FREE(here->inst_var[i]);
    }
    FREE(here->inst_var);

    /* ************************************************************* */
    /* Dont free params here.  They are not currently implemented on */
    /* a per-instance basis, so their allocated space is owned by    */
    /* the parent model, not the instance. Param stuff will be freed */
    /* by MIFmDelete                                                 */
    /* ************************************************************* */

    /* Free the stuff used by the cm_... functions */

    if (here->num_state && here->state)
        FREE(here->state);
    if (here->num_intgr && here->intgr)
        FREE(here->intgr);
    if (here->num_conv && here->conv)
        FREE(here->conv);

    return OK;
}
