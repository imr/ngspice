/*============================================================================
FILE    MIFload.c

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

    This file contains the driver function for calling code model evaluation
    functions.  This is one of the most important, complex, and often called
    functions in the model interface package.  It iterates through all models
    and all instances of a specified code model device type, fills in the
    inputs for the model, calls the model, and then uses the outputs and
    partials returned by the model to load the matrix.

INTERFACES

    MIFload()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/


/* #include "prefix.h"  */
#include "ngspice/ngspice.h"

#include <stdio.h>
#include <math.h>

#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#include "ngspice/evt.h"

#include "ngspice/mifproto.h"
#include "ngspice/mifparse.h"
#include "ngspice/mifdefs.h"
#include "ngspice/mifcmdat.h"
#include "ngspice/mif.h"

#include "ngspice/enh.h"
#include "ngspice/cm.h"

/*  #include "suffix.h"  */



static void MIFauto_partial(
    MIFinstance     *here,
    void            (*cm_func) (Mif_Private_t *),
    Mif_Private_t   *cm_data
);






/*
MIFload

This function is called by the CKTload() driver function to call
the C function for each instance of a code model type.  It loops
through all models of that type and all instances of each model.
For each instance, it prepares the structure that is passed to
the code model by filling it with the input values for that
instance.  The code model's C function is then called, and the
outputs and partial derivatives computed by the C function are
used to fill the matrix for the next solution attempt.
*/


int
MIFload(
    GENmodel      *inModel,  /* The head of the model list */
    CKTcircuit    *ckt)      /* The circuit structure */
{

    MIFmodel    *model;
    MIFinstance *here;

    Mif_Private_t   cm_data;   /* data to be passed to/from code model */
    Mif_Port_Type_t type;
    Mif_Port_Data_t *fast;

    Mif_Smp_Ptr_t  *smp_data_out;

    Mif_Port_Ptr_t  *smp_ptr;

    Mif_Port_Type_t in_type;
    Mif_Port_Type_t out_type;

    Mif_Boolean_t  is_input;
    Mif_Boolean_t  is_output;

    Mif_Cntl_Src_Type_t  cntl_src_type;

    Mif_Analysis_t  anal_type;

    Mif_Complex_t   czero;
    Mif_Complex_t   ac_gain;

    int         mod_type;
    int         num_conn;
    int         num_port;
    int         num_port_k;
    int         i;
    int         j;
    int         k;
    int         l;

    /*int         tag;*/

    double      *rhs;
    double      *rhsOld;
    double      partial;
    double      temp;

    double      *double_ptr0;
    double      *double_ptr1;

    /*double      *input;*/
    /*    double      *oldinput;*/

    char        *byte_ptr0;
    char        *byte_ptr1;

    double      last_input;
    double      conv_limit;

    double      cntl_input;


    Evt_Node_Data_t     *node_data;


    /* Prepare a zero complex number for AC gain initializations */
    czero.real = 0.0;
    czero.imag = 0.0;

    /* Setup for access into MIF specific model data */
    model = (MIFmodel *) inModel;
    mod_type = model->MIFmodType;

    /* Setup pointers for fast access to rhs and rhsOld elements of ckt struct */
    rhs = ckt->CKTrhs;
    rhsOld = ckt->CKTrhsOld;

    node_data = ckt->evt->data.node;

    /* *********************************************************************** */
    /* Setup the circuit data in the structure to be passed to the code models */
    /* *********************************************************************** */

    /* anal_init is set if this is the first iteration at any step in */
    /* an analysis */
    if(!(ckt->CKTmode & MODEINITFLOAT))
        g_mif_info.circuit.anal_init = MIF_TRUE;
    cm_data.circuit.anal_init = g_mif_info.circuit.anal_init;

    /* anal_type is determined by CKTload */
    anal_type = g_mif_info.circuit.anal_type;
    cm_data.circuit.anal_type = anal_type;

    /* get the analysis freq from the ckt struct if this is an AC analysis */
    /* otherwise, set the freq to zero */
    if(anal_type == MIF_AC)
       cm_data.circuit.frequency = ckt->CKTomega;
    else
       cm_data.circuit.frequency = 0.0;

    /* get the analysis times from the ckt struct if this is a transient analysis */
    /* otherwise, set the times to zero */
    if(anal_type == MIF_TRAN) {
       cm_data.circuit.time = ckt->CKTtime;
       cm_data.circuit.t[0] = ckt->CKTtime;
       for(i = 1; i < 8; i++) {
           cm_data.circuit.t[i] = cm_data.circuit.t[i-1] - ckt->CKTdeltaOld[i-1];
           if(cm_data.circuit.t[i] < 0.0)
              cm_data.circuit.t[i] = 0.0;
       }
    }
    else {
       cm_data.circuit.time = 0.0;
       for(i = 0; i < 8; i++) {
           cm_data.circuit.t[i] = 0.0;
       }
    }
    
    cm_data.circuit.call_type = MIF_ANALOG;
    cm_data.circuit.temperature = ckt->CKTtemp - 273.15;

    g_mif_info.circuit.call_type = MIF_ANALOG;
    g_mif_info.ckt = ckt;


    /* ***************************************************************** */
    /* loop through all models of this type */
    /* ***************************************************************** */
    for( ; model != NULL; model = MIFnextModel(model)) {
      
        /* If not an analog or hybrid model, continue to next */
        if(! model->analog)
	  continue;
	
        /* ***************************************************************** */
        /* loop through all instances of this model */
        /* ***************************************************************** */
        for(here = MIFinstances(model); here != NULL; here = MIFnextInstance(here)) {
            /* If not an analog or hybrid instance, continue to next */
            if(! here->analog)
                continue;

            /* ***************************************************************** */
            /* Prepare the data needed by the cm_.. functions                    */
            /* ***************************************************************** */
            g_mif_info.instance = here;
            g_mif_info.errmsg = "";

            if(here->initialized) {
                cm_data.circuit.init = MIF_FALSE;
                g_mif_info.circuit.init = MIF_FALSE;
            }
            else {
                cm_data.circuit.init = MIF_TRUE;
                g_mif_info.circuit.init = MIF_TRUE;
            }


            /* ***************************************************************** */
            /* if tran analysis and anal_init is true, copy state 1 to state 0   */
            /* Otherwise the data in state 0 would be invalid                    */
            /* ***************************************************************** */

            if((anal_type == MIF_TRAN) && g_mif_info.circuit.anal_init) {
                for(i = 0; i < here->num_state; i++) {
                    double_ptr0 = ckt->CKTstate0 + here->state[i].index;
                    double_ptr1 = ckt->CKTstate1 + here->state[i].index;
                    byte_ptr0   = (char *) double_ptr0;
                    byte_ptr1   = (char *) double_ptr1;
                    for(j = 0; j < here->state[i].bytes; j++)
                        byte_ptr0[j] = byte_ptr1[j];
                }
            }

            /* ***************************************************************** */
            /* If not AC analysis, loop through all connections on this instance */
            /* and load the input values for each input port of each connection  */
            /* ***************************************************************** */

            num_conn = here->num_conn;
            for(i = 0; i < num_conn; i++) {

                /* If AC analysis, skip getting input values.  The input values */
                /* should stay the same as they were at the last iteration of   */
                /* the operating point analysis */
                if(anal_type == MIF_AC)
                    break;

                /* if the connection is null, skip to next connection */
                if(here->conn[i]->is_null)
                    continue;

                /* if this connection is not an input, skip to next connection */
                if(! here->conn[i]->is_input)
                    continue;

                /* Get number of ports on this connection */
                num_port = here->conn[i]->size;

                /* loop through all ports on this connection */
                for(j = 0; j < num_port; j++) {

                    /*setup a pointer for fast access to port data */
                    fast = here->conn[i]->port[j];

                    /* skip if this port is null */
                    if(fast->is_null)
                        continue;

                    /* determine the type of this port */
                    type = fast->type;

                    /* If port type is Digital or User-Defined, we only need */
                    /* to get the total load.  The input values are pointers */
                    /* already set by EVTsetup() */
                    if((type == MIF_DIGITAL) || (type == MIF_USER_DEFINED)) {
                        fast->total_load =
                                node_data->total_load[fast->evt_data.node_index];
                    }
                    /* otherwise, it is an analog node and we get the input value */
                    else {
                        /* load the input values based on type and mode */
                        if(ckt->CKTmode & MODEINITJCT)
                            /* first iteration step for DC */
                            fast->input.rvalue = 0.0;
                        else if((ckt->CKTmode & MODEINITTRAN) ||
                                (ckt->CKTmode & MODEINITPRED))
                            /* first iteration step at timepoint */
                            fast->input.rvalue = ckt->CKTstate1[fast->old_input];
                        else {
                            /* subsequent iterations */

                            /* record last iteration's input value for convergence limiting */
                            last_input = fast->input.rvalue;

                            /* get the new input value */
                            switch(type) {
                            case MIF_VOLTAGE:
                            case MIF_DIFF_VOLTAGE:
                            case MIF_CONDUCTANCE:
                            case MIF_DIFF_CONDUCTANCE:
                                fast->input.rvalue = rhsOld[fast->smp_data.pos_node] -
                                                      rhsOld[fast->smp_data.neg_node];
                                break;
                            case MIF_CURRENT:
                            case MIF_DIFF_CURRENT:
                            case MIF_VSOURCE_CURRENT:
                            case MIF_RESISTANCE:
                            case MIF_DIFF_RESISTANCE:
                                fast->input.rvalue = rhsOld[fast->smp_data.ibranch];
                                break;
			    case MIF_DIGITAL:
			    case MIF_USER_DEFINED:
			      break;
                            } /* end switch on type of port */

                            /* If convergence limiting enabled, limit maximum input change */
                            if(ckt->enh->conv_limit.enabled) {
                                /* compute the maximum the input is allowed to change */
                                conv_limit = fabs(last_input) * ckt->enh->conv_limit.step;
                                if(conv_limit < ckt->enh->conv_limit.abs_step)
                                    conv_limit = ckt->enh->conv_limit.abs_step;
                                /* if input has changed too much, limit it and signal not converged */
                                if(fabs(fast->input.rvalue - last_input) > conv_limit) {
                                    if((fast->input.rvalue - last_input) > 0.0)
                                        fast->input.rvalue = last_input + conv_limit;
                                    else
                                        fast->input.rvalue = last_input - conv_limit;
                                    (ckt->CKTnoncon)++;
                                    /* report convergence problem if last call */
                                    if(ckt->enh->conv_debug.report_conv_probs) {
                                        ENHreport_conv_prob(ENH_ANALOG_INSTANCE,
                                                            here->MIFname, "");
                                    }
                                }
                            }

                        } /* end else */

                        /* Save value of input for use with MODEINITTRAN */
                        ckt->CKTstate0[fast->old_input] = fast->input.rvalue;

                    } /* end else analog type */
                } /* end for number of ports */
            } /* end for number of connections */

            /* ***************************************************************** */
            /* loop through all connections on this instance and zero out all    */
            /* outputs/partials/AC gains for each output port of each connection */
            /* ***************************************************************** */
            num_conn = here->num_conn;
            for(i = 0; i < num_conn; i++) {

                /* if the connection is null or is not an output */
                /* skip to next connection */
                if(here->conn[i]->is_null || (! here->conn[i]->is_output))
                    continue;

                /* loop through all ports on this connection */
                num_port = here->conn[i]->size;
                for(j = 0; j < num_port; j++) {

                    /*setup a pointer for fast access to port data */
                    fast = here->conn[i]->port[j];

                    /* skip if this port is null */
                    if(fast->is_null)
                        continue;

                    /* determine the type of this port */
                    type = fast->type;

                    /* If not an analog node, continue to next port */
                    if((type == MIF_DIGITAL) || (type == MIF_USER_DEFINED))
                        continue;

                    /* initialize the output to zero */
                    fast->output.rvalue = 0.0;

                    /* loop through all connections and ports that */
                    /* could be inputs for this port and zero the partials */
                    for(k = 0; k < num_conn; k++) {
                        if(here->conn[k]->is_null || (! here->conn[k]->is_input))
                            continue;
                        num_port_k = here->conn[k]->size;
                        for(l = 0; l < num_port_k; l++) {
                            /* skip if this port is null */
                            if(here->conn[k]->port[l]->is_null)
                                continue;
                            fast->partial[k].port[l] = 0.0;
                            fast->ac_gain[k].port[l] = czero;
                        } /* end for number of ports */
                    } /* end for number of connections */
                } /* end for number of ports */
            } /* end for number of connections */


            /* ***************************************************************** */
            /* Prepare the structure to be passed to the code model */
            /* ***************************************************************** */
            cm_data.num_conn = here->num_conn;
            cm_data.conn = here->conn;
            cm_data.num_param = here->num_param;
            cm_data.param = here->param;
            cm_data.num_inst_var = here->num_inst_var;
            cm_data.inst_var = here->inst_var;
            cm_data.callback = &(here->callback);

            /* Initialize the auto_partial flag to false */
            g_mif_info.auto_partial.local = MIF_FALSE;

            /* ******************* */
            /* Call the code model */
            /* ******************* */
            DEVices[mod_type]->DEVpublic.cm_func (&cm_data);

            /* Automatically compute partials if requested by .options auto_partial */
            /* or by model through call to cm_analog_auto_partial() in DC or TRAN analysis */
            if((anal_type != MIF_AC) &&
               (g_mif_info.auto_partial.global || g_mif_info.auto_partial.local))
                    MIFauto_partial(here, DEVices[mod_type]->DEVpublic.cm_func, &cm_data);

            /* ***************************************************************** */
            /* Loop through all connections on this instance and */
            /* load the data into the matrix for each output port */
            /* and for each V source associated with a current input. */
            /* For AC analysis, we only load the +-1s required to satisfy */
            /* KCL and KVL in the matrix equations.  */
            /* ***************************************************************** */

            num_conn = here->num_conn;
            for(i = 0; i < num_conn; i++) {

                /* if the connection is null, skip to next connection */
                if(here->conn[i]->is_null)
                    continue;

                /* prepare things for convenient access later */
                is_input = here->conn[i]->is_input;
                is_output = here->conn[i]->is_output;

                /* loop through all ports on this connection */
                num_port = here->conn[i]->size;
                for(j = 0; j < num_port; j++) {

                    /*setup a pointer for fast access to port data */
                    fast = here->conn[i]->port[j];

                    /* skip if this port is null */
                    if(fast->is_null)
                        continue;

                    /* determine the type of this port */
                    type = fast->type;

                    /* If not an analog node, continue to next port */
                    if((type == MIF_DIGITAL) || (type == MIF_USER_DEFINED))
                        continue;

                    /* create a pointer to the smp data for quick access */
                    smp_data_out = &(fast->smp_data);

                    /* if it is a current input */
                    /* load the matrix data needed for the associated zero-valued V source */
                    if(is_input && (type == MIF_CURRENT || type == MIF_DIFF_CURRENT)) {
                        *(smp_data_out->pos_ibranch) += 1.0;
                        *(smp_data_out->neg_ibranch) -= 1.0;
                        *(smp_data_out->ibranch_pos) += 1.0;
                        *(smp_data_out->ibranch_neg) -= 1.0;
                        /* rhs[smp_data_out->ibranch] += 0.0; */
                    } /* end if current input */

                    /* if it has a voltage source output, */
                    /* load the matrix with the V source output data */
                    if( (is_output && (type == MIF_VOLTAGE || type == MIF_DIFF_VOLTAGE)) ||
                                     (type == MIF_RESISTANCE || type == MIF_DIFF_RESISTANCE) ) {
                        *(smp_data_out->pos_branch) += 1.0;
                        *(smp_data_out->neg_branch) -= 1.0;
                        *(smp_data_out->branch_pos) += 1.0;
                        *(smp_data_out->branch_neg) -= 1.0;
                        if(anal_type != MIF_AC)
                           rhs[smp_data_out->branch] += fast->output.rvalue;
                    } /* end if V source output */

                    /* if it has a current source output, */
                    /* load the matrix with the V source output data */
                    if( (is_output && (type == MIF_CURRENT || type == MIF_DIFF_CURRENT)) ||
                                     (type == MIF_CONDUCTANCE || type == MIF_DIFF_CONDUCTANCE) ) {
                        if(anal_type != MIF_AC) {
                           rhs[smp_data_out->pos_node] -= fast->output.rvalue;
                           rhs[smp_data_out->neg_node] += fast->output.rvalue;
                        }
                    } /* end if current output */

                } /* end for number of ports */
            } /* end for number of connections */


            /* ***************************************************************** */
            /* loop through all output connections on this instance and */
            /* load the partials/AC gains into the matrix */
            /* ***************************************************************** */
            for(i = 0; i < num_conn; i++) {

                /* if the connection is null or is not an output */
                /* skip to next connection */
                if((here->conn[i]->is_null) || (! here->conn[i]->is_output))
                    continue;

                /* loop through all ports on this connection */
                num_port = here->conn[i]->size;
                for(j = 0; j < num_port; j++) {

                    /*setup a pointer for fast access to port data */
                    fast = here->conn[i]->port[j];

                    /* skip if this port is null */
                    if(fast->is_null)
                        continue;

                    /* determine the type of this output port */
                    out_type = fast->type;

                    /* If not an analog node, continue to next port */
                    if((out_type == MIF_DIGITAL) || (out_type == MIF_USER_DEFINED))
                        continue;

                    /* create a pointer to the smp data for quick access */
                    smp_data_out = &(fast->smp_data);

                    /* for this port, loop through all connections */
                    /* and all ports to touch on each possible input */
                    for(k = 0; k < num_conn; k++) {

                        /* if the connection is null or is not an input */
                        /* skip to next connection */
                        if((here->conn[k]->is_null) || (! here->conn[k]->is_input))
                            continue;

                        num_port_k = here->conn[k]->size;
                        /* loop through all the ports of this connection */
                        for(l = 0; l < num_port_k; l++) {

                            /* skip if this port is null */
                            if(here->conn[k]->port[l]->is_null)
                                continue;

                            /* determine the type of this input port */
                            in_type = here->conn[k]->port[l]->type;

                            /* If not an analog node, continue to next port */
                            if((in_type == MIF_DIGITAL) || (in_type == MIF_USER_DEFINED))
                                continue;

                            /* get the partial to local variable for fast access */
                            partial = fast->partial[k].port[l];
                            ac_gain = fast->ac_gain[k].port[l];

                            /* create a pointer to the matrix pointer data for quick access */
                            smp_ptr = &(smp_data_out->input[k].port[l]);

                            /* get the input value */
                            cntl_input = here->conn[k]->port[l]->input.rvalue;

                            /* determine type of controlled source according */
                            /* to input and output types */
                            cntl_src_type = MIFget_cntl_src_type(in_type, out_type);

                            switch(cntl_src_type) {
                            case MIF_VCVS:
                                if(anal_type == MIF_AC) {
                                   smp_ptr->e.branch_poscntl[0] -= ac_gain.real;
                                   smp_ptr->e.branch_negcntl[0] += ac_gain.real;
                                   smp_ptr->e.branch_poscntl[1] -= ac_gain.imag;
                                   smp_ptr->e.branch_negcntl[1] += ac_gain.imag;
                                }
                                else {
                                   smp_ptr->e.branch_poscntl[0] -= partial;
                                   smp_ptr->e.branch_negcntl[0] += partial;
                                   rhs[smp_data_out->branch] -= partial * cntl_input;
                                }
                                break;
                            case MIF_ICIS:
                                if(anal_type == MIF_AC) {
                                   smp_ptr->f.pos_ibranchcntl[0] += ac_gain.real;
                                   smp_ptr->f.neg_ibranchcntl[0] -= ac_gain.real;
                                   smp_ptr->f.pos_ibranchcntl[1] += ac_gain.imag;
                                   smp_ptr->f.neg_ibranchcntl[1] -= ac_gain.imag;
                                }
                                else {
                                   smp_ptr->f.pos_ibranchcntl[0] += partial;
                                   smp_ptr->f.neg_ibranchcntl[0] -= partial;
                                   temp = partial * cntl_input;
                                   rhs[smp_data_out->pos_node] += temp;
                                   rhs[smp_data_out->neg_node] -= temp;
                                }
                                break;
                            case MIF_VCIS:
                                if(anal_type == MIF_AC) {
                                   smp_ptr->g.pos_poscntl[0] += ac_gain.real;
                                   smp_ptr->g.pos_negcntl[0] -= ac_gain.real;
                                   smp_ptr->g.neg_poscntl[0] -= ac_gain.real;
                                   smp_ptr->g.neg_negcntl[0] += ac_gain.real;
                                   smp_ptr->g.pos_poscntl[1] += ac_gain.imag;
                                   smp_ptr->g.pos_negcntl[1] -= ac_gain.imag;
                                   smp_ptr->g.neg_poscntl[1] -= ac_gain.imag;
                                   smp_ptr->g.neg_negcntl[1] += ac_gain.imag;
                                }
                                else {
                                   smp_ptr->g.pos_poscntl[0] += partial;
                                   smp_ptr->g.pos_negcntl[0] -= partial;
                                   smp_ptr->g.neg_poscntl[0] -= partial;
                                   smp_ptr->g.neg_negcntl[0] += partial;
                                   temp = partial * cntl_input;
                                   rhs[smp_data_out->pos_node] += temp;
                                   rhs[smp_data_out->neg_node] -= temp;
                                }
                                break;
                            case MIF_ICVS:
                                if(anal_type == MIF_AC) {
                                   smp_ptr->h.branch_ibranchcntl[0] -= ac_gain.real;
                                   smp_ptr->h.branch_ibranchcntl[1] -= ac_gain.imag;
                                }
                                else {
                                   smp_ptr->h.branch_ibranchcntl[0] -= partial;
                                   rhs[smp_data_out->branch] -= partial * cntl_input;
                                }
                                break;
                            case MIF_minus_one:
                                break;
                            } /* end switch on controlled source type */
                        } /* end for number of input ports */
                    } /* end for number of input connections */
                } /* end for number of output ports */
            } /* end for number of output connections */

            here->initialized = MIF_TRUE;

        } /* end for all instances */

    } /* end for all models */

    return(OK);
}




/*
MIFauto_partial

This function is called by MIFload() when a code model requests
that partial derivatives be computed automatically.  It calls
the code model additional times with an individual input to the
model varied by a small amount at each call.  Partial
derivatives of each output with respect to the varied input
are then computed by divided differences.
*/


static void MIFauto_partial(
    MIFinstance     *here,         /* The instance structure */
    void            (*cm_func) (Mif_Private_t *),  /* The code model function to be called */
    Mif_Private_t   *cm_data)      /* The data to be passed to the code model */
{

    Mif_Port_Data_t *fast;
    Mif_Port_Data_t *out_fast;

    Mif_Port_Type_t type;
    Mif_Port_Type_t out_type;

    int         num_conn;
    int         num_port;
    int         num_port_k;

    int         i;
    int         j;
    int         k;
    int         l;

    double      epsilon;
    double      nominal_input;


    /* Reset init and anal_init flags before making additional calls */
    /* to the model */
    cm_data->circuit.init = MIF_FALSE;
    g_mif_info.circuit.init = MIF_FALSE;

    cm_data->circuit.anal_init = MIF_FALSE;
    g_mif_info.circuit.anal_init = MIF_FALSE;


    /* *************************** */
    /* Save nominal analog outputs */
    /* *************************** */

    /* loop through all connections */
    num_conn = here->num_conn;
    for(i = 0; i < num_conn; i++) {

        /* if the connection is null or is not an output */
        /* skip to next connection */
        if(here->conn[i]->is_null || (! here->conn[i]->is_output))
            continue;

        /* loop through all ports on this connection */
        num_port = here->conn[i]->size;
        for(j = 0; j < num_port; j++) {

            /*setup a pointer for fast access to port data */
            fast = here->conn[i]->port[j];

            /* skip if this port is null */
            if(fast->is_null)
                continue;

            /* determine the type of this port */
            type = fast->type;

            /* If not an analog port, continue to next port */
            if((type == MIF_DIGITAL) || (type == MIF_USER_DEFINED))
                continue;

            /* copy the output for use in computing output deltas */
            fast->nominal_output = fast->output.rvalue;

        } /* end for number of output ports */
    } /* end for number of output connections */


    /* ***************************************************************** */
    /* Change each analog input by a small amount and call the model to  */
    /* compute new outputs.                                              */
    /* ***************************************************************** */

    /* loop through all connections */
    num_conn = here->num_conn;
    for(i = 0; i < num_conn; i++) {

        /* if the connection is null, skip to next connection */
        if(here->conn[i]->is_null)
            continue;

        /* if this connection is not an input, skip to next connection */
        if(! here->conn[i]->is_input)
            continue;

        /* Get number of ports on this connection */
        num_port = here->conn[i]->size;

        /* loop through all ports on this connection */
        for(j = 0; j < num_port; j++) {

            /*setup a pointer for fast access to port data */
            fast = here->conn[i]->port[j];

            /* skip if this port is null */
            if(fast->is_null)
                continue;

            /* determine the type of this port */
            type = fast->type;

            /* If port type is Digital or User-Defined, skip it */
            if((type == MIF_DIGITAL) || (type == MIF_USER_DEFINED))
                continue;

            /* otherwise, it is an analog port and we need to perturb it and */
            /* then call the model */

            /* compute the perturbation amount depending on type of input */
            switch(type) {
                case MIF_VOLTAGE:
                case MIF_DIFF_VOLTAGE:
                case MIF_CONDUCTANCE:
                case MIF_DIFF_CONDUCTANCE:
                    epsilon = 1.0e-6;
                    break;

                case MIF_CURRENT:
                case MIF_DIFF_CURRENT:
                case MIF_VSOURCE_CURRENT:
                case MIF_RESISTANCE:
                case MIF_DIFF_RESISTANCE:
                    epsilon = 1.0e-12;
                    break;

                default:
                    printf("INTERNAL ERROR - MIFauto_partial.  Invalid port type\n");
                    epsilon = 1.0e-30;
                    break;
            } /* end switch on type of port */

            /* record and perturb input value */
            nominal_input = fast->input.rvalue;
            fast->input.rvalue += epsilon;


            /* call model to compute new outputs */
            cm_func (cm_data);


            /* ******************************************************* */
            /* Compute the partials of each output with respect to the */
            /* perturbed input by divided differences.                 */
            /* ******************************************************* */

            /* loop through all analog output connections */
            for(k = 0; k < num_conn; k++) {

                /* if the connection is null or is not an output */
                /* skip to next connection */
                if((here->conn[k]->is_null) || (! here->conn[k]->is_output))
                    continue;

                /* loop through all the ports of this connection */
                num_port_k = here->conn[k]->size;
                for(l = 0; l < num_port_k; l++) {

                    /*setup a pointer for out_fast access to port data */
                    out_fast = here->conn[k]->port[l];

                    /* skip if this port is null */
                    if(out_fast->is_null)
                        continue;

                    /* determine the out_type of this port */
                    out_type = out_fast->type;

                    /* If port type is Digital or User-Defined, skip it */
                    if((out_type == MIF_DIGITAL) || (out_type == MIF_USER_DEFINED))
                        continue;

                    /* compute partial by divided differences */
                    out_fast->partial[i].port[j] =
                        (out_fast->output.rvalue - out_fast->nominal_output) / epsilon;

                    /* zero the output in preparation for next call */
                    out_fast->output.rvalue = 0.0;

                } /* end for number of output ports */
            } /* end for number of output connections */

            /* restore nominal input value */
            fast->input.rvalue = nominal_input;

        } /* end for number of input ports */
    } /* end for number of input connections */


    /* *************************************************** */
    /* Call model one last time to recompute nominal case. */
    /* *************************************************** */

    /* This is needed even though the outputs are recorded, because */
    /* the model may compute other state values that cannot be restored */
    /* to the nominal condition from here */

    cm_func (cm_data);

}



