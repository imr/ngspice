/* ===========================================================================
FILE    CMmeters.c

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

    This file contains functions callable from code models.
    These functions are primarily designed for use by the
    "cmeter" and "lmeter" models provided in the XSPICE
    code model library.

INTERFACES

    cm_netlist_get_c()
    cm_netlist_get_l()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */
#include "ngspice/ngspice.h"
#include "ngspice/cm.h"
#include "ngspice/mif.h"

#include "ngspice/cktdefs.h"

#include "ngspice/mifdefs.h"
#include "cap/capdefs.h"
#include "ind/inddefs.h"
#include "vsrc/vsrcdefs.h"
#include "ngspice/inpdefs.h"



/*
cm_netlist_get_c()

This is a special function designed for use with the c_meter
model.  It returns the parallel combination of the capacitance
connected to the first port on the instance.
*/

double cm_netlist_get_c(void)
{
    CKTcircuit          *ckt;
    
    MIFinstance         *cmeter_inst;
    CAPinstance         *cap_inst;
    VSRCinstance        *vsrc_inst;

    CAPmodel            *cap_head;
    CAPmodel            *cap_model;
    VSRCmodel           *vsrc_head;
    VSRCmodel           *vsrc_model;

    int                 cap_type;
    int                 vsrc_type;

    int                 cmeter_node;
    int                 vsrc_node;

    double              c;


    /* Get the circuit data structure and current instance */
    ckt = g_mif_info.ckt;
    cmeter_inst = g_mif_info.instance;

    /* Get internal node number for positive node of cmeter input */
    cmeter_node = cmeter_inst->conn[0]->port[0]->smp_data.pos_node;

    /* Initialize total capacitance value to zero */
    c = 0.0;


    /* ****************************************************** */
    /* Look for capacitors connected directly to cmeter input */
    /* ****************************************************** */

    /* Get the head of the list of capacitor models in the circuit */
    cap_type = INPtypelook("Capacitor");
    if(cap_type < 0) {
        printf("\nERROR - Capacitor type not supported in this binary\n");
        return(0);
    }
    cap_head = (CAPmodel *) ckt->CKThead[cap_type];

    /* Scan through all capacitor instances and add in values */
    /* of any capacitors connected to cmeter input */

    for(cap_model = cap_head; cap_model; cap_model = CAPnextModel(cap_model)) {
        for(cap_inst = CAPinstances(cap_model);
                cap_inst;
                cap_inst = CAPnextInstance(cap_inst)) {
            if((cmeter_node == cap_inst->CAPposNode) ||
                    (cmeter_node == cap_inst->CAPnegNode)) {
                c += cap_inst->CAPcapac;
            }
        }
    }


    /* ***************************************************************** */
    /* Look for capacitors connected through zero-valued voltage sources */
    /* ***************************************************************** */

    /* Get the head of the list of voltage source models in the circuit */
    vsrc_type = INPtypelook("Vsource");
    if(vsrc_type < 0) {
        printf("\nERROR - Vsource type not supported in this binary\n");
        return(0);
    }
    vsrc_head = (VSRCmodel *) ckt->CKThead[vsrc_type];

    /* Scan through all voltage source instances and add in values */
    /* of any capacitors connected to cmeter input through voltage source */

    for(vsrc_model = vsrc_head; vsrc_model; vsrc_model = VSRCnextModel(vsrc_model)) {
        for(vsrc_inst = VSRCinstances(vsrc_model);
                vsrc_inst;
                vsrc_inst = VSRCnextInstance(vsrc_inst)) {

            /* Skip to next if not DC source with value = 0.0 */
            if((vsrc_inst->VSRCfunctionType != 0) ||
                    (vsrc_inst->VSRCdcValue != 0.0))
                continue;

            /* See if voltage source is connected to cmeter input */
            /* If so, get other node voltage source is connected to */
            /* If not, skip to next source */
            if(cmeter_node == vsrc_inst->VSRCposNode)
                vsrc_node = vsrc_inst->VSRCnegNode;
            else if(cmeter_node == vsrc_inst->VSRCnegNode)
                vsrc_node = vsrc_inst->VSRCposNode;
            else
                continue;


            /* Scan through all capacitor instances and add in values */
            /* of any capacitors connected to the voltage source node */

            for(cap_model = cap_head; cap_model; cap_model = CAPnextModel(cap_model)) {
                for(cap_inst = CAPinstances(cap_model);
                        cap_inst;
                        cap_inst = CAPnextInstance(cap_inst)) {
                    if((vsrc_node == cap_inst->CAPposNode) ||
                            (vsrc_node == cap_inst->CAPnegNode)) {
                        c += cap_inst->CAPcapac;
                    }
                }
            }


        } /* end for all vsrc instances */
    } /* end for all vsrc models */


    /* Return the total capacitance value */
    return(c);
}




/*
cm_netlist_get_l()

This is a special function designed for use with the l_meter
model.  It returns the equivalent value of inductance
connected to the first port on the instance.
*/


double cm_netlist_get_l(void)
{
    CKTcircuit          *ckt;
    
    MIFinstance         *lmeter_inst;
    INDinstance         *ind_inst;
    VSRCinstance        *vsrc_inst;

    INDmodel            *ind_head;
    INDmodel            *ind_model;
    VSRCmodel           *vsrc_head;
    VSRCmodel           *vsrc_model;

    int                 ind_type;
    int                 vsrc_type;

    int                 lmeter_node;
    int                 vsrc_node;

    double              l;


    /* Get the circuit data structure and current instance */
    ckt = g_mif_info.ckt;
    lmeter_inst = g_mif_info.instance;

    /* Get internal node number for positive node of lmeter input */
    lmeter_node = lmeter_inst->conn[0]->port[0]->smp_data.pos_node;

    /* Initialize total inductance to infinity */
    l = 1.0e12;


    /* ****************************************************** */
    /* Look for inductors connected directly to lmeter input */
    /* ****************************************************** */

    /* Get the head of the list of inductor models in the circuit */
    ind_type = INPtypelook("Inductor");
    if(ind_type < 0) {
        printf("\nERROR - Inductor type not supported in this binary\n");
        return(0);
    }
    ind_head = (INDmodel *) ckt->CKThead[ind_type];

    /* Scan through all inductor instances and add in values */
    /* of any inductors connected to lmeter input */

    for(ind_model = ind_head; ind_model; ind_model = INDnextModel(ind_model)) {
        for(ind_inst = INDinstances(ind_model);
                ind_inst;
                ind_inst = INDnextInstance(ind_inst)) {
            if((lmeter_node == ind_inst->INDposNode) ||
                    (lmeter_node == ind_inst->INDnegNode)) {
                l = 1.0 / ( (1.0 / l) + (1.0 / ind_inst->INDinduct) );
            }
        }
    }


    /* ***************************************************************** */
    /* Look for inductors connected through zero-valued voltage sources */
    /* ***************************************************************** */

    /* Get the head of the list of voltage source models in the circuit */
    vsrc_type = INPtypelook("Vsource");
    if(vsrc_type < 0) {
        printf("\nERROR - Vsource type not supported in this binary\n");
        return(0);
    }
    vsrc_head = (VSRCmodel *) ckt->CKThead[vsrc_type];

    /* Scan through all voltage source instances and add in values */
    /* of any inductors connected to lmeter input through voltage source */

    for(vsrc_model = vsrc_head; vsrc_model; vsrc_model = VSRCnextModel(vsrc_model)) {
        for(vsrc_inst = VSRCinstances(vsrc_model);
                vsrc_inst;
                vsrc_inst = VSRCnextInstance(vsrc_inst)) {

            /* Skip to next if not DC source with value = 0.0 */
            if((vsrc_inst->VSRCfunctionType != 0) ||
                    (vsrc_inst->VSRCdcValue != 0.0))
                continue;

            /* See if voltage source is connected to lmeter input */
            /* If so, get other node voltage source is connected to */
            /* If not, skip to next source */
            if(lmeter_node == vsrc_inst->VSRCposNode)
                vsrc_node = vsrc_inst->VSRCnegNode;
            else if(lmeter_node == vsrc_inst->VSRCnegNode)
                vsrc_node = vsrc_inst->VSRCposNode;
            else
                continue;


            /* Scan through all inductor instances and add in values */
            /* of any inductors connected to the voltage source node */

            for(ind_model = ind_head; ind_model; ind_model = INDnextModel(ind_model)) {
                for(ind_inst = INDinstances(ind_model);
                        ind_inst;
                        ind_inst = INDnextInstance(ind_inst)) {
                    if((vsrc_node == ind_inst->INDposNode) ||
                            (vsrc_node == ind_inst->INDnegNode)) {
                        l = 1.0 / ( (1.0 / l) + (1.0 / ind_inst->INDinduct) );
                    }
                }
            }


        } /* end for all vsrc instances */
    } /* end for all vsrc models */


    /* Return the total capacitance value */
    return(l);
}


