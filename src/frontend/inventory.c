/**********
Copyright 2010 Paolo Nenzi.  All rights reserved.
Author:   2010 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/optdefs.h"
#include "ngspice/dvec.h"
#include "ftehelp.h"
#include "ngspice/hlpdefs.h"

#include "circuits.h"
#include "where.h"

/*
 The inventory command shows the number of instances for each device
 in the current circuit.
*/

void
com_inventory(wordlist *wl)
{
    CKTcircuit *circuit  = NULL;
    STATistics *stat     = NULL;
    STATdevList *devList = NULL;
    int k;

    NG_IGNORE(wl);

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "There is no current circuit\n");
        return;
    }

    circuit = ft_curckt->ci_ckt;
    stat    = circuit->CKTstat;
    devList = stat->STATdevNum;

    out_init();
    out_send("\nCircuit Inventory\n\n");
    for (k = 0; k < ft_sim->numDevices; k++)
        if (ft_sim->devices[k] && devList[k].instNum > 0)
            out_printf("%s: %d\n",
                       ft_sim->devices[k]->name,
                       devList[k].instNum);
    out_send("\n");
}
