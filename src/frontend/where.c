/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ftehelp.h"
#include "ngspice/hlpdefs.h"

#include "circuits.h"
#include "where.h"


void
com_where(wordlist *wl)
{
    char    *msg;

    NG_IGNORE(wl);

    /*CDHW typing where with no current circuit caused crashes CDHW*/
    if (!ft_curckt) {
        fprintf(cp_err, "There is no current circuit\n");
        return;
    } else if (ft_curckt->ci_ckt != NULL) {
        fprintf(cp_err, "No unconverged node found.\n");
        return;
    }

    msg = ft_sim->nonconvErr (ft_curckt->ci_ckt, NULL);

    printf("%s", msg);

    /*
    if (ft_curckt) {
        msg = ft_sim->nonconvErr (ft_curckt->ci_ckt, 0);
        fprintf(cp_out, "%s", msg);
    } else {
        fprintf(cp_err, "Error: no circuit loaded.\n");
    }
    */
}
