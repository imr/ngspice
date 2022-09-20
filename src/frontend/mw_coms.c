/* Michael Widlok               2 Jun 1999 */
/* New commands for unloading circuits */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/ftedev.h"
#include "ngspice/ftedebug.h"
#include "ngspice/dvec.h"

#include "circuits.h"
#include "mw_coms.h"
#include "variable.h"
#include "runcoms.h"
#include "spiceif.h"

/* Clears ckt and removes current circ. form database */

void
com_removecirc(wordlist *wl)
{
    struct variable *v, *next;
    struct circ *ct;
    struct circ *caux = NULL;
    struct plot *p;
    struct plot *paux;
    int auxCir = 1, i, auxPlot;

    char* namecircuit;

    NG_IGNORE(wl);

    if (!ft_curckt) {
        fprintf(cp_err, "Warning: there is no circuit loaded.\n");
        fprintf(cp_err, "    Command 'remcirc' is ignored.\n");
        return;
    }

    ct = ft_curckt;

    if_cktfree(ct->ci_ckt, ct->ci_symtab);

    for (v = ct->ci_vars; v; v = next) {
        next = v->va_next;
        tfree(v);
    }

    /* PN FTESTATS*/
    tfree(ct->FTEstats);

    ct->ci_vars = NULL;
    caux = ft_circuits;
    namecircuit = copy(ft_curckt->ci_name);

    /* The circuit  being removed is the first loaded and you have more circuits */
    if (ft_curckt == ft_circuits  &&  ft_circuits->ci_next)
        ft_circuits = ft_circuits->ci_next;

    /* The circuit being removed id the first loaded and there are no more circuits */
    else if (ft_circuits->ci_next == NULL)
        ft_circuits = NULL;

    else {

        /* Run over the circuit list to find how many of them are
         * in front of the one to be removed
         */
        for (; ft_curckt != caux && caux; caux = caux->ci_next)
            auxCir++;

        caux = ft_circuits;

        /* Remove the circuit and move pointer to the next one */
        for (i = 1; i < auxCir-1; i++)
            caux = caux->ci_next;

        caux->ci_next = caux->ci_next->ci_next;
        /* ft_curckt = ft_circuits; */

    }


    /* If the plot is the first one and there are no other plots */
    if (!plot_list->pl_next && strcmp(plot_list->pl_title, namecircuit) == 0)
        plot_list = NULL;

    else if (plot_list && plot_list->pl_next) {
        p = plot_list;
        while (p) {
            auxPlot = 1;
            /* If the plot is in the first position */
            if (plot_list->pl_next && strcmp(plot_list->pl_title, namecircuit) == 0)
                plot_list = plot_list->pl_next;
            /* otherwise we run over the list of plots */
            else {
                for (; strcmp(p->pl_title, namecircuit) != 0 && p->pl_next; p = p->pl_next)
                    auxPlot++;
                if (strcmp(p->pl_title, namecircuit) == 0) {
                    paux = plot_list;
                    for (i = 1; i < auxPlot-1; i++)
                        paux = paux->pl_next;
                    paux->pl_next = paux->pl_next->pl_next;
                }
            }
            p = p->pl_next;
        }
    }

    /* if (ft_curckt) {
        ft_curckt->ci_devices = cp_kwswitch(CT_DEVNAMES, ft_circuits->ci_devices);
        ft_curckt->ci_nodes = cp_kwswitch(CT_NODENAMES, ft_circuits->ci_nodes);
    } */

    if (ft_circuits && caux->ci_next) {
        struct wordlist *wlist;
        wlist = wl_cons(tprintf("%d", auxCir), NULL);
        com_scirc(wlist);
        wl_free(wlist);
    }
    else if (ft_circuits) {
        struct wordlist *wlist;
        wlist = wl_cons(tprintf("%d", auxCir - 1), NULL);
        com_scirc(wlist);
        wl_free(wlist);
    }
    else
        ft_curckt = NULL;
}
