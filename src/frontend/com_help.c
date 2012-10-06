#include "ngspice/ngspice.h"

#include "ngspice/macros.h"
#include "ngspice/wordlist.h"
#include "ngspice/cpdefs.h"
#include "ngspice/bool.h"

#include "hcomp.h"
#include "com_help.h"
#include "ngspice/fteext.h"


void
com_help(wordlist *wl)
{
    struct comm *c;
    struct comm *ccc[512];  /* Should be enough. */
    int numcoms, i;
    bool allflag = FALSE;

    if (wl && eq(wl->wl_word, "all")) {
        allflag = TRUE;
        wl = NULL;  /* XXX Probably right */
    }

    /* We want to use more mode whether "moremode" is set or not. */
    out_moremode = TRUE;
    out_init();
    out_moremode = FALSE;
    if (wl == NULL) {
        out_printf("For a complete description "
                   "read the Spice3 User's Manual.\n");

        if (!allflag) {
            out_printf("For a list of all commands "
                       "type \"help all\", for a short\n"
                       "description of \"command\", "
                       "type \"help command\".\n");
        }

        /* Sort the commands */
        for (numcoms = 0; cp_coms[numcoms].co_func != NULL; numcoms++)
            ccc[numcoms] = &cp_coms[numcoms];
        qsort(ccc, (size_t) numcoms, sizeof(struct comm *), hcomp);

        for (i = 0; i < numcoms; i++) {
            if ((ccc[i]->co_spiceonly && ft_nutmeg) ||
                (ccc[i]->co_help == NULL) ||
                (!allflag && !ccc[i]->co_major))
                continue;
            out_printf("%s ", ccc[i]->co_comname);
            out_printf(ccc[i]->co_help, cp_program);
            out_send("\n");
        }
    } else {
        while (wl != NULL) {
            for (c = &cp_coms[0]; c->co_func != NULL; c++)
                if (eq(wl->wl_word, c->co_comname)) {
                    out_printf("%s ", c->co_comname);
                    out_printf(c->co_help, cp_program);
                    if (c->co_spiceonly && ft_nutmeg)
                        out_send(" (Not available in nutmeg)");
                    out_send("\n");
                    break;
                }
            if (c->co_func == NULL) {
                /* See if this is aliased. */
                struct alias *al;

                for (al = cp_aliases; al; al = al->al_next)
                    if (eq(al->al_name, wl->wl_word))
                        break;

                if (al == NULL) {
                    fprintf(cp_out, "Sorry, no help for %s.\n", wl->wl_word);
                } else {
                    out_printf("%s is aliased to ", wl->wl_word);
                    /* Minor badness here... */
                    wl_print(al->al_text, cp_out);
                    out_send("\n");
                }
            }
            wl = wl->wl_next;
        }
    }

    out_send("\n");
}
