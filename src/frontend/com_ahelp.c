#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"
#include "ngspice/fteext.h"

#include "variable.h"
#include "com_help.h"
#include "com_ahelp.h"
#include "hcomp.h"
#include "ftehelp.h"

#include "plotting/plotting.h"


void
com_ahelp(wordlist *wl)
{
    int i, n;
    /* assert: number of commands must be less than 512 */
    struct comm *cc[512];
    unsigned int env = 0;
    struct comm *com;
    unsigned int level;
    char slevel[256];

    if (wl) {
        com_help(wl);
        return;
    }

    out_init();

    /* determine environment */
    if (plot_list->pl_next)     /* plots load */
        env |= E_HASPLOTS;
    else
        env |= E_NOPLOTS;

    /* determine level */
    if (cp_getvar("level", CP_STRING, slevel, sizeof(slevel))) {
        switch (*slevel) {
        case 'b':   level = 1;
            break;
        case 'i':   level = 2;
            break;
        case 'a':   level = 4;
            break;
        default:    level = 1;
            break;
        }
    } else {
        level = 1;
    }

    out_printf(
        "For a complete description read the Spice3 User's Manual manual.\n");
    out_printf(
        "For a list of all commands type \"help all\", for a short\n");
    out_printf(
        "description of \"command\", type \"help command\".\n");

    /* sort the commands */
    for (n = 0; cp_coms[n].co_func != NULL; n++)
        cc[n] = &cp_coms[n];

    qsort(cc, (size_t) n, sizeof(struct comm *), hcomp);

    /* filter the commands */
    for (i = 0; i < n; i++) {
        com = cc[i];
        if ((com->co_env < (level << 13)) &&
            (!(com->co_env & 4095) || (env & com->co_env)))
        {
            if ((com->co_spiceonly && ft_nutmeg) || (com->co_help == NULL))
                continue;

            out_printf("%s ", com->co_comname);
            out_printf(com->co_help, cp_program);
            out_send("\n");
        }
    }

    out_send("\n");
}
