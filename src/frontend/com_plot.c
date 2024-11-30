#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "plotting/plotit.h"

#include "com_plot.h"

extern bool ft_batchmode;

/* Utility function to check for batch mode. */

int check_batch(const char *cmd)
{
    if (ft_batchmode) {
        fprintf(stderr,
                "\nWarning: command '%s' is not available during "
                "batch simulation, ignored!\n",
                cmd);
        fprintf(stderr, "    You may use Gnuplot instead.\n\n");
        return 1;
    }
    return 0;
}

/* plot name ... [xl[imit]] xlo xhi] [yl[imit ylo yhi] [vs xname] */
void
com_plot(wordlist *wl)
{
    if (check_batch("plot"))
        return;
    plotit(wl, NULL, NULL);
}

#ifdef TCL_MODULE
void
com_bltplot(wordlist *wl)
{
    if (check_batch("bltplot"))
        return;
    plotit(wl, NULL, "blt");
}

#endif

