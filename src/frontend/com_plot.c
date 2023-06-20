#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "plotting/plotit.h"

#include "com_plot.h"

extern bool ft_batchmode;

/* plot name ... [xl[imit]] xlo xhi] [yl[imit ylo yhi] [vs xname] */
void
com_plot(wordlist *wl)
{
    if (ft_batchmode) {
        fprintf(stderr, "\nWarning: command 'plot' is not available during batch simulation, ignored!\n");
        fprintf(stderr, "    You may use Gnuplot instead.\n\n");
        return;
    }
    plotit(wl, NULL, NULL);
}

#ifdef TCL_MODULE
void
com_bltplot(wordlist *wl)
{
    plotit(wl, NULL, "blt");
}

#endif

