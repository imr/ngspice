#include "ngspice/config.h"
#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "plotting/plotit.h"

#include "com_plot.h"


/* plot name ... [xl[imit]] xlo xhi] [yl[imit ylo yhi] [vs xname] */
void
com_plot(wordlist *wl)
{
    plotit(wl, NULL, NULL);
}

#ifdef TCL_MODULE
void
com_bltplot(wordlist *wl)
{
    plotit(wl, NULL, "blt");
}

#endif

