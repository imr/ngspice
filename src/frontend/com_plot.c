#include <config.h>
#include <ngspice.h>

#include <bool.h>
#include <wordlist.h>

#include "plotting/plotit.h"

#include "com_plot.h"


/* plot name ... [xl[imit]] xlo xhi] [yl[imit ylo yhi] [vs xname] */
void
com_plot(wordlist *wl)
{
    plotit(wl, (char *) NULL, (char *) NULL);
    return;
}
