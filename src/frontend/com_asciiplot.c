#include <config.h>
#include <ngspice.h>

#include <bool.h>
#include <wordlist.h>

#include "plotting/plotit.h"

#include "com_asciiplot.h"


void
com_asciiplot(wordlist *wl)
{
    plotit(wl, (char *) NULL, "lpr");
    return;
}
