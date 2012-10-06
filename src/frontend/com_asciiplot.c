#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "plotting/plotit.h"

#include "com_asciiplot.h"


void
com_asciiplot(wordlist *wl)
{
    plotit(wl, NULL, "lpr");
}
