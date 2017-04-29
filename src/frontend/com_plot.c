#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"
#include "ngspice/fteext.h"

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

void
com_clip(wordlist *wl)
{
	int err;
	double xmin, xmax;
	char *vecname = wl->wl_word;
	wl = wl->wl_next;
	char *mword = wl->wl_word;
	xmin = INPevaluate(&mword, &err, TRUE);
	wl = wl->wl_next;
	mword = wl->wl_word;
	xmax = INPevaluate(&mword, &err, TRUE);
	vec_clip(vecname, xmin, xmax);
}

