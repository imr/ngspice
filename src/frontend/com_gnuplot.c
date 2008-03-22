#include <stddef.h>

#include <ngspice.h>
#include <bool.h>
#include <wordlist.h>

#include "plotting/plotit.h"

#include "com_gnuplot.h"


/* gnuplot file plotargs */
void
com_gnuplot(wordlist *wl)
{
    char *fname;
    bool tempf = FALSE;

    if (wl) {
        fname = wl->wl_word;
        wl = wl->wl_next;
    }
    if (!wl) {
	return;
    }
    if (cieq(fname, "temp") || cieq(fname, "tmp")) {
        fname = smktemp("gp"); /* Is this the correct name ? */
        tempf = TRUE;
    }

    (void) plotit(wl, fname, "gnuplot");

#if 0
    /* Leave temp file sitting around so gnuplot can grab it from
       background. */
    if (tempf)
        (void) unlink(fname);
#endif

    return;
}
