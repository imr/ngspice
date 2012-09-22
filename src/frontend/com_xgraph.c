#include <stddef.h>

#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "plotting/plotit.h"
#include "../misc/mktemp.h"

#include "com_xgraph.h"


/* xgraph file plotargs */
void
com_xgraph(wordlist *wl)
{
    char *fname = NULL;
    bool tempf = FALSE;

    if (wl) {
        fname = wl->wl_word;
        wl = wl->wl_next;
    }

    if (!wl)
        return;

    if (cieq(fname, "temp") || cieq(fname, "tmp")) {
        fname = smktemp("xg");
        tempf = TRUE;
    }

    (void) plotit(wl, fname, "xgraph");

#if 0
    /* Leave temp file sitting around so xgraph can grab it from
       background. */
    if (tempf)
        (void) unlink(fname);
#endif

}
