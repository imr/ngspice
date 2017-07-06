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

/* Evaluate the command clip v(1) xmin xmax
   xmin and xmax may be expressions of single valued vectors or real numbers.
   Then call vec_clip() to do the clipping */
void
com_clip(wordlist *wl)
{
    struct pnode *names, *delnames;
    struct dvec *vec;
    double xmin, xmax;
    char *vecname = wl->wl_word;
    wl = wl->wl_next;
    delnames = names = ft_getpnames(wl, TRUE);
    vec = ft_evaluate(names);
    if (!vec) {
        fprintf(cp_err, "Error: Cannot evaluate xmin for %s.\n  No clipping done!\n", vecname);
        return;
    }
    xmin = vec->v_realdata[0];
    if (names->pn_value == NULL)
        vec_free(vec);
    names = names->pn_next;
    vec = ft_evaluate(names);
    if (!vec) {
        fprintf(cp_err, "Error: Cannot evaluate xmax for %s.\n  No clipping done!\n", vecname);
        return;
    }
    xmax = vec->v_realdata[0];
    if (names->pn_value == NULL)
        vec_free(vec);
    vec_clip(vecname, xmin, xmax);
    free_pnode(delnames);
}

