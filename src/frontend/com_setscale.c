#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/dvec.h"

#include "com_setscale.h"
#include "ngspice/cpextern.h"
#include "vectors.h"
#include "plotting/plotting.h"
#include "plotting/pvec.h"
#include "ngspice/fteext.h"

static struct dvec *find_vec(wordlist *wl)
{
    struct  dvec *d;
    char   *s;

    s = cp_unquote(wl->wl_word);
    if (s) {
        d = vec_get(s);
        tfree(s);       /*DG to avoid the cp_unquote memory leak */
    } else {
        d = NULL;
    }

    if (d == NULL)
        fprintf(cp_err, "Error: no such vector as %s.\n", wl->wl_word);
    return d;
}

/* Set the default scale to the named vector.  If no vector named,
 * find and print the default scale.  */

void
com_setscale(wordlist *wl)
{
    struct dvec *d, *ds;

    if (!plot_cur) {
        fprintf(cp_err, "Error: no current plot.\n");
        return;
    }

    if (!wl) {
        if (plot_cur->pl_scale)
            pvec(plot_cur->pl_scale);
        return;
    }

    d = find_vec(wl);
    if (d == NULL)
        return;

    /* Two-word form for altering the scale of a specific vector?
     * Keyword "none" clears the scale so that the plot's default scale
     * will be used.
     */

    wl = wl->wl_next;
    if (wl) {
        if (!strcmp(wl->wl_word, "none")) {
            d->v_scale = NULL;
            return;
        }

        ds = find_vec(wl);
        if (ds == NULL)
            return;
        d->v_scale = ds;
    } else {
        plot_cur->pl_scale = d;
    }
}
