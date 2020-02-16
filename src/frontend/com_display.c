#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/wordlist.h"
#include "ngspice/fteext.h"
#include "ngspice/cpextern.h"

#include "com_display.h"
#include "variable.h"
#include "plotting/plotting.h"
#include "plotting/pvec.h"


/* For the sort in display. */
static int
dcomp(const void *d1, const void *d2)
{
    struct dvec **v1 = (struct dvec **) d1;
    struct dvec **v2 = (struct dvec **) d2;

    return (strcmp((*v1)->v_name, (*v2)->v_name));
}


/* Display vector status, etc.  Note that this only displays stuff
 * from the current plot, and you must do a setplot to see the rest of
 * it.  */
void
com_display(wordlist *wl)
{
    struct dvec *d;
    struct dvec **dvs;
    int len = 0, i = 0;
    char *s;

    /* Maybe he wants to know about just a few vectors. */

    out_init();

    while (wl) {
        s = cp_unquote(wl->wl_word);
        d = vec_get(s);
        tfree(s);               /*DG to avoid the cp_unquote memory leak */
        if (d == NULL)
            fprintf(cp_err, "Error: no such vector as %s.\n", wl->wl_word);
        else if (d->v_plot == NULL)
            fprintf(cp_err, "Error: no analog vector as %s.\n", wl->wl_word);
        else
            while (d) {
                pvec(d);
                d = d->v_link2;
            }
        if (wl->wl_next == NULL)
            return;
        wl = wl->wl_next;
    }

    if (plot_cur)
        for (d = plot_cur->pl_dvecs; d; d = d->v_next)
            len++;

    if (len == 0) {
        fprintf(cp_out, "There are no vectors currently active.\n");
        return;
    }

    out_printf("Here are the vectors currently active:\n\n");
    dvs = TMALLOC(struct dvec *, len);
    for (d = plot_cur->pl_dvecs, i = 0; d; d = d->v_next, i++)
        dvs[i] = d;
    if (!cp_getvar("nosort", CP_BOOL, NULL, 0))
        qsort(dvs, (size_t) len, sizeof(struct dvec *), dcomp);

    out_printf("Title: %s\n", plot_cur->pl_title);
    out_printf("Name: %s (%s)\nDate: %s\n\n",
               plot_cur->pl_typename, plot_cur->pl_name,
               plot_cur->pl_date);

    for (i = 0; i < len; i++) {
        d = dvs[i];
        pvec(d);
    }
    tfree(dvs);
}
