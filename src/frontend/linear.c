/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"

#include "circuits.h"
#include "linear.h"
#include "interp.h"


/* Interpolate all the vectors in a plot to a linear time scale, which
 * we determine by looking at the transient parameters in the CKT struct.
 */

void
com_linearize(wordlist *wl)
{
    double tstart, tstop, tstep, d;
    struct plot *new, *old;
    struct dvec *newtime, *v;
    struct dvec *oldtime;
    int len, i;
    char buf[BSIZE_SP];

    if (!ft_curckt || !ft_curckt->ci_ckt ||
        !if_tranparams(ft_curckt, &tstart, &tstop, &tstep)) {
        fprintf(cp_err,
                "Error: can't get transient parameters from circuit\n");
        return;
    }
    if (((tstop - tstart) * tstep <= 0.0) || ((tstop - tstart) < tstep)) {
        fprintf(cp_err,
                "Error: bad parameters -- start = %G, stop = %G, step = %G\n",
                tstart, tstop, tstep);
        return;
    }
    if (!plot_cur || !plot_cur->pl_dvecs || !plot_cur->pl_scale) {
        fprintf(cp_err, "Error: no vectors available\n");
        return;
    }
    if (!isreal(plot_cur->pl_scale)) {
        fprintf(cp_err, "Error: non-real time scale for %s\n",
                plot_cur->pl_typename);
        return;
    }
    if (!ciprefix("tran", plot_cur->pl_typename)) {
        fprintf(cp_err, "Error: plot must be a transient analysis\n");
        return;
    }
    old = plot_cur;
    oldtime = old->pl_scale;
    new = plot_alloc("transient");
    (void) sprintf(buf, "%s (linearized)", old->pl_name);
    new->pl_name = copy(buf);
    new->pl_title = copy(old->pl_title);
    new->pl_date = copy(old->pl_date);
    new->pl_next = plot_list;
    plot_new(new);
    plot_setcur(new->pl_typename);
    plot_list = new;
    len = (int)((tstop - tstart) / tstep + 1.5);
    newtime = alloc(struct dvec);
    newtime->v_name = copy(oldtime->v_name);
    newtime->v_type = oldtime->v_type;
    newtime->v_flags = oldtime->v_flags;
    newtime->v_flags |= VF_PERMANENT;
    newtime->v_length = len;
    newtime->v_plot = new;
    newtime->v_realdata = TMALLOC(double, len);
    for (i = 0, d = tstart; i < len; i++, d += tstep)
        newtime->v_realdata[i] = d;
    new->pl_scale = new->pl_dvecs = newtime;

    if (wl) {
        while (wl) {
            v = vec_fromplot(wl->wl_word, old);
            if (!v) {
                fprintf(cp_err, "Error: no such vector %s\n",
                        wl->wl_word);
                wl = wl->wl_next;
                continue;
            }
            lincopy(v, newtime->v_realdata, len, oldtime);
            wl = wl->wl_next;
        }
    } else {
        for (v = old->pl_dvecs; v; v = v->v_next) {
            if (v == old->pl_scale)
                continue;
            lincopy(v, newtime->v_realdata, len, oldtime);
        }
    }
}
