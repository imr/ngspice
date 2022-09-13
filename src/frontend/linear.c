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
 * If no circuit is loaded, e.g. because the 'load' command has been used
 * to obtain data, try to get parameters from scale vector.
 * Interpolation may be restricted to only a region of the input vector,
 * thus creating a cutout of the original vector.
 */
void
com_linearize(wordlist *wl)
{
    double tstart, tstop, tstep, d;
    struct plot *new, *old;
    struct dvec *newtime, *v;
    struct dvec *oldtime;
    struct dvec *lin;
    int len, i;

    if (!plot_cur || !plot_cur->pl_typename || !ciprefix("tran", plot_cur->pl_typename)) {
        fprintf(cp_err, "Error: plot must be a transient analysis\n");
        return;
    }
    if (!plot_cur->pl_dvecs || !plot_cur->pl_scale) {
        fprintf(cp_err, "Error: no vectors available\n");
        return;
    }
    if (!isreal(plot_cur->pl_scale)) {
        fprintf(cp_err, "Error: non-real time scale for %s\n",
                plot_cur->pl_typename);
        return;
    }

    /* check if circuit is loaded and TSTART, TSTOP, TSTEP are available
       if no circuit is loaded, but vectors are available, obtain
       start, stop, step data from scale vector */
    if (!ft_curckt || !ft_curckt->ci_ckt ||
        !if_tranparams(ft_curckt, &tstart, &tstop, &tstep)) {
        fprintf(cp_err,
                "Warning: Can't get transient parameters from circuit.\n"
                "         Use transient analysis scale vector data instead.\n");
        int length = plot_cur->pl_scale->v_length;
        if (length < 1) {
            fprintf(cp_err, "Error: no data in vector\n");
            return;
        }
        tstart = plot_cur->pl_scale->v_realdata[0];
        tstop = plot_cur->pl_scale->v_realdata[length - 1];
        tstep = (tstop - tstart) / (double)length;
    }

    /* if this plot contains special vectors lin-tstart, lin-tstop or lin-tstep, use these instead */
    lin = vec_fromplot("lin-tstart", plot_cur);
    if (lin) {
        fprintf(cp_out, "linearize tstart is set to: %8e\n", lin->v_realdata[0]);
        tstart = lin->v_realdata[0];
    }

    lin = vec_fromplot("lin-tstop", plot_cur);
    if (lin) {
        fprintf(cp_out, "linearize tstop is set to: %8e\n", lin->v_realdata[0]);
        tstop = lin->v_realdata[0];
    }

    lin = vec_fromplot("lin-tstep", plot_cur);
    if (lin) {
        fprintf(cp_out, "linearize tstep is set to: %8e\n", lin->v_realdata[0]);
        tstep = lin->v_realdata[0];
    }

    /* finally check if tstart, tstop and tstep are reasonable */
    if (((tstop - tstart) * tstep <= 0.0) || ((tstop - tstart) < tstep)) {
        fprintf(cp_err,
            "Error: bad parameters -- start = %G, stop = %G, step = %G\n",
            tstart, tstop, tstep);
        return;
    }
    old = plot_cur;
    oldtime = old->pl_scale;
    new = plot_alloc("transient");
    new->pl_name = tprintf("%s (linearized)", old->pl_name);
    new->pl_title = copy(old->pl_title);
    new->pl_date = copy(old->pl_date);
    new->pl_next = plot_list;
    plot_new(new);
    plot_setcur(new->pl_typename);
    plot_list = new;
    len = (int)((tstop - tstart) / tstep + 1.5);
    newtime = dvec_alloc(copy(oldtime->v_name),
                         oldtime->v_type,
                         oldtime->v_flags | VF_PERMANENT,
                         len, NULL);

    newtime->v_plot = new;
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


/* Cut out part of tran vectors from cut-tstart to cut-tstop and copy these
   into a new plot. A new scale vector 'time' will be there as well. Vectors
   that are shorter than the new scale vector will not be copied. */
void
com_cutout(wordlist* wl)
{
    double tstart, tstop;
    struct plot* new, * old;
    struct dvec* newtime, * v, *nv;
    struct dvec* oldtime;
    struct dvec* sta, *sto;
    int len, i, istart, istop;

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

    int length = plot_cur->pl_scale->v_length;
    if (length < 1) {
    fprintf(cp_err, "Error: no data in vector\n");
        return;
    }
    /* if this plot contains special vectors cut-tstart or cut-tstop, use these */
    sta = vec_fromplot("cut-tstart", plot_cur);
    if (sta) {
        tstart = sta->v_realdata[0];
        for (istart = 0; istart < length - 1; istart++) {
            if (plot_cur->pl_scale->v_realdata[istart] > tstart) {
//                istart--;
                break;
            }
        }
    }
    else {
        tstart = plot_cur->pl_scale->v_realdata[0];
        istart = 0;
    }
    sto = vec_fromplot("cut-tstop", plot_cur);
    if (sto) {
        tstop = sto->v_realdata[0];
        for (istop = 0; istop < length - 1; istop++) {
            if (plot_cur->pl_scale->v_realdata[istop] > tstop)
                break;
        }
    }
    else {
        tstop = plot_cur->pl_scale->v_realdata[length - 1];
        istop = length - 1;
    }


    /* finally check if tstart, tstop, istart and istop are reasonable */
    if (((tstop - tstart)  <= 0.0) || ((istop - istart) <= 0)) {
        fprintf(cp_err,
            "Error: bad parameters -- start = %G, stop = %G\n",
            tstart, tstop);
        return;
    }

    /* create a new plot */
    old = plot_cur;
    oldtime = old->pl_scale;
    new = plot_alloc("transient");
    if (!sta && !sto)
        new->pl_name = tprintf("%s (copy)", old->pl_name);
    else
        new->pl_name = tprintf("%s (cut out)", old->pl_name);
    new->pl_title = copy(old->pl_title);
    new->pl_date = copy(old->pl_date);
    new->pl_next = plot_list;
    plot_new(new);
    plot_setcur(new->pl_typename);
    plot_list = new;

    /* copy a new scale vector (time) */
    len = istop - istart;
    newtime = dvec_alloc(copy(oldtime->v_name),
        oldtime->v_type,
        oldtime->v_flags | VF_PERMANENT,
        len, NULL);

    newtime->v_plot = new;
    for (i = 0; i < len; i++)
        newtime->v_realdata[i] = oldtime->v_realdata[istart + i];
    new->pl_scale = new->pl_dvecs = newtime;

    /* copy the vectors from tstart to tstop */
    if (wl) {
        while (wl) {
            v = vec_fromplot(wl->wl_word, old);
            if (!v) {
                fprintf(cp_err, "Error: no such vector %s\n",
                    wl->wl_word);
                wl = wl->wl_next;
                continue;
            }
            nv = copycut(v, newtime, istart, istop);
            vec_new(nv);
            wl = wl->wl_next;
        }
    }
    else {
        for (v = old->pl_dvecs; v; v = v->v_next) {
            if (v == old->pl_scale)
                continue;
            /* Don't copy vectors that are too short */
            if (v->v_length < istop)
                continue;
            nv = copycut(v, newtime, istart, istop);
            vec_new(nv);
        }
    }
}
