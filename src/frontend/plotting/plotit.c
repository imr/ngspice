#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/wordlist.h"
#include "ngspice/graph.h"
#include "ngspice/cpdefs.h"
#include "ngspice/pnode.h"
#include "ngspice/sim.h"
#include "ngspice/fteext.h"

#include <circuits.h>

#include "plotit.h"
#include "agraf.h"
#include "xgraph.h"
#include "gnuplot.h"
#include "graf.h"

static bool sameflag;

#ifdef TCL_MODULE
#include "ngspice/tclspice.h"
#endif


/* This routine gets parameters from the command line, which are of
 * the form "name number ..." It returns a pointer to the parameter
 * values.  */

static double *
getlims(wordlist *wl, char *name, int number)
{
    double *d;
    wordlist *beg, *wk;
    int n;

    if (number < 1)
        return NULL;

    beg = wl_find(name, wl->wl_next);

    if (!beg)
        return NULL;

    wk = beg->wl_next;

    d = TMALLOC(double, number);

    for (n = 0; n < number; n++) {

        char *ss;
        double *td;

        if (!wk) {
            fprintf(cp_err,
                    "Syntax error: not enough parameters for \"%s\".\n", name);
            txfree(d);
            return NULL;
        }

        ss = wk->wl_word;
        td = ft_numparse(&ss, FALSE);

        if (!td) {
            fprintf(cp_err,
                    "Syntax error: bad parameters for \"%s\".\n", name);
            txfree(d);
            return NULL;
        }

        d[n] = *td;

        wk = wk->wl_next;
    }

    wl_delete_slice(beg, wk);

    return d;
}


/* Extend a data vector to length by replicating the last element, or
 * truncate it if it is too long.  */

static void
xtend(struct dvec *v, int length)
{
    int i;

    if (v->v_length == length)
        return;

    if (v->v_length > length) {
        dvec_trunc(v, length);
        return;
    }

    i = v->v_length;

    dvec_realloc(v, length, NULL);

    if (isreal(v)) {
        double d = NAN;
        if (i > 0)
            d = v->v_realdata[i - 1];
        while (i < length)
            v->v_realdata[i++] = d;
    } else {
        ngcomplex_t c = {NAN, NAN};
        if (i > 0)
            c = v->v_compdata[i - 1];
        while (i < length)
            v->v_compdata[i++] = c;
    }
}


/* Collapse every *xcomp elements into one, and use only the elements
 * between xind[0] and xind[1].
 */

static void
compress(struct dvec *d, double *xcomp, double *xind)
{
    int cfac, ihi, ilo, newlen, i;

    if (xind) {
        ilo = (int) xind[0];
        ihi = (int) xind[1];
        if ((ihi >= ilo) && (ilo > 0) && (ilo < d->v_length) &&
            (ihi > 1) && (ihi <= d->v_length)) {
            newlen = ihi - ilo;
            if (isreal(d)) {
                double *dd = TMALLOC(double, newlen);
                memcpy(dd, d->v_realdata + ilo, (size_t) newlen * sizeof(double));
                dvec_realloc(d, newlen, dd);
            } else {
                ngcomplex_t *cc = TMALLOC(ngcomplex_t, newlen);
                memcpy(cc, d->v_compdata + ilo, (size_t) newlen * sizeof(ngcomplex_t));
                dvec_realloc(d, newlen, cc);
            }
        }
    }

    if (xcomp) {
        cfac = (int) *xcomp;
        if ((cfac > 1) && (cfac < d->v_length)) {
            for (i = 0; i * cfac < d->v_length; i++)
                if (isreal(d))
                    d->v_realdata[i] = d->v_realdata[i * cfac];
                else
                    d->v_compdata[i] = d->v_compdata[i * cfac];
            dvec_trunc(d, i);
        }
    }
}


/* Check for and remove a one-word keyword. */

static bool
getflag(wordlist *wl, char *name)
{
    wl = wl_find(name, wl->wl_next);

    if (!wl)
        return FALSE;

    wl_delete_slice(wl, wl->wl_next);

    return TRUE;
}


/* Return a parameter of the form "xlabel foo" */

static char *
getword(wordlist *wl, char *name)
{
    wordlist *beg;
    char *s;

    beg = wl_find(name, wl->wl_next);

    if (!beg)
        return NULL;

    if (!beg->wl_next) {
        fprintf(cp_err,
                "Syntax error: looking for plot keyword at \"%s\".\n", name);
        return NULL;
    }

    s = copy(beg->wl_next->wl_word);

    wl_delete_slice(beg, beg->wl_next->wl_next);

    return s;
}


/* The common routine for all plotting commands. This does hardcopy
 * and graphics plotting.  */

bool
plotit(wordlist *wl, char *hcopy, char *devname)
{
    /* All these things are static so that "samep" will work. */
    static double *xcompress = NULL, *xindices = NULL;
    static double *xlim = NULL, *ylim = NULL, *xynull;
    static double *xdelta = NULL, *ydelta = NULL;
    static char *xlabel = NULL, *ylabel = NULL, *title = NULL;
    static bool nointerp = FALSE;
    static GRIDTYPE gtype = GRID_LIN;
    static PLOTTYPE ptype = PLOT_LIN;

    bool gfound = FALSE, pfound = FALSE, oneval = FALSE;
    double *dd, ylims[2], xlims[2];
    struct pnode *pn, *names;
    struct dvec *d = NULL, *vecs = NULL, *lv, *lastvs = NULL;
    char *xn;
    int i, y_type, xt;
    wordlist *wwl, *tail;
    char *cline = NULL, buf[BSIZE_SP], *pname;
    char *nxlabel = NULL, *nylabel = NULL, *ntitle = NULL;

    double tstep, tstart, tstop, ttime;

    /* return value, error by default */
    bool rtn = FALSE;

    if (!wl)
        return rtn;

    /*
     * we need a preceding element here,
     * and we will destructively modify the wordlist in stupid ways,
     * thus lets make our own copy which fits our purpose
     */
    wl = wl_cons(NULL, wl_copy(wl));

    /* First get the command line, without the limits.
       Wii be used for zoomed windows */
    wwl = wl_copy(wl);
    xynull = getlims(wwl, "xl", 2);
    tfree(xynull);
    xynull = getlims(wwl, "xlimit", 2);
    tfree(xynull);
    xynull = getlims(wwl, "yl", 2);
    tfree(xynull);
    xynull = getlims(wwl, "ylimit", 2);
    tfree(xynull);
    /* remove tile, xlabel, ylabel */
    nxlabel = getword(wwl, "xlabel");
    nylabel = getword(wwl, "ylabel");
    ntitle = getword(wwl, "title");
    pname = wl_flatten(wwl->wl_next);
    tail = wl_cons(tprintf("plot %s", pname), NULL);
    tfree(pname);
    wl_free(wwl);

    /* add title, xlabel or ylabel, if available, with quotes '' */
    if (nxlabel) {
        tail = wl_cons(tprintf("xlabel '%s'", nxlabel), tail);
        tfree(nxlabel);
    }
    if (nylabel) {
        tail = wl_cons(tprintf("ylabel '%s'", nylabel), tail);
        tfree(nylabel);
    }
    if (ntitle) {
        tail = wl_cons(tprintf("title '%s'", ntitle), tail);
        tfree(ntitle);
    }

    tail = wl_reverse(tail);
    cline = wl_flatten(tail);
    wl_free(tail);

    /* Now extract all the parameters. */

    sameflag = getflag(wl, "samep");

    if (!sameflag || !xlim) {
        xlim = getlims(wl, "xl", 2);
        if (!xlim)
            xlim = getlims(wl, "xlimit", 2);
    } else {
        txfree(getlims(wl, "xl", 2));
        txfree(getlims(wl, "xlimit", 2));
    }

    if (!sameflag || !ylim) {
        ylim = getlims(wl, "yl", 2);
        if (!ylim)
            ylim = getlims(wl, "ylimit", 2);
    } else {
        txfree(getlims(wl, "yl", 2));
        txfree(getlims(wl, "ylimit", 2));
    }

    if (!sameflag || !xcompress) {
        xcompress = getlims(wl, "xcompress", 1);
        if (!xcompress)
            xcompress = getlims(wl, "xcomp", 1);
    } else {
        txfree(getlims(wl, "xcompress", 1));
        txfree(getlims(wl, "xcomp", 1));
    }

    if (!sameflag || !xindices) {
        xindices = getlims(wl, "xindices", 2);
        if (!xindices)
            xindices = getlims(wl, "xind", 2);
    } else {
        txfree(getlims(wl, "xindices", 2));
        txfree(getlims(wl, "xind", 2));
    }

    if (!sameflag || !xdelta) {
        xdelta = getlims(wl, "xdelta", 1);
        if (!xdelta)
            xdelta = getlims(wl, "xdel", 1);
    } else {
        txfree(getlims(wl, "xdelta", 1));
        txfree(getlims(wl, "xdel", 1));
    }

    if (!sameflag || !ydelta) {
        ydelta = getlims(wl, "ydelta", 1);
        if (!ydelta)
            ydelta = getlims(wl, "ydel", 1);
    } else {
        txfree(getlims(wl, "ydelta", 1));
        txfree(getlims(wl, "ydel", 1));
    }

    /* Get the grid type and the point type.  Note we can't do if-else
     * here because we want to catch all the grid types.
     */
    if (getflag(wl, "lingrid")) {
        if (gfound) {
            fprintf(cp_err, "Warning: too many grid types given\n");
        } else {
            gtype = GRID_LIN;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "loglog")) {
        if (gfound) {
            fprintf(cp_err, "Warning: too many grid types given\n");
        } else {
            gtype = GRID_LOGLOG;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "nogrid")) {
        if (gfound) {
            fprintf(cp_err, "Warning: too many grid types given\n");
        } else {
            gtype = GRID_NONE;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "linear")) {
        if (gfound) {
            fprintf(cp_err, "Warning: too many grid types given\n");
        } else {
            gtype = GRID_LIN;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "xlog")) {
        if (gfound) {
            if (gtype == GRID_YLOG)
                gtype = GRID_LOGLOG;
            else
                fprintf(cp_err, "Warning: too many grid types given\n");
        } else {
            gtype = GRID_XLOG;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "ylog")) {
        if (gfound) {
            if (gtype == GRID_XLOG)
                gtype = GRID_LOGLOG;
            else
                fprintf(cp_err, "Warning: too many grid types given\n");
        } else {
            gtype = GRID_YLOG;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "polar")) {
        if (gfound) {
            fprintf(cp_err, "Warning: too many grid types given\n");
        } else {
            gtype = GRID_POLAR;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "smith")) {
        if (gfound) {
            fprintf(cp_err, "Warning: too many grid types given\n");
        } else {
            gtype = GRID_SMITH;
            gfound = TRUE;
        }
    }
    if (getflag(wl, "smithgrid")) {
        if (gfound) {
            fprintf(cp_err, "Warning: too many grid types given\n");
        } else {
            gtype = GRID_SMITHGRID;
            gfound = TRUE;
        }
    }

    if (!sameflag && !gfound) {
        if (cp_getvar("gridstyle", CP_STRING, buf)) {
            if (eq(buf, "lingrid"))
                gtype = GRID_LIN;
            else if (eq(buf, "loglog"))
                gtype = GRID_LOGLOG;
            else if (eq(buf, "xlog"))
                gtype = GRID_XLOG;
            else if (eq(buf, "ylog"))
                gtype = GRID_YLOG;
            else if (eq(buf, "smith"))
                gtype = GRID_SMITH;
            else if (eq(buf, "smithgrid"))
                gtype = GRID_SMITHGRID;
            else if (eq(buf, "polar"))
                gtype = GRID_POLAR;
            else if (eq(buf, "nogrid"))
                gtype = GRID_NONE;
            else {
                fprintf(cp_err, "Warning: strange grid type %s\n", buf);
                gtype = GRID_LIN;
            }
            gfound = TRUE;
        } else {
            gtype = GRID_LIN;
        }
    }

    /* Now get the point type.  */

    if (getflag(wl, "linplot")) {
        if (pfound) {
            fprintf(cp_err, "Warning: too many plot types given\n");
        } else {
            ptype = PLOT_LIN;
            pfound = TRUE;
        }
    }
    if (getflag(wl, "noretraceplot")) {
        if (pfound) {
            fprintf(cp_err, "Warning: too many plot types given\n");
        } else {
            ptype = PLOT_MONOLIN;
            pfound = TRUE;
        }
    }
    if (getflag(wl, "combplot")) {
        if (pfound) {
            fprintf(cp_err, "Warning: too many plot types given\n");
        } else {
            ptype = PLOT_COMB;
            pfound = TRUE;
        }
    }
    if (getflag(wl, "pointplot")) {
        if (pfound) {
            fprintf(cp_err, "Warning: too many plot types given\n");
        } else {
            ptype = PLOT_POINT;
            pfound = TRUE;
        }
    }

    if (!sameflag && !pfound) {
        if (cp_getvar("plotstyle", CP_STRING, buf)) {
            if (eq(buf, "linplot"))
                ptype = PLOT_LIN;
            else if (eq(buf, "noretraceplot"))
                ptype = PLOT_MONOLIN;
            else if (eq(buf, "combplot"))
                ptype = PLOT_COMB;
            else if (eq(buf, "pointplot"))
                ptype = PLOT_POINT;
            else {
                fprintf(cp_err, "Warning: strange plot type %s\n", buf);
                ptype = PLOT_LIN;
            }
            pfound = TRUE;
        } else {
            ptype = PLOT_LIN;
        }
    }

    if (!sameflag || !xlabel)
        xlabel = getword(wl, "xlabel");
    else
        txfree(getword(wl, "xlabel"));

    if (!sameflag || !ylabel)
        ylabel = getword(wl, "ylabel");
    else
        txfree(getword(wl, "ylabel"));

    if (!sameflag || !title)
        title = getword(wl, "title");
    else
        txfree(getword(wl, "title"));

    if (!sameflag)
        nointerp = getflag(wl, "nointerp");
    else if (getflag(wl, "nointerp"))
        nointerp = TRUE;

    if (!wl->wl_next) {
        fprintf(cp_err, "Error: no vectors given\n");
        goto quit1;
    }

    /* Now parse the vectors.  We have a list of the form
     * "a b vs c d e vs f g h".  Since it's a bit of a hassle for
     * us to parse the vector boundaries here, we do this -- call
     * ft_getpnames() without the check flag, and then look for 0-length
     * vectors with the name "vs"...  This is a sort of a gross hack,
     * since we have to check for 0-length vectors ourselves after
     * evaulating the pnodes...
     */

    names = ft_getpnames(wl->wl_next, FALSE);
    if (names == NULL)
        goto quit1;

    /* Now evaluate the names. */
    for (pn = names, lv = NULL; pn; pn = pn->pn_next)
        if (pn->pn_value && (pn->pn_value->v_length == 0) &&
            eq(pn->pn_value->v_name, "vs"))
        {
            struct dvec *dv;

            if (!lv) {
                fprintf(cp_err, "Error: misplaced vs arg\n");
                goto quit;
            }
            if ((pn = pn->pn_next) == NULL) {
                fprintf(cp_err, "Error: missing vs arg\n");
                goto quit;
            }

            dv = ft_evaluate(pn);
            if (!dv)
                goto quit;

            if (lastvs)
                lv = lastvs->v_link2;
            else
                lv = vecs;

            while (lv) {
                lv->v_scale = dv;
                lastvs = lv;
                lv = lv->v_link2;
            }

        } else {

            struct dvec *dv;

            dv = ft_evaluate(pn);
            if (!dv)
                goto quit;

            if (!d)
                vecs = dv;
            else
                d->v_link2 = dv;

            for (d = dv; d->v_link2; d = d->v_link2)
                ;

            lv = dv;
        }

    /* free_pnode(names); pn:really should be commented out ? */
    d->v_link2 = NULL;

    /* Now check for 0-length vectors. */
    for (d = vecs; d; d = d->v_link2)
        if (!d->v_length) {
            fprintf(cp_err, "Error(plotit.c--plotit): %s: no such vector\n",
                    d->v_name);
            goto quit;
        }

    /* If there are higher dimensional vectors, transform them into a
     * family of vectors.
     */
    for (d = vecs, lv = NULL; d; d = d->v_link2)
        if (d->v_numdims > 1) {
            if (lv)
                lv->v_link2 = vec_mkfamily(d);
            else
                vecs = lv = vec_mkfamily(d);
            while (lv->v_link2)
                lv = lv->v_link2;
            lv->v_link2 = d->v_link2;
            d = lv;
        } else {
            lv = d;
        }

    /* Now fill in the scales for vectors who aren't already fixed up. */
    for (d = vecs; d; d = d->v_link2)
        if (!d->v_scale) {
            if (d->v_plot->pl_scale)
                d->v_scale = d->v_plot->pl_scale;
            else
                d->v_scale = d;
        }

    /* The following line displays the unit at the time of
       temp-sweep and res-sweep. This may not be a so good solution. by H.T */
    if (strcmp(vecs->v_scale->v_name, "temp-sweep") == 0)
        vecs->v_scale->v_type = SV_TEMP;
    if (strcmp(vecs->v_scale->v_name, "res-sweep") == 0)
        vecs->v_scale->v_type = SV_RES;

    /* See if the log flag is set anywhere... */
    if (!gfound) {
        for (d = vecs; d; d = d->v_link2)
            if (d->v_scale && (d->v_scale->v_gridtype == GRID_XLOG))
                gtype = GRID_XLOG;
        for (d = vecs; d; d = d->v_link2)
            if (d->v_gridtype == GRID_YLOG) {
                if ((gtype == GRID_XLOG) || (gtype == GRID_LOGLOG))
                    gtype = GRID_LOGLOG;
                else
                    gtype = GRID_YLOG;
            }
        for (d = vecs; d; d = d->v_link2)
            if (d->v_gridtype == GRID_SMITH ||
                d->v_gridtype == GRID_SMITHGRID ||
                d->v_gridtype == GRID_POLAR)
            {
                gtype = d->v_gridtype;
                break;
            }
    }

    /* See if there are any default plot types...  Here, like above, we
     * don't do entirely the best thing when there is a mixed set of
     * default plot types...
     */
    if (!sameflag && !pfound) {
        ptype = PLOT_LIN;
        for (d = vecs; d; d = d->v_link2)
            if (d->v_plottype != PLOT_LIN) {
                ptype = d->v_plottype;
                break;
            }
    }

    /* Check and see if this is pole zero stuff. */
    if ((vecs->v_type == SV_POLE) || (vecs->v_type == SV_ZERO))
        oneval = TRUE;

    for (d = vecs; d; d = d->v_link2)
        if (((d->v_type == SV_POLE) || (d->v_type == SV_ZERO)) !=
            oneval ? 1 : 0) {
            fprintf(cp_err,
                    "Error: plot must be either all pole-zero or contain no poles or zeros\n");
            goto quit;
        }

    if ((gtype == GRID_POLAR) || (gtype == GRID_SMITH || gtype == GRID_SMITHGRID))
        oneval = TRUE;

    /* If we are plotting scalars, make sure there is enough
     * data to fit on the screen.
     */
    for (d = vecs; d; d = d->v_link2)
        if (d->v_length == 1)
            xtend(d, d->v_scale->v_length);

    /* Now patch up each vector with the compression and thestrchr
     * selection.
     */
    if (xcompress || xindices)
        for (d = vecs; d; d = d->v_link2) {
            compress(d, xcompress, xindices);
            d->v_scale = vec_copy(d->v_scale);
            compress(d->v_scale, xcompress, xindices);
        }

    /* Transform for smith plots */
    if (gtype == GRID_SMITH) {
        double  re, im, rex, imx;
        double  r;
        struct dvec **prevvp, *n;
        int     j;

        prevvp = &vecs;

        for (d = vecs; d; d = d->v_link2) {
            if (d->v_flags & VF_PERMANENT) {
                n = vec_copy(d);
                n->v_flags &= ~VF_PERMANENT;
                n->v_link2 = d->v_link2;
                d = n;
                *prevvp = d;
            }
            prevvp = &d->v_link2;

            if (isreal(d)) {
                fprintf(cp_err,
                        "Warning: plotting real data \"%s\" on a smith grid\n",
                        d->v_name);

                for (j = 0; j < d->v_length; j++) {
                    r = d->v_realdata[j];
                    d->v_realdata[j] = (r - 1) / (r + 1);
                }
            } else {
                for (j = 0; j < d->v_length; j++) {
                    /* (re - 1, im) / (re + 1, im) */

                    re = realpart(d->v_compdata[j]);
                    im = imagpart(d->v_compdata[j]);

                    rex = re + 1;
                    imx = im;
                    re = re - 1;

                    /* (re, im) / (rex, imx) */
                    /* x = 1 - (imx / rex) * (imx / rex);
                     * r = re / rex + im / rex * imx / rex;
                     * i = im / rex - re / rex * imx / rex;
                     *
                     *
                     * realpart(d->v_compdata[j]) = r / x;
                     * imagpart(d->v_compdata[j]) = i / x;
                     */
                    realpart(d->v_compdata[j]) = (rex*re+imx*imx) / (rex*rex+imx*imx);
                    imagpart(d->v_compdata[j]) = (2*imx) / (rex*rex+imx*imx);
                }
            }
        }
    }

    /* Figure out the proper x- and y-axis limits. */
    if (ylim) {
        ylims[0] = ylim[0];
        ylims[1] = ylim[1];
    } else if (oneval) {
        ylims[0] = HUGE;
        ylims[1] = - ylims[0];
        for (d = vecs; d; d = d->v_link2) {
            /* dd = ft_minmax(d, TRUE); */
            /* With this we seek the maximum and minimum of imaginary part
             * that will go to Y axis
             */
            dd = ft_minmax(d, FALSE);
            if (ylims[0] > dd[0])
                ylims[0] = dd[0];
            if (ylims[1] < dd[1])
                ylims[1] = dd[1];
        }
    } else {
        ylims[0] = HUGE;
        ylims[1] = - ylims[0];
        for (d = vecs; d; d = d->v_link2) {
            dd = ft_minmax(d, TRUE);
            if (ylims[0] > dd[0])
                ylims[0] = dd[0];
            if (ylims[1] < dd[1])
                ylims[1] = dd[1];
        }

        /* XXX */
        for (d = vecs; d; d = d->v_link2) {
            if (d->v_flags & VF_MINGIVEN)
                if (ylims[0] < d->v_minsignal)
                    ylims[0] = d->v_minsignal;
            if (d->v_flags & VF_MAXGIVEN)
                if (ylims[1] > d->v_maxsignal)
                    ylims[1] = d->v_maxsignal;
        }
    }

    if (xlim) {
        xlims[0] = xlim[0];
        xlims[1] = xlim[1];
    } else if (oneval) {
        xlims[0] = HUGE;
        xlims[1] = - xlims[0];
        for (d = vecs; d; d = d->v_link2) {
            /* dd = ft_minmax(d, FALSE); */
            /* With this we seek the maximum and minimum of imaginary part
             * that will go to Y axis
             */
            dd = ft_minmax(d, TRUE);

            if (xlims[0] > dd[0])
                xlims[0] = dd[0];
            if (xlims[1] < dd[1])
                xlims[1] = dd[1];
        }
    } else {
        xlims[0] = HUGE;
        xlims[1] = - xlims[0];
        for (d = vecs; d; d = d->v_link2) {
            dd = ft_minmax(d->v_scale, TRUE);
            if (xlims[0] > dd[0])
                xlims[0] = dd[0];
            if (xlims[1] < dd[1])
                xlims[1] = dd[1];
        }
        for (d = vecs; d; d = d->v_link2) {
            if (d->v_scale->v_flags & VF_MINGIVEN)
                if (xlims[0] < d->v_scale->v_minsignal)
                    xlims[0] = d->v_scale->v_minsignal;
            if (d->v_scale->v_flags & VF_MAXGIVEN)
                if (xlims[1] > d->v_scale->v_maxsignal)
                    xlims[1] = d->v_scale->v_maxsignal;
        }
    }

    /* Do some coercion of the limits to make them reasonable. */
    if ((xlims[0] == 0) && (xlims[1] == 0)) {
        xlims[0] = -1.0;
        xlims[1] = 1.0;
    }
    if ((ylims[0] == 0) && (ylims[1] == 0)) {
        ylims[0] = -1.0;
        ylims[1] = 1.0;
    }
    if (xlims[0] > xlims[1]) {
        SWAP(double, xlims[0], xlims[1]);
    }
    if (ylims[0] > ylims[1]) {
        SWAP(double, ylims[0], ylims[1]);
    }
    if (AlmostEqualUlps(xlims[0], xlims[1], 10)) {
        xlims[0] *= (xlims[0] > 0) ? 0.9 : 1.1;
        xlims[1] *= (xlims[1] > 0) ? 1.1 : 0.9;
    }
    if (AlmostEqualUlps(ylims[0], ylims[1], 10)) {
        ylims[0] *= (ylims[0] > 0) ? 0.9 : 1.1;
        ylims[1] *= (ylims[1] > 0) ? 1.1 : 0.9;
    }

    if ((xlims[0] <= 0.0) && ((gtype == GRID_XLOG) ||
                              (gtype == GRID_LOGLOG))) {
        fprintf(cp_err, "Error: X values must be > 0 for log scale\n");
        goto quit;
    }
    if ((ylims[0] <= 0.0) && ((gtype == GRID_YLOG) ||
                              (gtype == GRID_LOGLOG))) {
        fprintf(cp_err, "Error: Y values must be > 0 for log scale\n");
        goto quit;
    }

    /* Fix the plot limits for smith and polar grids. */
    if ((!xlim || !ylim) && (gtype == GRID_POLAR)) {
        double mx, my, rad;
        /* (0,0) must be in the center of the screen. */
        mx = (fabs(xlims[0]) > fabs(xlims[1])) ? fabs(xlims[0]) : fabs(xlims[1]);
        my = (fabs(ylims[0]) > fabs(ylims[1])) ? fabs(ylims[0]) : fabs(ylims[1]);
        /* rad = (mx > my) ? mx : my; */
        /* AM.Roldán
         * Change this reason that this was discussed, as in the case of 1 + i want to plot point
         * is outside the drawing area so I'll stay as the maximum size of the hypotenuse of
         * the complex value
         */
        rad = hypot(mx, my);
        xlims[0] = - rad;
        xlims[1] = rad;
        ylims[0] = - rad;
        ylims[1] = rad;
    } else if ((!xlim || !ylim) && (gtype == GRID_SMITH || gtype == GRID_SMITHGRID))
    {
        xlims[0] = -1.0;
        xlims[1] = 1.0;
        ylims[0] = -1.0;
        ylims[1] = 1.0;
    }

    if (xlim)
        tfree(xlim);
    if (ylim)
        tfree(ylim);

    /* We don't want to try to deal with smith plots for asciiplot. */
    if (devname && eq(devname, "lpr")) {
        /* check if we should (can) linearize */
        if (ft_curckt && ft_curckt->ci_ckt &&
            (strcmp(ft_curckt->ci_name, plot_cur->pl_title) == 0) &&
            if_tranparams(ft_curckt, &tstart, &tstop, &tstep) &&
            ((tstop - tstart) * tstep > 0.0) &&
            ((tstop - tstart) >= tstep) &&
            plot_cur && plot_cur->pl_dvecs &&
            plot_cur->pl_scale &&
            isreal(plot_cur->pl_scale) &&
            ciprefix("tran", plot_cur->pl_typename))
        {
            int newlen = (int)((tstop - tstart) / tstep + 1.5);

            double *newscale;

            struct dvec *v, *newv_scale =
                dvec_alloc(copy(vecs->v_scale->v_name),
                           vecs->v_scale->v_type,
                           vecs->v_scale->v_flags,
                           newlen, NULL);

            newv_scale->v_gridtype = vecs->v_scale->v_gridtype;

            newscale = newv_scale->v_realdata;
            for (i = 0, ttime = tstart; i < newlen; i++, ttime += tstep)
                newscale[i] = ttime;

            for (v = vecs; v; v = v->v_link2) {
                double *newdata = TMALLOC(double, newlen);

                if (!ft_interpolate(v->v_realdata, newdata,
                                    v->v_scale->v_realdata, v->v_scale->v_length,
                                    newscale, newlen, 1)) {
                    fprintf(cp_err, "Error: can't interpolate %s\n", v->v_name);
                    goto quit;
                }

                dvec_realloc(v, newlen, newdata);

                /* Why go to all this trouble if agraf ignores it? */
                nointerp = TRUE;
            }

            vecs->v_scale = newv_scale;

        }

        ft_agraf(xlims, ylims,
                 vecs->v_scale, vecs->v_plot, vecs,
                 xdelta ? *xdelta : 0.0,
                 ydelta ? *ydelta : 0.0,
                 ((gtype == GRID_XLOG) || (gtype == GRID_LOGLOG)),
                 ((gtype == GRID_YLOG) || (gtype == GRID_LOGLOG)),
                 nointerp);
        rtn = TRUE;
        goto quit;
    }

    /* See if there is one common v_type we can give for the y scale... */
    for (d = vecs->v_link2; d; d = d->v_link2)
        if (d->v_type != vecs->v_type)
            break;

    y_type = d ? SV_NOTYPE : vecs->v_type;

#ifndef X_DISPLAY_MISSING
    if (devname && eq(devname, "xgraph")) {
        /* Interface to XGraph-11 Plot Program */
        ft_xgraph(xlims, ylims, hcopy,
                  title ? title : vecs->v_plot->pl_title,
                  xlabel ? xlabel : ft_typabbrev(vecs->v_scale->v_type),
                  ylabel ? ylabel : ft_typabbrev(y_type),
                  gtype, ptype, vecs);
        rtn = TRUE;
        goto quit;
    }
#endif

    if (devname && eq(devname, "gnuplot")) {
        /* Interface to Gnuplot Plot Program */
        ft_gnuplot(xlims, ylims, hcopy,
                   title ? title : vecs->v_plot->pl_title,
                   xlabel ? xlabel : ft_typabbrev(vecs->v_scale->v_type),
                   ylabel ? ylabel : ft_typabbrev(y_type),
                   gtype, ptype, vecs);
        rtn = TRUE;
        goto quit;
    }

    if (devname && eq(devname, "writesimple")) {
        /* Interface to simple write output */
        ft_writesimple(xlims, ylims, hcopy,
                       title ? title : vecs->v_plot->pl_title,
                       xlabel ? xlabel : ft_typabbrev(vecs->v_scale->v_type),
                       ylabel ? ylabel : ft_typabbrev(y_type),
                       gtype, ptype, vecs);
        rtn = TRUE;
        goto quit;
    }

#ifdef TCL_MODULE
    if (devname && eq(devname, "blt")) {
        /* Just send the pairs to Tcl/Tk */
        for (d = vecs; d; d = d->v_link2)
            blt_plot(d, oneval ? NULL : d->v_scale, (d == vecs) ? 1 : 0);
        rtn = TRUE;
        goto quit;
    }
#endif

    for (d = vecs, i = 0; d; d = d->v_link2)
        i++;

    /* Figure out the X name and the X type.  This is sort of bad... */
    xn = vecs->v_scale->v_name;
    xt = vecs->v_scale->v_type;

    pname = plot_cur->pl_typename;

    if (!gr_init(xlims, ylims, (oneval ? NULL : xn),
                 title ? title : vecs->v_plot->pl_title,
                 hcopy, i,
                 xdelta ? *xdelta : 0.0,
                 ydelta ? *ydelta : 0.0,
                 gtype, ptype, xlabel, ylabel, xt, y_type, pname, cline))
        goto quit;

    /* Now plot all the graphs. */
    for (d = vecs; d; d = d->v_link2)
        ft_graf(d, oneval ? NULL : d->v_scale, FALSE);

    gr_clean();
    rtn = TRUE;

quit:
    tfree(cline);
    free_pnode(names);
    FREE(title);
quit1:
    wl_free(wl);
    return rtn;
}
