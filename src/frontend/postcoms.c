/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Various post-processor commands having to do with vectors.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/sim.h"
#include "ngspice/plot.h"
#include "ngspice/graph.h"
#include "ngspice/ftedbgra.h"
#include "com_display.h"

#include "completion.h"
#include "postcoms.h"
#include "quote.h"
#include "variable.h"
#include "parser/complete.h" /* va: throwaway */
#include "plotting/plotting.h"


static void killplot(struct plot *pl);
static void DelPlotWindows(struct plot *pl);


void
com_unlet(wordlist *wl)
{
    while (wl) {
        vec_remove(wl->wl_word);
        wl = wl->wl_next;
    }
}


/* Load in a file. */

void
com_load(wordlist *wl)
{
    char *copypath;
    if (!wl)
        ft_loadfile(ft_rawfile);
    else
        while (wl) {
            /*ft_loadfile(cp_unquote(wl->wl_word)); DG: bad memory leak*/
            copypath = cp_unquote(wl->wl_word);/*DG*/
            ft_loadfile(copypath);
            tfree(copypath);
            wl = wl->wl_next;
        }

    /* note: default is to display the vectors in the last (current) plot */
    com_display(NULL);
}


/* Print out the value of an expression. When we are figuring out what to
 * print, link the vectors we want with v_link2... This has to be done
 * because of the way temporary vectors are linked together with permanent
 * ones under the plot.
 */

void
com_print(wordlist *wl)
{
    struct dvec *v, *lv = NULL, *bv, *nv, *vecs = NULL;
    int i, j, ll, width = DEF_WIDTH, height = DEF_HEIGHT, npoints, lineno;
    struct pnode *pn, *names;
    struct plot *p;
    bool col = TRUE, nobreak = FALSE, noprintscale, plotnames = FALSE;
    bool optgiven = FALSE;
    char *s, *buf, *buf2; /*, buf[BSIZE_SP], buf2[BSIZE_SP];*/
    char numbuf[BSIZE_SP], numbuf2[BSIZE_SP]; /* Printnum buffers */
    int ngood;

    if (wl == NULL)
        return;

    buf = TMALLOC(char, BSIZE_SP);
    buf2 = TMALLOC(char, BSIZE_SP);

    if (eq(wl->wl_word, "col")) {
        col = TRUE;
        optgiven = TRUE;
        wl = wl->wl_next;
    } else if (eq(wl->wl_word, "line")) {
        col = FALSE;
        optgiven = TRUE;
        wl = wl->wl_next;
    }

    ngood = 0;
    names = ft_getpnames(wl, TRUE);
    for (pn = names; pn; pn = pn->pn_next) {
        if ((v = ft_evaluate(pn)) == NULL)
            continue;
        if (!vecs)
            vecs = lv = v;
        else
            lv->v_link2 = v;
        for (lv = v; lv->v_link2; lv = lv->v_link2)
            ;
        ngood += 1;
    }

    if (!ngood)
        goto done;

    /* See whether we really have to print plot names. */
    for (v = vecs; v; v = v->v_link2)
        if (vecs->v_plot != v->v_plot) {
            plotnames = TRUE;
            break;
        }

    if (!optgiven) {
        /* Figure out whether col or line should be used... */
        col = FALSE;
        for (v = vecs; v; v = v->v_link2)
            if (v->v_length > 1) {
                col = TRUE;
                /* Improvement made to print cases @[sin] = (0 12 13 100K) */
                if ((v->v_plot->pl_scale && v->v_length != v->v_plot->pl_scale->v_length) && (*(v->v_name) == '@'))
                {
                    col = FALSE;
                }
                break;
            }
        /* With this I have found that the vector has less elements than the SCALE vector
         * in the linked PLOT. But now I must make sure in case of a print @vin[sin] or
         * @vin[pulse]
         * for it appear that the v->v_name begins with '@'
         * And then be in this case.
         */
    }

    out_init();
    if (!col) {
        if (cp_getvar("width", CP_NUM, &i))
            width = i;
        if (width < 60)
            width = 60;
        if (width > BSIZE_SP - 2)
            buf = TREALLOC(char, buf, width + 1);
        for (v = vecs; v; v = v->v_link2) {
            char *basename = vec_basename(v);
            if (plotnames)
                (void) sprintf(buf, "%s.%s", v->v_plot->pl_typename, basename);
            else
                (void) strcpy(buf, basename);
            tfree(basename);

            for (s = buf; *s; s++)
                ;
            s--;
            while (isspace_c(*s)) {
                *s = '\0';
                s--;
            }
            ll = 10;

            /* v->v_rlength = 1 when it comes to make a print @ M1 and does not want to come out on screen
             * Multiplier factor [m]=1
             *  @M1 = 0,00e+00
             * In any other case rlength not used for anything and only applies in the copy of the vectors.
             */
            if (v->v_rlength == 0) {
                if (v->v_length == 1) {
                    if (isreal(v)) {
                        printnum(numbuf, *v->v_realdata);
                        out_printf("%s = %s\n", buf, numbuf);
                    } else {
                        printnum(numbuf, realpart(v->v_compdata[0]));
                        printnum(numbuf2, imagpart(v->v_compdata[0]));
                        out_printf("%s = %s,%s\n", buf, numbuf, numbuf2);
                    }
                } else {
                    out_printf("%s = (  ", buf);
                    for (i = 0; i < v->v_length; i++)
                        if (isreal(v)) {

                            printnum(numbuf, v->v_realdata[i]);
                            (void) strcpy(buf, numbuf);
                            out_send(buf);
                            ll += (int) strlen(buf);
                            ll = (ll + 7) / 8;
                            ll = ll * 8 + 1;
                            if (ll > width) {
                                out_send("\n\t");
                                ll = 9;
                            } else {
                                out_send("\t");
                            }
                        } else {
                            /*DG*/
                            printnum(numbuf, realpart(v->v_compdata[i]));
                            printnum(numbuf2, imagpart(v->v_compdata[i]));
                            (void) sprintf(buf, "%s,%s", numbuf, numbuf2);
                            out_send(buf);
                            ll += (int) strlen(buf);
                            ll = (ll + 7) / 8;
                            ll = ll * 8 + 1;
                            if (ll > width) {
                                out_send("\n\t");
                                ll = 9;
                            } else {
                                out_send("\t");
                            }
                        }
                    out_send(")\n");
                } //end if (v->v_length == 1)
            }  //end  if (v->v_rlength == 1)
        }  // end for loop
    } else {    /* Print in columns. */
        if (cp_getvar("width", CP_NUM, &i))
            width = i;
        if (width < 40)
            width = 40;
        if (width > BSIZE_SP - 2) {
            buf = TREALLOC(char, buf, width + 1);
            buf2 = TREALLOC(char, buf2, width + 1);
        }
        if (cp_getvar("height", CP_NUM, &i))
            height = i;
        if (height < 20)
            height = 20;
        nobreak = cp_getvar("nobreak", CP_BOOL, NULL);
        if (!nobreak && !ft_nopage)
            nobreak = FALSE;
        else
            nobreak = TRUE;
        noprintscale = cp_getvar("noprintscale", CP_BOOL, NULL);
        bv = vecs;
    nextpage:
        /* Make the first vector of every page be the scale... */
        /* XXX But what if there is no scale?  e.g. op, pz */
        if (!noprintscale && bv->v_plot->pl_ndims)
            if (bv->v_plot->pl_scale && !vec_eq(bv, bv->v_plot->pl_scale)) {
                nv = vec_copy(bv->v_plot->pl_scale);
                vec_new(nv);
                nv->v_link2 = bv;
                bv = nv;
            }

        ll = 8;
        for (lv = bv; lv; lv = lv->v_link2) {
            if (isreal(lv))
                ll += 16;   /* Two tabs for real, */
            else
                ll += 32;   /* 4 for complex. */
            /* Make sure we have at least 2 vectors per page... */
            if ((ll > width) && (lv != bv) && (lv != bv->v_link2))
                break;
        }

        /* Print the header on the first page only. */
        p = bv->v_plot;
        j = (width - (int) strlen(p->pl_title)) / 2;    /* Yes, keep "(int)" */
        if (j < 0)
            j = 0;
        for (i = 0; i < j; i++)
            buf2[i] = ' ';
        buf2[j] = '\0';
        out_send(buf2);
        out_send(p->pl_title);
        out_send("\n");
        out_send(buf2);
        (void) sprintf(buf, "%s  %s", p->pl_name, p->pl_date);
        j = (width - (int) strlen(buf)) / 2;
        out_send(buf);
        out_send("\n");
        for (i = 0; i < width; i++)
            buf2[i] = '-';
        buf2[width] = '\n';
        buf2[width+1] = '\0';
        out_send(buf2);
        (void) sprintf(buf, "Index   ");
        for (v = bv; v && (v != lv); v = v->v_link2) {
            if (isreal(v)) {
                (void) sprintf(buf2, "%-16.15s", v->v_name);
            } else {
                /* The frequency vector is complex but often with imaginary part = 0,
                 * this prevents to print two columns.
                 */
                if (eq(v->v_name, "frequency")) {
                    if (imagpart(v->v_compdata[0]) == 0.0)
                        (void) sprintf(buf2, "%-16.15s", v->v_name);
                    else
                        (void) sprintf(buf2, "%-32.31s", v->v_name);
                } else {
                    (void) sprintf(buf2, "%-32.31s", v->v_name);
                }
            }
            (void) strcat(buf, buf2);
        }
        lineno = 3;
        j = 0;
        npoints = 0;
        for (v = bv; (v && (v != lv)); v = v->v_link2)
            if (v->v_length > npoints)
                npoints = v->v_length;
    pbreak:     /* New page. */
        out_send(buf);
        out_send("\n");
        for (i = 0; i < width; i++)
            buf2[i] = '-';
        buf2[width] = '\n';
        buf2[width+1] = '\0';
        out_send(buf2);
        lineno += 2;
    loop:
        while ((j < npoints) && (lineno < height)) {
            out_printf("%d\t", j);
            for (v = bv; (v && (v != lv)); v = v->v_link2) {
                if (v->v_length <= j) {
                    if (isreal(v))
                        out_send("\t\t");
                    else
                        out_send("\t\t\t\t");
                } else {
                    if (isreal(v)) {
                        printnum(numbuf, v->v_realdata[j]);
                        out_printf("%s\t", numbuf);
                    } else {
                        /* In case of a single frequency and have a real part avoids print imaginary part equals 0. */
                        if (eq(v->v_name, "frequency") &&
                            imagpart(v->v_compdata[j]) == 0.0)
                        {
                            printnum(numbuf, realpart(v->v_compdata[j]));
                            out_printf("%s\t", numbuf);
                        } else {
                            printnum(numbuf, realpart(v->v_compdata[j]));
                            printnum(numbuf2, imagpart(v->v_compdata[j]));
                            out_printf("%s,\t%s\t", numbuf, numbuf2);
                        }
                    }
                }
            }
            out_send("\n");
            j++;
            lineno++;
        }
        if ((j == npoints) && (lv == NULL)) /* No more to print. */
            goto done;
        if (j == npoints) { /* More vectors to print. */
            bv = lv;
            out_send("\f\n");   /* Form feed. */
            goto nextpage;
        }

        /* Otherwise go to a new page. */
        lineno = 0;
        if (nobreak)
            goto loop;
        else
            out_send("\f\n");   /* Form feed. */
        goto pbreak;
    }
done:
    /* Get rid of the vectors. */
    free_pnode(names);
    tfree(buf);
    tfree(buf2);
}


/* Write out some data. write filename expr ... Some cleverness here is
 * required.  If the user mentions a few vectors from various plots,
 * probably he means for them to be written out seperate plots.  In any
 * case, we have to be sure to write out the scales for everything we
 * write...
 */

void
com_write(wordlist *wl)
{
    char *file, buf[BSIZE_SP];
    struct pnode *pn;
    struct dvec *d, *vecs = NULL, *lv = NULL, *end, *vv;
    static wordlist all = { "all", NULL, NULL };
    struct pnode *names;
    bool ascii = AsciiRawFile;
    bool scalefound, appendwrite;
    struct plot *tpl, newplot;

    if (wl) {
        file = wl->wl_word;
        wl = wl->wl_next;
    } else {
        file = ft_rawfile;
    }

    if (cp_getvar("filetype", CP_STRING, buf)) {
        if (eq(buf, "binary"))
            ascii = FALSE;
        else if (eq(buf, "ascii"))
            ascii = TRUE;
        else
            fprintf(cp_err, "Warning: strange file type %s\n", buf);
    }
    appendwrite = cp_getvar("appendwrite", CP_BOOL, NULL);

    if (wl)
        names = ft_getpnames(wl, TRUE);
    else
        names = ft_getpnames(&all, TRUE);

    if (names == NULL)
        return;

    for (pn = names; pn; pn = pn->pn_next) {
        d = ft_evaluate(pn);
        if (!d)
            goto done;
        if (vecs)
            lv->v_link2 = d;
        else
            vecs = d;
        for (lv = d; lv->v_link2; lv = lv->v_link2)
            ;
    }

    /* Now we have to write them out plot by plot. */

    while (vecs) {
        tpl = vecs->v_plot;
        tpl->pl_written = TRUE;
        end = NULL;
        memcpy(&newplot, tpl, sizeof(struct plot));
        scalefound = FALSE;

        /* Figure out how many vectors are in this plot. Also look
         * for the scale, or a copy of it, which may have a different
         * name.
         */
        for (d = vecs; d; d = d->v_link2) {
            if (d->v_plot == tpl) {
                char *basename = vec_basename(d);
                vv = vec_copy(d);
                /* Note that since we are building a new plot
                 * we don't want to vec_new this one...
                 */
                tfree(vv->v_name);
                vv->v_name = basename;

                if (end)
                    end->v_next = vv;
                else
                    end = newplot.pl_dvecs = vv;
                end = vv;

                if (vec_eq(d, tpl->pl_scale)) {
                    newplot.pl_scale = vv;
                    scalefound = TRUE;
                }
            }
        }
        end->v_next = NULL;

        /* Maybe we shouldn't make sure that the default scale is
         * present if nobody uses it.
         */
        if (!scalefound) {
            newplot.pl_scale = vec_copy(tpl->pl_scale);
            newplot.pl_scale->v_next = newplot.pl_dvecs;
            newplot.pl_dvecs = newplot.pl_scale;
        }

        /* Now let's go through and make sure that everything that
         * has its own scale has it in the plot.
         */
        for (;;) {
            scalefound = FALSE;
            for (d = newplot.pl_dvecs; d; d = d->v_next) {
                if (d->v_scale) {
                    for (vv = newplot.pl_dvecs; vv; vv = vv->v_next)
                        if (vec_eq(vv, d->v_scale))
                            break;
                    if (!vv) {
                        /* We have to grab it... */
                        vv = vec_copy(d->v_scale);
                        vv->v_next = newplot.pl_dvecs;
                        newplot.pl_dvecs = vv;
                        scalefound = TRUE;
                    }
                }
            }

            if (!scalefound)
                break;
            /* Otherwise loop through again... */
        }

        if (ascii)
            raw_write(file, &newplot, appendwrite, FALSE);
        else
            raw_write(file, &newplot, appendwrite, TRUE);

        for (vv = newplot.pl_dvecs; vv;) {
            struct dvec *next_vv = vv->v_next;
            vv->v_plot = NULL;
            vec_free(vv);
            vv = next_vv;
        }

        /* Now throw out the vectors we have written already... */
        for (d = vecs, lv = NULL;  d; d = d->v_link2)
            if (d->v_plot == tpl) {
                if (lv) {
                    lv->v_link2 = d->v_link2;
                    d = lv;
                } else {
                    vecs = d->v_link2;
                }
            } else {
                lv = d;
            }
        /* If there are more plots we want them appended... */
        appendwrite = TRUE;
    }

done:
    free_pnode(names);
}


/* Write scattering parameters into a file with Touchstone File Format Version 1
   with command wrs2p file .
   Format info from http://www.eda.org/ibis/touchstone_ver2.0/touchstone_ver2_0.pdf
   See example 13 on page 15: Two port, ASCII, real-imaginary
   Check if S11, S21, S12, S22 and frequency vectors are available
   Check if vector Rbase is available
   Call spar_write()
*/

void
com_write_sparam(wordlist *wl)
{
    char *file;
    char *sbuf[6];
    wordlist *wl_sparam;
    struct pnode *pn;
    struct dvec *d, *vecs = NULL, *lv = NULL, *end, *vv, *Rbasevec = NULL;
    struct pnode *names;
    bool scalefound;
    struct plot *tpl, newplot;
    double Rbaseval;

    if (wl)
        file = wl->wl_word;
    else
        file = "s_param.s2p";

    /* generate wordlist with all vectors required*/
    sbuf[0] = "frequency";
    sbuf[1] = "S11";
    sbuf[2] = "S21";
    sbuf[3] = "S12";
    sbuf[4] = "S22";
    sbuf[5] = NULL;
    wl_sparam = wl_build(sbuf);

    names = ft_getpnames(wl_sparam, TRUE);
    if (names == NULL)
        goto done;

    for (pn = names; pn; pn = pn->pn_next) {
        d = ft_evaluate(pn);
        if (!d)
            goto done;

        if (vecs)
            lv->v_link2 = d;
        else
            vecs = d;

        for (lv = d; lv->v_link2; lv = lv->v_link2)
            ;
    }

    Rbasevec = vec_get("Rbase");
    if (Rbasevec) {
        Rbaseval = Rbasevec->v_realdata[0];
    } else {
        fprintf(stderr, "Error: No Rbase vector given\n");
        goto done;
    }

    /* Now we have to write them out plot by plot. */

    while (vecs) {
        tpl = vecs->v_plot;
        tpl->pl_written = TRUE;
        end = NULL;
        memcpy(&newplot, tpl, sizeof(struct plot));
        scalefound = FALSE;

        /* Figure out how many vectors are in this plot. Also look
         * for the scale, or a copy of it, which may have a different
         * name.
         */
        for (d = vecs; d; d = d->v_link2) {
            if (d->v_plot == tpl) {
                char *basename = vec_basename(d);
                vv = vec_copy(d);
                /* Note that since we are building a new plot
                 * we don't want to vec_new this one...
                 */
                tfree(vv->v_name);
                vv->v_name = basename;

                if (end)
                    end->v_next = vv;
                else
                    end = newplot.pl_dvecs = vv;
                end = vv;

                if (vec_eq(d, tpl->pl_scale)) {
                    newplot.pl_scale = vv;
                    scalefound = TRUE;
                }
            }
        }
        end->v_next = NULL;

        /* Maybe we shouldn't make sure that the default scale is
         * present if nobody uses it.
         */
        if (!scalefound) {
            newplot.pl_scale = vec_copy(tpl->pl_scale);
            newplot.pl_scale->v_next = newplot.pl_dvecs;
            newplot.pl_dvecs = newplot.pl_scale;
        }

        /* Now let's go through and make sure that everything that
         * has its own scale has it in the plot.
         */
        for (;;) {
            scalefound = FALSE;
            for (d = newplot.pl_dvecs; d; d = d->v_next) {
                if (d->v_scale) {
                    for (vv = newplot.pl_dvecs; vv; vv = vv->v_next)
                        if (vec_eq(vv, d->v_scale))
                            break;
                    if (!vv) {
                        /* We have to grab it... */
                        vv = vec_copy(d->v_scale);
                        vv->v_next = newplot.pl_dvecs;
                        newplot.pl_dvecs = vv;
                        scalefound = TRUE;
                    }
                }
            }
            if (!scalefound)
                break;
            /* Otherwise loop through again... */
        }

        spar_write(file, &newplot, Rbaseval);

        for (vv = newplot.pl_dvecs; vv;) {
            struct dvec *next_vv = vv->v_next;
            vv->v_plot = NULL;
            vec_free(vv);
            vv = next_vv;
        }

        /* Now throw out the vectors we have written already... */
        for (d = vecs, lv = NULL;  d; d = d->v_link2)
            if (d->v_plot == tpl) {
                if (lv) {
                    lv->v_link2 = d->v_link2;
                    d = lv;
                } else {
                    vecs = d->v_link2;
                }
            } else {
                lv = d;
            }
    }

done:
    free_pnode(names);
    wl_free(wl_sparam);
}


/* If the named vectors have more than 1 dimension, then consider
 * to be a collection of one or more matrices.  This command transposes
 * each named matrix.
 */
void
com_transpose(wordlist *wl)
{
    struct dvec *d;
    char *s;

    while (wl) {
        s = cp_unquote(wl->wl_word);
        d = vec_get(s);
        tfree(s); /*DG: Avoid Memory Leak */
        if (d == NULL)
            fprintf(cp_err, "Error: no such vector as %s.\n", wl->wl_word);
        else
            while (d) {
                vec_transpose(d);
                d = d->v_link2;
            }
        if (wl->wl_next == NULL)
            return;
        wl = wl->wl_next;
    }
}


/* Take a set of vectors and form a new vector of the nth elements of each. */
void
com_cross(wordlist *wl)
{
    char *newvec, *s;
    struct dvec *n, *v, *vecs = NULL, *lv = NULL;
    struct pnode *pn, *names;
    int i, ind;
    bool comp = FALSE;
    double *d;

    newvec = wl->wl_word;
    wl = wl->wl_next;
    s = wl->wl_word;
    if ((d = ft_numparse(&s, FALSE)) == NULL) {
        fprintf(cp_err, "Error: bad number %s\n", wl->wl_word);
        return;
    }
    if ((ind = (int)*d) < 0) {
        fprintf(cp_err, "Error: badstrchr %d\n", ind);
        return;
    }
    wl = wl->wl_next;
    names = ft_getpnames(wl, TRUE);
    for (pn = names; pn; pn = pn->pn_next) {
        if ((n = ft_evaluate(pn)) == NULL)
            goto done;

        if (!vecs)
            vecs = lv = n;
        else
            lv->v_link2 = n;

        for (lv = n; lv->v_link2; lv = lv->v_link2)
            ;
    }

    for (n = vecs, i = 0; n; n = n->v_link2) {
        if (iscomplex(n))
            comp = TRUE;
        i++;
    }

    vec_remove(newvec);
    v = dvec_alloc(copy(newvec),
                   vecs
                   ? vecs->v_type
                   : SV_NOTYPE,
                   comp
                   ? (VF_COMPLEX | VF_PERMANENT)
                   : (VF_REAL | VF_PERMANENT),
                   i, NULL);

    /* Now copy the ind'ths elements into this one. */
    for (n = vecs, i = 0; n; n = n->v_link2, i++)
        if (n->v_length > ind) {
            if (comp) {
                v->v_compdata[i] = n->v_compdata[ind];
            } else {
                v->v_realdata[i] = n->v_realdata[ind];
            }
        } else {
            if (comp) {
                realpart(v->v_compdata[i]) = 0.0;
                imagpart(v->v_compdata[i]) = 0.0;
            } else {
                v->v_realdata[i] = 0.0;
            }
        }
    vec_new(v);
    cp_addkword(CT_VECTOR, v->v_name);

done:
    free_pnode(names);
}


void
com_destroy(wordlist *wl)
{
    struct plot *pl, *npl = NULL;

    if (!wl) {
        DelPlotWindows(plot_cur);
        killplot(plot_cur);
    } else if (eq(wl->wl_word, "all")) {
        for (pl = plot_list; pl; pl = npl) {
            npl = pl->pl_next;
            if (!eq(pl->pl_typename, "const")) {
                DelPlotWindows(pl);
                killplot(pl);
            } else {
                plot_num = 1;
            }
        }
    } else {
        while (wl) {
            for (pl = plot_list; pl; pl = pl->pl_next)
                if (eq(pl->pl_typename, wl->wl_word))
                    break;
            if (pl) {
                DelPlotWindows(pl);
                killplot(pl);
            } else {
                fprintf(cp_err, "Error: no such plot %s\n", wl->wl_word);
            }
            wl = wl->wl_next;
        }
    }
}


static void
killplot(struct plot *pl)
{
    struct dvec *v, *nv = NULL;
    struct plot *op;

    if (eq(pl->pl_typename, "const")) {
        fprintf(cp_err, "Error: can't destroy the constant plot\n");
        return;
    }
    /*  pl_dvecs, pl_scale */
    for (v = pl->pl_dvecs; v; v = nv) {
        nv = v->v_next;
        vec_free(v);
    }
    /* unlink from plot_list (linked via pl_next) */
    if (pl == plot_list) {
        plot_list = pl->pl_next;
        if (pl == plot_cur)
            plot_cur = plot_list;
    } else {
        for (op = plot_list; op; op = op->pl_next)
            if (op->pl_next == pl)
                break;
        if (!op)
            fprintf(cp_err,
                    "Internal Error: kill plot -- not in list\n");
        op->pl_next = pl->pl_next;
        if (pl == plot_cur)
            plot_cur = op;
    }
    tfree(pl->pl_title);
    tfree(pl->pl_name);
    tfree(pl->pl_typename);
    wl_free(pl->pl_commands);
    tfree(pl->pl_date);
    if (pl->pl_ccom)
        throwaway(pl->pl_ccom);

    if (pl->pl_env)
        free_struct_variable(pl->pl_env);
    if (pl->pl_lookup_table)
        nghash_free(pl->pl_lookup_table, NULL, NULL);
    tfree(pl);
}


/* shared ngspice: remove plot 'const' upon 'quit'
   constanplot is not dynamically allocated on the heap */
void
destroy_const_plot(void)
{
    struct dvec *v, *nv = NULL;
    struct plot *pl = &constantplot;

    /*  pl_dvecs, pl_scale */
    for (v = pl->pl_dvecs; v; v = nv) {
        nv = v->v_next;
        vec_free(v);
    }
    wl_free(pl->pl_commands);
    if (pl->pl_ccom)
        throwaway(pl->pl_ccom);

    if (pl->pl_env)
        free_struct_variable(pl->pl_env);
    if (pl->pl_lookup_table)
        nghash_free(pl->pl_lookup_table, NULL, NULL);
}


/* delete all windows with graphs dedrived from a given plot */
static void
DelPlotWindows(struct plot *pl)
{
    /* do this only if windows or X11 is defined */
#if defined(HAS_WINGUI) || !defined(X_DISPLAY_MISSING)
    GRAPH *dgraph;
    int n;
    /* find and remove all graph structures derived from a given plot */
    for (n = 1; n < 100; n++) { /* should be no more than 100 */
        dgraph = FindGraph(n);
        if (dgraph) {
            if (ciprefix(pl->pl_typename, dgraph->plotname))
                RemoveWindow(dgraph);
        }
        /* We have to run through all potential graph ids. If some numbers are
           already missing, 'else break;' might miss the plotwindow to be removed. */
        /* else
           break;
        */
    }
#else
    NG_IGNORE(pl);
#endif
}


/*
 * command 'setplot'
 *   print a list of plots available
 * command 'setplot <plotname>'
 *   make <plotname> the current plot
 * command 'setplot new'
 *   create a new plot
 */

void
com_splot(wordlist *wl)
{
    struct plot *pl;

    if (wl) {
        plot_setcur(wl->wl_word);
        return;
    }

    fprintf(cp_out, "List of plots available:\n\n");
    for (pl = plot_list; pl; pl = pl->pl_next)
        fprintf(cp_out, "%s%s\t%s (%s)\n",
                (pl == plot_cur) ? "Current " : "\t",
                pl->pl_typename, pl->pl_title, pl->pl_name);
}
