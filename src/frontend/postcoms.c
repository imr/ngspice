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
#include "variable.h"
#include "ngspice/stringskip.h"
#include "../misc/misc_time.h"
#include "parser/complete.h" /* va: throwaway */
#include "plotting/plotting.h"

#include "ngspice/compatmode.h"
#include "ngspice/dstring.h"
#include "numparam/general.h"
#include "sndprint.h"

static void killplot(struct plot *pl);
static void DelPlotWindows(struct plot *pl);

/* check if the user want's to delete the scale vector of the current plot.
   This should not happen, because then redrawing the graph crashes ngspice */
static bool
is_scale_vec_of_current_plot(const char *v_name)
{
    if (!plot_cur) { /* no current plot */
        return FALSE;
    }

    const struct dvec * const pl_scale = plot_cur->pl_scale;
    if (!pl_scale) { /* no scale vector */
        return FALSE;
    }

    /* Test if this vector's name matches the scale vector's name */
    return cieq(v_name, pl_scale->v_name);
} /* end of function is_scale_vec_of_current_plot */


/* Remove vectors in the wordlist from the current plot */
void
com_unlet(wordlist *wl)
{
    for ( ; wl != (wordlist *) NULL; wl = wl->wl_next) {
        /* Don't delete the scale vector of the current plot */
        const char * const vector_name = wl->wl_word;
        if (is_scale_vec_of_current_plot(vector_name)) {
            /* If it is the scale vector of the current plot, print a
             * warning. Note that if it is true,  the scale vector name must
             * exist, so no part of plot_cur->pl_scale->v_name can be null. */
            fprintf(cp_err,
                    "\nWarning: Scale vector '%s' of the current plot "
                    "cannot be deleted!\n"
                    "Command 'unlet %s' is ignored.\n\n",
                    plot_cur->pl_scale->v_name, vector_name);
        }
        else {
            vec_remove(vector_name);
        }
    } /* end of loop over vectors to delete */
} /* end of function com_unlet */


/* Remove zero length vectors from the current plot */
void
com_remzerovec(wordlist* wl)
{
    NG_IGNORE(wl);
    
    struct dvec* ov;

    for (ov = plot_cur->pl_dvecs; ov; ov = ov->v_next) {
        if (ov->v_length == 0) {
            ov->v_flags &= ~VF_PERMANENT;
            /* Remove from the keyword list. */
            cp_remkword(CT_VECTOR, ov->v_name);
        }
    } /* end of loop over vectors to delete */
} /* end of function com_remzerovec */


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
    int i, j, ll, width = DEF_WIDTH, height = DEF_HEIGHT, npoints, lineno, npages = 0;
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

    names = ft_getpnames_quotes(wl, TRUE);

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
        if (cp_getvar("width", CP_NUM, &i, 0))
            width = i;
        if (width < 60)
            width = 60;
        if (width > BSIZE_SP - 2)
            buf = TREALLOC(char, buf, (size_t) width + 1);
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
        if (cp_getvar("width", CP_NUM, &i, 0))
            width = i;
        if (width < 40)
            width = 40;
        if (width > BSIZE_SP - 2) {
            buf = TREALLOC(char, buf, (size_t) width + 1);
            buf2 = TREALLOC(char, buf2, (size_t) width + 1);
        }
        if (cp_getvar("height", CP_NUM, &i, 0))
            height = i;
        if (height < 20)
            height = 20;
        nobreak = cp_getvar("nobreak", CP_BOOL, NULL, 0);
        if (!nobreak && !ft_nopage)
            nobreak = FALSE;
        else
            nobreak = TRUE;
        noprintscale = cp_getvar("noprintscale", CP_BOOL, NULL, 0);
        bv = vecs;
    nextpage:
        npages++;
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

        /* Print the header on the first page only, if 'option nopage'. */
        if (!ft_nopage || npages == 1) {
            /* print the header */
            p = bv->v_plot;
            j = (width - (int)strlen(p->pl_title)) / 2;    /* Yes, keep "(int)" */
            if (j < 0)
                j = 0;
            for (i = 0; i < j; i++)
                buf2[i] = ' ';
            buf2[j] = '\0';
            out_send(buf2);
            out_send(p->pl_title);
            out_send("\n");
            out_send(buf2);
            (void)sprintf(buf, "%s  %s", p->pl_name, p->pl_date);
            out_send(buf);
            out_send("\n");
        }
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
            if(nobreak)
                out_send("\n");   /* return without form feed. */
            else
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

#if defined(HAVE_LIBSNDFILE) && defined(HAVE_LIBSAMPLERATE)

/* tweaked version of print - write sound-files
*/
void
com_sndprint(wordlist* wl)
{
    struct dvec* v, * lv = NULL, * bv, * vecs = NULL;
    int i, j, npoints;
    struct pnode* nn;
    int ngood;

    if (wl == NULL)
        return;

#ifdef HAS_PROGREP
    SetAnalyse("Wav out", 0);
#endif

    if (eq(wl->wl_word, "col")) {
        wl = wl->wl_next;
    }
    else if (eq(wl->wl_word, "line")) {
        wl = wl->wl_next;
    }

    ngood = 0;
    for (nn = ft_getpnames(wl, TRUE); nn; nn = nn->pn_next) {
        v = ft_evaluate(nn);
        if (!v)
            continue;
        if (!vecs)
            vecs = lv = v;
        else
            lv->v_link2 = v;
        for (lv = v; lv->v_link2; lv = lv->v_link2)
            ;
        ngood += 1;
    }

    if (!vecs || vecs->v_plot->pl_scale->v_type != SV_TIME)
        return;

    if (!ngood) return;

    snd_init(ngood);
    bv = vecs;

    i = j = 0;
    npoints = 0;
    for (v = bv; v; v = v->v_link2)
        if (v->v_length > npoints)
            npoints = v->v_length;
    double samplerate = snd_get_samplerate();
    while ((j < npoints)) {

        double tme;
        if (isreal(bv->v_plot->pl_scale))
            tme = bv->v_plot->pl_scale->v_realdata[j] * samplerate;
        else
            tme = realpart(bv->v_plot->pl_scale->v_compdata[j]) * samplerate;
        int c = 0;
        for (v = bv; v; v = v->v_link2) {
            if (v->v_length <= j) {
                i += snd_send(tme, c, 0.0);
            }
            else {
                if (isreal(v))
                    i += snd_send(tme, c, v->v_realdata[j]);
                else
                    i += snd_send(tme, c, realpart(v->v_compdata[j]));
            }
            c++;
        }
        j++;
    }
    snd_close();
    printf("Info: wrote %i audio-samples from %i data-points\n", i / ngood, j);
    /* Get rid of the vectors. */
    return;
 }

/* Configure sndprint. */
void
com_sndparam(wordlist* wl)
{
	char* copypath;
	int i = 0;
	char* file = NULL;
	int srate = 48000;
	int fmt = -1;
	double mult = 1.0;
	double off = 0.0;
	int oversampling = 4;

	while (wl) {
		copypath = cp_unquote(wl->wl_word);
		switch (++i) {
		case 1:
			file = strdup(copypath);
			break;
		case 2:
			srate = atoi(copypath);
			break;
		case 3:
			fmt = snd_format(copypath);
			break;
		case 4:
			mult = atof(copypath);
			break;
		case 5:
			off = atof(copypath);
			break;
		case 6:
			oversampling = atoi(copypath);
			break;
		default:
			printf("Warning: unknown argument\n");

		}
		tfree(copypath);
		wl = wl->wl_next;

	}

	if (file)
		snd_configure(file, srate, fmt, mult, off, oversampling);
	return;
}
#endif // HAVE_LIBSNDFILE


/* Write out some data into a ngspice raw file with 'write filename expr'.
 * If vectors (expr) from various plots are selected, they are written
 * out as seperate plots.  In any case, we have to be sure to write out
 * the scales for everything we write. If expr is omitted, all vectors
 * of the current plot are written.
 */
void
com_write(wordlist *wl)
{
    char *file, buf[BSIZE_SP];
    struct pnode *pn;
    struct dvec *d, *vecs = NULL, *lv = NULL, *end, *vv;
    static wordlist all = { "all", NULL, NULL };
    struct pnode *names = NULL;
    bool ascii = AsciiRawFile;
    bool scalefound, appendwrite, plainwrite = FALSE;
    struct plot *tpl, newplot;

    if (wl) {
        file = wl->wl_word;
        wl = wl->wl_next;
    } else {
        file = ft_rawfile;
    }

    if (cp_getvar("filetype", CP_STRING, buf, sizeof(buf))) {
        if (eq(buf, "binary"))
            ascii = FALSE;
        else if (eq(buf, "ascii"))
            ascii = TRUE;
        else
            fprintf(cp_err, "Warning: strange file type %s\n", buf);
    }
    appendwrite = cp_getvar("appendwrite", CP_BOOL, NULL, 0);

    plainwrite = cp_getvar("plainwrite", CP_BOOL, NULL, 0);

    /* If variable plainwrite is set, we do not expand equations, serve v vs vs etc.
       We offer plain writing of the vectors. This enables node names containing +, -, / etc. */
    if (!plainwrite) {
        if (wl)
            names = ft_getpnames_quotes(wl, TRUE);
        else
            names = ft_getpnames_quotes(&all, TRUE);

        if (names == NULL) {
            fprintf(stderr, "Error during 'write': no writable vector found.\n");
            return;
        }

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
    }
    else {
        wordlist* wli;
        if (!wl)
            wl = &all;
        for (wli = wl; wli; wli = wli->wl_next) {
            d = vec_get(wli->wl_word);
            if (!d) {
                fprintf(stderr, "Error during 'write': vector %s not found\n", wli->wl_word);
                goto done;
            }
            if (vecs)
                lv->v_link2 = d;
            else
                vecs = d;
            for (lv = d; lv->v_link2; lv = lv->v_link2)
                ;
        }
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
                txfree(vv->v_name);
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

        raw_write(file, &newplot, appendwrite, !ascii);

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


/* Enhancement-64: write an N-port Touchstone v1 file (.sNp) directly from
   the S_i_j vectors of the current .sp plot. Layout per the Touchstone 1.x
   spec: `# Hz S RI R <Rbase>` option line; for N >= 3 the matrix is
   row-major with at most FOUR complex pairs per data line and every matrix
   row starting on a new line (the first row follows the frequency value);
   a 1-port is a single pair per line. (The classic 2-port S11 S21 S12 S22
   column order is handled by the original spar_write() path.) */
/* Enhancement-72: `rdsnp <file> [nports]` -- read a Touchstone v1 file
   into a new plot ("Touchstone import") holding a real `frequency` scale
   in Hz plus complex `S_i_j` (or Y/Z) vectors matching the .sp plot's
   conventions (Y/Z de-normalized back to absolute values), so imported
   measurement data can be compared 1:1 against simulated vectors. The
   port count comes from the `.sNp` extension unless given explicitly. */
void
com_read_sparam(wordlist *wl)
{
    FILE *fp;
    char line[4096];
    char *file;
    int nports = 0;
    double fscale = 1.0, rbase = 50.0;
    char fmt = 'r', param = 's';
    bool have_opt_line = FALSE;
    double *data = NULL;
    size_t ndata = 0, adata = 0;
    int per_block, npts, i, j, k;
    struct plot *new;
    struct dvec *freqv, *last;

    if (!wl) {
        fprintf(stderr, "Error: rdsnp requires a file name\n");
        return;
    }
    file = wl->wl_word;
    if (wl->wl_next)
        nports = atoi(wl->wl_next->wl_word);
    if (nports <= 0) {
        /* infer from the .sNp extension */
        char *dot = strrchr(file, '.');
        if (dot && (dot[1] == 's' || dot[1] == 'S')) {
            nports = atoi(dot + 2);
        }
    }
    if (nports <= 0) {
        fprintf(stderr,
                "Error: cannot infer the port count from '%s'; use rdsnp <file> <nports>\n",
                file);
        return;
    }

    if ((fp = fopen(file, "r")) == NULL) {
        perror(file);
        return;
    }

    while (fgets(line, sizeof(line), fp)) {
        char *t = skip_ws(line);
        if (*t == '\0' || *t == '!')
            continue;
        if (*t == '#') {
            /* option line: [unit] [param] [format] [R n], any order */
            char tok[64];
            int pos = 1;
            while (sscanf(t + pos, " %63s%n", tok, &i) == 1) {
                pos += i;
                if (cieq(tok, "hz"))
                    fscale = 1.0;
                else if (cieq(tok, "khz"))
                    fscale = 1e3;
                else if (cieq(tok, "mhz"))
                    fscale = 1e6;
                else if (cieq(tok, "ghz"))
                    fscale = 1e9;
                else if (cieq(tok, "s") || cieq(tok, "y") || cieq(tok, "z"))
                    param = (char) tolower_c(tok[0]);
                else if (cieq(tok, "ri"))
                    fmt = 'r';
                else if (cieq(tok, "ma"))
                    fmt = 'm';
                else if (cieq(tok, "db"))
                    fmt = 'd';
                else if (cieq(tok, "r")) {
                    if (sscanf(t + pos, " %lg%n", &rbase, &i) == 1)
                        pos += i;
                }
            }
            have_opt_line = TRUE;
            continue;
        }
        /* data line: append every number */
        {
            char *q = t;
            double v;
            while (sscanf(q, " %lg%n", &v, &i) == 1) {
                if (ndata == adata) {
                    adata = adata ? 2 * adata : 1024;
                    data = TREALLOC(double, data, adata);
                }
                data[ndata++] = v;
                q += i;
            }
        }
    }
    (void) fclose(fp);

    if (!have_opt_line)
        fprintf(stderr, "Warning: no '#' option line in %s; assuming Hz S RI R 50\n", file);

    per_block = 1 + 2 * nports * nports;
    if (ndata == 0 || ndata % (size_t) per_block != 0) {
        fprintf(stderr,
                "Error: %s holds %zu numbers, not a multiple of %d (1 + 2*%d^2) -- wrong port count?\n",
                file, ndata, per_block, nports);
        tfree(data);
        return;
    }
    npts = (int) (ndata / (size_t) per_block);

    /* build the plot (same pattern as com_linearize) */
    new = plot_alloc("sp");
    new->pl_name = tprintf("Touchstone import %s", file);
    new->pl_title = copy(file);
    new->pl_date = copy(datestring());
    new->pl_next = plot_list;
    plot_new(new);
    plot_setcur(new->pl_typename);
    plot_list = new;

    freqv = dvec_alloc(copy("frequency"), SV_FREQUENCY, VF_REAL | VF_PERMANENT, npts, NULL);
    freqv->v_plot = new;
    for (k = 0; k < npts; k++)
        freqv->v_realdata[k] = data[(size_t) k * (size_t) per_block] * fscale;
    new->pl_scale = new->pl_dvecs = freqv;
    last = freqv;

    for (i = 0; i < nports; i++) {
        for (j = 0; j < nports; j++) {
            char nb[40];
            struct dvec *v;
            int pair;
            /* position of pair (i,j) within a block, matching the writer */
            if (nports == 2) {
                static const int order[2][2] = {{0, 2}, {1, 3}};
                pair = order[i][j];
            } else {
                pair = i * nports + j;
            }
            (void) sprintf(nb, "%c_%d_%d", toupper_c(param), i + 1, j + 1);
            v = dvec_alloc(copy(nb), SV_NOTYPE, VF_COMPLEX | VF_PERMANENT, npts, NULL);
            v->v_plot = new;
            for (k = 0; k < npts; k++) {
                size_t base = (size_t) k * (size_t) per_block + 1 + 2 * (size_t) pair;
                double a = data[base], b = data[base + 1];
                double re, im;
                if (fmt == 'r') {
                    re = a;
                    im = b;
                } else {
                    double mag = (fmt == 'd') ? pow(10.0, a / 20.0) : a;
                    re = mag * cos(b * M_PI / 180.0);
                    im = mag * sin(b * M_PI / 180.0);
                }
                /* de-normalize back to absolute Y/Z (v1 files carry Y*R, Z/R) */
                if (param == 'y') {
                    re /= rbase;
                    im /= rbase;
                } else if (param == 'z') {
                    re *= rbase;
                    im *= rbase;
                }
                v->v_compdata[k].cx_real = re;
                v->v_compdata[k].cx_imag = im;
            }
            last->v_next = v;
            last = v;
        }
    }

    /* publish Rbase so the imported plot round-trips through wrsnp */
    {
        struct dvec *rv = dvec_alloc(copy("Rbase"), SV_NOTYPE, VF_REAL | VF_PERMANENT, 1, NULL);
        rv->v_plot = new;
        rv->v_realdata[0] = rbase;
        last->v_next = rv;
    }

    fprintf(stdout, "%d-port %c-parameters (%d points) read from %s into plot '%s'\n",
            nports, toupper_c(param), npts, file, new->pl_typename);
    tfree(data);
}


/* Enhancement-72: one complex value from an sp-plot vector. */
static void
spar_get(struct dvec *v, int k, double *re, double *im)
{
    if (isreal(v)) {
        *re = v->v_realdata[k];
        *im = 0.0;
    } else {
        *re = realpart(v->v_compdata[k]);
        *im = imagpart(v->v_compdata[k]);
    }
}

/* Enhancement-72: emit one network value in the requested Touchstone
   format (RI / MA / DB) after applying the v1 normalization for Y (x R)
   and Z (/ R) parameters. */
static void
spar_emit(FILE *fp, int prec, double re, double im, char fmt)
{
    if (fmt == 'r') {
        fprintf(fp, "  % .*e % .*e", prec, re, prec, im);
    } else {
        double mag = hypot(re, im);
        double ang = (mag > 0.0) ? atan2(im, re) * 180.0 / M_PI : 0.0;
        if (fmt == 'd')
            fprintf(fp, "  % .*e % .*e", prec,
                    20.0 * log10(mag > 0.0 ? mag : 1e-300), prec, ang);
        else
            fprintf(fp, "  % .*e % .*e", prec, mag, prec, ang);
    }
}

/* Enhancement-72: generalized Touchstone v1 writer -- any port count,
   RI/MA/DB formats, S/Y/Z parameters (Y/Z normalized to Rbase per the
   v1 spec), and Hz/kHz/MHz/GHz frequency units. The 2-port matrix uses
   the Touchstone-special S11 S21 S12 S22 column order on one line;
   N >= 3 is row-major with at most four pairs per data line and each
   matrix row on its own line; a 1-port is one pair per line. */
static void
spar_write_np(const char *file, int nports, double Rbaseval,
              char fmt, char param, double fscale, const char *funit)
{
    struct dvec *freqv = vec_get("frequency");
    struct dvec **sv;
    FILE *fp;
    int i, j, k, npts, prec, inrow;
    const char *fmt_str = (fmt == 'm') ? "MA" : (fmt == 'd') ? "DB" : "RI";
    char param_uc = (char) toupper_c(param);

    if (!freqv) {
        fprintf(stderr, "Error: no frequency vector (run a .sp analysis first)\n");
        return;
    }
    npts = freqv->v_length;
    prec = 6;

    sv = TMALLOC(struct dvec *, (size_t) (nports * nports));
    for (i = 0; i < nports; i++)
        for (j = 0; j < nports; j++) {
            char nb[40];
            (void) sprintf(nb, "%c_%d_%d", param_uc, i + 1, j + 1);
            sv[i * nports + j] = vec_get(nb);
            if (!sv[i * nports + j] || sv[i * nports + j]->v_length != npts) {
                fprintf(stderr, "Error: vector %s missing or of wrong length\n", nb);
                tfree(sv);
                return;
            }
        }

    if ((fp = fopen(file, "w")) == NULL) {
        perror(file);
        tfree(sv);
        return;
    }

    fprintf(fp, "!%d-port %c-parameter file\n", nports, param_uc);
    fprintf(fp, "!Title: %s\n", freqv->v_plot ? freqv->v_plot->pl_title : "");
    fprintf(fp, "!Generated by ngspice at %s\n",
            freqv->v_plot ? freqv->v_plot->pl_date : "");
    fprintf(fp, "# %s %c %s R %g\n", funit, param_uc, fmt_str, Rbaseval);
    if (nports == 2)
        fprintf(fp, "!freq  %c11  %c21  %c12  %c22  (%s pairs)\n",
                param_uc, param_uc, param_uc, param_uc, fmt_str);

    for (k = 0; k < npts; k++) {
        double f = isreal(freqv) ? freqv->v_realdata[k]
                                 : realpart(freqv->v_compdata[k]);
        fprintf(fp, "% .*e", prec, f / fscale);
        if (nports == 2) {
            /* Touchstone 2-port column order: 11, 21, 12, 22 */
            static const int order[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
            int n;
            for (n = 0; n < 4; n++) {
                double re, im;
                spar_get(sv[order[n][0] * 2 + order[n][1]], k, &re, &im);
                if (param == 'y') {
                    re *= Rbaseval;
                    im *= Rbaseval;
                } else if (param == 'z') {
                    re /= Rbaseval;
                    im /= Rbaseval;
                }
                spar_emit(fp, prec, re, im, fmt);
            }
            fprintf(fp, "\n");
            continue;
        }
        for (i = 0; i < nports; i++) {
            if (i > 0)
                fprintf(fp, "%*s", prec + 9, "");   /* align continuation rows */
            inrow = 0;
            for (j = 0; j < nports; j++) {
                double re, im;
                spar_get(sv[i * nports + j], k, &re, &im);
                if (param == 'y') {
                    re *= Rbaseval;
                    im *= Rbaseval;
                } else if (param == 'z') {
                    re /= Rbaseval;
                    im /= Rbaseval;
                }
                if (inrow == 4) {           /* max 4 pairs per line */
                    fprintf(fp, "\n%*s", prec + 9, "");
                    inrow = 0;
                }
                spar_emit(fp, prec, re, im, fmt);
                inrow++;
            }
            fprintf(fp, "\n");             /* each matrix row on its own line */
        }
    }

    (void) fclose(fp);
    fprintf(stdout, "%d-port %c-parameters written to %s (%s, %s)\n",
            nports, param_uc, file, fmt_str, funit);
    tfree(sv);
}


/* Write scattering parameters into a file with Touchstone File Format Version 1
   with command wrs2p file (2 ports) or wrsnp file (any port count).
   Format info from http://www.eda.org/ibis/touchstone_ver2.0/touchstone_ver2_0.pdf
   See example 13 on page 15: Two port, ASCII, real-imaginary
   Check if S_1_1, S_2_1, S_1_2, S_2_2 and frequency vectors are available
   Check if vector Rbase is available (the .sp analysis publishes it since
   Enhancement-64, so no manual `let Rbase = ...` is needed anymore)
   Call spar_write() (2-port) or spar_write_np() (N-port)
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

    int nports = 0;
    /* Enhancement-72: output options -- format (ri|ma|db), parameter
       (s|y|z) and frequency unit (hz|khz|mhz|ghz), any order after the
       file name. Defaults preserve the classic wrs2p output exactly. */
    char fmt = 'r', param = 's';
    double fscale = 1.0;
    const char *funit = "Hz";
    bool have_opts = FALSE;

    if (wl)
        file = wl->wl_word;
    else
        file = "s_param.s2p";

    if (wl) {
        wordlist *w;
        for (w = wl->wl_next; w; w = w->wl_next) {
            char *t = w->wl_word;
            if (cieq(t, "ri") || cieq(t, "ma") || cieq(t, "db")) {
                fmt = (char) tolower_c(t[0]);   /* 'r' | 'm' | 'd' */
                have_opts = TRUE;
            } else if (cieq(t, "s") || cieq(t, "y") || cieq(t, "z")) {
                param = (char) tolower_c(t[0]);
                have_opts = TRUE;
            } else if (cieq(t, "hz")) {
                fscale = 1.0; funit = "Hz"; have_opts = TRUE;
            } else if (cieq(t, "khz")) {
                fscale = 1e3; funit = "kHz"; have_opts = TRUE;
            } else if (cieq(t, "mhz")) {
                fscale = 1e6; funit = "MHz"; have_opts = TRUE;
            } else if (cieq(t, "ghz")) {
                fscale = 1e9; funit = "GHz"; have_opts = TRUE;
            } else {
                fprintf(stderr,
                        "Error: unknown wrsnp option '%s' (expected ri|ma|db, s|y|z, hz|khz|mhz|ghz)\n",
                        t);
                return;
            }
        }
    }

    /* Enhancement-64: how many ports does the current sp plot hold? */
    while (nports < 99) {
        char nb[40];
        (void) sprintf(nb, "S_%d_%d", nports + 1, nports + 1);
        if (!vec_get(nb))
            break;
        nports++;
    }
    if (nports == 0) {
        fprintf(stderr, "Error: no S-parameter vectors found (run a .sp analysis first)\n");
        return;
    }

    /* Enhancement-64: the sp analysis publishes Rbase (the ports'
       reference resistance); a user-defined `let Rbase = ...` still
       overrides. The sp plot is complex, so read either data form. */
    Rbasevec = vec_get("Rbase");
    if (Rbasevec) {
        Rbaseval = isreal(Rbasevec) ? Rbasevec->v_realdata[0]
                                    : realpart(Rbasevec->v_compdata[0]);
    } else {
        fprintf(stderr, "Error: No Rbase vector given\n");
        return;
    }

    if (nports != 2 || have_opts) {
        spar_write_np(file, nports, Rbaseval, fmt, param, fscale, funit);
        return;
    }

    /* generate wordlist with all vectors required*/
    sbuf[0] = "frequency";
    sbuf[1] = "S_1_1";
    sbuf[2] = "S_2_1";
    sbuf[3] = "S_1_2";
    sbuf[4] = "S_2_2";
    sbuf[5] = NULL;
    wl_sparam = wl_build((const char * const *) sbuf);

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

    /* For each vector named in the wordlist, perform the transform to
     * it and the vectors associated with it through v_link2 */
    for ( ; wl != (wordlist *) NULL; wl = wl->wl_next) {
        s = cp_unquote(wl->wl_word);
        d = vec_get(s);
        tfree(s); /*DG: Avoid Memory Leak */
        if (d == NULL) {
            /* Print error message, but continue with other vectors */
            fprintf(cp_err, "Error: no such vector as %s.\n", wl->wl_word);
       }
        else {
            /* Transpose the named vector and vectors tied to it
             * through v_link2 */
            while (d) {
                vec_transpose(d);
                d = d->v_link2;
            }
        }
    } /* end of loop over words in wordlist */
} /* end of function com_transpose */



/* Take a set of vectors and form a new vector of the nth elements of each. */
void
com_cross(wordlist *wl)
{
    char *newvec, *s;
    struct dvec *n, *v, *vecs = NULL, *lv = NULL;
    struct pnode *pn, *names;
    int i, ind;
    bool comp = FALSE;

    newvec = wl->wl_word;
    wl = wl->wl_next;
    s = wl->wl_word;

    {
        double val;
        if (ft_numparse(&s, FALSE, &val) <= 0) {
            fprintf(cp_err, "Error: bad index value %s\n", wl->wl_word);
            return;
        }
        if ((ind = (int) val) < 0) {
            fprintf(cp_err, "Error: badstrchr %d\n", ind);
            return;
        }
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
            (int) (vecs ? vecs->v_type : SV_NOTYPE),
            comp ? (VF_COMPLEX | VF_PERMANENT) : (VF_REAL | VF_PERMANENT),
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

/* Free resources associated with "plot" datasets. The wordlist contains
 * the names of the plots to delete or the word "all" to delete all but the
 * default "const" plot, which cannot be deleted, even by name. If there are
 * no names given, the current plot is deleted */
void com_destroy(wordlist *wl)
{
    /* If no name given, delete the current output data */
    if (!wl) {
        DelPlotWindows(plot_cur);
        killplot(plot_cur);
    }
    else if (eq(wl->wl_word, "all")) { /* "all" -> all plots deleted */
        struct plot *pl, *npl = NULL;
        for (pl = plot_list; pl; pl = npl) {
            npl = pl->pl_next;
            if (!eq(pl->pl_typename, "const")) {
                DelPlotWindows(pl);
                killplot(pl);
            }
            else {
                plot_num = 1;
            }
        }
    }
    else { /* list of plots by name */
        while (wl) {
            struct plot *pl;
            for (pl = plot_list; pl; pl = pl->pl_next) {
                if (eq(pl->pl_typename, wl->wl_word)) {
                    break;
                }
            }
            if (pl) {
                DelPlotWindows(pl);
                killplot(pl);
            }
            else {
                fprintf(cp_err, "Error: no such plot %s\n", wl->wl_word);
            }
            wl = wl->wl_next;
        }
    }
} /* end of function com_destroy */



static void killplot(struct plot *pl)
{
    if (eq(pl->pl_typename, "const")) {
        fprintf(cp_err, "Error: can't destroy the constant plot\n");
        return;
    }
    /*  pl_dvecs, pl_scale */
    {
        struct dvec *v;
        struct dvec *nv;
        for (v = pl->pl_dvecs; v; v = nv) {
            nv = v->v_next;
            vec_free(v);
        }
    }

    /* unlink from plot_list (linked via pl_next) */
    if (pl == plot_list) { /* First in list */
        plot_list = pl->pl_next;
        if (pl == plot_cur) {
            plot_cur = plot_list;
        }
    }
    else { /* inside list */
        struct plot *op;
        for (op = plot_list; op; op = op->pl_next) {
            if (op->pl_next == pl) {
                break;
            }
        }
        if (!op) {
            fprintf(cp_err,
                    "Internal Error: kill plot -- not in list\n");
            return;
        }
        op->pl_next = pl->pl_next;
        if (pl == plot_cur) {
            plot_cur = op;
        }
    }
    /* delete the hash table entry for this plot */
    if (pl->pl_lookup_table) {
        nghash_free(pl->pl_lookup_table, NULL, NULL);
        pl->pl_lookup_table = NULL;
    }
    txfree(pl->pl_title);
    txfree(pl->pl_name);
    txfree(pl->pl_typename);
    wl_free(pl->pl_commands);
    txfree(pl->pl_date); /* va: also tfree (memory leak) */
    if (pl->pl_ccom)  { /* va: also tfree (memory leak) */
        throwaway(pl->pl_ccom);
    }

    if (pl->pl_env) { /* The 'environment' for this plot. */
        /* va: HOW to do? */
        printf("va: killplot should tfree pl->pl_env=(%p)\n", pl->pl_env);
        fflush(stdout);
    }
    txfree(pl); /* va: also tfree pl itself (memory leak) */
}

/* delete the const plot (called from com_quit) */
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
    /* delete the hash table entry for the const plot */
    if (pl->pl_lookup_table) {
        nghash_free(pl->pl_lookup_table, NULL, NULL);
        pl->pl_lookup_table = NULL;
    }
    wl_free(pl->pl_commands);
    if (pl->pl_ccom)    /* va: also tfree (memory leak) */
        throwaway(pl->pl_ccom);

    if (pl->pl_env) { /* The 'environment' for this plot. */
        /* va: HOW to do? */
        printf("va: killplot should tfree pl->pl_env=(%p)\n", pl->pl_env);
        fflush(stdout);
    }
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

/* Helper for com_splot(). */

static int new_str(wordlist **pwl, char **ps)
{
    *pwl = (*pwl)->wl_next;
    if (!*pwl)
        return 1;
    tfree(*ps);
    *ps = copy((*pwl)->wl_word);
    return 0;
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

        if (cieq(wl->wl_word, "new")) {
            /* The user may also supply, name, title and typename strings.
             * The typename is the 'true name' used in commands!
             */

            if (new_str(&wl, &plot_cur->pl_typename))
                return;
            if (new_str(&wl, &plot_cur->pl_title))
                return;
            new_str(&wl, &plot_cur->pl_name);
        }
        return;
    }

    fprintf(cp_out, "List of plots available:\n\n");
    for (pl = plot_list; pl; pl = pl->pl_next)
        fprintf(cp_out, "%s%s\t%s (%s)\n",
                (pl == plot_cur) ? "Current " : "\t",
                pl->pl_typename, pl->pl_title, pl->pl_name);
}
