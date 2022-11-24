/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000 AlansFixes
**********/

/*
 * Spice-2 compatibility stuff for .plot, .print, .four, and .width.
 */

#include "ngspice/ngspice.h"
#include <assert.h>
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dstring.h"
#include "ngspice/dvec.h"
#include "ngspice/fteinp.h"
#include "ngspice/sim.h"
#include "circuits.h"
#include "dotcards.h"
#include "variable.h"
#include "fourier.h"
#include "breakp2.h"
#include "com_measure2.h"
#include "com_commands.h"
#include "com_asciiplot.h"
#include "resource.h"
#include "postcoms.h"

/* Extract all the .save lines */

static void fixdotplot(wordlist *wl);
static void fixdotprint(wordlist *wl);
static char *fixem(char *string);
void ft_savemeasure(void);


static struct plot *
setcplot(char *name)
{
    struct plot *pl;

    for (pl = plot_list; pl; pl = pl->pl_next)
        if (ciprefix(name, pl->pl_typename))
            return pl;

    return NULL;
}


/* All lines with .width, .plot, .print, .save, .op, .meas, .tf
   have been assembled into a wordlist (wl_first) in inp.c:inp_spsource(),
   and then stored to ci_commands in inp.c:inp_dodeck().
   The .save lines are selected, com_save will put the commands into dbs.
*/

void
ft_dotsaves(void)
{
    wordlist *iline, *wl = NULL;
    char *s;

    if (!ft_curckt) /* Shouldn't happen. */
        return;

    for (iline = ft_curckt->ci_commands; iline; iline = iline->wl_next)
        if (ciprefix(".save", iline->wl_word)) {
            s = iline->wl_word;
            /* skip .save */
            s = nexttok(s);
            wl = wl_append(wl, gettoks(s));
        }

    com_save(wl);
    wl_free(wl);
}


/* Go through the dot lines given and make up a big "save" command with
 * all the node names mentioned. Note that if a node is requested for
 * one analysis, it is saved for all of them.
 */

static char *plot_opts[ ] = {
    "linear",
    "xlog",
    "ylog",
    "loglog"
};


int
ft_savedotargs(void)
{
    wordlist *w, *wl = NULL, *iline, **prev_wl, *w_next;
    char *name;
    char *s;
    int some = 0;
    static wordlist all = { "all", NULL, NULL };
    int isaplot;
    int i;
    int status;

    if (!ft_curckt) /* Shouldn't happen. */
        return 0;

    for (iline = ft_curckt->ci_commands; iline; iline = iline->wl_next) {
        s = iline->wl_word;
        if (ciprefix(".plot", s))
            isaplot = 1;
        else
            isaplot = 0;

        if (isaplot || ciprefix(".print", s)) {
            s = nexttok(s);
            name = gettok(&s);

            if ((w = gettoks(s)) == NULL) {
                fprintf(cp_err, "Warning: no nodes given: %s\n", iline->wl_word);
            } else {
                if (isaplot) {
                    prev_wl = &w;
                    for (wl = w; wl; wl = w_next) {
                        w_next = wl->wl_next;
                        for (i = 0; (size_t) i < NUMELEMS(plot_opts); i++) {
                            if (!strcmp(wl->wl_word, plot_opts[i])) {
                                /* skip it */
                                *prev_wl = w_next;
                                tfree(wl);
                                break;
                            }
                        }
                        if (i == NUMELEMS(plot_opts))
                            prev_wl = &wl->wl_next;
                    }
                }
                some = 1;
                com_save2(w, name);
            }
        } else if (ciprefix(".four", s)) {
            s = nexttok(s);
            s = nexttok(s);
            if ((w = gettoks(s)) == NULL) {
                fprintf(cp_err, "Warning: no nodes given: %s\n", iline->wl_word);
            } else {
                some = 1;
                com_save2(w, "TRAN");       /* A hack */
            }
        } else if (ciprefix(".meas", s)) {
            status = measure_extract_variables(s);
            if (!(status)) {
                some = 1;
            }
        } else if (ciprefix(".op", s)) {
            some = 1;
            com_save2(&all, "OP");
        } else if (ciprefix(".tf", s)) {
            some = 1;
            com_save2(&all, "TF");
        }
    }
    return some;
}


void
ft_savemeasure(void)
{
    char *s;
    wordlist *iline;

    if (!ft_curckt) /* Shouldn't happen. */
        return;

    for (iline = ft_curckt->ci_commands; iline; iline = iline->wl_next) {
        s = iline->wl_word;
        if (ciprefix(".measure", s)) {
            (void) measure_extract_variables(s);
        }
    }
}


/* Execute the .whatever lines found in the deck, after we are done running.
 * We'll be cheap and use cp_lexer to get the words... This should make us
 * spice-2 compatible.  If terse is TRUE then there was a rawfile, so don't
 * print lots of junk.
 */

int
ft_cktcoms(bool terse)
{
    wordlist *coms, *command, all;
    char *plottype, *s;
    struct dvec *v;
    static wordlist twl = { "col", NULL, NULL };
    struct plot *pl;
    int i, found;
    char numbuf[BSIZE_SP]; /* For printnum*/

    all.wl_next = NULL;
    all.wl_word = "all";

    if (!ft_curckt) {
        return 1;
    }

    plot_cur = setcplot("op");
    if (!ft_curckt->ci_commands && !plot_cur)
        goto nocmds;
    coms = ft_curckt->ci_commands;
    cp_interactive = FALSE;

    /* Listing */
    if (ft_listprint) {
        if (terse)
            fprintf(cp_err, ".options: no listing, rawfile was generated.\n");
        else
            inp_list(cp_out, ft_curckt->ci_deck, ft_curckt->ci_options, LS_DECK);
    }

    /* If there was a .op line, then we have to do the .op output. */
    plot_cur = setcplot("op");
    if (plot_cur != NULL) {
        assert(plot_cur->pl_dvecs != NULL);
        if (plot_cur->pl_dvecs->v_realdata != NULL) {
            if (terse) {
                fprintf(cp_out, "OP information in rawfile.\n");
            } else {
                fprintf(cp_out, "\t%-30s%15s\n", "Node", "Voltage");
                fprintf(cp_out, "\t%-30s%15s\n", "----", "-------");
                fprintf(cp_out, "\t----\t-------\n");
                for (v = plot_cur->pl_dvecs; v; v = v->v_next) {
                    if (!isreal(v)) {
                        fprintf(cp_err,
                                "Internal error: op vector %s not real\n",
                                v->v_name);
                        continue;
                    }
                    if ((v->v_type == SV_VOLTAGE) && (*(v->v_name) != '@')) {
                        printnum(numbuf, v->v_realdata[0]);
                        fprintf(cp_out, "\t%-30s%15s\n", v->v_name, numbuf);
                    }
                }
                fprintf(cp_out, "\n\tSource\tCurrent\n");
                fprintf(cp_out, "\t------\t-------\n\n");
                for (v = plot_cur->pl_dvecs; v; v = v->v_next)
                    if (v->v_type == SV_CURRENT) {
                        printnum(numbuf, v->v_realdata[0]);
                        fprintf(cp_out, "\t%-30s%15s\n", v->v_name, numbuf);
                    }
                fprintf(cp_out, "\n");

                if (!ft_nomod) {
                    com_showmod(&all);
                }
                com_show(&all);
            }
        }
    }

    for (pl = plot_list; pl; pl = pl->pl_next)
        if (ciprefix("tf", pl->pl_typename)) {
            if (terse) {
                fprintf(cp_out, "TF information in rawfile.\n");
                break;
            }
            plot_cur = pl;
            fprintf(cp_out, "Transfer function information:\n");
            com_print(&all);
            fprintf(cp_out, "\n");
        }

    /* Now all the '.' lines */
    while (coms) {
        wordlist* freecom;
        freecom = command = cp_lexer(coms->wl_word);
        if (!command) {
            /* Line not converted to a wordlist */
            goto bad;
        }
        if (command->wl_word == (char*)NULL) {
            /* Line not converted to a wordlist */
            wl_free(freecom);
            goto bad;
        }
        if (eq(command->wl_word, ".width")) {
            do
                command = command->wl_next;
            while (command && !ciprefix("out", command->wl_word));
            if (command) {
                s = strchr(command->wl_word, '=');
                if (!s || !s[1]) {
                    fprintf(cp_err, "Error: bad line %s\n", coms->wl_word);
                    coms = coms->wl_next;
                    wl_free(freecom);
                    continue;
                }
                i = atoi(++s);
                cp_vset("width", CP_NUM, &i);
            }
        } else if (eq(command->wl_word, ".print")) {
            if (terse) {
                fprintf(cp_out,
                        ".print line ignored since rawfile was produced.\n");
            } else {
                command = command->wl_next;
                if (!command) {
                    fprintf(cp_err, "Error: bad line %s\n", coms->wl_word);
                    coms = coms->wl_next;
                    wl_free(freecom);
                    continue;
                }
                plottype = command->wl_word;
                command = command->wl_next;
                fixdotprint(command);
                twl.wl_next = command;
                found = 0;
                for (pl = plot_list; pl; pl = pl->pl_next)
                    if (ciprefix(plottype, pl->pl_typename)) {
                        plot_cur = pl;
                        com_print(&twl);
                        fprintf(cp_out, "\n");
                        found = 1;
                    }
                if (!found)
                    fprintf(cp_err, "Error: .print: no %s analysis found.\n",
                            plottype);
            }
        } else if (eq(command->wl_word, ".plot")) {
            if (terse) {
                fprintf(cp_out,
                        ".plot line ignored since rawfile was produced.\n");
            } else {
                command = command->wl_next;
                if (!command) {
                    fprintf(cp_err, "Error: bad line %s\n",
                            coms->wl_word);
                    coms = coms->wl_next;
                    wl_free(freecom);
                    continue;
                }
                plottype = command->wl_word;
                command = command->wl_next;
                fixdotplot(command);
                found = 0;
                for (pl = plot_list; pl; pl = pl->pl_next)
                    if (ciprefix(plottype, pl->pl_typename)) {
                        plot_cur = pl;
                        com_asciiplot(command);
                        fprintf(cp_out, "\n");
                        found = 1;
                    }
                if (!found)
                    fprintf(cp_err, "Error: .plot: no %s analysis found.\n",
                            plottype);
            }
        } else if (ciprefix(".four", command->wl_word)) {
            if (terse) {
                fprintf(cp_out,
                        ".fourier line ignored since rawfile was produced.\n");
            } else {
                int err;

                plot_cur = setcplot("tran");
                err = fourier(command->wl_next, plot_cur);
                if (!err)
                    fprintf(cp_out, "\n\n");
                else
                    fprintf(cp_err, "No transient data available for "
                            "fourier analysis");
            }
        } else if (!eq(command->wl_word, ".save") &&
                   !eq(command->wl_word, ".op") &&
                   !ciprefix(".meas", command->wl_word) &&
                   !eq(command->wl_word, ".tf")) {
            wl_free(freecom);
            goto bad;
        }
        coms = coms->wl_next; /* go to next line */
        wl_free(freecom);
    } /* end of loop over '.' lines */

nocmds:
    /* Now the node table
       if (ft_nodesprint)
       ;
    */

    /* The options */
    if (ft_optsprint) {
        fprintf(cp_out, "Options:\n\n");
        cp_vprint();
        (void) putc('\n', cp_out);
    }

    /* And finally the accounting info. */
    if (ft_acctprint) {
        static wordlist ww = { "everything", NULL, NULL };
        com_rusage(&ww);
    } else if ((!ft_noacctprint) && (!ft_acctprint)) {
        com_rusage(NULL);
    }
    /* absolutely no accounting if noacct is given */

    putc('\n', cp_out);
    return 0;

bad:
    fprintf(cp_err, "Internal Error: ft_cktcoms: bad commands\n");
    return 1;
}


/* These routines make sure that the arguments to .plot and .print in
 * spice2 decks are acceptable to spice3. The things we look for are
 *  trailing (a,b) in .plot -> xlimit a b
 *  vm(x) -> mag(v(x))
 *  vp(x) -> ph(v(x))
 *  v(x,0) -> v(x)
 *  v(0,x) -> -v(x)
 */

static void
fixdotplot(wordlist *wl)
{
    /* Create a buffer for printing numbers */
    DS_CREATE(numbuf, 100);

    while (wl) {
        wl->wl_word = fixem(wl->wl_word);

        /* Is this a trailing "(a,b)"? Note that we require it to be
         * one word. */
        if (!wl->wl_next && (*wl->wl_word == '(')) {
            double d1, d2;
            char *s = wl->wl_word + 1;
            if (ft_numparse(&s, FALSE, &d1) < 0 ||
                    *s != ',') {
                fprintf(cp_err, "Error: bad limits \"%s\"\n",
                        wl->wl_word);
                goto EXITPOINT;
            }
            s++; /* step past comma */
            if (ft_numparse(&s, FALSE, &d2) < 0 ||
                    *s != ')' || s[1] != '\0') { /* must end with ")" */
                fprintf(cp_err, "Error: bad limits \"%s\"\n",
                        wl->wl_word);
                goto EXITPOINT;
            }

            tfree(wl->wl_word);
            wl->wl_word = copy("xlimit");
            ds_clear(&numbuf);
            if (printnum_ds(&numbuf, d1) != 0) {
                fprintf(cp_err, "Unable to print limit 1: %g\n", d1);
                goto EXITPOINT;
            }
            wl_append_word(NULL, &wl, copy(ds_get_buf(&numbuf)));
            ds_clear(&numbuf);
            if (printnum_ds(&numbuf, d2) != 0) {
                fprintf(cp_err, "Unable to print limit 2: %g\n", d2);
                goto EXITPOINT;
            }
            wl_append_word(NULL, &wl, copy(ds_get_buf(&numbuf)));
        } /* end of case of start of potential (a,b) */
        wl = wl->wl_next;
    } /* end of loop over words */

EXITPOINT:
    ds_free(&numbuf); /* Free DSTRING resources */
} /* end of function fixdotplot */



static void fixdotprint(wordlist *wl)
{
    /* Process each word in the wordlist */
    while (wl) {
        wl->wl_word = fixem(wl->wl_word);
        wl = wl->wl_next;
    }
} /* end of function fixdotprint */



static char *fixem(char *string)
{
    char buf[BSIZE_SP], *s, *t;
    char *ss = string; /* save addr of string in case it is freed */

    if (ciprefix("v(", string) &&strchr(string, ',')) {
        for (s = string; *s && (*s != ','); s++)
            ;
        *s++ = '\0';
        for (t = s; *t && (*t != ')'); t++)
            ;
        *t   = '\0';
        if (eq(s, "0"))
            (void) sprintf(buf, "v(%s)", string + 2);
        else if (eq(string + 2, "0"))
            (void) sprintf(buf, "-v(%s)", s);
        else
            (void) sprintf(buf, "v(%s)-v(%s)", string + 2, s);
    } else if (ciprefix("vm(", string) &&strchr(string, ',')) {
        for (s = string; *s && (*s != ','); s++)
            ;
        *s++ = '\0';
        for (t = s;      *t && (*t != ')'); t++)
            ;
        *t   = '\0';
        if (eq(s, "0"))
            (void) sprintf(buf, "mag(v(%s))", string + 3);
        else if (eq(string + 3, "0"))
            (void) sprintf(buf, "mag(-v(%s))", s);
        else
            (void) sprintf(buf, "mag(v(%s)-v(%s))", string + 3, s);
    } else if (ciprefix("vp(", string) &&strchr(string, ',')) {
        for (s = string; *s && (*s != ','); s++)
            ;
        *s++ = '\0';
        for (t = s;      *t && (*t != ')'); t++)
            ;
        *t   = '\0';
        if (eq(s, "0"))
            (void) sprintf(buf, "ph(v(%s))", string + 3);
        else if (eq(string + 3, "0"))
            (void) sprintf(buf, "ph(-v(%s))", s);
        else
            (void) sprintf(buf, "ph(v(%s)-v(%s))", string + 3, s);
    } else if (ciprefix("vi(", string) &&strchr(string, ',')) {
        for (s = string; *s && (*s != ','); s++)
            ;
        *s++ = '\0';
        for (t = s;      *t && (*t != ')'); t++)
            ;
        *t   = '\0';
        if (eq(s, "0"))
            (void) sprintf(buf, "imag(v(%s))", string + 3);
        else if (eq(string + 3, "0"))
            (void) sprintf(buf, "imag(-v(%s))", s);
        else
            (void) sprintf(buf, "imag(v(%s)-v(%s))", string + 3, s);
    } else if (ciprefix("vr(", string) &&strchr(string, ',')) {
        for (s = string; *s && (*s != ','); s++)
            ;
        *s++ = '\0';
        for (t = s;      *t && (*t != ')'); t++)
            ;
        *t   = '\0';
        if (eq(s, "0"))
            (void) sprintf(buf, "real(v(%s))", string + 3);
        else if (eq(string + 3, "0"))
            (void) sprintf(buf, "real(-v(%s))", s);
        else
            (void) sprintf(buf, "real(v(%s)-v(%s))", string + 3, s);
    } else if (ciprefix("vdb(", string) &&strchr(string, ',')) {
        for (s = string; *s && (*s != ','); s++)
            ;
        *s++ = '\0';
        for (t = s;      *t && (*t != ')'); t++)
            ;
        *t   = '\0';
        if (eq(s, "0"))
            (void) sprintf(buf, "db(v(%s))", string + 4);
        else if (eq(string + 4, "0"))
            (void) sprintf(buf, "db(-v(%s))", s);
        else
            (void) sprintf(buf, "db(v(%s)-v(%s))", string + 4, s);
    } else if (ciprefix("i(", string)) {
        for (s = string; *s && (*s != ')'); s++)
            ;
        *s = '\0';
        string += 2;
        (void) sprintf(buf, "%s#branch", string);
    } else {
        return string;
    }

    txfree(ss);
    string = copy(buf);

    return string;
} /* end of function fixem */



wordlist *
gettoks(char *s)
{
    char        *t, *s0;
    char        *l, *r, *c;     /* left, right, center/comma */
    wordlist    *wl, *list, **prevp;


    list = NULL;
    prevp = &list;

    /* stripWhite.... uses copy() to return a malloc'ed s, so we have to free it,
       using s0 as its starting address */
    if (strchr(s, '('))
        s0 = s = stripWhiteSpacesInsideParens(s);
    else
        s0 = s = copy(s);

    while ((t = gettok(&s)) != NULL) {
        if (*t == '(') {
            /* gettok uses copy() to return a malloc'ed t, so we have to free it */
            tfree(t);
            continue;
        }
        l = strrchr(t, '(');
        if (!l) {
            wl = wl_cons(copy(t), NULL);
            *prevp = wl;
            prevp = &wl->wl_next;
            tfree(t);
            continue;
        }

        r = strchr(t, ')');

        c = strchr(t, ',');
        if (!c)
            c = r;

        if (c)
            *c = '\0';

        wl = wl_cons(NULL, NULL);
        *prevp = wl;
        prevp = &wl->wl_next;

        /* Transfer i(xx) to xxx#branch only when i is the first
           character of the token or preceeded by a space. */
        if ((*(l - 1) == 'i' ||
             ((*(l - 1) == 'I') && (l - 1 == t))) ||
            ((l > t + 1) && isspace(*(l-2)))) {
            char buf[513];
            sprintf(buf, "%s#branch", l + 1);
            wl->wl_word = copy(buf);
            c = r = NULL;
        }
        else {
            wl->wl_word = copy(l + 1);
        }

        if (c != r) {
            *r = '\0';
            wl = wl_cons(copy(c + 1), NULL);
            *prevp = wl;
            prevp = &wl->wl_next;
        }
        tfree(t);
    } /* end of loop parsing string */

    txfree(s0);
    return list;
} /* end of function gettoks */



