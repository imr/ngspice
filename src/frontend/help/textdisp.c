/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group 
**********/


/*
 *   faustus@cad.berkeley.edu, ucbvax!faustus
 * Permission is granted to modify and re-distribute this code in any manner
 * as long as this notice is preserved.  All standard disclaimers apply.
 *
 */

#include "ngspice.h"
#include "cpdefs.h"
#include "hlpdefs.h"
#include "suffix.h"

static topic *curtop;
static bool quitflag;

static void putline(char *s);
static int putstuff(toplink *tl, int base);

int hlp_width = 72;

bool
hlp_tdisplay(topic *top)
{
    wordlist *wl;
    int i = 0;

    curtop = top;

    out_init();
    out_printf("\n\t%s\n", top->title);
    for (wl = top->text; wl; wl = wl->wl_next)
        putline(wl->wl_word);
    if (top->subtopics) {
        out_printf("\tSub-Topics:\n\n");
        i = putstuff(top->subtopics, 0);
    }
    if (top->seealso) {
        out_printf("\n\tSee Also:\n\n");
        (void) putstuff(top->seealso, i);
    }
    out_printf("\n");
    return (TRUE);
}

toplink *
hlp_thandle(topic **parent)
{
    char buf[BSIZE_SP], *s;
    toplink *tl;
    int num;

    quitflag = FALSE;
    if (!curtop) {
        *parent = NULL;
        return (NULL);
    }
    for (;;) {
        fprintf(cp_out, "Selection (`?' for help): ");
        (void) fflush(cp_out);
        if (!fgets(buf, BSIZE_SP, cp_in)) {
            clearerr(stdin);
            quitflag = TRUE;
            *parent = NULL;
            return (NULL);
        }

        for (s = buf; *s && isspace(*s); s++)
            ;
        switch (*s) {
            case '?':
            fprintf(cp_out,
"\nType the number of a sub-topic or see also, or one of:\n\
\tr\tReprint the current topic\n\
\tp or CR\tReturn to the previous topic\n\
\tq\tQuit help\n\
\t?\tPrint this message\n\n");
            continue;

            case 'r':
            (void) hlp_tdisplay(curtop);
            continue;

            case 'q':
            quitflag = TRUE;
            *parent = NULL;
            return (NULL);

            case 'p':
            case '\n':
            case '\r':
            case '\0':
            *parent = curtop;
            return (NULL);
        }
        if (!isdigit(*s)) {
            fprintf(cp_err, "Invalid command\n");
            continue;
        }
        num = atoi(s);
        if (num <= 0) {
            fprintf(cp_err, "Bad choice.\n");
            continue;
        }
        for (tl = curtop->subtopics; tl; tl = tl->next)
            if (--num == 0)
                break;
        if (num) {
            for (tl = curtop->seealso; tl; tl = tl->next)
                if (--num == 0)
                    break;
        }
        if (num) {
            fprintf(cp_err, "Bad choice.\n");
            continue;
        }
        *parent = curtop;
        return (tl);
    }
}

/* ARGSUSED */
void
hlp_tkillwin(topic *top)
{
    if (curtop)
        curtop = curtop->parent;
    if (curtop && !quitflag)
        (void) hlp_tdisplay(curtop);
    return;
}

/* This has to rip out the font changes from the lines... */

static void
putline(char *s)
{
    char buf[BSIZE_SP];
    int i = 0;

    while (*s) {
        if (((*s == '\033') && s[1]) ||
                ((*s == '_') && (s[1] == '\b')))
            s += 2;
        else
            buf[i++] = *s++;
    }
    buf[i] = '\0';
    out_printf("%s\n", buf);
    return;
}

/* Figure out the number of columns we can use.  Assume an entry like
 * nn) word -- add 5 characters to the width...
 */

static int
putstuff(toplink *tl, int base)
{
    int maxwidth = 0, ncols, nrows, nbuts = 0, i, j, k;
    toplink *tt;

    for (tt = tl; tt; tt = tt->next) {
        if (strlen(tt->description) + 5 > maxwidth)
            maxwidth = strlen(tt->description) + 5;
        nbuts++;
    }
    ncols = hlp_width / maxwidth;
    if (!ncols) {
        fprintf(stderr, "Help, button too big!!\n");
        return (0);
    }
    if (ncols > nbuts)
        ncols = nbuts;
    maxwidth = hlp_width / ncols;
    nrows = nbuts / ncols;
    if (nrows * ncols < nbuts)
        nrows++;

    for (i = 0; i < nrows; i++) {
        for (tt = tl, j = 0; j < i; j++, tt = tt->next)
            ;
        for (j = 0; j < ncols; j++) {
            if (tt)
                out_printf("%2d) %-*s ", base + j * nrows + i +
                    1, maxwidth - 5, tt->description);
            for (k = 0; k < nrows; k++)
                if (tt)
                    tt = tt->next;
            
        }
        out_printf("\n");
    }
    return (nbuts);
}

