/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher
**********/

/*
 * For dealing with nutmeg input decks and command scripts
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteinp.h"
#include "ngspice/stringskip.h"
#include "nutinp.h"
#include "variable.h"
#include "../misc/mktemp.h"
#include "subckt.h"

/* The routine to source a spice input deck. We read the deck in, take out
 * the front-end commands, and create a CKT structure. Also we filter out
 * the following lines: .save, .width, .four, .print, and .plot, to perform
 * after the run is over.
 */

void
inp_nutsource(FILE *fp, bool comfile, char *filename)
{
    struct card *deck, *dd, *ld;
    struct card *realdeck, *options = NULL;
    char *tt = NULL, name[BSIZE_SP], *s, *t;
    bool commands = FALSE;
    wordlist *wl = NULL, *end = NULL;
    wordlist *controls = NULL;
    FILE *lastin, *lastout, *lasterr;

    deck = inp_readall(fp, NULL, comfile, FALSE, NULL); /* still to check if . or filename instead of NULL */
    if (!deck)
        return;

    realdeck = inp_deckcopy(deck);

    if (!comfile) {
        /* Save the title before INPgetTitle gets it. */
        tt = copy(deck->line);
        if (!deck->nextcard)
            fprintf(cp_err, "Warning: no lines in deck...\n");
    }
    (void) fclose(fp);

    /* Now save the IO context and start a new control set...  After
     * we are done with the source we'll put the old file descriptors
     * back.  I guess we could use a FILE stack, but since this routine
     * is recursive anyway...
     */
    lastin = cp_curin;
    lastout = cp_curout;
    lasterr = cp_curerr;
    cp_curin = cp_in;
    cp_curout = cp_out;
    cp_curerr = cp_err;

    cp_pushcontrol();

    /* We should now go through the deck and execute front-end
     * commands and remove them. Front-end commands are enclosed by
     * the lines .control and .endc, unless comfile
     * is TRUE, in which case every line must be a front-end command.
     * There are too many problems with matching the first word on
     * the line.
     */
    ld = deck;
    if (comfile) {
        /* This is easy. */
        for (dd = deck; dd; dd = ld) {
            ld = dd->nextcard;
            if ((dd->line[0] == '*') && (dd->line[1] != '#'))
                continue;
            if (!ciprefix(".control", dd->line) &&
                !ciprefix(".endc", dd->line)) {
                if (dd->line[0] == '*')
                    (void) cp_evloop(dd->line + 2);
                else
                    (void) cp_evloop(dd->line);
            }
            tfree(dd->line);
            tfree(dd);
        }
    } else {
        for (dd = deck->nextcard; dd; dd = ld->nextcard) {
            if ((dd->line[0] == '*') && (dd->line[1] != '#')) {
                ld = dd;
                continue;
            }
            (void) strncpy(name, dd->line, BSIZE_SP - 1);
            s = skip_ws(name);
            t = skip_non_ws(s);
            *t = '\0';

            if (ciprefix(".control", dd->line)) {
                ld->nextcard = dd->nextcard;
                tfree(dd->line);
                tfree(dd);
                if (commands)
                    fprintf(cp_err, "Warning: redundant .control line\n");
                else
                    commands = TRUE;
            } else if (ciprefix(".endc", dd->line)) {
                ld->nextcard = dd->nextcard;
                tfree(dd->line);
                tfree(dd);
                if (commands)
                    commands = FALSE;
                else
                    fprintf(cp_err, "Warning: misplaced .endc line\n");
            } else if (commands || prefix("*#", dd->line)) {
                controls = wl_cons(NULL, controls);
                wl = controls;
                if (prefix("*#", dd->line))
                    wl->wl_word = copy(dd->line + 2);
                else
                    wl->wl_word = dd->line;
                ld->nextcard = dd->nextcard;
                tfree(dd);
            } else if (!*dd->line) {
                /* So blank lines in com files don't get
                 * considered as circuits.
                 */
                ld->nextcard = dd->nextcard;
                tfree(dd->line);
                tfree(dd);
            } else {
                inp_casefix(s);
                inp_casefix(dd->line);
                if (eq(s, ".width") || ciprefix(".four", s) ||
                    eq(s, ".plot")  ||
                    eq(s, ".print") ||
                    eq(s, ".save"))
                {
                    wl_append_word(&wl, &end, copy(dd->line));
                    ld->nextcard = dd->nextcard;
                    tfree(dd->line);
                    tfree(dd);
                } else {
                    ld = dd;
                }
            }
        }
        if (deck->nextcard) {
            /* There is something left after the controls. */
            fprintf(cp_out, "\nCircuit: %s\n\n", tt);
            fprintf(stderr, "\nCircuit: %s\n\n", tt);

            /* Now expand subcircuit macros. Note that we have to
             * fix the case before we do this but after we
             * deal with the commands.
             */
            if (!cp_getvar("nosubckt", CP_BOOL, NULL, 0))
                deck->nextcard = inp_subcktexpand(deck->nextcard);
            deck->actualLine = realdeck;
            nutinp_dodeck(deck, tt, wl, FALSE, options, filename);
        }

        /* Now that the deck is loaded, do the commands... */
        controls = wl_reverse(controls);
        for (wl = controls; wl; wl = wl->wl_next)
            (void) cp_evloop(wl->wl_word);
        wl_free(controls);
    }

    /* Now reset everything.  Pop the control stack, and fix up the IO
     * as it was before the source.
     */
    cp_popcontrol();

    cp_curin = lastin;
    cp_curout = lastout;
    cp_curerr = lasterr;

    tfree(tt);
}


void
nutcom_source(wordlist *wl)
{
    FILE *fp, *tp;
    char buf[BSIZE_SP];
    bool inter;
    char *tempfile = NULL;
    wordlist *owl = wl;
    size_t n;

    inter = cp_interactive;
    cp_interactive = FALSE;
    if (wl->wl_next) {
        /* There are several files -- put them into a temp file...  */
        tempfile = smktemp("sp");
        if ((fp = inp_pathopen(tempfile, "w+")) == NULL) {
            perror(tempfile);
            cp_interactive = TRUE;
            return;
        }
        while (wl) {
            if ((tp = inp_pathopen(wl->wl_word, "r")) == NULL) {
                perror(wl->wl_word);
                (void) fclose(fp);
                cp_interactive = TRUE;
                (void) unlink(tempfile);
                return;
            }
            while ((n = fread(buf, 1, BSIZE_SP, tp)) > 0)
                (void) fwrite(buf, 1, n, fp);
            (void) fclose(tp);
            wl = wl->wl_next;
        }
        (void) fseek(fp, 0L, SEEK_SET);
    } else {
        fp = inp_pathopen(wl->wl_word, "r");
    }

    if (fp == NULL) {
        perror(wl->wl_word);
        cp_interactive = TRUE;
        return;
    }

    /* Don't print the title if this is a .spiceinit file. */
    if (ft_nutmeg ||
        substring(INITSTR, owl->wl_word) ||
        substring(ALT_INITSTR, owl->wl_word))
    {
        inp_nutsource(fp, TRUE, tempfile ? NULL : wl->wl_word);
    } else {
        inp_nutsource(fp, FALSE, tempfile ? NULL : wl->wl_word);
    }

    cp_interactive = inter;
    if (tempfile)
        (void) unlink(tempfile);
}


void
nutinp_source(char *file)
{
    static struct wordlist wl = { NULL, NULL, NULL };

    wl.wl_word = file;
    nutcom_source(&wl);
}


/* This routine is cut in half here because com_rset has to do what follows
 * also. End is the list of commands we execute when the job is finished:
 * we only bother with this if we might be running in batch mode, since
 * it isn't much use otherwise.
 */

void
nutinp_dodeck(struct card *deck, char *tt, wordlist *end, bool reuse, struct card *options, char *filename)
{
    NG_IGNORE(filename);
    NG_IGNORE(options);
    NG_IGNORE(reuse);
    NG_IGNORE(end);
    NG_IGNORE(tt);
    NG_IGNORE(deck);

    /* This was "ifdef notdef"-ed out, so I tossed it */
}
