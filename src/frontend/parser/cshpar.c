/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * The main entry point for cshpar.
 */


#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include <signal.h>
#include "cshpar.h"

#ifdef HAVE_SGTTY_H
#include <sgtty.h>
#else
#ifdef HAVE_TERMIO_H
#include <termio.h>
#else
#ifdef HAVE_TERMIOS_H
#include <termios.h>
#endif
#endif
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif

/* perform history substitution only when variable 'histsubst' is set */
bool cp_no_histsubst = TRUE;

/* Things go as follows:
 * (1) Read the line and do some initial quoting (by setting the 8th bit),
 *  and command ignoring. Also deal with command completion.
 * (2) Do history substitutions. (!, ^)
 * (3) Do alias substitution.
 *
 * In front.c these things get done:
 * (4) Do variable substitution. ($varname)
 * (5) Do backquote substitution. (``)
 * (6) Do globbing. (*, ?, [], {}, ~)
 * (7) Do io redirection.
 */

static void pwlist(wordlist *wlist, char *name);


wordlist *cp_parse(char *string)
{
    wordlist *wlist;

    wlist = cp_lexer(string);

    /* Test for valid wordlist */
    if (!wlist) {
        return (wordlist *) NULL;
    }
    if (!wlist->wl_word) {
        wl_free(wlist);
        return (wordlist *) NULL;
    }

    if (!string) { /* cp_lexer read user data */
        cp_event++;
    }

    pwlist(wlist, "Initial parse");

    /* Do history substitution (!1, etc.) if enabled */
    if (!cp_no_histsubst) {
        wlist = cp_histsubst(wlist);

        /* Test for valid wordlist */
        if (!wlist) {
            return (wordlist *) NULL;
        }
        if (!wlist->wl_word) {
            wl_free(wlist);
            return (wordlist *) NULL;
        }

        pwlist(wlist, "After history substitution");
        if (cp_didhsubst) {
            wl_print(wlist, stdout);
            putc('\n', stdout);
        }
    } /* end of case that history substitutions are allowed */



    /* Add the word list to the history. */
    /* MW. If string==NULL we do not have to do this, and then play
     * with cp_lastone is not needed, but watch out cp_doalias */
    if ((*wlist->wl_word) && !(string))
        cp_addhistent(cp_event - 1, wlist);

    wlist = cp_doalias(wlist);
    pwlist(wlist, "After alias substitution");
    pwlist(wlist, "Returning ");
    return wlist;
} /* end of function cp_parse */



static void
pwlist(wordlist *wlist, char *name)
{
    wordlist *wl;

    if (!cp_debug)
        return;
    fprintf(cp_err, "%s : [ ", name);
    for (wl = wlist; wl; wl = wl->wl_next)
        fprintf(cp_err, "%s ", wl->wl_word);
    fprintf(cp_err, "]\n");
}
