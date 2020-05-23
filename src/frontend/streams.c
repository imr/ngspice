/*************
 * streams.c
 ************/
#include <errno.h>
#include <string.h>

#include "ngspice/ngspice.h"
#include "ngspice/wordlist.h"
#include "ngspice/bool.h"

#include "variable.h"
#include "terminal.h"
#include "ngspice/cpextern.h"
#include "streams.h"


bool cp_debug = FALSE;
char cp_gt = '>';
char cp_lt = '<';
char cp_amp = '&';

FILE *cp_in = NULL;
FILE *cp_out = NULL;
FILE *cp_err = NULL;

/* These are the fps that cp_ioreset resets the cp_* to.  They are
 * changed by the source routines.  */
FILE *cp_curin = NULL;
FILE *cp_curout = NULL;
FILE *cp_curerr = NULL;


static bool fileexists(const char *name)
{
#ifdef HAVE_ACCESS
    if (access(name, 0) == 0) {
        return TRUE;
    }
#endif
    return FALSE;
}



/* This routine sets the cp_{in,out,err} pointers and takes the io
 * directions out of the command line.  */
wordlist *cp_redirect(wordlist *wl)
{
    int gotinput = 0, gotoutput = 0, goterror = 0, append = 0;
    wordlist *w;
    FILE *fp;

    w = wl->wl_next;    /* Don't consider empty commands. */

    while (w) {
        char *fname;
        if (*w->wl_word == cp_lt && w->wl_word[1] == '\0') {

            wordlist *beg = w;

            if (gotinput++) {
                fprintf(cp_err, "Error: ambiguous input redirect.\n");
                goto error;
            }
            w = w->wl_next;

            if (w && *w->wl_word == cp_lt && w->wl_word[1] == '\0') {
                fprintf(cp_err, "Error: `<<' redirection is not implemented.\n");
                goto error;
            }

            if (!w) {
                fprintf(cp_err, "Error: missing name for input.\n");
                return (wordlist *) NULL;
            }

            fname = cp_unquote(w->wl_word);
            w = w->wl_next;

#ifdef CPDEBUG
            if (cp_debug) {
                fprintf(cp_err, "Input file is %s...\n", fname);
            }
#endif

            fp = fopen(fname, "r");
            if (!fp) {
                perror(fname);
                tfree(fname);
                goto error;
            }

            tfree(fname);
            cp_in = fp;

            /* special case for set command:
               keep i/o information (handled in com_set.c)  */
            wordlist* bw = beg->wl_prev->wl_prev;
            if (!(bw && cieq(bw->wl_word, "set"))) {
                wl_delete_slice(beg, w);
            }

        }
        else if (*w->wl_word == cp_gt && w->wl_word[1] == '\0') {
            wordlist *beg = w;

            if (gotoutput++) {
                fprintf(cp_err, "Error: ambiguous output redirect.\n");
                goto error;
            }
            w = w->wl_next;

            if (w && *w->wl_word == cp_gt && w->wl_word[1] == '\0') {
                append++;
                w = w->wl_next;
            }

            if (w && *w->wl_word == cp_amp) {
                if (goterror++) {
                    fprintf(cp_err, "Error: ambiguous error redirect.\n");
                    return (wordlist *) NULL;
                }
                w = w->wl_next;
            }

            if (!w) {
                fprintf(cp_err, "Error: missing name for output.\n");
                return (wordlist *) NULL;
            }

            fname = cp_unquote(w->wl_word);
            w = w->wl_next;

#ifdef CPDEBUG
            if (cp_debug)
                fprintf(cp_err, "Output file is %s... %s\n", fname,
                        append ? "(append)" : "");
#endif

            if (cp_noclobber && fileexists(fname)) {
                fprintf(stderr, "Error: %s: file exists\n", fname);
                goto error;
            }

            fp = fopen(fname, append ? "a" : "w+");

            if (!fp) {
                tfree(fname);
                perror(fname);
                goto error;
            }
            tfree(fname);

            cp_out = fp;
            if (goterror) {
                cp_err = fp;
            }

            out_isatty = FALSE;

            wl_delete_slice(beg, w);

        }
        else {
            w = w->wl_next;
        }
    } /* end of loop over arguments */
    return wl;

error:
    wl_free(wl);                /* FIXME, Ouch !! */
    return (wordlist *) NULL;
} /* end of function cp_redirect */



/* Reset the cp_* FILE pointers to the standard ones.  This is tricky,
 * since if we are sourcing a command file, and io has been redirected
 * from inside the file, we have to reset it back to what it was for
 * the source, not for the top level.  That way if you type "foo >
 * bar" where foo is a script, and it has redirections of its own
 * inside of it, none of the output from foo will get sent to
 * stdout...  */
void cp_ioreset(void)
{
    if (cp_in != cp_curin) {
        if (cp_in) {
            fclose(cp_in);
        }
    }
    if (cp_out != cp_curout) {
        if (cp_out) {
            fclose(cp_out);
        }
    }
    if (cp_err != cp_curerr) {
        if (cp_err  &&  cp_err != cp_out) {
            fclose(cp_err);
        }
    }

    cp_in  = cp_curin;
    cp_out = cp_curout;
    cp_err = cp_curerr;

    /*** Minor bug here... */
    out_isatty = TRUE;
} /* end of function cp_ioreset */



/* Do this only right before an exec, since we lose the old std*'s. */
void fixdescriptors(void)
{
    bool dup2_fail = FALSE;
    if (cp_in != stdin) {
        dup2_fail |= dup2(fileno(cp_in), fileno(stdin)) == -1;
    }
    if (cp_out != stdout) {
        dup2_fail |= dup2(fileno(cp_out), fileno(stdout)) == -1;
    }
    if (cp_err != stderr) {
        dup2_fail |= dup2(fileno(cp_err), fileno(stderr)) == -1;
    }

    /* Warn if there was some failure */
    if (dup2_fail) {
        (void) fprintf(cp_err,
                "I/O descriptor failure: %s.\n", strerror(errno));
    }
} /* end of function fixdescriptors */



