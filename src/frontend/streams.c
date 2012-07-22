/*************
* streams.c
************/

#include "ngspice/config.h"
#include "ngspice/ngspice.h"
#include "ngspice/wordlist.h"
#include "ngspice/bool.h"

#include "variable.h"
#include "terminal.h"
#include "quote.h"
#include "ngspice/cpextern.h"
#include "streams.h"

bool cp_debug = FALSE;
char cp_gt = '>';
char cp_lt = '<';
char cp_amp = '&';

FILE *cp_in=NULL;
FILE *cp_out=NULL;
FILE *cp_err=NULL;

/* These are the fps that cp_ioreset resets the cp_* to.  They are
 * changed by the source routines.  */
FILE *cp_curin = NULL;
FILE *cp_curout = NULL;
FILE *cp_curerr = NULL;

/* static functions */
static bool fileexists(char *name);

static bool
fileexists(char *name)
{
#ifdef HAVE_ACCESS
    if (access(name, 0) == 0)
        return (TRUE);
#endif
    return (FALSE);
}


/* This routine sets the cp_{in,out,err} pointers and takes the io
 * directions out of the command line.  */
wordlist *
cp_redirect(wordlist *wl)
{
    bool gotinput = FALSE, gotoutput = FALSE, goterror = FALSE;
    bool app = FALSE, erralso = FALSE;
    wordlist *w, *bt, *nw;
    char *s,*copyword;
    FILE *tmpfp;

    w = wl->wl_next;    /* Don't consider empty commands. */
    while (w) {
        if (*w->wl_word == cp_lt) {
            bt = w;
            if (gotinput) {
                fprintf(cp_err, 
                "Error: ambiguous input redirect.\n");
                goto error;
            }
            gotinput = TRUE;
            w = w->wl_next;
            if (w == NULL) {
                fprintf(cp_err, 
                "Error: missing name for input.\n");
                return (NULL);
            }
            if (*w->wl_word == cp_lt) {
                /* Do reasonable stuff here... */
            } else {
                /*tmpfp = fopen(cp_unquote(w->wl_word), "r"); DG very bad: memory leak the string allocated by cp_unquote is lost*/
                copyword=cp_unquote(w->wl_word);/*DG*/
                tmpfp = fopen(copyword, "r"); 
                tfree(copyword);
  
                if (!tmpfp) {
                    perror(w->wl_word);
                    goto error;
                } else
                    cp_in = tmpfp;
            }
#ifdef CPDEBUG
            if (cp_debug)
                fprintf(cp_err, "Input file is %s...\n",
                    w->wl_word);
#endif
            bt->wl_prev->wl_next = w->wl_next;
            if (w->wl_next)
                w->wl_next->wl_prev = bt->wl_prev;
            nw = w->wl_next;
            w->wl_next = NULL;
            w = nw;
            wl_free(bt);
        } else if (*w->wl_word == cp_gt) {
            bt = w;
            if (gotoutput) {
                fprintf(cp_err, 
                "Error: ambiguous output redirect.\n");
                goto error;
            }
            gotoutput = TRUE;
            w = w->wl_next;
            if (w == NULL) {
                fprintf(cp_err, 
                "Error: missing name for output.\n");
                return (NULL);
            }
            if (*w->wl_word == cp_gt) {
                app = TRUE;
                w = w->wl_next;
                if (w == NULL) {
                    fprintf(cp_err, 
                    "Error: missing name for output.\n");
                    return (NULL);
                }
            }
            if (*w->wl_word == cp_amp) {
                erralso = TRUE;
                if (goterror) {
                    fprintf(cp_err, 
                "Error: ambiguous error redirect.\n");
                    return (NULL);
                }
                goterror = TRUE;
                w = w->wl_next;
                if (w == NULL) {
                    fprintf(cp_err, 
                    "Error: missing name for output.\n");
                    return (NULL);
                }
            }
            s = cp_unquote(w->wl_word);
            if (cp_noclobber && fileexists(s)) {
                fprintf(stderr, "Error: %s: file exists\n", s);
                goto error;
            }
            if (app)
                tmpfp = fopen(s, "a");
            else
                tmpfp = fopen(s, "w+");
            tfree(s);/*DG cp_unquote memory leak*/
            if (!tmpfp) {
                perror(w->wl_word);
                goto error;
            } else {
                cp_out = tmpfp;
                out_isatty = FALSE;
            }
#ifdef CPDEBUG
            if (cp_debug)
                fprintf(cp_err, "Output file is %s... %s\n",
                    w->wl_word, app ? "(append)" : "");
#endif
            bt->wl_prev->wl_next = w->wl_next;
            if (w->wl_next)
                w->wl_next->wl_prev = bt->wl_prev;
            w = w->wl_next;
            if (w)
                w->wl_prev->wl_next = NULL;
            wl_free(bt);
            if (erralso)
                cp_err = cp_out;
        } else
            w = w->wl_next;
    }
    return (wl);

error:  wl_free(wl);
    return (NULL);
}

/* Reset the cp_* FILE pointers to the standard ones.  This is tricky,
 * since if we are sourcing a command file, and io has been redirected
 * from inside the file, we have to reset it back to what it was for
 * the source, not for the top level.  That way if you type "foo >
 * bar" where foo is a script, and it has redirections of its own
 * inside of it, none of the output from foo will get sent to
 * stdout...  */

void
cp_ioreset(void)
{
    if (cp_in != cp_curin)
        if (cp_in)
            fclose(cp_in);
    if (cp_out != cp_curout)
        if (cp_out)
            fclose(cp_out);
    if (cp_err != cp_curerr)
        if (cp_err  &&  cp_err != cp_out)
            fclose(cp_err);

    cp_in  = cp_curin;
    cp_out = cp_curout;
    cp_err = cp_curerr;

    /*** Minor bug here... */
    out_isatty = TRUE;
    return;
}


/* Do this only right before an exec, since we lose the old std*'s. */

void
fixdescriptors(void)
{
    if (cp_in != stdin)
        dup2(fileno(cp_in), fileno(stdin));
    if (cp_out != stdout)
        dup2(fileno(cp_out), fileno(stdout));
    if (cp_err != stderr)
        dup2(fileno(cp_err), fileno(stderr));
    return;
}
