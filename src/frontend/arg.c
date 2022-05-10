/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Jeffrey M. Hsu
**********/

/*
  This files contains the routines to evalute arguments to a command
  and prompt the user if necessary.
*/

#include "ngspice/ngspice.h"
#include "ngspice/fteinput.h"
#include "ngspice/cpdefs.h"
#include "ngspice/fteext.h"

#include "arg.h"
#include "variable.h"


static void common(const char *string, const struct wordlist *wl,
                   const struct comm *command);
static void common_list(const char *string, const struct wordlist *wl,
                        const struct comm *command);
static int countargs(const wordlist *wl);


/* returns a private copy of the string */

char *
prompt(FILE *fp)
{
    char    buf[100];
    char    *p;
    size_t  n;

    if (!fgets(buf, sizeof(buf), fp))
        return NULL;
    n = strlen(buf) - 1;
    buf[n] = '\0';      /* fgets leaves the \n */
    p = TMALLOC(char, n + 1);
    strcpy(p, buf);
    return p;
}


static int countargs(const wordlist *wl)
{
    int number = 0;
    const wordlist *w;

    for (w = wl; w; w = w->wl_next)
        number++;

    return (number);
}


wordlist *
process(wordlist *wlist)
{
    wlist = cp_variablesubst(wlist);
    wlist = cp_bquote(wlist);
    wlist = cp_doglob(wlist);
    return (wlist);
}


void
arg_print(const wordlist *wl, const struct comm *command)
{
    common("which variable", wl, command);
}


void
arg_plot(const wordlist *wl, const struct comm *command)
{
    common("which variable", wl, command);
}


void
arg_load(const wordlist *wl_in, const struct comm *command)
{
    /* just call com_load */
    wordlist * const wl = wl_copy(wl_in);
    command->co_func(wl);
    wl_free(wl);
}


void arg_let(const wordlist *wl, const struct comm *command)
{
    common("which vector", wl, command);
}


void arg_set(const wordlist *wl, const struct comm *command)
{
    common("which variable", wl, command);
}


void arg_display(const wordlist *wl, const struct comm *command)
{
    NG_IGNORE(wl);
    NG_IGNORE(command);

    /* just return; display does the right thing */
}


void arg_enodes(const wordlist *wl, const struct comm *command)
{
    common_list("which event nodes", wl, command);
}


/* a common prompt routine */
static void common(const char *string, const struct wordlist *wl,
        const struct comm *command)
{
    struct wordlist *w;
    char *buf;

    if (!countargs(wl)) {
        outmenuprompt(string);
        if ((buf = prompt(cp_in)) == NULL) /* prompt aborted */
            return;               /* don't execute command */
        /* do something with the wordlist */
        w = wl_cons(buf, NULL);

        w = process(w);
        /* O.K. now call fn */
        command->co_func(w);
        wl_free(w);
    }
} /* end of function common */


/* A common prompt routine for commands that take a list. */
static void common_list(const char *string, const struct wordlist *wl,
                        const struct comm *command)
{
    struct wordlist *w;
    char *buf;

    if (!countargs(wl)) {
        outmenuprompt(string);
        if ((buf = prompt(cp_in)) == NULL) /* prompt aborted */
            return;               /* don't execute command */
        /* do something with the wordlist */
        w = cp_lexer(buf);
        if (!w)
            return;
        if (w->wl_word) {
            /* O.K. now call fn */
            command->co_func(w);
        }
        wl_free(w);
    }
}


void
outmenuprompt(const char *string)
{
    fprintf(cp_out, "%s: ", string);
    fflush(cp_out);
    return;
}
