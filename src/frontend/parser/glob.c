/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Expand global characters.
 */

#include "ngspice/config.h"
#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "glob.h"

#ifdef HAVE_SYS_DIR_H
#include <sys/types.h>
#include <sys/dir.h>
#else

#ifdef HAVE_DIRENT_H
#include <sys/types.h>
#include <dirent.h>
#ifndef direct
#define direct dirent
#endif
#endif

#endif

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif



char cp_comma = ',';
char cp_ocurl = '{';
char cp_ccurl = '}';
char cp_til = '~';

static wordlist *bracexpand(char *string);
static wordlist *brac1(char *string);
static wordlist *brac2(char *string);


/* For each word, go through two steps: expand the {}'s, and then do ?*[]
 * globbing in them. Sort after the second phase but not the first...
 */

/* MW. Now only tilde is supported, {}*? don't work */

wordlist *
cp_doglob(wordlist *wlist)
{
  wordlist *wl;
    char *s;

    /* Expand {a,b,c} */

    for (wl = wlist; wl; wl = wl->wl_next) {
        wordlist *nwl, *w = bracexpand(wl->wl_word);
        if (!w) {
            wlist->wl_word = NULL; /* XXX */
            return (wlist);
        }
        nwl = wl_splice(wl, w);
        if (wlist == wl)
            wlist = w;
        wl = nwl;
    }

    /* Do tilde expansion. */

    for (wl = wlist; wl; wl = wl->wl_next)
        if (*wl->wl_word == cp_til) {
            s = cp_tildexpand(wl->wl_word);
            txfree(wl->wl_word);    /* sjb - fix memory leak */
            if (!s)
                *wl->wl_word = '\0';	/* MW. We Con't touch tmalloc addres */
            else
		wl->wl_word = s;
        }

    return (wlist);
}

static wordlist *
bracexpand(char *string)
{
    wordlist *wl, *w;
    char *s;

    if (!string)
        return (NULL);
    wl = brac1(string);
    if (!wl)
        return (NULL);
    for (w = wl; w; w = w->wl_next) {
        s = w->wl_word;
        w->wl_word = copy(s);
        tfree(s);
    }
    return (wl);
}

/* Given a string, returns a wordlist of all the {} expansions. This is
 * called recursively by cp_brac2(). All the words here will be of size
 * BSIZE_SP, so it is a good idea to copy() and free() the old words.
 */

static wordlist *
brac1(char *string)
{
    wordlist *words, *wl, *w, *nw, *nwl, *newwl;
    char *s;
    int nb;

    words = wl_cons(TMALLOC(char, BSIZE_SP), NULL);
    words->wl_word[0] = '\0';
    for (s = string; *s; s++) {
        if (*s == cp_ocurl) {
            nwl = brac2(s);
            nb = 0;
            for (;;) {
                if (*s == cp_ocurl)
                    nb++;
                if (*s == cp_ccurl)
                    nb--;
                if (*s == '\0') {   /* { */
                    fprintf(cp_err, "Error: missing }.\n");
                    return (NULL);
                }
                if (nb == 0)
                    break;
                s++;
            }
            /* Add nwl to the rest of the strings in words. */
            newwl = NULL;
            for (wl = words; wl; wl = wl->wl_next)
                for (w = nwl; w; w = w->wl_next) {
                    nw = wl_cons(TMALLOC(char, BSIZE_SP), NULL);
                    (void) strcpy(nw->wl_word, wl->wl_word);
                    (void) strcat(nw->wl_word, w->wl_word);
                    newwl = wl_append(newwl, nw);
                }
            wl_free(words);
            words = newwl;
        } else
            for (wl = words; wl; wl = wl->wl_next)
                appendc(wl->wl_word, *s);
    }
    return (words);
}

/* Given a string starting with a {, return a wordlist of the expansions
 * for the text until the matching }.
 */

static wordlist *
brac2(char *string)
{
    wordlist *wlist = NULL, *nwl;
    char buf[BSIZE_SP], *s;
    int nb;
    bool eflag = FALSE;

    string++;   /* Get past the first open brace... */
    for (;;) {
        (void) strcpy(buf, string);
        nb = 0;
        s = buf;
        for (;;) {
            if ((*s == cp_ccurl) && (nb == 0)) {
                eflag = TRUE;
                break;
            }
            if ((*s == cp_comma) && (nb == 0))
                break;
            if (*s == cp_ocurl)
                nb++;
            if (*s == cp_ccurl)
                nb--;
            if (*s == '\0') {       /* { */
                fprintf(cp_err, "Error: missing }.\n");
                return (NULL);
            }
            s++;
        }
        *s = '\0';
        nwl = brac1(buf);
        wlist = wl_append(wlist, nwl);
        string += s - buf + 1;
        if (eflag)
            return (wlist);
    }
}

/* Expand tildes. */

char *
cp_tildexpand(char *string)
{
    char	*result;

    result = tildexpand(string);

    if (!result) {
	if (cp_nonomatch) {
	    return copy(string);
	} else {
	    return NULL;
	}
    }
    return result;
}


/* Say whether the pattern p can match the string s. */

/* MW. Now simply compare strings */

bool
cp_globmatch(char *p, char *s)
{
    return(!(strcmp(p, s)));
}

