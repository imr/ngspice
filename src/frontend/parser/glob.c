/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Expand global characters.
 */

#include <config.h>
#include "ngspice.h"
#include "cpdefs.h"
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



bool noglobs();

char cp_comma = ',';
char cp_til = '~';

/* For each word, go through two steps: expand the {}'s, and then do ?*[]
 * globbing in them. Sort after the second phase but not the first...
 */

/* MW. Now only tilde is supported, {}*? don't work */

wordlist *
cp_doglob(wordlist *wlist)
{
  wordlist *wl;
    char *s;


    /* Do tilde expansion. */

    for (wl = wlist; wl; wl = wl->wl_next)
        if (*wl->wl_word == cp_til) {
            s = cp_tildexpand(wl->wl_word);
            if (!s)
                *wl->wl_word = '\0';	/* MW. We Con't touch tmalloc addres */
            else
		wl->wl_word = s;
        }

    return (wlist);
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

