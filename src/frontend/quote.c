/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 *
 * Various things for quoting words. If this is not ascii, quote and
 * strip are no-ops, so '' and \ quoting won't work. To fix this, sell
 * your IBM machine and buy a vax.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "quote.h"


/* Strip all the 8th bits from a string (destructively). */

void
cp_wstrip(char *str)
{
    char c, d;

    if (str)
        while ((c = *str) != '\0') {   /* assign and test */
            d = (char) strip(c);
            if (c != d)
                *str = d;
            str++;
        }
}


/* Quote all characters in a word. */

void
cp_quoteword(char *str)
{
    if (str)
        while (*str) {
            *str = (char) quote(*str);
            str++;
        }
}


/* Print a word (strip the word first). */

void
cp_printword(char *string, FILE *fp)
{
    char *s;

    if (string)
        for (s = string; *s; s++)
            (void) putc((strip(*s)), fp);
}


/* (Destructively) strip all the words in a wlist. */

void
cp_striplist(wordlist *wlist)
{
    wordlist *wl;

    for (wl = wlist; wl; wl = wl->wl_next)
        cp_wstrip(wl->wl_word);
}


/* Remove the "" from a string. */

char *
cp_unquote(char *string)
{
    char *s;
    size_t l;

    if (string) {
        l = strlen(string);
        s = TMALLOC(char, l + 1);

        if (l >= 2 && *string == '"' && string[l-1] == '"') {
            strncpy(s, string+1, l-2);
            s[l-2] = '\0';
        } else {
            strcpy(s, string);
        }

        return (s);
    } else {
        return 0;
    }
}
