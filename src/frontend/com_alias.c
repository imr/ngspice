/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* Do alias substitution.  */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "com_alias.h"

struct alias *cp_aliases = NULL;


/* Return NULL if no alias was found. We can get away with just
 * calling cp_histsubst now because the line will have gone onto the
 * history list by now and cp_histsubst will look in the right place.  */
static wordlist *
asubst(wordlist *wlist)
{
    struct alias *al;
    wordlist *wl;
    char *word;

    word = wlist->wl_word;
    if (*word == '\\') {
        while ((word[0] = word[1]) != '\0')
            word++;
        return (NULL);
    }

    for (al = cp_aliases; al; al = al->al_next)
        if (eq(word, al->al_name))
            break;
    if (!al)
        return (NULL);

    wl = cp_histsubst(wl_copy(al->al_text));

    if (cp_didhsubst) {
        /* Make sure that we have an up-to-date last history entry. */
        wl_free(cp_lastone->hi_wlist);
        cp_lastone->hi_wlist = wl_copy(wl);
    } else {
        /* If it had no history args, then append the rest of the wl */
        wl_append(wl, wl_copy(wlist->wl_next));
    }

    return (wl);
}


/* MW. This function should not use cp_lastone, see cp_parse in cpshar.c
 *      Many things are deleted here and memory leak closed */
wordlist *
cp_doalias(wordlist *wlist)
{
    wordlist *comm;

    /* The alias process is going to modify the "last" line typed, so
     * save a copy of what it really is and restore it after aliasing
     * is done. We have to do tricky things do get around the problems
     * with ; ...  */
    comm = wlist;

    while (comm) {

        int ntries;
        wordlist *end, *nextc;

        nextc = wl_find(cp_csep, comm);

        if (nextc == comm) {     /* skip leading `;' */
            comm = comm->wl_next;
            continue;
        }

        /* Temporarily hide the rest of the command... */
        end = comm->wl_prev;
        wl_chop(comm);
        wl_chop(nextc);

        for (ntries = 21; ntries; ntries--) {
            wordlist *nwl = asubst(comm);
            if (nwl == NULL)
                break;
            if (eq(nwl->wl_word, comm->wl_word)) {
                /* Just once through... */
                wl_free(comm);
                comm = nwl;
                break;
            } else {
                wl_free(comm);
                comm = nwl;
            }
        }

        if (!ntries) {
            fprintf(cp_err, "Error: alias loop.\n");
            wl_free(comm);
            return wl_cons(NULL, NULL);
        }

        wl_append(end, comm);
        wl_append(comm, nextc);

        if (!end)
            wlist = comm;

        comm = nextc;
    }

    return (wlist);
}


/* If we use this, aliases will be in alphabetical order. */
void
cp_setalias(char *word, wordlist *wlist)
{
    struct alias *al, *ta;

    cp_unalias(word);
    cp_addkword(CT_ALIASES, word);

    if (cp_aliases == NULL) {
        al = cp_aliases = TMALLOC(struct alias, 1);
        al->al_next = NULL;
        al->al_prev = NULL;
    } else {
        for (al = cp_aliases; al->al_next; al = al->al_next) {
            if (strcmp(al->al_name, word) > 0)
                break;
        }
        /* The new one goes before al */
        if (al->al_prev) {
            al = al->al_prev;
            ta = al->al_next;
            al->al_next = TMALLOC(struct alias, 1);
            al->al_next->al_prev = al;
            al = al->al_next;
            al->al_next = ta;
            ta->al_prev = al;
        } else {
            cp_aliases = TMALLOC(struct alias, 1);
            cp_aliases->al_next = al;
            cp_aliases->al_prev = NULL;
            al->al_prev = cp_aliases;
            al = cp_aliases;
        }
    }

    al->al_name = copy(word);
    al->al_text = wl_copy(wlist);
    /* We can afford to not worry about the bits, because before the
     * keyword lookup is done the alias is evaluated.  Make everything
     * file completion, just in case...  */
    cp_addcomm(word, (long) 1, (long) 1, (long) 1, (long) 1);
}


void
cp_unalias(char *word)
{
    struct alias *al;

    cp_remkword(CT_ALIASES, word);

    for (al = cp_aliases; al; al = al->al_next)
        if (eq(word, al->al_name))
            break;

    if (al == NULL)
        return;

    if (al->al_next)
        al->al_next->al_prev = al->al_prev;

    if (al->al_prev) {
        al->al_prev->al_next = al->al_next;
    } else {
        al->al_next->al_prev = NULL;
        cp_aliases = al->al_next;
    }

    wl_free(al->al_text);
    tfree(al->al_name);
    tfree(al);
    cp_remcomm(word);
}


void
cp_paliases(char *word)
{
    struct alias *al;

    for (al = cp_aliases; al; al = al->al_next)
        if ((word == NULL) || eq(al->al_name, word)) {
            if (!word)
                fprintf(cp_out, "%s\t", al->al_name);
            wl_print(al->al_text, cp_out);
            (void) putc('\n', cp_out);
        }
}


/* The routine for the "alias" command. */

void
com_alias(wordlist *wl)
{
    if (wl == NULL)
        cp_paliases(NULL);
    else if (wl->wl_next == NULL)
        cp_paliases(wl->wl_word);
    else
        cp_setalias(wl->wl_word, wl->wl_next);
}


void
com_unalias(wordlist *wl)
{
    struct alias *al, *na;

    if (eq(wl->wl_word, "*")) {
        for (al = cp_aliases; al; al = na) {
            na = al->al_next;
            wl_free(al->al_text);
            tfree(al->al_name);
            tfree(al);
        }
        cp_aliases = NULL;
        wl = wl->wl_next;
    }

    while (wl != NULL) {
        cp_unalias(wl->wl_word);
        wl = wl->wl_next;
    }
}

