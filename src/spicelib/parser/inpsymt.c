/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Stuff for the terminal and node symbol tables.
 * Defined: INPtabInit, INPinsert, INPtermInsert, INPtabEnd
 */
/* MW. Special INPinsertNofree for routines from spiceif.c and outif.c */

#include "ngspice/ngspice.h"
#include <stdio.h>		/* Take this out soon. */
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/inpdefs.h"
#include "ngspice/cpstd.h"
#include "ngspice/fteext.h"
#include "inpxx.h"


static int hash(char *name, int tsize);

/* Initialize the symbol tables. */

INPtables *INPtabInit(int numlines)
{
    INPtables *tab;

    tab = TMALLOC(INPtables, 1);
    tab->INPsymtab = TMALLOC(struct INPtab *, numlines / 4 + 1);
    ZERO(tab->INPsymtab, (numlines / 4 + 1) * sizeof(struct INPtab *));
    tab->INPtermsymtab = TMALLOC(struct INPnTab *, numlines);
    ZERO(tab->INPtermsymtab, numlines * sizeof(struct INPnTab *));
    tab->INPsize = numlines / 4 + 1;
    tab->INPtermsize = numlines;
    return (tab);
}

/* insert 'token' into the terminal symbol table */
/* create a NEW NODE and return a pointer to it in *node */

int INPtermInsert(CKTcircuit *ckt, char **token, INPtables * tab, CKTnode **node)
{
    int key;
    int error;
    struct INPnTab *t;

    key = hash(*token, tab->INPtermsize);
    for (t = tab->INPtermsymtab[key]; t; t = t->t_next) {
        if (!strcmp(*token, t->t_ent)) {
            FREE(*token);
            *token = t->t_ent;
            if (node)
                *node = t->t_node;
            return (E_EXISTS);
        }
    }
    t = TMALLOC(struct INPnTab, 1);
    if (t == NULL)
        return (E_NOMEM);
    ZERO(t, struct INPnTab);
    error = ft_sim->newNode (ckt, &(t->t_node), *token);
    if (error)
        return (error);
    if (node)
        *node = t->t_node;
    t->t_ent = *token;
    t->t_next = tab->INPtermsymtab[key];
    tab->INPtermsymtab[key] = t;
    return (OK);
}


/* insert 'token' into the terminal symbol table */
/* USE node as the node pointer */


int INPmkTerm(CKTcircuit *ckt, char **token, INPtables * tab, CKTnode **node)
{
    int key;
    struct INPnTab *t;

    NG_IGNORE(ckt);

    key = hash(*token, tab->INPtermsize);
    for (t = tab->INPtermsymtab[key]; t; t = t->t_next) {
        if (!strcmp(*token, t->t_ent)) {
            FREE(*token);
            *token = t->t_ent;
            if (node)
                *node = t->t_node;
            return (E_EXISTS);
        }
    }
    t = TMALLOC(struct INPnTab, 1);
    if (t == NULL)
        return (E_NOMEM);
    ZERO(t, struct INPnTab);
    t->t_node = *node;
    t->t_ent = *token;
    t->t_next = tab->INPtermsymtab[key];
    tab->INPtermsymtab[key] = t;
    return (OK);
}

/* insert 'token' into the terminal symbol table as a name for ground*/

int INPgndInsert(CKTcircuit *ckt, char **token, INPtables * tab, CKTnode **node)
{
    int key;
    int error;
    struct INPnTab *t;

    key = hash(*token, tab->INPtermsize);
    for (t = tab->INPtermsymtab[key]; t; t = t->t_next) {
        if (!strcmp(*token, t->t_ent)) {
            FREE(*token);
            *token = t->t_ent;
            if (node)
                *node = t->t_node;
            return (E_EXISTS);
        }
    }
    t = TMALLOC(struct INPnTab, 1);
    if (t == NULL)
        return (E_NOMEM);
    ZERO(t, struct INPnTab);
    error = ft_sim->groundNode (ckt, &(t->t_node), *token);
    if (error)
        return (error);
    if (node)
        *node = t->t_node;
    t->t_ent = *token;
    t->t_next = tab->INPtermsymtab[key];
    tab->INPtermsymtab[key] = t;
    return (OK);
}

/* retrieve 'token' from the symbol table */

int INPretrieve(char **token, INPtables * tab)
{
    struct INPtab *t;
    int key;

    key = hash(*token, tab->INPsize);
    for (t = tab->INPsymtab[key]; t; t = t->t_next)
        if (!strcmp(*token, t->t_ent)) {
            *token = t->t_ent;
            return (OK);
        }
    return (E_BADPARM);
}


/* insert 'token' into the symbol table */

int INPinsert(char **token, INPtables * tab)
{
    struct INPtab *t;
    int key;

    key = hash(*token, tab->INPsize);
    for (t = tab->INPsymtab[key]; t; t = t->t_next)
        if (!strcmp(*token, t->t_ent)) {
            FREE(*token);
            *token = t->t_ent;
            return (E_EXISTS);
        }
    t = TMALLOC(struct INPtab, 1);
    if (t == NULL)
        return (E_NOMEM);
    ZERO(t, struct INPtab);
    t->t_ent = *token;
    t->t_next = tab->INPsymtab[key];
    tab->INPsymtab[key] = t;
    return (OK);
}


/* MW. insert 'token' into the symbol table but no free() token pointer.
*	Calling routine should take care for this */

int INPinsertNofree(char **token, INPtables * tab)
{
    struct INPtab *t;
    int key;

    key = hash(*token, tab->INPsize);
    for (t = tab->INPsymtab[key]; t; t = t->t_next)
        if (!strcmp(*token, t->t_ent)) {

            /* MW. We can't touch memory pointed by token now */
            *token = t->t_ent;
            return (E_EXISTS);
        }
    t = TMALLOC(struct INPtab, 1);
    if (t == NULL)
        return (E_NOMEM);
    ZERO(t, struct INPtab);
    t->t_ent = *token;
    t->t_next = tab->INPsymtab[key];
    tab->INPsymtab[key] = t;
    return (OK);
}

/* remove 'token' from the symbol table */
int INPremove(char *token, INPtables * tab)
{
    struct INPtab *t, **prevp;
    int key;

    key = hash(token, tab->INPsize);
    prevp = &tab->INPsymtab[key];
    for (t = *prevp; t && token != t->t_ent; t = t->t_next)
        prevp = &t->t_next;
    if (!t)
        return OK;

    *prevp = t->t_next;
    tfree(t->t_ent);
    tfree(t);

    return OK;
}

/* remove 'token' from the symbol table */
int INPremTerm(char *token, INPtables * tab)
{
    struct INPnTab *t, **prevp;
    int key;

    key = hash(token, tab->INPtermsize);
    prevp = &tab->INPtermsymtab[key];
    for (t = *prevp; t && token != t->t_ent; t = t->t_next)
        prevp = &t->t_next;
    if (!t)
        return OK;

    *prevp = t->t_next;
    tfree(t->t_ent);
    tfree(t);

    return OK;
}

/* Free the space used by the symbol tables. */

void INPtabEnd(INPtables * tab)
{
    struct INPtab *t, *lt;
    struct INPnTab *n, *ln;
    int i;

    for (i = 0; i < tab->INPsize; i++)
        for (t = tab->INPsymtab[i]; t; t = lt) {
            lt = t->t_next;
            FREE(t->t_ent);
            FREE(t);
        }
    FREE(tab->INPsymtab);
    for (i = 0; i < tab->INPtermsize; i++)
        for (n = tab->INPtermsymtab[i]; n; n = ln) {
            ln = n->t_next;
            FREE(n->t_ent);
            FREE(n);		/* But not t_node ! */
        }
    FREE(tab->INPtermsymtab);
    FREE(tab);
    return;
}

static int hash(char *name, int tsize)
{
    unsigned int hash = 5381;
    char c;

    while ((c = *name++) != '\0')
        hash = (hash * 33) ^ (unsigned) c;

    return (int) (hash % (unsigned) tsize);
}

/* Just tests for the existence of a node. If node is found, its
   token and node adresses are fed back.
   Return value 0 if no node is found, E_EXISTS if its already there */
int INPtermSearch(CKTcircuit* ckt, char** token, INPtables* tab, CKTnode** node)
{
    int key;
    struct INPnTab* t;
    NG_IGNORE(ckt);

    key = hash(*token, tab->INPtermsize);
    for (t = tab->INPtermsymtab[key]; t; t = t->t_next) {
        if (!strcmp(*token, t->t_ent)) {
            FREE(*token);
            *token = t->t_ent;
            if (node)
                *node = t->t_node;
            return (E_EXISTS);
        }
    }
    return (0);
}
