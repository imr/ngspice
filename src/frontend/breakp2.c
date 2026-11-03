/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Code to deal with breakpoints and tracing.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/ftedebug.h"
#include "breakp2.h"


/* global linked list to store .save data and breakpoint data */
struct dbcomm *dbs = NULL;      /* export for iplot */

/* used in breakp.c and breakp2.c */
int debugnumber = 1;

static char *copynode(char* s);


/* Analyse the data given by the .save card or 'save' command.
   Store the data in the global dbs struct.
*/

/* Save a vector. */

void
com_save(wordlist *wl)
{
    settrace(wl, VF_ACCUM, NULL);
}


/* Save a vector with the analysis type given (name). */
void
com_save2(wordlist *wl, char *name)
{
    settrace(wl, VF_ACCUM, name);
}


void
settrace(wordlist *wl, int what, char *name)
{
    struct dbcomm *d, *last, *dbcheck, **dpp;


    if (!ft_curckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
    }

    /* Find last entry for list extension. */

    last = NULL;
    if (what == VF_ACCUM) {
        /* Also remove any remaining "save none". */

        for (dpp = &dbs; *dpp; ) {
            dbcheck = *dpp;
            if (dbcheck->db_type == DB_SAVE &&
                eq(dbcheck->db_nodename1, "none")) {
                *dpp = dbcheck->db_next;
                ft_curckt->ci_dbs = dbs;
                dbfree1(dbcheck);
            } else {
                last = dbcheck;
                dpp = &dbcheck->db_next;
            }
        }
    } else if (dbs) {
        for (last = dbs; last->db_next; last = last->db_next)
            ;
    }

    for (; wl; wl = wl->wl_next) {
        char *s = cp_unquote(wl->wl_word);
        char *db_nodename1 = NULL;
        char  db_type = 0;

        if (eq(s, "all") || eq(s, "nosub")) {
            switch (what) {
            case VF_PRINT:
                db_type = DB_TRACEALL;
                tfree(s);
                break;
            case VF_ACCUM:
                db_nodename1 = s;
                db_type = DB_SAVE;
                break;
            }
        } else {
            switch (what) {
            case VF_PRINT:
                db_type = DB_TRACENODE;
                break;
            case VF_ACCUM:
                db_type = DB_SAVE;
                break;
            }

            /* v(2) --> 2, i(vds) --> vds#branch */

            if (s) {
                db_nodename1 = copynode(s);
                if (db_nodename1 != s)
                    tfree(s);
            }
            if (!db_nodename1)  /* skip on error */
                continue;
        }

        if (db_type == DB_SAVE) {
            if (eq("none", db_nodename1)) {
                /* "save none" purges the list of saves. */

                last = NULL;
                for (dpp = &dbs; *dpp; ) {
                    dbcheck = *dpp;
                    if (dbcheck->db_type == DB_SAVE) {
                        *dpp = dbcheck->db_next;
                        dbfree1(dbcheck);
                        continue;
                    } else {
                        last = dbcheck;
                        dpp = &dbcheck->db_next;
                    }
                }
                if (wl->wl_next) {
                    // More to come, forget this.

                    tfree(db_nodename1);
                    continue;
                }
            } else {
                /* Ignore duplicates. */

                for (dbcheck = dbs; dbcheck; dbcheck = dbcheck->db_next) {
                    if (dbcheck->db_type == DB_SAVE &&
                        eq(dbcheck->db_nodename1, db_nodename1)) {
                        tfree(db_nodename1);
                        break;
                    }
                }
                if (dbcheck)
                    continue;
            }
        }

        d = TMALLOC(struct dbcomm, 1);
        d->db_analysis = name;
        d->db_type = db_type;
        d->db_nodename1 = db_nodename1;
        d->db_number = debugnumber++;

        if (last)
            last->db_next = d;
        else
            ft_curckt->ci_dbs = dbs = d;
        last = d;
    }
}


/* retrieve the save nodes from dbs into an array */
int
ft_getSaves(struct save_info **savesp)
/* global variable: dbs */
{
    struct dbcomm *d;
    int count = 0, i = 0;
    struct save_info *array;

    for (d = dbs; d; d = d->db_next)
        if (d->db_type == DB_SAVE)
            count++;

    if (!count)
        return (0);

    *savesp = array = TMALLOC(struct save_info, count);

    for (d = dbs; d; d = d->db_next)
        if (d->db_type == DB_SAVE) {
            if (d->db_analysis)
                array[i].analysis = copy(d->db_analysis);
            else
                array[i].analysis = NULL;
            array[i++].name = copy(d->db_nodename1);
        }

    return (count);
}


/* v(2) --> 2, i(vds) --> vds#branch, 3 --> 3, @mn1[vth0] --> @mn1[vth0]
 *   derived from wordlist *gettoks(char *s)
 */

static char*
copynode(char *s)
{
    char *l, *r;
    char *ret = NULL;

    l = strrchr(s, '(');
    if (!l)
        return s;
    if (!strchr(l, ')')) {
        fprintf(cp_err, "Warning: Missing ')' in %s\n  Not saved!\n", s);
        return NULL;
    }
    s = stripWhiteSpacesInsideParens(s);
    l = strrchr(s, '(');
    r = strchr(s, ')');

    *r = '\0';
    if (*(l - 1) == 'i' || *(l - 1) == 'I')
        ret = tprintf("%s#branch", l + 1);
    else
        ret = copy(l + 1);
    free(s);    // Not the original!
    return ret;
}
