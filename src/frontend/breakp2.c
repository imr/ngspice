/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
$Id$
**********/

/*
 * Code to deal with breakpoints and tracing.
 */

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "dvec.h"
#include "ftedebug.h"
#include "quote.h"
#include "breakp2.h"

struct dbcomm *dbs = NULL;      /* export for iplot */
int debugnumber = 1;

/* Set a breakpoint. Possible commands are:
 *  stop after n
 *  stop when var cond val
 *
 * If more than one is given on a command line, then this is a conjunction.
 */

/* Save a vector. */

void
com_save(wordlist *wl)
{
    settrace(wl, VF_ACCUM, NULL);
    return;
}

void
com_save2(wordlist *wl, char *name)
{
    settrace(wl, VF_ACCUM, name);
    return;
}

void
settrace(wordlist *wl, int what, char *name)
{
    struct dbcomm *d, *td;
    char *s;

    while (wl) {
        s = cp_unquote(wl->wl_word);
        d = alloc(struct dbcomm);
        d->db_number = debugnumber++;
        d->db_analysis = name;
        if (eq(s, "all")) {
            switch (what) {
                case VF_PRINT:
                    d->db_type = DB_TRACEALL;
                    break;
/*              case VF_PLOT:
                    d->db_type = DB_IPLOTALL;
                    break; */
                case VF_ACCUM:
                    /* d->db_type = DB_SAVEALL; */
		    d->db_nodename1 = copy(s);
                    d->db_type = DB_SAVE;
                    break;
            }
/*          wrd_chtrace((char *) NULL, TRUE, what); */
        } else {
            switch (what) {
                case VF_PRINT:
                    d->db_type = DB_TRACENODE;
                    break;
/*              case VF_PLOT:
                    d->db_type = DB_IPLOT;
                    break; */
                case VF_ACCUM:
                    d->db_type = DB_SAVE;
                    break;
            }
            d->db_nodename1 = copy(s);
/*          wrd_chtrace(s, TRUE, what); */
        }
         tfree(s);/*DG avoid memoy leak */
        if (dbs) {
            for (td = dbs; td->db_next; td = td->db_next)
                ;
            td->db_next = d;
        } else
            dbs = d;
        wl = wl->wl_next;
    }
    return;
}

int
ft_getSaves(struct save_info **savesp)
{
    struct dbcomm *d;
    int count = 0, i = 0;
    struct save_info *array;

    for (d = dbs; d; d = d->db_next)
        if (d->db_type == DB_SAVE)
            count++;
    
    if (!count)
        return (0);
    
    *savesp = array = (struct save_info *)
	    tmalloc(sizeof (struct save_info) * count);

    for (d = dbs; d; d = d->db_next)
        if (d->db_type == DB_SAVE) {
            array[i].used = 0;
	    if (d->db_analysis)
		array[i].analysis = (IFuid *) copy(d->db_analysis);
	    else
		array[i].analysis = NULL;
            array[i++].name = copy(d->db_nodename1);
	}
    
    return (count);
}
