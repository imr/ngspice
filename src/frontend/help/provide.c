/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified 1999 Emmanuel Rouat
**********/

/*
 *   faustus@cad.berkeley.edu, ucbvax!faustus
 * Permission is granted to modify and re-distribute this code in any manner
 * as long as this notice is preserved.  All standard disclaimers apply.
 *
 * Toss the help window up on the screen, and deal with the graph...
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpstd.h"
#include "ngspice/hlpdefs.h"
#include "ngspice/suffix.h"

bool hlp_usex = FALSE;


void
hlp_provide(topic *top)
{
    toplink *res;
    topic *parent, *newtop;

    if (!top)
        return;

#ifndef X_DISPLAY_MISSING
    if (getenv("DISPLAY") || hlp_displayname)
        hlp_usex = TRUE;
#endif

    top->xposition = top->yposition = 0;
    if (hlp_usex) {
        if (!hlp_xdisplay(top)) {
            fprintf(stderr, "Couldn't open X display.\n");
            return;
        }
    } else {
        if (!hlp_tdisplay(top)) {
            fprintf(stderr, "Couldn't display text\n");
            return;
        }
    }

#ifndef X_DISPLAY_MISSING       /* X11 does this asynchronously */
    if (hlp_usex)
        return;
#endif

    for (;;) {
        if (hlp_usex)
            res = hlp_xhandle(&parent);
        else
            res = hlp_thandle(&parent);
        if (!res && !parent) {
            /* No more windows. */
            hlp_killfamily(top);
            if (hlp_usex)
                hlp_xclosedisplay(); /* need to change
                                        display pointer back J.H. */
            return;
        }
        if (res) {
            /* Create a new window... */
            if (hlp_usex)
                hlp_xwait(parent, TRUE);
            if ((newtop = hlp_read(res->place)) == NULL) {
                fprintf(stderr, "Internal error: bad link\n");
                hlp_xwait(parent, FALSE);
                continue;
            }
            if (hlp_usex)
                hlp_xwait(parent, FALSE);
            newtop->next = parent->children;
            parent->children = newtop;
            newtop->parent = parent;
            newtop->xposition = parent->xposition + 50;
            newtop->yposition = parent->yposition + 50;
            if (hlp_usex) {
                if (!hlp_xdisplay(newtop)) {
                    fprintf(stderr, "Couldn't open win\n");
                    return;
                }
            } else {
                if (!hlp_tdisplay(newtop)) {
                    fprintf(stderr, "Couldn't display\n");
                    return;
                }
            }
        } else {
            /* Blow this one and its descendants away. */
            hlp_killfamily(parent);
            hlp_fixchildren(parent);
            if (parent == top)
                return;
        }
    }
}


void
hlp_fixchildren(topic *parent)
{

    topic *pa;

    if (parent->parent) {
        if (parent->parent->children == parent)
            parent->parent->children = parent->next;
        else {
            for (pa = parent->parent->children; pa->next; pa = pa->next)
                if (pa->next == parent)
                    break;
            if (!pa->next)
                fprintf(stderr, "bah...\n");
            pa->next = pa->next->next;
        }
    }
}


/* Note that this doesn't actually free the data structures, just gets
 * rid of the window.
 */

void
hlp_killfamily(topic *top)
{
    topic *ch;

    for (ch = top->children; ch; ch = ch->next)
        hlp_killfamily(ch);

    if (hlp_usex)
        hlp_xkillwin(top);
    else
        hlp_tkillwin(top);

    top->children = NULL;
}

