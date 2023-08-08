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
#include "breakp.h"
#include "breakp2.h"
#include "runcoms2.h"

#include "completion.h"


static bool satisfied(struct dbcomm *d, struct plot *plot);
static void printcond(struct dbcomm *d, FILE *fp);

static int howmanysteps = 0;
static int steps = 0;
static bool interpolated = FALSE;


/* Set a breakpoint. Possible commands are:
 *  stop after n
 *  stop when var cond val
 *
 * If more than one is given on a command line, then this is a conjunction.
 */

void
com_stop(wordlist *wl)
{
    /* Check for an active circuit */
    if (ft_curckt == (struct circ *) NULL) {
        fprintf(cp_err, "No circuit loaded. Stopping is not possible.\n");
        return;
    }

    /* Check to see if we have to consider interpolated data. */
    if (cp_getvar("interp", CP_BOOL, NULL, 0)) {
        interpolated = TRUE;
        fprintf(cp_out, "Note: Stop condition has to fit the interpolated time data!\n\n");
    }
    else
        interpolated = FALSE;

    struct dbcomm *thisone = NULL;
    struct dbcomm *d = NULL;
    char *s, buf[64];
    int i;

    while (wl) {
        if (thisone == NULL) {
            thisone = d = TMALLOC(struct dbcomm, 1);
        } else {
            d->db_also = TMALLOC(struct dbcomm, 1);
            d = d->db_also;
        }
        d->db_also = NULL;

        /* Figure out what the first condition is. */
        d->db_analysis = NULL;
        if (eq(wl->wl_word, "after") && wl->wl_next) {
            d->db_type = DB_STOPAFTER;
            d->db_number = debugnumber;
            if (!wl->wl_next->wl_word) {
                i = 0;
            } else {
#ifdef HAVE_CTYPE_H
                for (s = wl->wl_next->wl_word, i = 0; *s; s++)
                    if (!isdigit_c(*s))
                        goto bad;
                    else
                        i = i * 10 + (*s - '0');
#else
                i = atoi(wl->wl_next->wl_word); /* etoi ??? */
#endif
            }
            d->db_iteration = i;
            wl = wl->wl_next->wl_next;
        } else if (eq(wl->wl_word, "when") && wl->wl_next) {
            /* cp_lexer(string) will not discriminate '=', so we have
               to do it here */
            if (strchr(wl->wl_next->wl_word, '=') &&
                    (!(wl->wl_next->wl_next) ||
                    strstr(wl->wl_next->wl_next->wl_word, "when") ||
                    strstr(wl->wl_next->wl_next->wl_word, "after"))) {
                /* we have vec=val in a single word */
                wordlist *wln;
                char **charr = TMALLOC(char*, 4);
                char *tok = copy(wl->wl_next->wl_word);
                char *tokeq = strchr(tok, '=');
                char *tokafter = copy(tokeq + 1);
                *tokeq = '\0';
                charr[0] = tok;
                charr[1] = copy("eq");
                charr[2] = tokafter;
                charr[3] = NULL;
                wln = wl_build((const char * const *) charr);
                wl_splice(wl->wl_next, wln);
            }

            /* continue with parsing the enhanced wordlist */
            if (wl->wl_next->wl_next && wl->wl_next->wl_next->wl_next) {
                wl = wl->wl_next;
                d->db_number = debugnumber;
                d->db_type = DB_STOPWHEN;
                s = wl->wl_word;

                {
                    double val;
                    if (ft_numparse(&s, FALSE, &val) >= 0) {
                        d->db_value1 = val;
                    }
                    else {
                        d->db_nodename1 = copy(wl->wl_word);
                    }
                }
                wl = wl->wl_next;

                /* Now get the condition */
                if (eq(wl->wl_word, "eq") || eq(wl->wl_word, "="))
                    d->db_op = DBC_EQU;
                else if (eq(wl->wl_word, "ne"))
                    d->db_op = DBC_NEQ;
                else if (eq(wl->wl_word, "gt") || eq(wl->wl_word, ">"))
                    d->db_op = DBC_GT;
                else if (eq(wl->wl_word, "lt"))
                    d->db_op = DBC_LT;
                else if (eq(wl->wl_word, "<")) {
                    /* "<>" is parsed as two words. */

                    if (eq(wl->wl_next->wl_word, ">")) {
                        if (!wl->wl_next->wl_next)
                            goto bad;
                        d->db_op = DBC_NEQ;
                        wl = wl->wl_next;
                    } else {
                        d->db_op = DBC_LT;
                    }
                } else if (eq(wl->wl_word, "ge") || eq(wl->wl_word, ">="))
                    d->db_op = DBC_GTE;
                else if (eq(wl->wl_word, "le") || eq(wl->wl_word, "<="))
                    d->db_op = DBC_LTE;
                else
                    goto bad;
                wl = wl->wl_next;

                /* Now see about the second one. */
                s = wl->wl_word;

                {
                    double val;
                    if (ft_numparse(&s, FALSE, &val) >= 0) {
                        d->db_value2 = val;
                    }
                    else {
                        d->db_nodename2 = copy(wl->wl_word);
                    }
                }
                wl = wl->wl_next;
            } else { // Neither "after" nor "when".
                goto bad;
            }
        } else {
            goto bad;
        }
    } /* end of loop over wordlist */

    if (thisone) {
        if (dbs) {
            for (d = dbs; d->db_next; d = d->db_next)
                ;
            d->db_next = thisone;
        } else {
            ft_curckt->ci_dbs = dbs = thisone;
        }
        (void) sprintf(buf, "%d", debugnumber);
        cp_addkword(CT_DBNUMS, buf);
        debugnumber++;
    }

    return;

bad:
    fprintf(cp_err, "Syntax error parsing breakpoint specification.\n");
    while (thisone) {
        d = thisone->db_also;
        txfree(thisone);
        thisone = d;
    }
} /* end of function com_stop */



/* Trace a node (have wrd_point print it). Usage is "trace node ..."*/
void
com_trce(wordlist *wl)
{
    settrace(wl, VF_PRINT, NULL);
}


/* Incrementally plot a value. This is just like trace. */

void
com_iplot(wordlist *wl)
{
    /* Check for an active circuit */
    if (ft_curckt == (struct circ *) NULL) {
        fprintf(cp_err, "No circuit loaded. "
                "Incremental plotting is not possible.\n");
        return;
    }

    /* settrace(wl, VF_PLOT); */

    struct dbcomm *d, *td, *currentdb = NULL;
    double         window = 0.0;
    int            initial_steps = IPOINTMIN;
    char          *s;

    /* Look for "-w window-size" at the front, indicating a windowed iplot
     * or "-d steps" to set the initial delay before the window appears.
     */

    while (wl && wl->wl_word[0] == '-') {
        if (wl->wl_word[1] == 'w' && !wl->wl_word[2]) {
            wl = wl->wl_next;
            if (wl) {
                char *cp;
                int   error;

                cp = wl->wl_word;
                window = INPevaluate(&cp, &error, 0);
                if (error || window <= 0) {
                    fprintf(cp_err,
                            "Incremental plot width must be positive.\n");
                    return;
                }
            }
        } else if (wl->wl_word[1] == 'd' && !wl->wl_word[2]) {
            wl = wl->wl_next;
            if (wl)
                initial_steps = atoi(wl->wl_word);
        } else {
            break;
        }
        wl = wl->wl_next;
    }

    /* We use a modified ad-hoc algorithm here where db_also denotes
       vectors on the same command line and db_next denotes
       separate iplot commands. */

    while (wl) {
        s = cp_unquote(wl->wl_word);
        d = TMALLOC(struct dbcomm, 1);
        d->db_analysis = NULL;
        d->db_number = debugnumber++;
        d->db_op = initial_steps; // Field re-use
        d->db_value1 = window;    // Field re-use
        if (eq(s, "all")) {
            d->db_type = DB_IPLOTALL;
        } else {
            d->db_type = DB_IPLOT;
            d->db_nodename1 = copy(s);
        }
        tfree(s);/*DG: avoid memory leak */
        d->db_also = currentdb;
        currentdb = d;
        wl = wl->wl_next;
    }

    if (dbs) {
        for (td = dbs; td->db_next; td = td->db_next)
            ;
        td->db_next = currentdb;
    } else {
        ft_curckt->ci_dbs = dbs = currentdb;
    }
}


/* Step a number of iterations. */

void
com_step(wordlist *wl)
{
    if (wl)
        steps = howmanysteps = atoi(wl->wl_word);
    else
        steps = howmanysteps = 1;

    com_resume(NULL);
}


/* Print out the currently active breakpoints and traces. If we are printing
 * to a file, assume that the file will be used for a later source and leave
 * off the event numbers (with UNIX, that is).  -- I don't like this.
 */

#undef  isatty
#define isatty(xxxx) 1


void
com_sttus(wordlist *wl)
{
    struct dbcomm *d, *dc;

    NG_IGNORE(wl);

    for (d = dbs; d; d = d->db_next) {
        if (d->db_type == DB_TRACENODE) {
            if (isatty(fileno(cp_out)))
                fprintf(cp_out, "%-4d trace %s", d->db_number, d->db_nodename1);
            else
                fprintf(cp_out, "trace %s", d->db_nodename1);
        } else if (d->db_type == DB_IPLOT) {
            if (isatty(fileno(cp_out)))
                fprintf(cp_out, "%-4d iplot %s", d->db_number, d->db_nodename1);
            else
                fprintf(cp_out, "iplot %s", d->db_nodename1);
            for (dc = d->db_also; dc; dc = dc->db_also)
                fprintf(cp_out, " %s", dc->db_nodename1);
        } else if (d->db_type == DB_SAVE) {
            if (isatty(fileno(cp_out)))
                fprintf(cp_out, "%-4d save %s", d->db_number, d->db_nodename1);
            else
                fprintf(cp_out, "save %s", d->db_nodename1);
        } else if (d->db_type == DB_TRACEALL) {
            if (isatty(fileno(cp_out)))
                fprintf(cp_out, "%-4d trace all", d->db_number);
            else
                fprintf(cp_out, "trace all");
        } else if (d->db_type == DB_IPLOTALL) {
            if (isatty(fileno(cp_out)))
                fprintf(cp_out, "%-4d iplot all", d->db_number);
            else
                fprintf(cp_out, "iplot all");
        } else if (d->db_type == DB_SAVEALL) {
            if (isatty(fileno(cp_out)))
                fprintf(cp_out, "%-4d save all", d->db_number);
            else
                fprintf(cp_out, "save all");
        } else if ((d->db_type == DB_STOPAFTER) || (d->db_type == DB_STOPWHEN)) {
            if (isatty(fileno(cp_out)))
                fprintf(cp_out, "%-4d stop", d->db_number);
            else
                fprintf(cp_out, "stop");
            printcond(d, cp_out);
        } else if (d->db_type == DB_DEADIPLOT) {
            if (isatty(fileno(cp_out))) {
                fprintf(cp_out, "%-4d exiting iplot %s", d->db_number,
                        d->db_nodename1);
            } else {
                fprintf(cp_out, "exiting iplot %s", d->db_nodename1);
            }
            for (dc = d->db_also; dc; dc = dc->db_also)
                fprintf(cp_out, " %s", dc->db_nodename1);
        } else {
            fprintf(cp_err,
                    "com_sttus: Internal Error: bad db %d\n", d->db_type);
        }
        (void) putc('\n', cp_out);
    }
}


/* free the dbcomm structure which has been defined in
 *   function settrace() in breakp2.c
 */

void
dbfree1(struct dbcomm *d)
{
    tfree(d->db_nodename1);
    tfree(d->db_nodename2);
    if (d->db_also)
        dbfree(d->db_also);
    tfree(d);
}


void
dbfree(struct dbcomm *d)
{
    while (d) {
        struct dbcomm *next_d = d->db_next;
        dbfree1(d);
        d = next_d;
    }
}


/* Delete breakpoints and traces. Usage is delete [number ...] */

void
com_delete(wordlist *wl)
{
    int i;
    char *s, buf[64];
    struct dbcomm *d, *dt;

    if (wl && eq(wl->wl_word, "all")) {
        dbfree(dbs);
        dbs = NULL;
        if (ft_curckt)
            ft_curckt->ci_dbs = NULL;
        return;
    } else if (!wl) {
        if (!dbs) {
            fprintf(cp_err, "Error: no debugs in effect\n");
            return;
        }
    }

    while (wl) {

        if (wl->wl_word) {
#ifdef HAVE_CTYPE_H
            for (s = wl->wl_word, i = 0; *s; s++)
                if (!isdigit_c(*s)) {
                    fprintf(cp_err, "Error: %s isn't a number.\n",
                            wl->wl_word);
                    goto bad;
                } else {
                    i = i * 10 + (*s - '0');
                }
#else
            i = atoi(wl->wl_next->wl_word); /* etoi ??? */
#endif
        } else {
            i = 0;
        }

        for (d = dbs, dt = NULL; d; d = d->db_next) {
            if (d->db_number == i) {
                if (dt)
                    dt->db_next = d->db_next;
                else
                    ft_curckt->ci_dbs = dbs = d->db_next;
                dbfree1(d);
                (void) sprintf(buf, "%d", i);
                cp_remkword(CT_DBNUMS, buf);
                break;
            }
            dt = d;
        }

    bad:
        wl = wl->wl_next;
    }
}


/* Writedata calls this routine to see if it should keep going. If it
 * returns TRUE, then the run should resume.
 */

bool
ft_bpcheck(struct plot *runplot, int iteration)
{
    struct dbcomm *d, *dt;

    if ((howmanysteps > 0) && (--howmanysteps == 0)) {
        if (steps > 1)
            fprintf(cp_err, "Stopped after %d steps.\n", steps);
        return (FALSE);
    }

    /* Check the debugs set. */
    for (d = dbs; d; d = d->db_next) {
        for (dt = d; dt; dt = dt->db_also) {
            switch (dt->db_type) {
            case DB_TRACENODE:
            case DB_TRACEALL:
            case DB_IPLOT:
            case DB_DEADIPLOT:
            case DB_IPLOTALL:
            case DB_SAVE:
            case DB_SAVEALL:
                goto more;
            case DB_STOPAFTER:
                if (iteration == dt->db_iteration)
                    break;
                else
                    goto more;
            case DB_STOPWHEN:
                /* See if the condition is TRUE. */
                if (satisfied(dt, runplot))
                    break;
                else
                    goto more;
            default:
                fprintf(cp_err, "ft_bpcheck: Internal Error: bad db %d\n",
                        dt->db_type);
            }
        }

        if (dt == NULL) {
            /* It made it */
            fprintf(cp_err, "%-2d: condition met: stop ", d->db_number);
            printcond(d, cp_err);
            (void) putc('\n', cp_err);
            return (FALSE);
        }

    more:   /* Just continue */
        ;
    }

    return (TRUE);
}


/* This is called to determine whether a STOPWHEN is TRUE. */

static bool
satisfied(struct dbcomm *d, struct plot *plot)
{
    struct dvec *v1 = NULL, *v2 = NULL;
    double d1, d2;
    static double laststoptime = 0.;

    if (d->db_nodename1) {
        if ((v1 = vec_fromplot(d->db_nodename1, plot)) == NULL) {
            fprintf(cp_err, "Error: %s: no such node\n", d->db_nodename1);
            return (FALSE);
        }
        if (v1->v_length == 0)
            return (FALSE);

        if (isreal(v1))
            d1 = v1->v_realdata[v1->v_length - 1];
        else
            d1 = realpart((v1->v_compdata[v1->v_length - 1]));

    } else {
        d1 = d->db_value1;
    }

    if (d->db_nodename2) {
        if ((v2 = vec_fromplot(d->db_nodename2, plot)) == NULL) {
            fprintf(cp_err, "Error: %s: no such node\n", d->db_nodename2);
            return (FALSE);
        }
        if (isreal(v2))
            d2 = v2->v_realdata[v2->v_length - 1];
        else
            d2 = realpart((v2->v_compdata[v2->v_length - 1]));
    /* option interp: no new time step since last stop */
    } else if (interpolated && AlmostEqualUlps(d1, laststoptime, 3)){    
        d2 = 0.;
    } else {
        d2 = d->db_value2;
    }

    switch (d->db_op) {
    case DBC_EQU:
    {
        bool hit = AlmostEqualUlps(d1, d2, 3) ? TRUE : FALSE;
        /* option interp: save the last stop time */
        if (interpolated && hit)
            laststoptime = d1;
        return hit;
    }
        // return ((d1 == d2) ? TRUE : FALSE);
    case DBC_NEQ:
        return ((d1 != d2) ? TRUE : FALSE);
    case DBC_GTE:
        return ((d1 >= d2) ? TRUE : FALSE);
    case DBC_LTE:
        return ((d1 <= d2) ? TRUE : FALSE);
    case DBC_GT:
        return ((d1 > d2) ? TRUE : FALSE);
    case DBC_LT:
        return ((d1 < d2) ? TRUE : FALSE);
    default:
        fprintf(cp_err,
                "satisfied: Internal Error: bad cond %d\n", d->db_op);
        return (FALSE);
    }
}


/* Writedata calls this before it starts a run, to set the proper flags
 * on the dvecs. If a change is made during a break, then the routine
 * wrd_chtrace is used from these routines. We have to be clever with save:
 * if there was no save given, then save everything. Unfortunately you
 * can't stop in the middle, do a save, and have the rest then discarded.
 */

void
ft_trquery(void)
{
}


static void
printcond(struct dbcomm *d, FILE *fp)
{
    struct dbcomm *dt;

    for (dt = d; dt; dt = dt->db_also) {
        if (dt->db_type == DB_STOPAFTER) {
            fprintf(fp, " after %d", dt->db_iteration);
        } else {
            if (dt->db_nodename1)
                fprintf(fp, " when %s", dt->db_nodename1);
            else
                fprintf(fp, " when %g", dt->db_value1);

            switch (dt->db_op) {
            case DBC_EQU:
                fputs(" =", fp);
                break;
            case DBC_NEQ:
                fputs(" <>", fp);
                break;
            case DBC_GT:
                fputs(" >", fp);
                break;
            case DBC_LT:
                fputs(" <", fp);
                break;
            case DBC_GTE:
                fputs(" >=", fp);
                break;
            case DBC_LTE:
                fputs(" <=", fp);
                break;
            default:
                fprintf(cp_err,
                        "printcond: Internal Error: bad cond %d", dt->db_op);
            }

            if (dt->db_nodename2)
                fprintf(fp, " %s", dt->db_nodename2);
            else
                fprintf(fp, " %g", dt->db_value2);
        }
    }
}
