/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* The front-end command loop.  */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"

#include "control.h"
#include "com_cdump.h"
#include "variable.h"
#include "ngspice/fteext.h"


/* Return values from doblock().  I am assuming that nobody will use
 * these characters in a string.  */
#define NORMAL      '\001'
#define BROKEN      '\002'
#define CONTINUED   '\003'
#define NORMAL_STR  "\001"
#define BROKEN_STR  "\002"
#define CONTINUED_STR   "\003"

static void cp_free_control(void); /* needed by resetcontrol */

/* Are we waiting for a command? This lets signal handling be
 * more clever. */

bool cp_cwait = FALSE;
char *cp_csep = ";"; /* character that separates commands */

bool cp_dounixcom = FALSE;

/* We have to keep the control structures in a stack, so that when we
 * do a 'source', we can push a fresh set onto the top...  Actually
 * there has to be two stacks -- one for the pointer to the list of
 * control structs, and one for the 'current command' pointer...  */
struct control *control[CONTROLSTACKSIZE];
struct control *cend[CONTROLSTACKSIZE];
int stackp = 0;


/* If there is an argument, give this to cshpar to use instead of
 * stdin. In a few places, we call cp_evloop again if it returns 1 and
 * exit (or close a file) if it returns 0... Because of the way
 * sources are done, we can't allow the control structures to get
 * blown away every time we return -- probably every time we type
 * source at the keyboard and every time a source returns to keyboard
 * input is ok though -- use ft_controlreset.  */

/* Notes by CDHW:
 * This routine leaked like a sieve because each getcommand() created a
 * wordlist that was never freed because it might have been added into
 * the control structure. I've tackled this by making sure that everything
 * put into the cend[stackp] is a copy. This means that wlist can be
 * destroyed safely
 */

/* no redirection after the following commands (we may need more to add here!) */
static char *noredirect[] = { "stop", "define", "circbyline", NULL};


/* This function returns the (first) structure wit the label s */
static struct control *findlabel(const char *s, struct control *ct)
{
    while (ct) {
        if ((ct->co_type == CO_LABEL) && eq(s, ct->co_text->wl_word)) {
            break;
        }
        ct = ct->co_next;
    }
    return (ct);
}


/* This is also in cshpar.c ... */
static void
pwlist(wordlist *wlist, char *name)
{
    wordlist *wl;

    if (!cp_debug)
        return;
    fprintf(cp_err, "%s : [ ", name);
    for (wl = wlist; wl; wl = wl->wl_next)
        fprintf(cp_err, "%s ", wl->wl_word);
    fprintf(cp_err, "]\n");
}


/* CDHW defined functions */

static void
pwlist_echo(wordlist *wlist, char *name)   /*CDHW used to perform function of set echo */
{
    wordlist *wl;

    if ((!cp_echo)||cp_debug) /* cpdebug prints the same info */
        return;
    fprintf(cp_err, "%s ", name);
    for (wl = wlist; wl; wl = wl->wl_next)
        fprintf(cp_err, "%s ", wl->wl_word);
    fprintf(cp_err, "\n");
}


/*CDHW Remove control structure and free the memory its hogging CDHW*/

static void
ctl_free(struct control *ctrl)
{
    if (!ctrl) {
        return;
    }

    wl_free(ctrl->co_cond);
    ctrl->co_cond = NULL;
    txfree(ctrl->co_foreachvar);
    ctrl->co_foreachvar = NULL;
    wl_free(ctrl->co_text);
    ctrl->co_text = NULL;
    ctl_free(ctrl->co_children);
    ctrl->co_children = NULL;
    ctl_free(ctrl->co_elseblock);
    ctrl->co_elseblock = NULL;
    ctl_free(ctrl->co_next);
    ctrl->co_next = NULL;
    txfree(ctrl);
}


/* Note that we only do io redirection when we get to here - we also
 * postpone some other things until now.  */
static void
docommand(wordlist *wlist)
{
    wordlist *rwlist;

    if (cp_debug) {
        printf("docommand ");
        wl_print(wlist, stdout);
        putc('\n', stdout);
    }

    /* Do all the things that used to be done by cshpar when the line
     * was read...  */
    wlist = cp_variablesubst(wlist);
    pwlist(wlist, "After variable substitution");

    wlist = cp_bquote(wlist);
    pwlist(wlist, "After backquote substitution");

    /* Do not expand braces after command circbyline, keep them intact */
    if (!eq(wlist->wl_word, "circbyline"))
        wlist = cp_doglob(wlist);
    pwlist(wlist, "After globbing");

    pwlist_echo(wlist, "Becomes >");

    if (!wlist || !wlist->wl_word) /*CDHW need to free wlist in second case? CDHW*/
        return;

    /* Now loop through all of the commands given. */
    rwlist = wlist;
    while (wlist) {

        char *s;
        int i;
        struct comm *command;
        wordlist *nextc, *ee;

        nextc = wl_find(cp_csep, wlist);

        if (nextc == wlist) {   /* skip leading `;' */
            wlist = wlist->wl_next;
            continue;
        }

        /* Temporarily hide the rest of the command... */
        ee = wlist->wl_prev;
        wl_chop(nextc);
        wl_chop(wlist);

        /* And do the redirection. */
        cp_ioreset();
        for (i = 0; noredirect[i]; i++)
            if (eq(wlist->wl_word, noredirect[i]))
                break;
        if (!noredirect[i])
            if ((wlist = cp_redirect(wlist)) == NULL) {
                cp_ioreset();
                return;
            }

        s = wlist->wl_word;

        /* Look for the command in the command list. */
        for (i = 0; cp_coms[i].co_comname; i++)
            if (strcasecmp(cp_coms[i].co_comname, s) == 0)
                break;

        command = &cp_coms[i];

        /* Now give the user-supplied command routine a try... */
        if (!command->co_func && cp_oddcomm(s, wlist->wl_next))
            goto out;

        /* If it's not there, try it as a unix command. */
        if (!command->co_comname) {
            if (cp_dounixcom && cp_unixcom(wlist))
                goto out;
            fprintf(cp_err, "%s: no such command available in %s\n",
                    s, cp_program);
            goto out;

            /* If it hasn't been implemented */
        } else if (!command->co_func) {
            fprintf(cp_err, "%s: command is not implemented\n", s);
            goto out;
            /* If it's there but spiceonly, and this is nutmeg, error. */
        } else if (ft_nutmeg && command->co_spiceonly) {
            fprintf(cp_err, "%s: command available only in spice\n", s);
            goto out;
        }

        /* The command was a valid spice/nutmeg command. */
        {
            int nargs = wl_length(wlist->wl_next);
            if (nargs < command->co_minargs) {
                if (command->co_argfn &&
                    cp_getvar("interactive", CP_BOOL, NULL, 0)) {
                    command->co_argfn (wlist->wl_next, command);
                } else {
                    fprintf(cp_err, "%s: too few args.\n", s);
                }
            } else if (nargs > command->co_maxargs) {
                fprintf(cp_err, "%s: too many args.\n", s);
            } else {
                command->co_func (wlist->wl_next);
            }
        }

    out:
        wl_append(ee, wlist);
        wl_append(wlist, nextc);

        if (!ee)
            rwlist = wlist;

        wlist = nextc;
    }

    wl_free(rwlist);

    /* Do periodic sorts of things... */
    cp_periodic();

    cp_ioreset();
}


/* Execute a block.  There can be a number of return values from this routine.
 *  NORMAL indicates a normal termination
 *  BROKEN indicates a break -- if the caller is a breakable loop,
 *      terminate it, otherwise pass the break upwards
 *  CONTINUED indicates a continue -- if the caller is a continuable loop,
 *      continue, else pass the continue upwards
 *  Any other return code is considered a pointer to a string which is
 *      a label somewhere -- if this label is present in the block,
 *      goto it, otherwise pass it up. Note that this prevents jumping
 *      into a loop, which is good.
 *
 * Note that here is where we expand variables, ``, and globs for
 * controls.
 *
 * The 'num' argument is used by break n and continue n.  */
static char *
doblock(struct control *bl, int *num)
{
    struct control *ch, *cn = NULL;
    wordlist *wl, *wltmp;
    char *i, *wlword;
    int nn;

    nn = *num + 1; /*CDHW this is a guess... CDHW*/

    switch (bl->co_type) {
    case CO_WHILE:
        if (!bl->co_children) {
            fprintf(cp_err, "Warning: Executing empty 'while' block.\n"
                    "         (Use a label statement as a no-op "
                    "to suppress this warning.)\n");
        }
        while (bl->co_cond && cp_istrue(bl->co_cond)) {
            if (!bl->co_children) cp_periodic();  /*CDHW*/
            for (ch = bl->co_children; ch; ch = cn) {
                cn = ch->co_next;
                i = doblock(ch, &nn);
                switch (*i) {

                case NORMAL:
                    break;

                case BROKEN:    /* Break. */
                    if (nn < 2) {
                        return (NORMAL_STR);
                    } else {
                        *num = nn - 1;
                        return (BROKEN_STR);
                    }

                case CONTINUED: /* Continue. */
                    if (nn < 2) {
                        cn = NULL;
                        break;
                    } else {
                        *num = nn - 1;
                        return (CONTINUED_STR);
                    }

                default:
                    cn = findlabel(i, bl->co_children);
                    if (!cn)
                        return (i);
                }
            }
        }
        break;

    case CO_DOWHILE:
        do {
            for (ch = bl->co_children; ch; ch = cn) {
                cn = ch->co_next;
                i = doblock(ch, &nn);
                switch (*i) {

                case NORMAL:
                    break;

                case BROKEN:    /* Break. */
                    if (nn < 2) {
                        return (NORMAL_STR);
                    } else {
                        *num = nn - 1;
                        return (BROKEN_STR);
                    }

                case CONTINUED: /* Continue. */
                    if (nn < 2) {
                        cn = NULL;
                        break;
                    } else {
                        *num = nn - 1;
                        return (CONTINUED_STR);
                    }

                default:
                    cn = findlabel(i, bl->co_children);
                    if (!cn)
                        return (i);
                }
            }
        } while (bl->co_cond && cp_istrue(bl->co_cond));
        break;

    case CO_REPEAT:
        if (!bl->co_children) {
            fprintf(cp_err, "Warning: Executing empty 'repeat' block.\n");
            fprintf(cp_err, "         (Use a label statement as a no-op to suppress this warning.)\n");
        }
        if (!bl->co_timestodo) bl->co_timestodo = bl->co_numtimes;
        /*bl->co_numtimes: total repeat count
          bl->co_numtimes = -1: repeat forever
          bl->co_timestodo: remaining repeats*/
        while ((bl->co_timestodo > 0) ||
               (bl->co_timestodo == -1)) {
            if (!bl->co_children) cp_periodic();  /*CDHW*/
            if (bl->co_timestodo != -1) bl->co_timestodo--;
            /* loop through all stements inside rpeat ... end */
            for (ch = bl->co_children; ch; ch = cn) {
                cn = ch->co_next;
                i = doblock(ch, &nn);
                switch (*i) {

                case NORMAL:
                    break;

                case BROKEN:    /* Break. */
                    /* before leaving repeat loop set remaining timestodo to 0 */
                    bl->co_timestodo = 0;
                    if (nn < 2) {
                        return (NORMAL_STR);
                    } else {
                        *num = nn - 1;
                        return (BROKEN_STR);
                    }

                case CONTINUED: /* Continue. */
                    if (nn < 2) {
                        cn = NULL;
                        break;
                    } else {
                        /* before leaving repeat loop set remaining timestodo to 0 */
                        bl->co_timestodo = 0;
                        *num = nn - 1;
                        return (CONTINUED_STR);
                    }

                default:
                    cn = findlabel(i, bl->co_children);

                    if (!cn) {
                        /* no label found inside repeat loop:
                           before leaving loop set remaining timestodo to 0 */
                        bl->co_timestodo = 0;
                        return (i);
                    }
                }
            }
        }
        break;

    case CO_IF:
        if (bl->co_cond && cp_istrue(bl->co_cond)) {
            for (ch = bl->co_children; ch; ch = cn) {
                cn = ch->co_next;
                i = doblock(ch, &nn);
                if (*i > 2) {
                    cn = findlabel(i,
                                   bl->co_children);
                    if (!cn)
                        return (i);
                    else
                        tfree(i);
                } else if (*i != NORMAL) {
                    *num = nn;
                    return (i);
                }
            }
        } else {
            for (ch = bl->co_elseblock; ch; ch = cn) {
                cn = ch->co_next;
                i = doblock(ch, &nn);
                if (*i > 2) {
                    cn = findlabel(i, bl->co_elseblock);
                    if (!cn)
                        return (i);
                } else if (*i != NORMAL) {
                    *num = nn;
                    return (i);
                }
            }
        }
        break;

    case CO_FOREACH:
        wltmp = cp_variablesubst(cp_bquote(cp_doglob(wl_copy(bl->co_text))));
        for (wl = wltmp; wl; wl = wl->wl_next) {
            cp_vset(bl->co_foreachvar, CP_STRING, wl->wl_word);
            for (ch = bl->co_children; ch; ch = cn) {
                cn = ch->co_next;
                i = doblock(ch, &nn);
                switch (*i) {

                case NORMAL:
                    break;

                case BROKEN:    /* Break. */
                    if (nn < 2) {
                        wl_free(wltmp);
                        return (NORMAL_STR);
                    } else {
                        *num = nn - 1;
                        wl_free(wltmp);
                        return (BROKEN_STR);
                    }

                case CONTINUED: /* Continue. */
                    if (nn < 2) {
                        cn = NULL;
                        break;
                    } else {
                        *num = nn - 1;
                        wl_free(wltmp);
                        return (CONTINUED_STR);
                    }

                default:
                    cn = findlabel(i, bl->co_children);
                    if (!cn) {
                        wl_free(wltmp);
                        return (i);
                    }
                }
            }
        }
        wl_free(wltmp);
        break;

    case CO_BREAK:
        if (bl->co_numtimes > 0) {
            *num = bl->co_numtimes;
            return (BROKEN_STR);
        } else {
            fprintf(cp_err, "Warning: break %d a no-op\n",
                    bl->co_numtimes);
            return (NORMAL_STR);
        }

    case CO_CONTINUE:
        if (bl->co_numtimes > 0) {
            *num = bl->co_numtimes;
            return (CONTINUED_STR);
        } else {
            fprintf(cp_err, "Warning: continue %d a no-op\n",
                    bl->co_numtimes);
            return (NORMAL_STR);
        }

    case CO_GOTO:
        wl = cp_variablesubst(cp_bquote(cp_doglob(wl_copy(bl->co_text))));
        wlword = wl->wl_word;
        wl->wl_word = NULL;
        wl_free(wl);
        return (wlword);

    case CO_LABEL:
        /* Do nothing. */
        cp_periodic();  /*CDHW needed to avoid lock-ups when loop contains only a label CDHW*/
        break;

    case CO_STATEMENT:
        docommand(wl_copy(bl->co_text));
        break;

    case CO_UNFILLED:
        /* There was probably an error here... */
        fprintf(cp_err, "Warning: ignoring previous error\n");
        break;

    default:
        fprintf(cp_err,
                "doblock: Internal Error: bad block type %d\n",
                bl->co_type);
        return (NORMAL_STR);
    }
    return (NORMAL_STR);
}


/* Maxiumum number of cheverons used for the alternative prompt */
#define MAX_CHEVRONS    16

/* Get the alternate prompt.
   Number of chevrons indicates stack depth.
   Returns NULL when there is no alternate prompt.
   SJB 28th April 2005 */
char *
get_alt_prompt(void)
{
    int i = 0;
    static char buf[MAX_CHEVRONS + 2];  /* includes terminating space & null */
    struct control *c;

    /* If nothing on the command stack return NULL */
    if (cend[stackp] == NULL)
        return NULL;

    /* measure stack depth */
    for (c = cend[stackp]->co_parent; c; c = c->co_parent)
        i++;

    if (i == 0) {
        return NULL;
    }

    /* Avoid overflow of buffer and
       indicate when we've limited the chevrons by starting with a '+' */
    if (i > MAX_CHEVRONS) {
        i = MAX_CHEVRONS;
        buf[0] = '+';
    } else {
        buf[0] = '>';
    }

    /* return one chevron per command stack depth */
    {
        int j;
        for (j = 1; j < i; j++)
            buf[j] = '>';

        /* Add space and terminate */
        buf[j] = ' ';
        buf[j + 1] = '\0';
    }

    return buf;
} /* end of function get_alt_prompt */



/* Get a command. This does all the bookkeeping things like turning
 * command completion on and off...  */
static wordlist *
getcommand(char *string)
{
    wordlist *wlist;

    if (cp_debug) {
        fprintf(cp_err, "calling getcommand %s\n", string ? string : "");
    }

#if !defined(HAVE_GNUREADLINE) && !defined(HAVE_BSDEDITLINE)
    /* set cp_altprompt for use by the lexer - see parser/lexical.c */
    cp_altprompt = get_alt_prompt();
#endif /* !defined(HAVE_GNUREADLINE) && !defined(HAVE_BSDEDITLINE) */

    cp_cwait = TRUE;
    wlist = cp_parse(string);
    cp_cwait = FALSE;
    if (cp_debug) {
        printf("getcommand ");
        wl_print(wlist, stdout);
        putc('\n', stdout);
    }
    return wlist;
}


/* va: TODO: free control structure(s) before overwriting (memory leakage) */
int
cp_evloop(char *string)
{
    wordlist *wlist, *ww, *freewl;
    struct control *x;
    char *i;

#define newblock                                                \
    do {                                                        \
        cend[stackp]->co_children = TMALLOC(struct control, 1); \
        ZERO(cend[stackp]->co_children, struct control);        \
        cend[stackp]->co_children->co_parent = cend[stackp];    \
        cend[stackp] = cend[stackp]->co_children;               \
        cend[stackp]->co_type = CO_UNFILLED;                    \
    } while(0)

    for (;;) {
        freewl = wlist = getcommand(string);
        if (wlist == NULL) { /* End of file or end of user input. */
            if (cend[stackp] && cend[stackp]->co_parent && !string) {
                cp_resetcontrol(TRUE);
                continue;
            }
            else {
                return (0);
            }
        }
        if ((wlist->wl_word == NULL) || (*wlist->wl_word == '\0')) {
            /* User just typed return. */
            wl_free(wlist); /* va, avoid memory leak */
            if (string) {
                return 1;
            }
            else {
                cp_event--;
                continue;
            }
        }

        /* Just a check... */
        for (ww = wlist; ww; ww = ww->wl_next) {
            if (!ww->wl_word) {
                fprintf(cp_err,
                        "cp_evloop: Internal Error: NULL word pointer\n");
                wl_free(wlist);
                continue;
            }
        }


        /* Add this to the control structure list. If cend->co_type is
         * CO_UNFILLED, the last line was the beginning of a block,
         * and this is the unfilled first statement.
         */
        /* va: TODO: free old structure and its content, before overwriting */
        if (cend[stackp] && (cend[stackp]->co_type != CO_UNFILLED)) {
            cend[stackp]->co_next = TMALLOC(struct control, 1);
            ZERO(cend[stackp]->co_next, struct control);
            cend[stackp]->co_next->co_prev = cend[stackp];
            cend[stackp]->co_next->co_parent = cend[stackp]->co_parent;
            cend[stackp] = cend[stackp]->co_next;
        } else if (!cend[stackp]) {
            control[stackp] = cend[stackp] = TMALLOC(struct control, 1);
            ZERO(cend[stackp], struct control);
        }

        if (eq(wlist->wl_word, "while")) {
            cend[stackp]->co_type = CO_WHILE;
            cend[stackp]->co_cond = wl_copy(wlist->wl_next); /* va, wl_copy */
            if (!cend[stackp]->co_cond) {
                fprintf(stderr,
                        "Error: missing while condition, 'false' will be assumed.\n");
            }
            newblock;
        } else if (eq(wlist->wl_word, "dowhile")) {
            cend[stackp]->co_type = CO_DOWHILE;
            cend[stackp]->co_cond = wl_copy(wlist->wl_next); /* va, wl_copy */
            if (!cend[stackp]->co_cond) {
                /* va: prevent misinterpretation as trigraph sequence with \-sign */
                fprintf(stderr,
                        "Error: missing dowhile condition, '?\?\?' will be assumed.\n");
            }
            newblock;
        } else if (eq(wlist->wl_word, "repeat")) {
            cend[stackp]->co_type = CO_REPEAT;
            if (!wlist->wl_next) {
                cend[stackp]->co_numtimes = -1;
            } else {
                char *s = "1";
                double val;

                struct wordlist *t;  /*CDHW*/
                /*CDHW wlist = cp_variablesubst(cp_bquote(cp_doglob(wl_copy(wlist)))); Wrong order? Leak? CDHW*/
                t = cp_doglob(cp_bquote(cp_variablesubst(wl_copy(wlist)))); /*CDHW leak from cp_doglob? */

                if (!t->wl_next) {
                    fprintf(cp_err, "Error: Undefined number after command 'repeat', assume 1\n");
                }
                else
                    s = t->wl_next->wl_word;

                if (ft_numparse(&s, FALSE, &val) > 0) {
                    /* Can be converted to int */
                    if (val < 0) {
                        fprintf(cp_err,
                                "Error: can't repeat a negative number of times\n");
                        val = 0.0;
                    }
                    cend[stackp]->co_numtimes = (int) val;
                }
                else {
                    fprintf(cp_err,
                            "Error: bad repeat argument %s\n",
                            t->wl_next->wl_word); /* CDHW */
                }
                wl_free(t);
                t = NULL;  /* CDHW */
            }
            newblock;

        } else if (eq(wlist->wl_word, "if")) {
            cend[stackp]->co_type = CO_IF;
            cend[stackp]->co_cond = wl_copy(wlist->wl_next); /* va, wl_copy */
            if (!cend[stackp]->co_cond) {
                fprintf(stderr,
                        "Error: missing if condition.\n");
            }
            newblock;

        } else if (eq(wlist->wl_word, "foreach")) {
            cend[stackp]->co_type = CO_FOREACH;
            if (wlist->wl_next) {
                wlist = wlist->wl_next;
                cend[stackp]->co_foreachvar =
                    copy(wlist->wl_word);
                wlist = wlist->wl_next;
            }
            else {
                fprintf(stderr,
                        "Error: missing foreach variable.\n");
                wl_free(wlist);
                continue;
            }
            wlist = cp_doglob(wlist);
            cend[stackp]->co_text = wl_copy(wlist);
            newblock;
        } else if (eq(wlist->wl_word, "label")) {
            cend[stackp]->co_type = CO_LABEL;
            if (wlist->wl_next) {
                cend[stackp]->co_text = wl_copy(wlist->wl_next);
                /* I think of everything, don't I? */
                cp_addkword(CT_LABEL, wlist->wl_next->wl_word);
                if (wlist->wl_next->wl_next)
                    fprintf(cp_err,
                            "Warning: ignored extra junk after label.\n");
            } else {
                fprintf(stderr, "Error: missing label.\n");
            }

        } else if (eq(wlist->wl_word, "goto")) {
            /* Incidentally, this won't work if the values 1 and 2 ever get
             * to be valid character pointers -- I think it's reasonably
             * safe to assume they aren't...  */
            cend[stackp]->co_type = CO_GOTO;
            if (wlist->wl_next) {
                cend[stackp]->co_text = wl_copy(wlist->wl_next);
                if (wlist->wl_next->wl_next)
                    fprintf(cp_err,
                            "Warning: ignored extra junk after goto.\n");
            } else {
                fprintf(stderr, "Error: missing label.\n");
            }
        } else if (eq(wlist->wl_word, "continue")) {
            cend[stackp]->co_type = CO_CONTINUE;
            if (wlist->wl_next) {
                cend[stackp]->co_numtimes = scannum(wlist->wl_next->wl_word);
                if (wlist->wl_next->wl_next)
                    fprintf(cp_err,
                            "Warning: ignored extra junk after continue %d.\n",
                            cend[stackp]->co_numtimes);
            } else {
                cend[stackp]->co_numtimes = 1;
            }
        } else if (eq(wlist->wl_word, "break")) {
            cend[stackp]->co_type = CO_BREAK;
            if (wlist->wl_next) {
                cend[stackp]->co_numtimes = scannum(wlist->wl_next->wl_word);
                if (wlist->wl_next->wl_next)
                    fprintf(cp_err,
                            "Warning: ignored extra junk after break %d.\n",
                            cend[stackp]->co_numtimes);
            } else {
                cend[stackp]->co_numtimes = 1;
            }
        } else if (eq(wlist->wl_word, "end")) {
            /* Throw away this thing if not in a block. */
            if (!cend[stackp]->co_parent) {
                fprintf(stderr, "Error: no block to end.\n");
                cend[stackp]->co_type = CO_UNFILLED;
            } else if (cend[stackp]->co_prev) {
                cend[stackp]->co_prev->co_next = NULL;
                x = cend[stackp];
                cend[stackp] = cend[stackp]->co_parent;
                tfree(x);
                x = NULL;
            } else {
                x = cend[stackp];
                cend[stackp] = cend[stackp]->co_parent;
                cend[stackp]->co_children = NULL;
                tfree(x);
                x = NULL;
            }
        } else if (eq(wlist->wl_word, "else")) {
            if (!cend[stackp]->co_parent ||
                    (cend[stackp]->co_parent->co_type != CO_IF)) {
                fprintf(stderr, "Error: misplaced else.\n");
                cend[stackp]->co_type = CO_UNFILLED;
            } else {
                if (cend[stackp]->co_prev)
                    cend[stackp]->co_prev->co_next = NULL;
                else
                    cend[stackp]->co_parent->co_children = NULL;
                cend[stackp]->co_parent->co_elseblock = cend[stackp];
                cend[stackp]->co_prev = NULL;
            }
        } else {
            cend[stackp]->co_type = CO_STATEMENT;
            cend[stackp]->co_text = wl_copy(wlist);
        }

        if (!cend[stackp]->co_parent) {
            x = cend[stackp];
            /* We have to toss this do-while loop in here so
             * that gotos at the top level will work.
             */
            do {
                int nn = 0; /* CDHW */
                i = doblock(x, &nn);
                switch (*i) {
                case NORMAL:
                    break;
                case BROKEN:
                    fprintf(cp_err,
                            "Error: break not in loop or too many break levels given\n");
                    break;
                case CONTINUED:
                    fprintf(cp_err,
                            "Error: continue not in loop or too many continue levels given\n");
                    break;
                default:
                    x = findlabel(i, control[stackp]);
                    if (!x)
                        fprintf(cp_err, "Error: label %s not found\n", i);
                    tfree(i);
                }
                if (x)
                    x = x->co_next;
            } while (x);
        }
        wl_free(freewl);
        if (string)
            return (1); /* The return value is irrelevant. */
    } /* end of unconditional loop */
} /* end of function cp_evloop */


/* This blows away the control structures... */
void cp_resetcontrol(bool warn)
{
    if (warn) {
        fprintf(cp_err, "Warning: clearing control structures\n");
        if (cend[stackp] && cend[stackp]->co_parent)
            fprintf(cp_err, "Warning: EOF before block terminated\n");
    }
    /* free the control structures */
    cp_free_control();
    control[0] = cend[0] = NULL;
    stackp = 0;
    cp_kwswitch(CT_LABEL, NULL);
}


/* Push or pop a new control structure set... */
void
cp_popcontrol(void)
{
    if (cp_debug)
        fprintf(cp_err, "pop: stackp: %d -> %d\n", stackp, stackp - 1);
    if (stackp < 1) {
        fprintf(cp_err, "cp_popcontrol: Internal Error: stack empty\n");
    } else {
        /* va: free unused control structure */
        ctl_free(control[stackp]);
        stackp--;
    }
}


void
cp_pushcontrol(void)
{
    if (cp_debug)
        fprintf(cp_err, "push: stackp: %d -> %d\n", stackp, stackp + 1);
    if (stackp > CONTROLSTACKSIZE - 2) {
        fprintf(cp_err, "Error: stack overflow -- max depth = %d\n",
                CONTROLSTACKSIZE);
        stackp = 0;
    } else {
        stackp++;
        control[stackp] = cend[stackp] = NULL;
    }
}


/* And this returns to the top level (for use in the interrupt handlers). */
void
cp_toplevel(void)
{
    stackp = 0;
    if (cend[stackp])
        while (cend[stackp]->co_parent)
            cend[stackp] = cend[stackp]->co_parent;
}


/* va: This totally frees the control structures */
static void
cp_free_control(void)
{
    int i;

    /* Free the control structures */
    for (i = stackp; i >= 0; i--) {
        ctl_free(control[i]);
    }

    control[0] = cend[0] = NULL;
    stackp = 0;
}
