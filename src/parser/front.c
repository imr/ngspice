/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * The front-end command loop.
 */

#include "ngspice.h"
#include "cpdefs.h"
#include "front.h"

/* Return values from doblock().  I am assuming that nobody will use
 * these characters in a string.  */
#define NORMAL      '\001'
#define BROKEN      '\002'
#define CONTINUED   '\003'
#define NORMAL_STR  "\001"
#define BROKEN_STR  "\002"
#define CONTINUED_STR   "\003"

/* Are we waiting for a command? This lets signal handling be
 * more clever. */

bool cp_cwait = FALSE;      
char *cp_csep = ";";

bool cp_dounixcom = FALSE;



enum co_command {
    CO_UNFILLED,
    CO_STATEMENT,
    CO_WHILE,
    CO_DOWHILE,
    CO_IF,
    CO_FOREACH,
    CO_BREAK,
    CO_CONTINUE,
    CO_LABEL,
    CO_GOTO,
    CO_REPEAT
};

/* We have to keep the control structures in a stack, so that when we do
 * a 'source', we can push a fresh set onto the top...  Actually there have
 * to be two stacks -- one for the pointer to the list of control structs,
 * and one for the 'current command' pointer...
 */

#define CONTROLSTACKSIZE 256    /* Better be enough. */
static struct control *control[CONTROLSTACKSIZE], *cend[CONTROLSTACKSIZE];


/* static declarations */
static char * doblock(struct control *bl, int *num);
static struct control * findlabel(char *s, struct control *ct);
static void docommand(register wordlist *wlist);
static wordlist * getcommand(char *string);
static void pwlist(wordlist *wlist, char *name);
static void dodump(struct control *cc);


static int stackp = 0;

/* If there is an argument, give this to cshpar to use instead of
 * stdin. In a few places, we call cp_evloop again if it returns 1 and
 * exit (or close a file) if it returns 0... Because of the way
 * sources are done, we can't allow the control structures to get
 * blown away every time we return -- probably every time we type
 * source at the keyboard and every time a source returns to keyboard
 * input is ok though -- use ft_controlreset.  */

static char *noredirect[] = { "stop", NULL } ;  /* Only one?? */





int
cp_evloop(char *string)
{
    wordlist *wlist, *ww;
    struct control *x;
    char *i;
    int nn;

#define newblock    cend[stackp]->co_children = alloc(struct control);      \
	    ZERO(cend[stackp]->co_children,struct control), \
            cend[stackp]->co_children->co_parent = cend[stackp]; \
            cend[stackp] = cend[stackp]->co_children;        \
            cend[stackp]->co_type = CO_UNFILLED;

    for (;;) {
	wlist = getcommand(string);
	if (wlist == NULL) {    /* End of file or end of user input. */
	    if (cend[stackp]->co_parent && !string) {
		cp_resetcontrol();
		continue;
	    } else
		return (0);
	}
	if ((wlist->wl_word == NULL) || (*wlist->wl_word == '\0')) {
	    /* User just typed return. */
	    if (string)
		return (1);
	    else {
		cp_event--;
		continue;
	    }
	}

	/* Just a check... */
	for (ww = wlist; ww; ww = ww->wl_next)
	    if (!ww->wl_word) {
		fprintf(cp_err, 
			"cp_evloop: Internal Error: NULL word pointer\n");
		continue;
	    }


	/* Add this to the control structure list. If cend->co_type is
	 * CO_UNFILLED, the last line was the beginning of a block,
	 * and this is the unfilled first statement.  */
	if (cend[stackp] && (cend[stackp]->co_type != CO_UNFILLED)) {
	    cend[stackp]->co_next = alloc(struct control);
	    ZERO(cend[stackp]->co_next, struct control);
	    cend[stackp]->co_next->co_prev = cend[stackp];
	    cend[stackp]->co_next->co_parent = 
		cend[stackp]->co_parent;
	    cend[stackp] = cend[stackp]->co_next;
	} else if (!cend[stackp]) {
	    control[stackp] = cend[stackp] = alloc(struct control);
	    ZERO(cend[stackp], struct control);
	}

	if (eq(wlist->wl_word, "while")) {
	    cend[stackp]->co_type = CO_WHILE;
	    cend[stackp]->co_cond = wlist->wl_next;
	    if (!cend[stackp]->co_cond) {
		fprintf(stderr,
			"Error: missing while condition.\n");
	    }
	    newblock;
	} else if (eq(wlist->wl_word, "dowhile")) {
	    cend[stackp]->co_type = CO_DOWHILE;
	    cend[stackp]->co_cond = wlist->wl_next;
	    if (!cend[stackp]->co_cond) {
		fprintf(stderr,
			"Error: missing dowhile condition.\n");
	    }
	    newblock;
	} else if (eq(wlist->wl_word, "repeat")) {
	    cend[stackp]->co_type = CO_REPEAT;
	    if (!wlist->wl_next) {
		cend[stackp]->co_numtimes = -1;
	    } else {
		char *s;
		double *dd;
		wlist = cp_variablesubst(cp_bquote(
		    cp_doglob(wl_copy(wlist))));
		s = wlist->wl_next->wl_word;

		dd = ft_numparse(&s, FALSE);
		if (dd) {
		    if (*dd < 0) {
			fprintf(cp_err, 
				"Error: can't repeat a negative number of times\n");
			*dd = 0.0;
		    }
		    cend[stackp]->co_numtimes = (int) *dd;
		} else
		    fprintf(cp_err, 
			    "Error: bad repeat argument %s\n",
			    wlist->wl_next->wl_word);
	    }
	    newblock;
	} else if (eq(wlist->wl_word, "if")) {
	    cend[stackp]->co_type = CO_IF;
	    cend[stackp]->co_cond = wlist->wl_next;
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
	    } else
		fprintf(stderr, 
			"Error: missing foreach variable.\n");
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
	    } else
		fprintf(stderr, "Error: missing label.\n");
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
	    } else
		fprintf(stderr, "Error: missing label.\n");
	} else if (eq(wlist->wl_word, "continue")) {
	    cend[stackp]->co_type = CO_CONTINUE;
	    if (wlist->wl_next) {
		cend[stackp]->co_numtimes = scannum(wlist->
						    wl_next->wl_word);
		if (wlist->wl_next->wl_next)
		    fprintf(cp_err, 
			    "Warning: ignored extra junk after continue %d.\n",
			    cend[stackp]->co_numtimes);
	    } else
		cend[stackp]->co_numtimes = 1;
	} else if (eq(wlist->wl_word, "break")) {
	    cend[stackp]->co_type = CO_BREAK;
	    if (wlist->wl_next) {
		cend[stackp]->co_numtimes = scannum(wlist->
						    wl_next->wl_word);
		if (wlist->wl_next->wl_next)
		    fprintf(cp_err, 
			    "Warning: ignored extra junk after break %d.\n",
			    cend[stackp]->co_numtimes);
	    } else
		cend[stackp]->co_numtimes = 1;
	} else if (eq(wlist->wl_word, "end")) {
	    /* Throw away this thing. */
	    if (!cend[stackp]->co_parent) {
		fprintf(stderr, "Error: no block to end.\n");
		cend[stackp]->co_type = CO_UNFILLED;
	    } else if (cend[stackp]->co_prev) {
		cend[stackp]->co_prev->co_next = NULL;
		x = cend[stackp];
		cend[stackp] = cend[stackp]->co_parent;
		tfree(x);
	    } else {
		x = cend[stackp];
		cend[stackp] = cend[stackp]->co_parent;
		cend[stackp]->co_children = NULL;
		tfree(x);
	    }
	} else if (eq(wlist->wl_word, "else")) {
	    if (!cend[stackp]->co_parent ||
		(cend[stackp]->co_parent->co_type !=
		 CO_IF)) {
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
	    cend[stackp]->co_text = wlist;
	}
	if (!cend[stackp]->co_parent) {
	    x = cend[stackp];
	    /* We have to toss this do-while loop in here so
	     * that gotos at the top level will work.
	     */
	    do {
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
		}
		if (x)
		    x = x->co_next;
	    } while (x);
	}
	if (string)
	    return (1); /* The return value is irrelevant. */
    }
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
 * Note that here is where we expand variables, ``, and globs for controls.
 * The 'num' argument is used by break n and continue n.
 */

static char *
doblock(struct control *bl, int *num)
{
    struct control *ch, *cn = NULL;
    wordlist *wl;
    char *i;
    int nn;

    switch (bl->co_type) {
    case CO_WHILE:
        while (bl->co_cond && cp_isTRUE(bl->co_cond)) {
            for (ch = bl->co_children; ch; ch = cn) {
                cn = ch->co_next;
                i = doblock(ch, &nn);
                switch (*i) {

		case NORMAL:
                    break;

		case BROKEN:    /* Break. */
                    if (nn < 2)
                        return (NORMAL_STR);
                    else {
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
                    if (nn < 2)
                        return (NORMAL_STR);
                    else {
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
        } while (bl->co_cond && cp_isTRUE(bl->co_cond));
        break;

    case CO_REPEAT:
        while ((bl->co_numtimes > 0) ||
	       (bl->co_numtimes == -1)) {
            if (bl->co_numtimes != -1)
                bl->co_numtimes--;
            for (ch = bl->co_children; ch; ch = cn) {
                cn = ch->co_next;
                i = doblock(ch, &nn);
                switch (*i) {

		case NORMAL:
                    break;

		case BROKEN:    /* Break. */
                    if (nn < 2)
                        return (NORMAL_STR);
                    else {
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

    case CO_IF:
        if (bl->co_cond && cp_isTRUE(bl->co_cond)) {
            for (ch = bl->co_children; ch; ch = cn) {
                cn = ch->co_next;
                i = doblock(ch, &nn);
                if (*i > 2) {
                    cn = findlabel(i,
				   bl->co_children);
                    if (!cn)
                        return (i);
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
                    cn = findlabel(i,
				   bl->co_elseblock);
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
        for (wl = cp_variablesubst(cp_bquote(cp_doglob(wl_copy(bl->co_text))));
	     wl;
	     wl = wl->wl_next) {
            cp_vset(bl->co_foreachvar, VT_STRING, wl->wl_word);
            for (ch = bl->co_children; ch; ch = cn) {
                cn = ch->co_next;
                i = doblock(ch, &nn);
                switch (*i) {

		case NORMAL:
                    break;

		case BROKEN:    /* Break. */
                    if (nn < 2)
                        return (NORMAL_STR);
                    else {
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
        wl = cp_variablesubst(cp_bquote(cp_doglob(
	    wl_copy(bl->co_text))));
        return (wl->wl_word);

    case CO_LABEL:
	/* Do nothing. */
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


static struct control *
findlabel(char *s, struct control *ct)
{
    while (ct) {
        if ((ct->co_type == CO_LABEL) && eq(s, ct->co_text->wl_word))
            break;
        ct = ct->co_next;
    }
    return (ct);
}


/* This blows away the control structures... */
void
cp_resetcontrol(void)
{
    if (cend[stackp] && cend[stackp]->co_parent)
        fprintf(cp_err, "Warning: EOF before block terminated\n");
    /* We probably should free the control structures... */
    control[0] = cend[0] = NULL;
    stackp = 0;
    cp_kwswitch(CT_LABEL, (char *) NULL);
    return;
}


/* Push or pop a new control structure set... */
void
cp_popcontrol(void)
{
    if (cp_debug)
        fprintf(cp_err, "pop: stackp: %d -> %d\n", stackp, stackp - 1);
    if (stackp < 1)
        fprintf(cp_err, "cp_popcontrol: Internal Error: stack empty\n");
    else
        stackp--;
    return;
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
    return;
}


/* And this returns to the top level (for use in the interrupt handlers). */
void
cp_toplevel(void)
{
    stackp = 0;
    if (cend[stackp])
        while (cend[stackp]->co_parent)
            cend[stackp] = cend[stackp]->co_parent;
    return;
}


/* Note that we only do io redirection when we get to here - we also
 * postpone some other things until now.  */
static void
docommand(register wordlist *wlist)
{
    register char *r, *s, *t;
    char *lcom;
    int nargs;
    register int i;
    struct comm *command;
    wordlist *wl, *nextc, *ee, *rwlist;

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

    wlist = cp_doglob(wlist);
    pwlist(wlist, "After globbing");

    if (!wlist || !wlist->wl_word)
        return;

    /* Now loop through all of the commands given. */
    rwlist = wlist;
    do {
        for (nextc = wlist; nextc; nextc = nextc->wl_next)
            if (eq(nextc->wl_word, cp_csep))
                break;

        /* Temporarily hide the rest of the command... */
        if (nextc && nextc->wl_prev)
            nextc->wl_prev->wl_next = NULL;
        ee = wlist->wl_prev;
        if (ee)
            wlist->wl_prev = NULL;

        if (nextc == wlist) {
            /* There was no text... */
            goto out;
        }

        /* And do the redirection. */
        cp_ioreset();
        for (i = 0; noredirect[i]; i++)
            if (eq(wlist->wl_word, noredirect[i]))
                break;
        if (!noredirect[i]) {
            if (!(wlist = cp_redirect(wlist))) {
                cp_ioreset();
                return;
            }
        }

        /* Get rid of all the 8th bits now... */
        cp_striplist(wlist);

        s = wlist->wl_word;

        /* Look for the command in the command list. */
        for (i = 0; cp_coms[i].co_comname; i++) {
            /* strcmp(cp_coms[i].co_comname, s) ... */
            for (t = cp_coms[i].co_comname, r = s; *t && *r;
		 t++, r++)
                if (*t != *r)
                    break;
            if (!*t && !*r)
                break; 
        }
        
        /* Now give the user-supplied command routine a try... */
        if (!cp_coms[i].co_func && cp_oddcomm(s, wlist->wl_next))
            goto out;

        /* If it's not there, try it as a unix command. */
        if (!cp_coms[i].co_comname) {
            if (cp_dounixcom && cp_unixcom(wlist))
                goto out;
            fprintf(cp_err,"%s: no such command available in %s\n",
		    s, cp_program);
            goto out;

	    /* If it's there but spiceonly, and this is nutmeg, error. */
        } else if (!cp_coms[i].co_func && ft_nutmeg && 
		   (cp_coms[i].co_spiceonly)) {
            fprintf(cp_err,"%s: command available only in spice\n",
                    s);
            goto out;
        }

        /* The command was a valid spice/nutmeg command. */
        command = &cp_coms[i];
        nargs = 0;
        for (wl = wlist->wl_next; wl; wl = wl->wl_next)
            nargs++;
        if (command->co_stringargs) {
            lcom = wl_flatten(wlist->wl_next);
            (*command->co_func) (lcom);
        } else {
            if (nargs < command->co_minargs) {
		if (command->co_argfn) {
		    (*command->co_argfn) (wlist->wl_next, command);
		} else {
		    fprintf(cp_err, "%s: too few args.\n", s);
		}
            } else if (nargs > command->co_maxargs) {
                fprintf(cp_err, "%s: too many args.\n", s);
            } else
                (*command->co_func) (wlist->wl_next);
        }

        /* Now fix the pointers and advance wlist. */
    out:        wlist->wl_prev = ee;
        if (nextc) {
            if (nextc->wl_prev)
                nextc->wl_prev->wl_next = nextc;
            wlist = nextc->wl_next;
        }
    } while (nextc && wlist);

    wl_free(rwlist);

    /* Do periodic sorts of things... */
    cp_periodic();

    cp_ioreset();
    return;
}


/* Get a command. This does all the bookkeeping things like turning
 * command completion on and off...  */
static wordlist *
getcommand(char *string)
{
    wordlist *wlist;
    int i = 0, j;
    static char buf[64];
    struct control *c;

    if (cp_debug)
        fprintf(cp_err, "calling getcommand %s\n", 
                string ? string : "");
    if (cend[stackp]) {
        for (c = cend[stackp]->co_parent; c; c = c->co_parent)
            i++;
        if (i) {
            for (j = 0; j < i; j++)
                buf[j] = '>';
            buf[j] = ' ';
            buf[j + 1] = '\0';
            cp_altprompt = buf;
        } else
            cp_altprompt = NULL;
    } else
        cp_altprompt = NULL;

    cp_cwait = TRUE;
    wlist = cp_parse(string);
    cp_cwait = FALSE;
    if (cp_debug) {
        printf("getcommand ");
        wl_print(wlist, stdout);
        putc('\n', stdout);
    }
    return (wlist);
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
    return;
}

static int indent;


void
com_cdump(wordlist *wl)
{
    struct control *c;

    indent = 0;
    for (c = control[stackp]; c; c = c->co_next)
        dodump(c);
    return;
}

#define tab(num)    for (i = 0; i < num; i++) putc(' ', cp_out);

static void
dodump(struct control *cc)
{
    int i;
    struct control *tc;

    switch (cc->co_type) {
    case CO_UNFILLED:
	tab(indent);
	fprintf(cp_out, "(unfilled)\n");
	break;
    case CO_STATEMENT:
	tab(indent);
	wl_print(cc->co_text, cp_out);
	putc('\n', cp_out);
	break;
    case CO_WHILE:
	tab(indent);
	fprintf(cp_out, "while ");
	wl_print(cc->co_cond, cp_out);
	putc('\n', cp_out);
	indent += 8;
	for (tc = cc->co_children; tc; tc = tc->co_next)
	    dodump(tc);
	indent -= 8;
	tab(indent);
	fprintf(cp_out, "end\n");
	break;
    case CO_REPEAT:
	tab(indent);
	fprintf(cp_out, "repeat ");
	if (cc->co_numtimes != -1)
	    fprintf(cp_out, "%d\n", cc->co_numtimes);
	else
	    putc('\n', cp_out);
	indent += 8;
	for (tc = cc->co_children; tc; tc = tc->co_next)
	    dodump(tc);
	indent -= 8;
	tab(indent);
	fprintf(cp_out, "end\n");
	break;
    case CO_DOWHILE:
	tab(indent);
	fprintf(cp_out, "dowhile ");
	wl_print(cc->co_cond, cp_out);
	putc('\n', cp_out);
	indent += 8;
	for (tc = cc->co_children; tc; tc = tc->co_next)
	    dodump(tc);
	indent -= 8;
	tab(indent);
	fprintf(cp_out, "end\n");
	break;
    case CO_IF:
	tab(indent);
	fprintf(cp_out, "if ");
	wl_print(cc->co_cond, cp_out);
	putc('\n', cp_out);
	indent += 8;
	for (tc = cc->co_children; tc; tc = tc->co_next)
	    dodump(tc);
	indent -= 8;
	tab(indent);
	fprintf(cp_out, "end\n");
	break;
    case CO_FOREACH:
	tab(indent);
	fprintf(cp_out, "foreach %s ", cc->co_foreachvar);
	wl_print(cc->co_text, cp_out);
	putc('\n', cp_out);
	indent += 8;
	for (tc = cc->co_children; tc; tc = tc->co_next)
	    dodump(tc);
	indent -= 8;
	tab(indent);
	fprintf(cp_out, "end\n");
	break;
    case CO_BREAK:
	tab(indent);
	if (cc->co_numtimes != 1)
	    fprintf(cp_out, "break %d\n", cc->co_numtimes);
	else
	    fprintf(cp_out, "break\n");
	break;
    case CO_CONTINUE:
	tab(indent);
	if (cc->co_numtimes != 1)
	    fprintf(cp_out, "continue %d\n",
		    cc->co_numtimes);
	else
	    fprintf(cp_out, "continue\n");
	break;
    case CO_LABEL:
	tab(indent);
	fprintf(cp_out, "label %s\n", cc->co_text->wl_word);
	break;
    case CO_GOTO:
	tab(indent);
	fprintf(cp_out, "goto %s\n", cc->co_text->wl_word);
	break;
    default:
	tab(indent);
	fprintf(cp_out, "bad type %d\n", cc->co_type);
	break;
    }
    return;
}

