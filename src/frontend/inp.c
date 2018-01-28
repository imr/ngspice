/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher
**********/

/*
 * Stuff for dealing with spice input decks and command scripts, and
 * the listing routines.
 */

#include "ngspice/ngspice.h"

#include "ngspice/cktdefs.h"
#include "ngspice/cpdefs.h"
#include "ngspice/inpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteinp.h"
#include "inp.h"

#include "runcoms.h"
#include "inpcom.h"
#include "circuits.h"
#include "completion.h"
#include "variable.h"
#include "breakp2.h"
#include "dotcards.h"
#include "../misc/util.h" /* ngdirname() */
#include "../misc/mktemp.h"
#include "../misc/misc_time.h"
#include "subckt.h"
#include "spiceif.h"
#include "com_let.h"
#include "com_commands.h"

#ifdef XSPICE
#include "ngspice/ipctiein.h"
#include "ngspice/enh.h"
#endif

#include "numparam/numpaif.h"
#include "ngspice/stringskip.h"
#include "ngspice/randnumb.h"


#define line_free(line, flag)                   \
    do {                                        \
        line_free_x(line, flag);                \
        line = NULL;                            \
    } while(0)

static char *upper(register char *string);
static bool doedit(char *filename);
static struct card *com_options = NULL;
static struct card *mc_deck = NULL;
static void cktislinear(CKTcircuit *ckt, struct card *deck);
static void dotifeval(struct card *deck);

static wordlist *inp_savecurrents(struct card *deck, struct card *options, wordlist *wl, wordlist *controls);

void line_free_x(struct card *deck, bool recurse);
void create_circbyline(char *line);

extern bool ft_batchmode;

/* structure used to save expression parse trees for .model and
 * device instance lines
 */

struct pt_temper {
    char *expression;
    wordlist *wl;
    wordlist *wlend;
    INPparseTree *pt;
    struct pt_temper *next;
};

static int inp_parse_temper(struct card *deck,
                            struct pt_temper **motdlist_p,
                            struct pt_temper **devtlist_p);
static void inp_parse_temper_trees(struct circ *ckt);


/*
 * create an unique artificial *unusable* FILE ptr
 *   meant to be used with Xprintf() only to eventually
 *   redirect output to the `out_vprintf()' family
 */

static FILE *cp_more;
static FILE *cp_more = (FILE*) &cp_more;

static void
Xprintf(FILE *fdst, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);

    if (fdst == cp_more)
        out_vprintf(fmt, ap);
    else
        vfprintf(fdst, fmt, ap);

    va_end(ap);
}


/* Do a listing. Use is listing [expanded] [logical] [physical] [deck] */
void
com_listing(wordlist *wl)
{
    int type = LS_LOGICAL;
    bool expand = FALSE, do_param_listing = FALSE;
    char *s;

    if (ft_curckt) {  /* if there is a current circuit . . . .  */
        while (wl) {
            s = wl->wl_word;
            if (strcmp(s, "param") == 0) {
                do_param_listing = TRUE;
            } else {
                switch (*s) {
                case 'l':
                case 'L':
                    type = LS_LOGICAL;
                    break;
                case 'p':
                case 'P':
                    type = LS_PHYSICAL;
                    break;
                case 'd':
                case 'D':
                    type = LS_DECK;
                    break;
                case 'e':
                case 'E':
                    expand = TRUE;
                    break;
                default:
                    fprintf(cp_err, "Error: bad listing type %s\n", s);
                    return; /* SJB - don't go on after an error */
                }
            }
            wl = wl->wl_next;
        }

        if (do_param_listing) {
            nupa_list_params(cp_out);
        } else {
            if (type != LS_DECK)
                fprintf(cp_out, "\t%s\n\n", ft_curckt->ci_name);
            inp_list(cp_out,
                     expand ? ft_curckt->ci_deck : ft_curckt->ci_origdeck,
                     ft_curckt->ci_options, type);
        }
    } else {
        fprintf(cp_err, "Error: no circuit loaded.\n");
    }
}


/* returns inp_casefix() or NULL */
static char *
upper(char *string)
{
    static char buf[BSIZE_SP];

    if (string) {
        strncpy(buf, string, BSIZE_SP - 1);
        buf[BSIZE_SP - 1] = '\0';
        inp_casefix(buf);
    } else {
        strcpy(buf, "<null>");
    }
    return buf;
}


/* Provide an input listing on the specified file of the given card
 * deck.  The listing should be of either LS_PHYSICAL or LS_LOGICAL or
 * LS_DECK lines as specified by the type parameter.  */
void
inp_list(FILE *file, struct card *deck, struct card *extras, int type)
{
    struct card *here;
    struct card *there;
    bool renumber;
    bool useout = (file == cp_out);
    int i = 1;

    /* gtri - wbk - 03/07/91 - Don't use 'more' type output if ipc enabled */
#ifdef XSPICE
    if (g_ipc.enabled)
        useout = FALSE;
#endif
    /* gtri - end - 03/07/91 */

    if (useout) {
        out_init();
        file = cp_more;
    }

    renumber = cp_getvar("renumber", CP_BOOL, NULL);

    if (type == LS_LOGICAL) {
    top1:
        for (here = deck; here; here = here->nextcard) {
            if (renumber)
                here->linenum = i;
            if (ciprefix(".end", here->line) && !isalpha_c(here->line[4]))
                continue;
            if (*here->line != '*') {
                Xprintf(file, "%6d : %s\n", here->linenum, upper(here->line));
                if (here->error)
                    Xprintf(file, "%s\n", here->error);
            }
            i++;
        }

        if (extras) {
            deck = extras;
            extras = NULL;
            goto top1;
        }

        Xprintf(file, "%6d : .end\n", i);

    } else if ((type == LS_PHYSICAL) || (type == LS_DECK)) {

    top2:
        for (here = deck; here; here = here->nextcard) {
            if ((here->actualLine == NULL) || (here == deck)) {
                if (renumber)
                    here->linenum = i;
                if (ciprefix(".end", here->line) && !isalpha_c(here->line[4]))
                    continue;
                if (type == LS_PHYSICAL)
                    Xprintf(file, "%6d : %s\n",
                            here->linenum, upper(here->line));
                else
                    Xprintf(file, "%s\n", upper(here->line));
                if (here->error && (type == LS_PHYSICAL))
                    Xprintf(file, "%s\n", here->error);
            } else {
                for (there = here->actualLine; there; there = there->nextcard) {
                    there->linenum = i++;
                    if (ciprefix(".end", here->line) && isalpha_c(here->line[4]))
                        continue;
                    if (type == LS_PHYSICAL)
                        Xprintf(file, "%6d : %s\n",
                                there->linenum, upper(there->line));
                    else
                        Xprintf(file, "%s\n", upper(there->line));
                    if (there->error && (type == LS_PHYSICAL))
                        Xprintf(file, "%s\n", there->error);
                }
                here->linenum = i;
            }
            i++;
        }
        if (extras) {
            deck = extras;
            extras = NULL;
            goto top2;
        }
        if (type == LS_PHYSICAL)
            Xprintf(file, "%6d : .end\n", i);
        else
            Xprintf(file, ".end\n");
    } else {
        fprintf(cp_err, "inp_list: Internal Error: bad type %d\n", type);
    }
}


/*
 * Free memory used by a line.
 * If recurse is TRUE then recursively free all lines linked via the ->nextcard field.
 * If recurse is FALSE free only this line.
 * All lines linked via the ->actualLine field are always recursivly freed.
 * SJB - 22nd May 2001
 */
void
line_free_x(struct card *deck, bool recurse)
{
    while (deck) {
        struct card *next_deck = deck->nextcard;
        line_free_x(deck->actualLine, TRUE);
        tfree(deck->line);
        tfree(deck->error);
        tfree(deck);
        if (!recurse)
            return;
        deck = next_deck;
    }
}


/* concatenate two lists, destructively altering the first one */
struct card *
line_nconc(struct card *head, struct card *rest)
{
    struct card *p = head;
    if (!rest)
        return head;
    if (!head)
        return rest;
    while (p->nextcard)
        p = p->nextcard;
    p->nextcard = rest;
    return head;
}


/* reverse the linked list struct card */
struct card *
line_reverse(struct card *head)
{
    struct card *prev = NULL;
    struct card *next;

    while (head) {
        next = head->nextcard;
        head->nextcard = prev;
        prev = head;
        head = next;
    }

    return prev;
}


/* free mc_deck */
void
mc_free(void)
{
    line_free_x(mc_deck, TRUE);
}


/* The routine to source a spice input deck. We read the deck in, take
 * out the front-end commands, and create a CKT structure. Also we
 * filter out the following cards: .save, .width, .four, .print, and
 * .plot, to perform after the run is over.
 * Then, we run dodeck, which parses up the deck.             */
void
inp_spsource(FILE *fp, bool comfile, char *filename, bool intfile)
/* arguments:
 *  *fp = pointer to the input file
 *  comfile = whether it is a command file.  Values are TRUE/FALSE
 *  *filename = name of input file
 *  intfile = whether input is from internal array.  Values are TRUE/FALSE
 */
{
    struct card *deck = NULL, *dd, *ld, *prev_param = NULL, *prev_card = NULL;
    struct card *realdeck = NULL, *options = NULL, *curr_meas = NULL;
    char *tt = NULL, name[BSIZE_SP], *s, *t, *temperature = NULL;
    double testemp = 0.0;
    bool commands = FALSE;
    wordlist *wl = NULL, *end = NULL, *wl_first = NULL;
    wordlist *controls = NULL, *pre_controls = NULL;
    FILE *lastin, *lastout, *lasterr;
    double temperature_value;
    bool expr_w_temper = FALSE;

    double startTime, endTime;

#ifdef HAS_PROGREP
    if (!comfile)
        SetAnalyse("Source Deck", 0);
#endif

    /* read in the deck from a file */
    char *dir_name = ngdirname(filename ? filename : ".");

    startTime = seconds();
    /* inp_source() called with fp: load from file, */
    /* called with *fp == NULL and intfile: we want to load circuit from circarray */
    if (fp || intfile) {
        deck = inp_readall(fp, dir_name, comfile, intfile, &expr_w_temper);

        /* files starting with *ng_script are user supplied command files */
        if (deck && ciprefix("*ng_script", deck->line))
            comfile = TRUE;
        /* save a copy of the deck for later reloading with 'mc_source' */
        if (deck && !comfile) {
            if (mc_deck)
                mc_free();
            mc_deck = inp_deckcopy_oc(deck);
        }
    }
    /* called with *fp == NULL and not intfile: we want to reload circuit from mc_deck */
    else {
        if (mc_deck)
            deck = inp_deckcopy(mc_deck);
        else {
            fprintf(stderr, "Error: No circuit loaded, cannot copy internally using mc_source\n");
            controlled_exit(1);
        }
    }
    endTime = seconds();
    tfree(dir_name);

    /* if nothing came back from inp_readall, just close fp and return to caller */
    if (!deck) {
        if (!intfile)
            fclose(fp);
        return;
    }

    /* files starting with *ng_script are user supplied command files */
    if (ciprefix("*ng_script", deck->line))
        comfile = TRUE;

    if (!comfile) {
        /* Extract the .option lines from the deck into 'options',
           and remove them from the deck. */
        options = inp_getopts(deck);

        /* copy a deck before subckt substitution. */
        realdeck = inp_deckcopy(deck);

        /* Save the title before INPgetTitle gets it. */
        tt = copy(deck->line);
        if (!deck->nextcard)
            fprintf(cp_err, "Warning: no lines in input\n");
    }
    if (fp && !intfile)
        fclose(fp);

    /* Now save the IO context and start a new control set.  After we
       are done with the source we'll put the old file descriptors
       back.  I guess we could use a FILE stack, but since this
       routine is recursive anyway.  */
    lastin = cp_curin;
    lastout = cp_curout;
    lasterr = cp_curerr;
    cp_curin = cp_in;
    cp_curout = cp_out;
    cp_curerr = cp_err;

    cp_pushcontrol();

    /* We should now go through the deck and execute front-end
     * commands and remove them. Front-end commands are enclosed by
     * the cards .control and .endc, unless comfile is TRUE, in which
     * case every line must be a front-end command.  There are too
     * many problems with matching the first word on the line.  */
    ld = deck;
    if (comfile) {
        /* Process each command, except 'option' which is assembled
           in a list and ingnored here */
        for (dd = deck; dd; dd = ld) {
            ld = dd->nextcard;
            if ((dd->line[0] == '*') && (dd->line[1] != '#'))
                continue;
            if (!ciprefix(".control", dd->line) && !ciprefix(".endc", dd->line)) {
                if (dd->line[0] == '*')
                    cp_evloop(dd->line + 2);
                /* option line stored but not processed */
                else if (ciprefix("option", dd->line))
                    com_options = inp_getoptsc(dd->line, com_options);
                else
                    cp_evloop(dd->line);
            }
        }
        /* free the control deck */
        line_free(deck, TRUE);
        /* set to NULL to allow generation of a new dbs */
        /* do this here and in the 'else' branch of 'if (comfile)' */
        dbs = NULL;
        ft_dotsaves();
    } /* end if (comfile) */

    else {    /* must be regular deck . . . . */
        /* loop through deck and handle control cards */
        for (dd = deck->nextcard; dd; dd = ld->nextcard) {
            /* get temp from deck */
            if (ciprefix(".temp", dd->line)) {
                s = skip_ws(dd->line + 5);
                if (temperature)
                    txfree(temperature);
                temperature = strdup(s);
            }
            /* Ignore comment lines, but not lines begining with '*#',
               but remove them, if they are in a .control ... .endc section */
            s = skip_ws(dd->line);
            if ((*s == '*') && ((s != dd->line) || (s[1] != '#'))) {
                if (commands) {
                    /* Remove comment lines in control sections, so they  don't
                     * get considered as circuits.  */
                    ld->nextcard = dd->nextcard;
                    line_free(dd, FALSE);
                    continue;
                }
                ld = dd;
                continue;
            }

            /* Put the first token from line into s */
            strncpy(name, dd->line, BSIZE_SP);
            s = skip_ws(name);
            t = skip_non_ws(s);
            *t = '\0';

            if (ciprefix(".control", dd->line)) {
                ld->nextcard = dd->nextcard;
                line_free(dd, FALSE); /* SJB - free this line's memory */
                if (commands)
                    fprintf(cp_err, "Warning: redundant .control card\n");
                else
                    commands = TRUE;
            } else if (ciprefix(".endc", dd->line)) {
                ld->nextcard = dd->nextcard;
                line_free(dd, FALSE); /* SJB - free this line's memory */
                if (commands)
                    commands = FALSE;
                else
                    fprintf(cp_err, "Warning: misplaced .endc card\n");
            } else if (commands || prefix("*#", dd->line)) {
                /* assemble all commands starting with pre_ after stripping pre_,
                to be executed before circuit parsing */
                if (ciprefix("pre_", dd->line)) {
                    s = copy(dd->line + 4);
                    pre_controls = wl_cons(s, pre_controls);
                }
                /* assemble all other commands to be executed after circuit parsing */
                else {
                    /* special control lines outside of .control section*/
                    if (prefix("*#", dd->line)) {
                        s = copy(dd->line + 2);
                    /* all commands from within .control section */
                    } else {
                        s = dd->line;
                        dd->line = NULL; /* SJB - prevent line_free() freeing the string (now pointed at by wl->wl_word) */
                    }
                    controls = wl_cons(s, controls);
                }
                ld->nextcard = dd->nextcard;
                line_free(dd, FALSE);
            } else if (!*dd->line) {
                /* So blank lines in com files don't get considered as circuits. */
                ld->nextcard = dd->nextcard;
                line_free(dd, FALSE);
            } else {
                /* lines .width, .four, .plot, .print, .save added to wl_first, removed from deck */
                /* lines .op, .meas, .tf added to wl_first */
                inp_casefix(s); /* s: first token from line */
                inp_casefix(dd->line);
                if (eq(s, ".width") ||
                    ciprefix(".four", s) ||
                    eq(s, ".plot") ||
                    eq(s, ".print") ||
                    eq(s, ".save") ||
                    eq(s, ".op") ||
                    ciprefix(".meas", s) ||
                    eq(s, ".tf"))
                {
                    wl_append_word(&wl_first, &end, copy(dd->line));

                    if (!eq(s, ".op") && !eq(s, ".tf") && !ciprefix(".meas", s)) {
                        ld->nextcard = dd->nextcard;
                        line_free(dd, FALSE);
                    } else {
                        ld = dd;
                    }
                } else {
                    ld = dd;
                }
            }
        }  /* end for (dd = deck->nextcard . . . .  */

        /* Now that the deck is loaded, do the pre commands, if there are any,
           before the circuit structure is set up */
        if (pre_controls) {
            pre_controls = wl_reverse(pre_controls);
            for (wl = pre_controls; wl; wl = wl->wl_next)
                cp_evloop(wl->wl_word);
            wl_free(pre_controls);
        }

        /* set temperature if defined to a preliminary variable which may be used
           in numparam evaluation */
        if (temperature) {
            temperature_value = atof(temperature);
            cp_vset("pretemp", CP_REAL, &temperature_value);
        }
        if (ft_ngdebug) {
            cp_getvar("pretemp", CP_REAL, &testemp);
            printf("test temperature %f\n", testemp);
        }

        /* We are done handling the control stuff.  Now process remainder of deck.
           Go on if there is something left after the controls.*/
        if (deck->nextcard) {
            fprintf(cp_out, "\nCircuit: %s\n\n", tt);
#ifdef HAS_PROGREP
            SetAnalyse("Prepare Deck", 0);
#endif
            /* Now expand subcircuit macros and substitute numparams.*/
            if (!cp_getvar("nosubckt", CP_BOOL, NULL))
                if ((deck->nextcard = inp_subcktexpand(deck->nextcard)) == NULL) {
                    line_free(realdeck, TRUE);
                    line_free(deck->actualLine, TRUE);
                    tfree(tt);
                    return;
                }

            /* Now handle translation of spice2c6 POLYs. */
#ifdef XSPICE
            /* Translate all SPICE 2G6 polynomial type sources */
            deck->nextcard = ENHtranslate_poly(deck->nextcard);
#endif

            line_free(deck->actualLine, FALSE);
            deck->actualLine = realdeck;

            /* print out the expanded deck into debug-out2.txt */
            if (ft_ngdebug) {
                /*debug: print into file*/
                FILE *fdo = fopen("debug-out2.txt", "w");
                if (fdo) {
                    struct card *t = NULL;
                    fprintf(fdo, "**************** uncommented deck **************\n\n");
                    /* always print first line */
                    fprintf(fdo, "%6d  %6d  %s\n", deck->linenum_orig, deck->linenum, deck->line);
                    /* here without out-commented lines */
                    for (t = deck->nextcard; t; t = t->nextcard) {
                        if (*(t->line) == '*')
                            continue;
                        fprintf(fdo, "%6d  %6d  %s\n", t->linenum_orig, t->linenum, t->line);
                    }
                    fprintf(fdo, "\n****************** complete deck ***************\n\n");
                    /* now completely */
                    for (t = deck; t; t = t->nextcard)
                        fprintf(fdo, "%6d  %6d  %s\n", t->linenum_orig, t->linenum, t->line);
                    fclose(fdo);
                }
                else
                    fprintf(stderr, "Warning: Cannot open file debug-out2.txt for saving debug info\n");
            }
            for (dd = deck; dd; dd = dd->nextcard) {
                /* get csparams and create vectors, being
                   available in .control section, in plot 'const' */
                if (ciprefix(".csparam", dd->line)) {
                    wordlist *wlist = NULL;
                    char *cstoken[3];
                    int i;
                    dd->line[0] = '*';
                    s = skip_ws(dd->line + 8);
                    cstoken[0] = gettok_char(&s, '=', FALSE, FALSE);
                    cstoken[1] = gettok_char(&s, '=', TRUE, FALSE);
                    cstoken[2] = gettok(&s);
                    for (i = 3; --i >= 0;)
                        wlist = wl_cons(cstoken[i], wlist);
                    com_let(wlist);
                    wl_free(wlist);
                }
            }

            /* handle .if ... .elseif ... .else ... .endif statements. */
            dotifeval(deck);

            /* merge the two option line structs
               com_options (comfile == TRUE, filled in from spinit, .spiceinit, and *ng_sript), and
               options (comfile == FALSE, filled in from circuit with .OPTIONS)
               into options, thus keeping com_options,
               options is loaded into circuit and freed when circuit is removed */
            options = line_reverse(line_nconc(options, inp_deckcopy(com_options)));

            /* List of all expressions found in instance and .model lines */
            struct pt_temper *devtlist = NULL;
            struct pt_temper *modtlist = NULL;

            /* prepare parse trees from 'temper' expressions */
            if (expr_w_temper)
                inp_parse_temper(deck, &modtlist, &devtlist);

            /* If user wants all currents saved (.options savecurrents), add .save 
            to wl_first with all terminal currents available on selected devices */
            wl_first = inp_savecurrents(deck, options, wl_first, controls);

            /* now load deck into ft_curckt -- the current circuit. */
            inp_dodeck(deck, tt, wl_first, FALSE, options, filename);

            ft_curckt->devtlist = devtlist;
            ft_curckt->modtlist = modtlist;

            /* inp_dodeck did take ownership */
            tt = NULL;
            options = NULL;

        }     /*  if (deck->nextcard) */

        /* look for and set temperature; also store param and .meas statements in circuit struct */
        if (ft_curckt) {
            ft_curckt->ci_param = NULL;
            ft_curckt->ci_meas  = NULL;
            /* PN add here stats*/
            ft_curckt->FTEstats->FTESTATnetLoadTime = endTime - startTime;
        }

        for (dd = deck; dd; dd = dd->nextcard) {
            /* all parameter lines should be sequentially ordered and placed at
               beginning of deck */
            if (ciprefix(".para", dd->line)) {
                ft_curckt->ci_param = dd;
                /* find end of .param statements */
                while (ciprefix(".para", dd->line)) {
                    prev_param = dd;
                    dd = dd->nextcard;
                    if (dd == NULL)
                        break; // no line after .param line
                }
                prev_card->nextcard  = dd;
                prev_param->nextcard = NULL;
                if (dd == NULL) {
                    fprintf(cp_err, "Warning: Missing .end card!\n");
                    break; // no line after .param line
                }
            }

            if (ciprefix(".meas", dd->line)) {
                if (cp_getvar("autostop", CP_BOOL, NULL)) {
                    if (strstr(dd->line, " max ") ||
                        strstr(dd->line, " min ") ||
                        strstr(dd->line, " avg ") ||
                        strstr(dd->line, " rms ") ||
                        strstr(dd->line, " integ "))
                    {
                        printf("Warning: .OPTION AUTOSTOP will not be effective because one of 'max|min|avg|rms|integ' is used in .meas\n");
                        printf("         AUTOSTOP being disabled...\n");
                        cp_remvar("autostop");
                    }
                }

                if (curr_meas == NULL) {
                    curr_meas = ft_curckt->ci_meas = dd;
                } else {
                    curr_meas->nextcard = dd;
                    curr_meas = dd;
                }
                prev_card->nextcard = dd->nextcard;
                curr_meas->nextcard = NULL;
                dd                 = prev_card;
            }
            prev_card = dd;
        }  //end of for-loop

        /* set temperature, if defined, to new value.
           cp_vset will set the variable "temp" and also set CKTtemp,
           so we can do it only here because the circuit has to be already there */
        if (temperature) {
            temperature_value = atof(temperature);
            cp_vset("temp", CP_REAL, &temperature_value);
            txfree(temperature);
        }

#ifdef TRACE
        /* SDB debug statement */
        printf("In inp_spsource, done with dodeck.\n");
#endif

        /* print out the expanded deck into debug-out3.txt */
        if (ft_ngdebug) {
            /*debug: print into file*/
            FILE *fdo = fopen("debug-out3.txt", "w");
            if (fdo) {
                struct card *t = NULL;
                fprintf(fdo, "**************** uncommented deck **************\n\n");
                /* always print first line */
                fprintf(fdo, "%6d  %6d  %s\n", deck->linenum_orig, deck->linenum, deck->line);
                /* here without out-commented lines */
                for (t = deck->nextcard; t; t = t->nextcard) {
                    if (*(t->line) == '*')
                        continue;
                    fprintf(fdo, "%6d  %6d  %s\n", t->linenum_orig, t->linenum, t->line);
                }
                fprintf(fdo, "\n****************** complete deck ***************\n\n");
                /* now completely */
                for (t = deck; t; t = t->nextcard)
                    fprintf(fdo, "%6d  %6d  %s\n", t->linenum_orig, t->linenum, t->line);
                fclose(fdo);
            }
            else
                fprintf(stderr, "Warning: Cannot open file debug-out3.txt for saving debug info\n");
        }

        /* Now the circuit is defined, so generate the parse trees */
        inp_parse_temper_trees(ft_curckt);
        /* Get the actual data for model and device instance parameters */
        inp_evaluate_temper(ft_curckt);

       /* linked list dbs is used to store the "save" or .save data (defined in breakp2.c),
          (When controls are executed later on, also stores TRACE, IPLOT, and STOP data) */
        /* set to NULL to allow generation of a new dbs */
        dbs = NULL;
        /* .save data stored in dbs.
           Do this here before controls are run: .save is thus recognized even if
           .control is used */
        ft_dotsaves();

        /* Now that the deck is loaded, do the commands, if there are any */
        controls = wl_reverse(controls);
        for (wl = controls; wl; wl = wl->wl_next)
            cp_evloop(wl->wl_word);
        wl_free(controls);
    }

    /* Now reset everything.  Pop the control stack, and fix up the IO
     * as it was before the source.  */
    cp_popcontrol();

    cp_curin = lastin;
    cp_curout = lastout;
    cp_curerr = lasterr;

    tfree(tt);
}


/* This routine is cut in half here because com_rset has to do what
 * follows also. End is the list of commands we execute when the job
 * is finished: we only bother with this if we might be running in
 * batch mode, since it isn't much use otherwise.  */
/*------------------------------------------------------------------
 * It appears that inp_dodeck adds the circuit described by *deck
 * to the current circuit (ft_curckt).
 *-----------------------------------------------------------------*/
void
inp_dodeck(
    struct card *deck,     /*in: the spice deck */
    char *tt,              /*in: the title of the deck */
    wordlist *end,         /*in: all lines with .width, .plot, .print, .save, .op, .meas, .tf */
    bool reuse,            /*in: TRUE if called from runcoms2.c com_rset,
                             FALSE if called from inp_spsource() */
    struct card *options,  /*in: all .option lines from deck */
    char *filename         /*in: input file of deck */
    )
{
    struct circ *ct;
    struct card *dd;
    CKTcircuit *ckt;
    char *s;
    INPtables *tab = NULL;
    struct variable *eev = NULL;
    wordlist *wl;
    bool noparse, ii;
    int print_listing;
    bool have_err = FALSE;
    int warn;          /* whether SOA check should be performed */
    int maxwarns = 0;  /* specifies the maximum number of SOA warnings */
    double startTime;

    /* First throw away any old error messages there might be and fix
       the case of the lines.  */
    for (dd = deck; dd; dd = dd->nextcard)
        if (dd->error) {
            tfree(dd->error);
            dd->error = NULL;
        }

    if (reuse) {
        /* re-use existing circuit structure */
        ct = ft_curckt;
    } else {
        if (ft_curckt) {
            ft_curckt->ci_devices = cp_kwswitch(CT_DEVNAMES, NULL);
            ft_curckt->ci_nodes   = cp_kwswitch(CT_NODENAMES, NULL);
        }
        /* create new circuit structure */
        ft_curckt = ct = TMALLOC(struct circ, 1);

        /*PN FTESTATS*/
        ft_curckt->FTEstats = TMALLOC(FTESTATistics, 1);
    }
    noparse = cp_getvar("noparse", CP_BOOL, NULL);


    /* We check preliminary for the scale option. This special processing
       is needed because we need the scale info BEFORE building the circuit
       and seems there is no other way to do this. */
    if (!noparse) {
        struct card *opt_beg = options;
        for (; options; options = options->nextcard) {
            s = skip_non_ws(options->line);

            ii = cp_interactive;
            cp_interactive = FALSE;
            wl = cp_lexer(s);
            cp_interactive = ii;
            if (!wl || !wl->wl_word || !*wl->wl_word)
                continue;
            if (eev)
                eev->va_next = cp_setparse(wl);
            else
                ct->ci_vars = eev = cp_setparse(wl);
            wl_free(wl);
            while (eev && (eev->va_next))
                eev = eev->va_next;
        }
        for (eev = ct->ci_vars; eev; eev = eev->va_next) {
            switch (eev->va_type) {
            case CP_BOOL:
                break;
            case CP_NUM:
                break;
            case CP_REAL:
                if (strcmp("scale", eev->va_name) == 0) {
                    cp_vset("scale", CP_REAL, &eev->va_real);
                    printf("Scale set\n");
                }
                break;
            case CP_STRING:
                break;
            default: {
                fprintf(stderr, "ERROR: enumeration value `CP_LIST' not handled in inp_dodeck\nAborting...\n");
                controlled_exit(EXIT_FAILURE);
            }
            } /* switch  . . . */
        }
        options = opt_beg; // back to the beginning
    } /* if (!noparse)  . . . */

    /*----------------------------------------------------
     * Now assuming that we wanna parse this deck, we call
     * if_inpdeck which takes the deck and returns a
     * a pointer to the circuit ckt.
     *---------------------------------------------------*/
    if (!noparse) {
        startTime = seconds();
        ckt = if_inpdeck(deck, &tab);
        ft_curckt->FTEstats->FTESTATnetParseTime = seconds() - startTime;
    } else {
        ckt = NULL;
    }

    /* set ckt->CKTisLinear=1 if circuit only contains R, L, C */
    if (ckt)
        cktislinear(ckt, deck);
    /* set some output terminal data */
    out_init();
    /* if_inpdeck() may return NULL upon error */
    if (ckt) {
        if (cp_getvar("warn", CP_NUM, &warn))
            ckt->CKTsoaCheck = warn;
        else
            ckt->CKTsoaCheck = 0;

        if (cp_getvar("maxwarns", CP_NUM, &maxwarns))
            ckt->CKTsoaMaxWarns = maxwarns;
        else
            ckt->CKTsoaMaxWarns = 5;
    }

    ft_curckt->FTEstats->FTESTATdeckNumLines = 0;
    /*----------------------------------------------------
     Now run through the deck and look to see if there are
     errors on any line (message contained in ->error).

     Error messages have been generated either by writing
     directly to ->error from a struct card or to
     ->error from a struct card , or by using one of the
     macros as defined in inpmacs.h. Functions INPerror(),
     INPerrCat(), and SPerror() are invoked.
     *---------------------------------------------------*/
    for (dd = deck; dd; dd = dd->nextcard) {

        ft_curckt->FTEstats->FTESTATdeckNumLines += 1;

#ifdef TRACE
        /* SDB debug statement */
        printf("In inp_dodeck, looking for errors and examining line %s . . . \n", dd->line);
#endif

        if (dd->error) {
            char *p, *q;
#ifdef XSPICE
            /* add setting of ipc syntax error flag */
            g_ipc.syntax_error = IPC_TRUE;
#endif
            p = dd->error;
            do {
                q = strchr(p, '\n');
                if (q)
                    *q = '\0';

                if (p == dd->error) {
                    if (strstr(dd->line, ".model"))
                        out_printf("Warning: Model issue on line %d :\n  %.*s ...\n%s\n",
                                   dd->linenum_orig, 72, dd->line, dd->error);
                    else {
                        out_printf("Error on line %d :\n  %s\n%s\n",
                                   dd->linenum_orig, dd->line, dd->error);
                        have_err = TRUE;
                    }
                    if (ft_stricterror)
                        controlled_exit(EXIT_BAD);
                } else {
                    out_printf("%s\n", p);
                }

                if (q)
                    *q++ = '\n';
                p = q;
            } while (p && *p);
        }  /* end  if (dd->error) */

    }   /* for (dd = deck; dd; dd = dd->nextcard) */

    /* Stop here and exit if error occurred in batch mode */
    if (have_err && ft_batchmode) {
        fprintf(stderr, "\nngspice stopped due to error, no simulation run!\n");
        controlled_exit(EXIT_BAD);
    }

    /* Only print out netlist if brief is FALSE */
    if (!cp_getvar("brief", CP_BOOL, NULL)) {
        /* output deck */
        out_printf("\nProcessed Netlist\n");
        out_printf("=================\n");
        print_listing = 1;
        for (dd = deck; dd; dd = dd->nextcard) {
            if (ciprefix(".prot", dd->line))
                print_listing = 0;
            if (print_listing == 1)
                out_printf("%s\n", dd->line);
            if (ciprefix(".unprot", dd->line))
                print_listing = 1;
        }
        out_printf("\n");
    }

    /* Add this circuit to the circuit list. If reuse is TRUE
       (command 'reset'), then use the existing ft_curckt structure.  */
    if (!reuse) {
        /* Be sure that ci_devices and ci_nodes are valid */
        ft_curckt->ci_devices = cp_kwswitch(CT_DEVNAMES, NULL);
        cp_kwswitch(CT_DEVNAMES, ft_curckt->ci_devices);
        ft_curckt->ci_nodes = cp_kwswitch(CT_NODENAMES, NULL);
        cp_kwswitch(CT_NODENAMES, ft_curckt->ci_nodes);
        ft_newcirc(ct);
        /* Assign current circuit */
        ft_curckt = ct;
    }
    ct->ci_name = tt;
    ct->ci_deck = deck;
    ct->ci_options = options;
    if (deck->actualLine)
        ct->ci_origdeck = deck->actualLine;
    else
        ct->ci_origdeck = ct->ci_deck;
    ct->ci_ckt = ckt;             /* attach the input ckt to the list of circuits */
    ct->ci_symtab = tab;
    ct->ci_inprogress = FALSE;
    ct->ci_runonce = FALSE;
    ct->ci_commands = end;
    if (reuse)
        tfree(ct->ci_filename);
    ct->ci_filename = copy(filename);

    if (!noparse) {
        /*
         * for (; options; options = options->nextcard) {
         *     s = skip_non_ws(options->line);
         *     ii = cp_interactive;
         *     cp_interactive = FALSE;
         *     wl = cp_lexer(s);
         *     cp_interactive = ii;
         *     if (!wl || !wl->wl_word || !*wl->wl_word)
         *         continue;
         *     if (eev)
         *         eev->va_next = cp_setparse(wl);
         *     else
         *         ct->ci_vars = eev = cp_setparse(wl);
         *     while (eev->va_next)
         *         eev = eev->va_next;
         * }
        */
        for (eev = ct->ci_vars; eev; eev = eev->va_next) {
            bool one = TRUE;   /* FIXME, actually eev->va_bool should be TRUE anyway */
            switch (eev->va_type) {
            case CP_BOOL:
                if_option(ct->ci_ckt, eev->va_name, eev->va_type, &one);
                break;
            case CP_NUM:
                if_option(ct->ci_ckt, eev->va_name, eev->va_type, &eev->va_num);
                break;
            case CP_REAL:
                if_option(ct->ci_ckt, eev->va_name, eev->va_type, &eev->va_real);
                break;
            case CP_STRING:
                if_option(ct->ci_ckt, eev->va_name, eev->va_type, eev->va_string);
                break;
            default: {
                fprintf(stderr, "ERROR: enumeration value `CP_LIST' not handled in inp_dodeck\nAborting...\n");
                controlled_exit(EXIT_FAILURE);
            }
            } // switch  . . .
        }
    } // if (!noparse)  . . .

    /* add title of deck to data base */
    /* this won't work if the title is the empty string
    *    cp_addkword() doesn't work for tt === ""
    *  since CT_CKTNAMES doesn't seem to be used anywhere
    *  I've disabled this piece.
    */
#if 0
    cp_addkword(CT_CKTNAMES, tt);
#endif
}


void
com_mc_source(wordlist *wl)
{
    NG_IGNORE(wl);
    inp_spsource(NULL, FALSE, NULL, FALSE);
}


/* Edit and re-load the current input deck.  Note that if these
 * commands are used on a non-unix machine, they will leave spice.tmp
 * junk files lying around.  */
void
com_edit(wordlist *wl)
{
    char *filename;
    FILE *fp;
    bool inter, permfile;
    char buf[BSIZE_SP];

    if (!cp_getvar("interactive", CP_BOOL, NULL)) {
        fprintf(cp_err,
                "Warning: `edit' is disabled because 'interactive' has not been set.\n"
                "  perhaps you want to 'set interactive'\n");
        return;
    }

    inter = cp_interactive;
    cp_interactive = FALSE;
    if (wl) {
        if (!doedit(wl->wl_word)) {
            cp_interactive = inter;
            return;
        }
        if ((fp = inp_pathopen(wl->wl_word, "r")) == NULL) {
            perror(wl->wl_word);
            cp_interactive = inter;
            return;
        }
        inp_spsource(fp, FALSE, wl->wl_word, FALSE);
    } else {
        /* If there is no circuit yet, then create one */
        if (ft_curckt && ft_curckt->ci_filename) {
            filename = ft_curckt->ci_filename;
            permfile = TRUE;
        } else {
            filename = smktemp("sp");
            permfile = FALSE;
        }
        if (ft_curckt && !ft_curckt->ci_filename) {
            if ((fp = fopen(filename, "w")) == NULL) {
                perror(filename);
                cp_interactive = inter;
                return;
            }
            inp_list(fp, ft_curckt->ci_deck, ft_curckt->ci_options, LS_DECK);
            fprintf(cp_err,
                    "Warning: editing a temporary file -- "
                    "circuit not saved\n");
            fclose(fp);
        } else if (!ft_curckt) {
            if ((fp = fopen(filename, "w")) == NULL) {
                perror(filename);
                cp_interactive = inter;
                return;
            }
            fprintf(fp, "SPICE 3 test deck\n");
            fclose(fp);
        }
        if (!doedit(filename)) {
            cp_interactive = inter;
            return;
        }

        if ((fp = fopen(filename, "r")) == NULL) {
            perror(filename);
            cp_interactive = inter;
            return;
        }
        inp_spsource(fp, FALSE, permfile ? filename : NULL, FALSE);

        /* fclose(fp);  */
        /*      MW. inp_spsource already closed fp */

        if (ft_curckt && !ft_curckt->ci_filename)
            unlink(filename);
    }

    cp_interactive = inter;

    /* note: default is to run circuit after successful edit */

    fprintf(cp_out, "run circuit? ");
    fflush(cp_out);
    fgets(buf, BSIZE_SP, stdin);
    if (buf[0] != 'n') {
        fprintf(cp_out, "running circuit\n");
        com_run(NULL);
    }
}


/* alter a parameter, either
   subckt param:  alterparam subcktname pname=vpval
   global .param: alterparam pname=pval
   Changes params in mc_deck
   To become effective, 'mc_source' has to be called after 'alterparam' */
void
com_alterparam(wordlist *wl)
{
    struct card *dd;
    char *pname, *pval, *tmp, *subcktname = NULL, *linein, *linefree, *s;
    bool found = FALSE;

    if (!mc_deck) {
        fprintf(cp_err, "Error: No internal deck available\n");
        return;
    }
    linefree = wl_flatten(wl);
    linein = skip_ws(linefree);
    s = tmp = gettok_char(&linein, '=', FALSE, FALSE);
    if (!s) {
        fprintf(cp_err, "\nError: Wrong format in line 'alterparam %s'\n   command 'alterparam' skipped\n", linefree);
        tfree(linefree);
        return;
    }
    linein++; /* skip the '=' */
    pval = gettok(&linein);
    subcktname = gettok(&tmp);
    if (!pval || !subcktname) {
        fprintf(cp_err, "\nError: Wrong format in line 'alterparam %s'\n   command 'alterparam' skipped\n", linefree);
        tfree(pval);
        tfree(subcktname);
        tfree(linefree);
        return;
    }
    pname = gettok(&tmp);
    if (!pname) {
        pname = subcktname;
        subcktname = NULL;
    }
    tfree(linefree);
    tfree(s);
    for (dd = mc_deck->nextcard; dd; dd = dd->nextcard) {
        char *curr_line = dd->line;
        /* alterparam subcktname pname=vpval
           Parameters from within subcircuit are no longer .param lines, but have been added to
           the .subckt line as pname=paval and to the x line as pval. pval in the x line takes
           precedence when subciruit is called, so has to be replaced here.
           Find subcircuit with subcktname.
           After params: Count the number of parameters (notok) until parameter pname is found.
           When found, search for x-line with subcktname.
           Replace parameter value number notok by pval.
        */
        if (subcktname) {
            /* find subcircuit */
            if (ciprefix(".subckt", curr_line)) {
                curr_line = nexttok(curr_line); /* skip .subckt */
                char *sname = gettok(&curr_line);
                if (eq(sname, subcktname)) {
                    tfree(sname);
                    curr_line = strstr(curr_line, "params:");
                    curr_line = skip_non_ws(curr_line); /* skip params: */
                    /* string to search for */
                    char *pname_eq = tprintf("%s=", pname);
                    int notok = 0;
                    while (*curr_line) {
                        char *token = gettok(&curr_line);
                        if (ciprefix(pname_eq, token)) {
                            tfree(token);
                            found = TRUE;
                            break;
                        }
                        notok++;
                        tfree(token);
                    }
                    tfree(pname_eq);
                    if (found) {
                        /* find x line with same subcircuit name */
                        struct card *xx;
                        char *bsubb = tprintf(" %s ", subcktname);
                        for (xx = mc_deck->nextcard; xx; xx = xx->nextcard) {
                            char *xline = xx->line;
                            if (*xline == 'x') {
                                xline = strstr(xline, bsubb);
                                if (xline) {
                                    xline = nexttok(xline); /* skip subcktname */
                                    int ii;
                                    for (ii = 0; ii < notok; ii++)
                                        xline = nexttok(xline); /* skip parameter values */
                                    char *beg = copy_substring(xx->line, xline);
                                    xline = nexttok(xline); /* skip parameter value to be replaced */
                                    char *newline = tprintf("%s %s %s", beg, pval, xline);
                                    tfree(xx->line);
                                    xx->line = newline;
                                    tfree(beg);
                                }
                                else
                                    continue;
                            }
                        }
                        tfree(bsubb);
                    }
                }
                else {
                    tfree(sname);
                    continue;
                }
            }
        } /* subcktname */
        /* alterparam pname=vpval */
        else {
            if (ciprefix(".para", curr_line)) {
                curr_line = nexttok(curr_line); /* skip .param */
                char *name = gettok_char(&curr_line, '=', FALSE, FALSE);
                if (eq(name, pname)) {
                    curr_line = dd->line;
                    char *start = gettok_char(&curr_line, '=', TRUE, FALSE);
                    tfree(dd->line);
                    dd->line = tprintf("%s%s", start, pval);
                    found = TRUE;
                    tfree(start);
                }
                tfree(name);
            }
        }
    }
    if (!found)
        fprintf(cp_err, "\nError: parameter '%s' not found,\n   command 'alterparam' skipped\n", pname);
    tfree(pval);
    tfree(pname);
    tfree(subcktname);
}


static bool
doedit(char *filename)
{
    char buf[BSIZE_SP], buf2[BSIZE_SP], *editor;

    if (cp_getvar("editor", CP_STRING, buf2)) {
        editor = buf2;
    } else {
        if ((editor = getenv("EDITOR")) == NULL) {
            if (Def_Editor && *Def_Editor)
                editor = Def_Editor;
            else
                editor = "/usr/bin/vi";
        }
    }
    sprintf(buf, "%s %s", editor, filename);
    return (system(buf) ? FALSE : TRUE);
}


void
com_source(wordlist *wl)
{
    FILE *fp, *tp;
    char buf[BSIZE_SP];
    bool inter;
    char *tempfile = NULL, *firstfile;

    wordlist *owl = wl;
    size_t n;

    inter = cp_interactive;
    cp_interactive = FALSE;

    firstfile = wl->wl_word;

    if (wl->wl_next) {
        /* There are several files -- put them into a temp file  */
        tempfile = smktemp("sp");
        if ((fp = inp_pathopen(tempfile, "w+")) == NULL) {
            perror(tempfile);
            cp_interactive = TRUE;
            return;
        }
        while (wl) {
            if ((tp = inp_pathopen(wl->wl_word, "r")) == NULL) {
                perror(wl->wl_word);
                fclose(fp);
                cp_interactive = TRUE;
                unlink(tempfile);
                return;
            }
            while ((n = fread(buf, 1, BSIZE_SP, tp)) > 0)
                fwrite(buf, 1, n, fp);
            fclose(tp);
            wl = wl->wl_next;
        }
        fseek(fp, 0L, SEEK_SET);
    } else {
        fp = inp_pathopen(wl->wl_word, "r");
    }

    if (fp == NULL) {
        perror(wl->wl_word);
        cp_interactive = TRUE;
        return;
    }

    /* Don't print the title if this is a spice initialisation file. */
    if (ft_nutmeg || substring(INITSTR, owl->wl_word) || substring(ALT_INITSTR, owl->wl_word))
        inp_spsource(fp, TRUE, tempfile ? NULL : wl->wl_word, FALSE);
    else {
        /* Save path name for use in XSPICE fopen_with_path() */
        if (Infile_Path)
            tfree(Infile_Path);
        Infile_Path = ngdirname(firstfile);
        inp_spsource(fp, FALSE, tempfile ? NULL : wl->wl_word, FALSE);
    }

    cp_interactive = inter;
    if (tempfile)
        unlink(tempfile);
}


void
inp_source(char *file)
{
    static struct wordlist wl = { NULL, NULL, NULL };
    wl.wl_word = file;
    com_source(&wl);
}


/* check the input deck (after inpcom and numparam extensions)
   for linear elements. If only linear elements are found,
   ckt->CKTisLinear is set to 1. Return immediately if a first
   non-linear element is found. */
static void
cktislinear(CKTcircuit *ckt, struct card *deck)
{
    struct card *dd;
    char firstchar;

    if (deck->nextcard)
        for (dd = deck->nextcard; dd; dd = dd->nextcard) {
            firstchar = *dd->line;
            switch (firstchar) {
                case 'r':
                case 'l':
                case 'c':
                case 'i':
                case 'v':
                case '*':
                case '.':
                case 'e':
                case 'g':
                case 'f':
                case 'h':
                    continue;
                    break;
                default:
                    ckt->CKTisLinear = 0;
                    return;
            }
        }

    ckt->CKTisLinear = 1;
}


/* global array for assembling circuit lines entered by fcn circbyline
   or receiving array from external caller. Array is created once per ngspice call.
   Last line of the array has to get the value NULL */
char **circarray;


void
create_circbyline(char *line)
{
    static int linec = 0;
    static int memlen = 256;
    FILE *fp = NULL;
    if (!circarray)
        circarray = TMALLOC(char*, memlen);
    char *p = skip_ws(line);
    if (line < p)
        memmove(line, p, strlen(p) + 1);
    circarray[linec++] = line;
    if (linec < memlen) {
        if (ciprefix(".end", line) && (line[4] == '\0' || isspace_c(line[4]))) {
            circarray[linec] = NULL;
            inp_spsource(fp, FALSE, "", TRUE);
            linec = 0;
        }
    }
    else {
        memlen += memlen;
        circarray = TREALLOC(char*, circarray, memlen);
    }
}


/* fcn called by command 'circbyline' */
void
com_circbyline(wordlist *wl)
{
    /* undo the automatic wordline creation.
       wl_flatten allocates memory on the heap for each newline.
       This memory will be released line by line in inp_source(). */

    char *newline = wl_flatten(wl);
    create_circbyline(newline);
}


/* handle .if('expr') ... .elseif('expr') ... .else ... .endif statements.
   numparam has evaluated .if('boolean expression') to
   .if (   1.000000000e+000  ) or .elseif (   0.000000000e+000  ) */
static void
dotifeval(struct card *deck)
{
    int iftrue = 0, elseiftrue = 0, elsetrue = 0, iffound = 0, elseiffound = 0, elsefound = 0;
    struct card *dd;
    char *dottoken;
    char *s, *t;

    /* skip the first line (title line) */
    for (dd = deck->nextcard; dd; dd = dd->nextcard) {

        s = t = dd->line;

        if (*s == '*')
            continue;

        dottoken = gettok(&t);
        /* find '.if' and read its parameter */
        if (cieq(dottoken, ".if")) {
            elsefound = 0;
            elseiffound = 0;
            iffound = 1;
            *s = '*';
            s = dd->line + 3;
            iftrue = atoi(s);
        }
        else if (cieq(dottoken, ".elseif")) {
            elsefound = 0;
            elseiffound = 1;
            iffound = 0;
            *s = '*';
            if (!iftrue) {
                s = dd->line + 7;
                elseiftrue = atoi(s);
            }
        }
        else if (cieq(dottoken, ".else")) {
            elsefound = 1;
            elseiffound = 0;
            iffound = 0;
            if (!iftrue && !elseiftrue)
                elsetrue = 1;
            *s = '*';
        }
        else if (cieq(dottoken, ".endif")) {
            elsefound = elseiffound = iffound = 0;
            elsetrue = elseiftrue = iftrue = 0;
            *s = '*';
//          inp_subcktexpand(dd);
        }
        else {
            if (iffound && !iftrue) {
                *s = '*';
            }
            else if (elseiffound && !elseiftrue)  {
                *s = '*';
            }
            else if (elsefound && !elsetrue)  {
                *s = '*';
            }
        }
        tfree(dottoken);
    }
}


/*
    Evaluate expressions containing 'temper' keyword, found in
    .model lines or device instance lines.
    Activity has four steps:
    1) Prepare the expressions to survive numparam expansion
       (see function inp_temper_compat() in inpcom.c). A global
       variable expr_w_temper is set TRUE if any expression with
       'temper' has been found.
    2) After numparam insertion and subcircuit expansion,
       get the expressions, store them with a place holder for the
       pointer to the expression parse tree and a wordlist containing
       device/model name, parameter name and a placeholder for the
       evaluation result ready to be used by com_alter(mod) functions,
       in linked lists modtlist (model) or devtlist (device instance).
       (done function inp_parse_temper()).
    3) After the circuit structure has been established, generate
       the parse trees. We can do it only then because pointers to
       ckt->CKTtemp and others are stored in the trees.
       (done in function inp_parse_temper_trees()).
    4) Evaluation  of the parse trees is requested by calling function
       inp_evaluate_temper(). The B Source parser is invoked here.
       ckt->CKTtemp is used to replace the 'temper' token by the actual
       circuit temperature. The evaluation results are added to the
       wordlist, com_alter(mod) is called to set the new parameters
       to the model parameters or device instance parameters.
*/

static int
inp_parse_temper(struct card *card, struct pt_temper **modtlist_p, struct pt_temper **devtlist_p)
{
    int error = 0;

    struct pt_temper *modtlist = NULL;
    struct pt_temper *devtlist = NULL;

    /* skip title line */
    card = card->nextcard;
    for (; card; card = card->nextcard) {

        char *curr_line = card->line;

        /* exclude some elements */
        if (strchr("*vbiegfh", curr_line[0]))
            continue;
        /* exclude all dot commands except .model */
        if (curr_line[0] == '.' && !prefix(".model", curr_line))
            continue;
        /* exclude lines not containing 'temper' */
        if (!strstr(curr_line, "temper"))
            continue;

        bool is_model = prefix(".model", curr_line);

        /* skip ".model" */
        if (is_model)
            curr_line = nexttok(curr_line);

        /* now start processing of the remaining lines containing 'temper' */
        char *name = gettok(&curr_line);
        char *t = curr_line;
        while ((t = search_identifier(t, "temper", curr_line)) != NULL) {
            struct pt_temper *alter = TMALLOC(struct pt_temper, 1);
            char *eq_ptr = find_back_assignment(t, curr_line);
            if (!eq_ptr) {
                t = t + 1;
                continue;
            }
            /* go back over param name */
            char *end_param = skip_back_ws(eq_ptr, curr_line);
            char *beg_param = eq_ptr;
            while (beg_param > curr_line && !isspace_c(beg_param[-1]) && beg_param[-1] != '(')
                beg_param--;
            /* find end of expression string */
            char *beg_expr = skip_ws(eq_ptr + 1);
            char *end_expr = find_assignment(beg_expr);
            if (end_expr) {
                end_expr = skip_back_ws(end_expr, curr_line);
                end_expr = skip_back_non_ws(end_expr, curr_line);
            } else {
                end_expr = strchr(beg_expr, '\0');
            }
            end_expr = skip_back_ws(end_expr, curr_line);
            /* overwrite this parameter assignment with ' '
             *   the backend will use a default
             * later, after evaluation, "alter" the parameter
             */
            alter->expression = copy_substring(beg_expr, end_expr);

            /* to be filled in by evaluation function */
            alter->wlend = wl_cons(NULL, NULL);
            /* create wordlist suitable for com_altermod */
            alter->wl =
                wl_cons(copy(name),
                        wl_cons(copy_substring(beg_param, end_param),
                                wl_cons(copy("="),
                                        alter->wlend)));

            memset(beg_param, ' ', (size_t) (end_expr - beg_param));

            /* fill in the linked parse tree list */
            if (is_model) {
                alter->next = modtlist;
                modtlist = alter;
            } else {
                alter->next = devtlist;
                devtlist = alter;
            }

            t = end_expr;
        }
        tfree(name);
    }

    *modtlist_p = modtlist;
    *devtlist_p = devtlist;

    return error;
}


static void
inp_parse_temper_trees(struct circ *circ)
{
    struct pt_temper *d;

    for(d = circ->devtlist; d; d = d->next) {
        char *expression = d->expression;
        INPgetTree(&expression, &d->pt, circ->ci_ckt, NULL);
    }

    for(d = circ->modtlist; d; d = d->next) {
        char *expression = d->expression;
        INPgetTree(&expression, &d->pt, circ->ci_ckt, NULL);
    }
}


void
rem_tlist(struct pt_temper *p)
{
    while (p) {
        struct pt_temper *next_p = p->next;
        tfree(p->expression);
        wl_free(p->wl);
        INPfreeTree((IFparseTree *) p->pt);
        tfree(p);
        p = next_p;
    }
}


void
inp_evaluate_temper(struct circ *circ)
{
    struct pt_temper *d;
    double result;

    for(d = circ->devtlist; d; d = d->next) {
        IFeval((IFparseTree *) d->pt, 1e-12, &result, NULL, NULL);
        if (d->wlend->wl_word)
            tfree(d->wlend->wl_word);
        d->wlend->wl_word = tprintf("%g", result);
        com_alter(d->wl);
    }

    for(d = circ->modtlist; d; d = d->next) {
        char *name = d->wl->wl_word;
        INPretrieve(&name, circ->ci_symtab);
        /* only evaluate models which have been entered into the
           hash table ckt->MODnameHash */
        if (ft_sim->findModel (circ->ci_ckt, name) == NULL)
            continue;
        IFeval((IFparseTree *) d->pt, 1e-12, &result, NULL, NULL);
        if (d->wlend->wl_word)
            tfree(d->wlend->wl_word);
        d->wlend->wl_word = tprintf("%g", result);
        com_altermod(d->wl);
    }
}


/*
 * Enable current measurements by the user
 *   if 'option savecurrents' is set by the user.
 * We have to check for this option here prematurely
 *   because options will be processed later.
 * Then append a
 *    .save all
 * statement to 'wl' if no other 'save' has been given so far.
 * Then scan the deck for known devices and append
 *   .save @q1[ib]
 * statements to 'wl' for all of their current vectors.
 */

static wordlist *
inp_savecurrents(struct card *deck, struct card *options, wordlist *wl, wordlist *controls)
{
    wordlist *p;

    /* check if option 'savecurrents' is set */
    for (; options; options = options->nextcard)
        if (strstr(options->line, "savecurrents"))
            break;

    if (!options)
        return wl;

    /* search for 'save' command in the .control section */
    for (p = controls; p; p = p->wl_next)
        if(prefix("save", p->wl_word))
            break;

    /* search for '.save' in the 'wl' list */
    if (!p)
        for (p = wl; p; p = p->wl_next)
            if(prefix(".save", p->wl_word))
                break;

    /* if not found, then add '.save all' */
    if (!p)
        p = wl_cons(copy(".save all"), NULL);
    else
        p = NULL;

    /* Scan the deck for devices with their terminals.
     * We currently serve bipolars, resistors, MOS1, capacitors, inductors,
     * controlled current sources. Others may follow.
     */
    for (deck = deck->nextcard; deck; deck = deck->nextcard) {
        char *newline, *devname, *devline = deck->line;

        switch (devline[0]) {
        case 'm':
            devname = gettok(&devline);
            newline = tprintf(".save @%s[id] @%s[is] @%s[ig] @%s[ib]",
                              devname, devname, devname, devname);
            break;
        case 'j':
            devname = gettok(&devline);
            newline = tprintf(".save @%s[id] @%s[is] @%s[ig] @%s[igd]",
                              devname, devname, devname, devname);
            break;
        case 'q':
            devname = gettok(&devline);
            newline = tprintf(".save @%s[ic] @%s[ie] @%s[ib] @%s[is]",
                              devname, devname, devname, devname);
            break;
        case 'd':
            devname = gettok(&devline);
            newline = tprintf(".save @%s[id]", devname);
            break;
        case 'r':
        case 'c':
        case 'l':
        case 'b':
        case 'f':
        case 'g':
        case 'w':
        case 's':
            devname = gettok(&devline);
            newline = tprintf(".save @%s[i]", devname);
            break;
        case 'i':
            devname = gettok(&devline);
            newline = tprintf(".save @%s[current]", devname);
            break;
        default:
            continue;
        }

        p = wl_cons(newline, p);
        tfree(devname);
    }

    return wl_append(wl, wl_reverse(p));
}
