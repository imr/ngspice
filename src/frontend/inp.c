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

#include "ngspice/osdiitf.h"
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
#include "ngspice/compatmode.h"

static struct card *com_options = NULL;
static struct card *mc_deck = NULL;
static struct card *recent_deck = NULL;

static void cktislinear(CKTcircuit *ckt, struct card *deck);
void create_circbyline(char *line, bool reset, bool lastline);
static bool doedit(char *filename);
static void dotifeval(struct card *deck);
static void eval_agauss(struct card *deck, char *fcn);
static wordlist *inp_savecurrents(struct card *deck, struct card *options,
        wordlist *wl, wordlist *controls);
static void recifeval(struct card *pdeck);
static char *upper(register char *string);
static void rem_unused_mos_models(struct card* deck);

extern void com_optran(wordlist * wl);
extern void tprint(struct card *deck);


//void inp_source_recent(void);
//void inp_mc_free(void);
//void inp_remove_recent(void);
static bool mc_reload = FALSE;
void eval_opt(struct card *deck);

extern bool ft_batchmode;

/* from inpcom.c */
extern struct nscope* inp_add_levels(struct card *deck);
extern void inp_rem_levels(struct nscope* root);
extern void comment_out_unused_subckt_models(struct card *deck);
extern void inp_rem_unused_models(struct nscope *root, struct card *deck);

extern void modprobenames(INPtables * tab);

#ifdef SHARED_MODULE
extern void exec_controls(wordlist *controls);
#endif

/* display the source file name in the source window */
#ifdef HAS_WINGUI
extern void SetSource(char *Name);
#endif

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


/* Do a listing. Use is listing [expanded] [logical] [physical] [deck] [runnable] */
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
                case 'r':
                case 'R':
                    expand = TRUE;
                    type = LS_RUNNABLE;
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
            if (type != LS_DECK && type != LS_RUNNABLE)
                fprintf(cp_out, "\t%s\n\n", ft_curckt->ci_name);
            inp_list(cp_out,
                     expand ? ft_curckt->ci_deck : ft_curckt->ci_origdeck,
                     ft_curckt->ci_options, type);
            if (expand && ft_curckt->ci_auto && type != LS_RUNNABLE)
                inp_list(cp_out, ft_curckt->ci_auto,
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
    static char buf[LBSIZE_SP];

    if (string) {
        if (strlen(string) > LBSIZE_SP - 1)
            fprintf(stderr, "Warning: output of command 'listing' will be truncated\n");
        strncpy(buf, string, LBSIZE_SP - 1);
        buf[LBSIZE_SP - 1] = '\0';
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

    renumber = cp_getvar("renumber", CP_BOOL, NULL, 0);

    if (type == LS_LOGICAL || type == LS_RUNNABLE) {
    top1:
        for (here = deck; here; here = here->nextcard) {
            if (renumber)
                here->linenum = i;
            if (ciprefix(".end", here->line) && !isalpha_c(here->line[4]))
                continue;
            if ((*here->line != '*') && (type == LS_LOGICAL)) {
                Xprintf(file, "%6d : %s\n", here->linenum, upper(here->line));
                if (here->error)
                    Xprintf(file, "%s\n", here->error);
            }
            else if ((*here->line != '*') && (type == LS_RUNNABLE)) {
                Xprintf(file, "%s\n", here->line);
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

        if (type == LS_LOGICAL)
            Xprintf(file, "%6d : .end\n", i);
        else if (type == LS_RUNNABLE)
            Xprintf(file, ".end\n");

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


/* store ft_curckt->ci_mcdeck into a 'previous' deck */
void
inp_mc_free(void)
{
    if (ft_curckt && ft_curckt->ci_mcdeck) {
        if (recent_deck && recent_deck != ft_curckt->ci_mcdeck) {
            struct circ *pp;
            /* NULL any ci_mcdeck entry from ft_circuits whose address equals recent_deck,
            then free this address */
            for (pp = ft_circuits; pp; pp = pp->ci_next)
                if (pp->ci_mcdeck == recent_deck) {
                    pp->ci_mcdeck = NULL;
                }
            line_free(recent_deck, TRUE);
        }
        recent_deck = ft_curckt->ci_mcdeck;
        ft_curckt->ci_mcdeck = NULL;
    }
}

/* called by com_rset: reload most recent circuit */
void
inp_source_recent(void) {
    mc_deck = recent_deck;
    mc_reload = TRUE;
    inp_spsource(NULL, FALSE, NULL, FALSE);
}

/* remove the 'recent' deck */
void
inp_remove_recent(void) {
    if (recent_deck)
        line_free(recent_deck, TRUE);
}


/* Check for .option seed=[val|random] and set the random number generator.
   Check for .option cshunt=val and set a global variable
   Input is the option deck (already sorted for .option) */
void
eval_opt(struct card* deck)
{
    struct card* card;
    bool has_seed = FALSE;
    bool has_cshunt = FALSE;

    for (card = deck; card; card = card->nextcard) {
        char* line = card->line;

        if (strstr(line, "seedinfo"))
            setseedinfo();
        char* begtok = strstr(line, "seed=");
        if (begtok)
            begtok = &begtok[5]; /*skip seed=*/
        if (begtok) {
            if (has_seed)
                fprintf(cp_err, "Warning: Multiple 'option seed=val|random' found!\n");
            char* token = gettok(&begtok);
            /* option seed=random [seed='random'] */
            if (eq(token, "random") || eq(token, "{random}")) {
                time_t acttime = time(NULL);
                /* get random value from time in seconds since 1.1.1970 */
                int rseed = (int)(acttime - 1600000000);
                cp_vset("rndseed", CP_NUM, &rseed);
                com_sseed(NULL);
                has_seed = TRUE;
            }
            /* option seed=val*/
            else {
                int sr = atoi(token);
                if (sr <= 0)
                    fprintf(cp_err, "Warning: Cannot convert 'option seed=%s' to seed value, skipped!\n", token);
                else {
                    cp_vset("rndseed", CP_NUM, &sr);
                    com_sseed(NULL);
                    has_seed = TRUE;
                }
            }
            tfree(token);
        }

        begtok = strstr(line, "cshunt=");
        if (begtok)
            begtok = &begtok[7]; /*skip cshunt=*/
        if (begtok) {
            int err = 0;
            if (has_cshunt)
                fprintf(cp_err, "Warning: Multiple '.option cshunt=val' found!\n");
            /* option cshunt=val*/
            double sr = INPevaluate(&begtok, &err, 0);
            if (sr <= 0 || err)
                fprintf(cp_err, "Warning: Cannot convert 'option cshunt=%s' to capacitor value, skipped!\n", begtok);
            else {
                cp_vset("cshunt_value", CP_REAL, &sr);
                has_cshunt = TRUE;
            }
        }
    }
}

/* The routine to source a spice input deck. We read the deck in, take
 * out the front-end commands, and create a CKT structure. Also we
 * filter out the following cards: .save, .width, .four, .print, and
 * .plot, to perform after the run is over.
 * Then, we run dodeck, which parses up the deck.             */
int
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
    char *tt = NULL, name[BSIZE_SP + 1], *s, *t, *temperature = NULL;
    bool commands = FALSE;
    wordlist *wl = NULL, *end = NULL, *wl_first = NULL;
    wordlist *controls = NULL, *pre_controls = NULL;
    FILE *lastin, *lastout, *lasterr;
    double temperature_value;
    bool expr_w_temper = FALSE;

    double startTime, loadTime = 0., endTime;

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
        /* stored to new circuit ci_mcdeck in fcn */
            mc_deck = inp_deckcopy_oc(deck);
        }
    }
    /* called with *fp == NULL and not intfile: we want to reload circuit from mc_deck */
    else {
        /* re-load deck due to command 'reset' via function inp_source_recent() */
        if (mc_reload && mc_deck) {
            deck = inp_deckcopy(mc_deck);
            expr_w_temper = TRUE;
            mc_reload = FALSE;
            fprintf(stdout, "Reset re-loads circuit %s\n", mc_deck->line);
        }
        /* re-load input deck from the current circuit structure */
        else if (ft_curckt && ft_curckt->ci_mcdeck) {
            deck = inp_deckcopy(ft_curckt->ci_mcdeck);
            expr_w_temper = TRUE;
        }
        /* re-load input deck from the recent circuit structure with mc_source */
        else if (!ft_curckt && mc_deck) {
            deck = inp_deckcopy(mc_deck);
            expr_w_temper = TRUE;
        }
        /* no circuit available, should not happen */
        else {
            fprintf(stderr, "Error: No circuit loaded, cannot copy internally using mc_source or reset\n");
            controlled_exit(1);
        }
        /* print out the re-loaded deck into debug-out-mc.txt */
        if (ft_ngdebug) {
            /*debug: print into file*/
            FILE *fdo = fopen("debug-out-mc.txt", "w");
            if (fdo) {
                struct card *tc = NULL;
                fprintf(fdo, "****************** complete mc deck ***************\n\n");
                /* now completely */
                for (tc = deck; tc; tc = tc->nextcard)
                    fprintf(fdo, "%6d  %6d  %s\n", tc->linenum_orig, tc->linenum, tc->line);
                fclose(fdo);
            }
            else
                fprintf(stderr, "Warning: Cannot open file debug-out-mc.txt for saving debug info\n");
        }
    }
    endTime = seconds();
    /* store input directory to a variable */
    if (fp) {
        cp_vset("inputdir", CP_STRING, dir_name);
    }

    /* if nothing came back from inp_readall, e.g. after calling ngspice without parameters,
       just close fp and return to caller */
    if (!deck) {
        if (!intfile && fp)
            fclose(fp);
        return 0;
    }

    /* files starting with *ng_script are user supplied command files */
    if (ciprefix("*ng_script", deck->line))
        comfile = TRUE;

    if (!comfile) {
        /* Extract the .option lines from the deck into 'options',
           and remove them from the deck. Exceptions are .option with params. */
        options = inp_getopts(deck);
        /* Check for .option seed=[val|random] and set the random number generator.
           Check for .option cshunt=val and set a global variable cshunt_value */
        eval_opt(options);
        /* copy a deck before subckt substitution. */
        realdeck = inp_deckcopy(deck);

        /* Save the title before INPgetTitle gets it. */
        tt = copy(deck->line);
        if (!deck->nextcard) {
            fprintf(cp_err, "Warning: no lines in input\n");
        }
    }
    if (fp && !intfile) {
        fclose(fp);
    }

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

    else {  /* must be regular deck . . . . */
        /* loop through deck and handle control cards */
        for (dd = deck->nextcard; dd; dd = ld->nextcard) {
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
                /* assemble all commands starting with pre_ after stripping
                 * pre_, to be executed before circuit parsing */
                if (ciprefix("pre_", dd->line)) {
                    s = copy(dd->line + 4);
                    pre_controls = wl_cons(s, pre_controls);
                }
                /* assemble all other commands to be executed after circuit
                 * parsing */
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
                /* Do not eliminate " around netnames, to allow '/' or '-' in netnames */
                if (!eq(s, ".plot") && !eq(s, ".print"))
                    inp_casefix(dd->line);
                if (eq(s, ".width") ||
                        ciprefix(".four", s) ||
                        eq(s, ".plot") ||
                        eq(s, ".print") ||
/*                        eq(s, ".save") || add .save only after subcircuit expansion */
                        eq(s, ".op") ||
                        ciprefix(".meas", s) ||
                        eq(s, ".tf")) {
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
            for (wl = pre_controls; wl; wl = wl->wl_next){
#ifdef OSDI
                inputdir = dir_name;
#endif
                cp_evloop(wl->wl_word);
            }

#ifdef OSDI
            inputdir = NULL;
#endif
            wl_free(pre_controls);
        }

        /* We are done handling the control stuff.  Now process remainder of deck.
           Go on if there is something left after the controls.*/
        if (deck->nextcard) {
            fprintf(cp_out, "\nCircuit: %s\n\n", tt);
#ifdef HAS_PROGREP
            SetAnalyse("Prepare Deck", 0);
#endif
            endTime = seconds();
            loadTime = endTime - startTime;
            startTime = endTime;

            /* If we have large PDK deck, search for scale option and set 
            the variable 'scale'*/
            if (newcompat.hs || newcompat.spe) {
                struct card* scan;
                double dscale = 1;
                /* from options in a script */
                for (scan = com_options; scan; scan = scan->nextcard) {
                    char* tmpscale = strstr(scan->line, "scale=");
                    if (tmpscale) {
                        int err;
                        tmpscale = tmpscale + 6;
                        dscale = INPevaluate(&tmpscale, &err, 1);
                        if (err == 0) {
                            cp_vset("scale", CP_REAL, &dscale);
                            printf("option SCALE: Scale is set to %g for instance and model parameters\n", dscale);
                        }
                        else
                            fprintf(stderr, "\nError: Could not set 'scale' variable\n");
                    }
                    tmpscale = strstr(scan->line, "scalm=");
                    if (tmpscale) {
                        int err;
                        tmpscale = tmpscale + 6;
                        dscale = INPevaluate(&tmpscale, &err, 1);
                        if (err == 0) {
                            cp_vset("scalm", CP_REAL, &dscale);
                            fprintf(stderr, "Warning: option SCALM is not supported.\n");
                        }
                        else
                            fprintf(stderr, "\nError: Could not set 'scalm' variable\n");
                    }
                }
                /* from .options (will override the previous settings) */
                for (scan = options; scan; scan = scan->nextcard) {
                    char* tmpscale = strstr(scan->line, "scale=");
                    if (tmpscale) {
                        int err;
                        tmpscale = tmpscale + 6;
                        dscale = INPevaluate(&tmpscale, &err, 1);
                        if (err == 0) {
                            cp_vset("scale", CP_REAL, &dscale);
                            printf("option SCALE: Scale is set to %g for instance and model parameters\n", dscale);
                        }
                        else
                            fprintf(stderr, "\nError: Could not set 'scale' variable\n");
                    }
                    tmpscale = strstr(scan->line, "scalm=");
                    if (tmpscale) {
                        int err;
                        tmpscale = tmpscale + 6;
                        dscale = INPevaluate(&tmpscale, &err, 1);
                        if (err == 0) {
                            cp_vset("scalm", CP_REAL, &dscale);
                            fprintf(stderr, "Warning: option SCALM is not supported\n");
                        }
                        else
                            fprintf(stderr, "\nError: Could not set 'scalm' variable\n");
                    }
                }
            }

            /* Now expand subcircuit macros and substitute numparams.*/
            if (!cp_getvar("nosubckt", CP_BOOL, NULL, 0))
                if ((deck->nextcard = inp_subcktexpand(deck->nextcard)) == NULL) {
                    line_free(realdeck, TRUE);
                    line_free(deck->actualLine, TRUE);
                    tfree(tt);
                    return 1;
                }

            /* replace agauss(x,y,z) in each b-line by suitable value, one for all */
            bool statlocal = cp_getvar("statlocal", CP_BOOL, NULL, 0);
            if (!statlocal) {
                static char* statfcn[] = { "agauss", "gauss", "aunif", "unif", "limit" };
                int ii;
                for (ii = 0; ii < 5; ii++)
                    eval_agauss(deck, statfcn[ii]);
            }

            /* Scan the deck again, now also adding .save commands to wl_first */
            for (dd = deck->nextcard; dd; dd = dd->nextcard) {
                char* curr_line = dd->line;
                if (ciprefix(".save", curr_line)) {
                    wl_append_word(&wl_first, &end, copy(dd->line));
                    *curr_line = '*';
                }
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
                    struct card *tc = NULL;
                    fprintf(fdo, "**************** uncommented deck **************\n\n");
                    /* always print first line */
                    fprintf(fdo, "%6d  %6d  %s\n", deck->linenum_orig, deck->linenum, deck->line);
                    /* here without out-commented lines */
                    for (tc = deck->nextcard; tc; tc = tc->nextcard) {
                        if (*(tc->line) == '*')
                            continue;
                        fprintf(fdo, "%6d  %6d  %s\n", tc->linenum_orig, tc->linenum, tc->line);
                    }
                    fprintf(fdo, "\n****************** complete deck ***************\n\n");
                    /* now completely */
                    for (tc = deck; tc; tc = tc->nextcard)
                        fprintf(fdo, "%6d  %6d  %s\n", tc->linenum_orig, tc->linenum, tc->line);
                    fclose(fdo);
                }
                else
                    fprintf(stderr, "Warning: Cannot open file debug-out2.txt for saving debug info\n");
            }

            /* handle .if ... .elseif ... .else ... .endif statements. */
            dotifeval(deck);

            /* get csparams and create vectors, available
               in plot 'const' of a .control section */
            for (dd = deck; dd; dd = dd->nextcard) {
                if (ciprefix(".csparam", dd->line)) {
                    wordlist *wlist = NULL;
                    char *cstoken[3];
                    int i;
                    dd->line[0] = '*';
                    s = skip_ws(dd->line + 8);
                    while (s && *s) {
                        char* nexttoken = s;
                        cstoken[0] = gettok_char(&s, '=', FALSE, FALSE);
                        cstoken[1] = gettok_char(&s, '=', TRUE, FALSE);
                        cstoken[2] = gettok(&s);
                        /* guard against buggy input line */
                        if (!cstoken[0] || !cstoken[1] || !cstoken[2] || strchr(cstoken[2],'=')) {
                            fprintf(stderr, "Warning: bad csparam definition, %s skipped!\n", nexttoken);
                            fprintf(stderr, "    See line %d, .%s\n\n", dd->linenum, dd->line + 1);
                            tfree(cstoken[0]);
                            tfree(cstoken[1]);
                            tfree(cstoken[2]);
                            break;
                        }
                        for (i = 3; --i >= 0; ) {
                            wlist = wl_cons(cstoken[i], wlist);
                        }
                        com_let(wlist);
                        wl_free(wlist);
                        wlist = NULL;
                    }
                }
            }

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

            /* replace agauss(x,y,z) in each b-line by suitable value */
            /* FIXME: This is for the local param setting (not yet implemented in
            inp_fix_agauss_in_param() for model parameters according to HSPICE manual)*/
            if (statlocal) {
                static char *statfcn[] = {"agauss", "gauss", "aunif", "unif", "limit"};
                int ii;
                for (ii = 0; ii < 5; ii++)
                    eval_agauss(deck, statfcn[ii]);
            }
            /* If user wants all currents saved (.options savecurrents), add .save 
            to wl_first with all terminal currents available on selected devices */
            wl_first = inp_savecurrents(deck, options, wl_first, controls);

            /* Circuit is flat, all numbers expanded.
               So again try to remove unused MOS models.
               All binning models are still here when w or l have been
               determined by an expression. */
           if (newcompat.hs || newcompat.spe)
              rem_unused_mos_models(deck->nextcard);

            /* now load deck into ft_curckt -- the current circuit. */
            if(inp_dodeck(deck, tt, wl_first, FALSE, options, filename) != 0)
                return 1;

            if (ft_curckt) {
                ft_curckt->devtlist = devtlist;
                ft_curckt->modtlist = modtlist;
            }

            /* inp_dodeck did take ownership */
            tt = NULL;
            options = NULL;

        }     /*  if (deck->nextcard) */

        /* look for and set temperature; also store param and .meas statements in circuit struct */
        if (ft_curckt) {
            ft_curckt->ci_param = NULL;
            ft_curckt->ci_meas  = NULL;
        }

        for (dd = deck; dd; dd = dd->nextcard) {
            /* first line is title line, skip it */
            if (deck == dd) {
                prev_card = dd;
                continue;
            }
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

            /* remove the .measure cards from the deckand store them in ft_curckt->ci_meas */
            if (ciprefix(".meas", dd->line)) {
                if (cp_getvar("autostop", CP_BOOL, NULL, 0)) {
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
            /* get temp from deck .temp 125 or .temp=125 */
            if (ciprefix(".temp", dd->line)) {
                s = skip_ws(dd->line + 5);
                if (*s == '=') {
                    s = skip_ws(s + 1);
                }
                if (temperature) {
                    txfree(temperature);
                }
                temperature = copy(s);
                *(dd->line) = '*';
            }
            prev_card = dd;
        }  //end of for-loop

        /* set temperature, if defined, to new value.
           cp_vset will set the variable "temp" and also set CKTtemp,
           so we can do it only here because the circuit has to be already existing */
        if (temperature) {
            char *endstr;
            temperature_value = strtod(temperature, &endstr);
            /* number strngs from numparam may contain trailing spaces */
            endstr = skip_ws(endstr);
            /* if endstr contains characters, temperature has not been a pure number string */
            if (*endstr != '\0') {
                fprintf(stderr, "Warning: Could not set temperature to %s\n   Set to default 27 C instead.\n", temperature);
                temperature_value = 27;
            }
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
                struct card *tc = NULL;
                fprintf(fdo, "**************** uncommented deck **************\n\n");
                /* always print first line */
                fprintf(fdo, "%6d  %6d  %s\n", deck->linenum_orig, deck->linenum, deck->line);
                /* here without out-commented lines */
                for (tc = deck->nextcard; tc; tc = tc->nextcard) {
                    if (*(tc->line) == '*')
                        continue;
                    fprintf(fdo, "%6d  %6d  %s\n", tc->linenum_orig, tc->linenum, tc->line);
                }
                fprintf(fdo, "\n****************** complete deck ***************\n\n");
                /* now completely */
                for (tc = deck; tc; tc = tc->nextcard)
                    fprintf(fdo, "%6d  %6d  %s\n", tc->linenum_orig, tc->linenum, tc->line);
                fclose(fdo);
            }
            else
                fprintf(stderr, "Warning: Cannot open file debug-out3.txt for saving debug info\n");
        }

        /* Remove comment lines 
        if (newcompat.hs || newcompat.spe) {
            struct card *prev, *fcard, *tmpdeck;
            prev = deck;
            tmpdeck = deck->nextcard;
            for (fcard = tmpdeck; fcard; fcard = fcard->nextcard) {
                if (*(prev->nextcard->line) == '*') {
                    struct card* tmpcard = fcard->nextcard;
                    line_free_x(prev->nextcard, FALSE);
                    fcard = prev->nextcard = tmpcard;
                }
                prev = fcard;
            }
        }*/

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

        /* statistics for preparing the deck */
        endTime = seconds();
        if (ft_curckt) {
            ft_curckt->FTEstats->FTESTATnetLoadTime = loadTime;
            ft_curckt->FTEstats->FTESTATnetPrepTime = seconds() - startTime;
        }

        /* in shared ngspice controls a execute in the primary thread, typically
           before the background thread has finished. This leads to premature execution
           of commands. Thus this is delegated to a function using a third thread, that
           only starts when the background thread has finished (sharedspice.c).*/
#ifdef SHARED_MODULE
        for (wl = controls; wl; wl = wl->wl_next){
#ifdef OSDI
            inputdir = dir_name;
#endif
            if (cp_getvar("controlswait", CP_BOOL, NULL, 0)) {
                if (wl)
                    exec_controls(wl_copy(wl));
                break;
            }
            else
                cp_evloop(wl->wl_word);
        }
#else
        for (wl = controls; wl; wl = wl->wl_next){
#ifdef OSDI
            inputdir = dir_name;
#endif
            cp_evloop(wl->wl_word);
        }
#endif
        wl_free(controls);
#ifdef OSDI
            inputdir = NULL;
#endif
    }

    /* Now reset everything.  Pop the control stack, and fix up the IO
     * as it was before the source.  */
    cp_popcontrol();

    cp_curin = lastin;
    cp_curout = lastout;
    cp_curerr = lasterr;

    tfree(tt);
    tfree(dir_name);



    return 0;
}


/* This routine is cut in half here because com_rset has to do what
 * follows also. End is the list of commands we execute when the job
 * is finished: we only bother with this if we might be running in
 * batch mode, since it isn't much use otherwise.  */
/*------------------------------------------------------------------
 * It appears that inp_dodeck adds the circuit described by *deck
 * to the current circuit (ft_curckt).
 *-----------------------------------------------------------------*/
int
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
    INPtables *tab = NULL;
    struct variable *eev = NULL;
    bool noparse;
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
    noparse = cp_getvar("noparse", CP_BOOL, NULL, 0);

    /* Read the options, create variables and store them
       in ftcurckt->ci_vars */
    if (!noparse) {
        char* s;
        bool ii;
        wordlist* wl;
        struct card* opt_beg = options;
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
                break;
            case CP_STRING:
                break;
            default: {
                fprintf(stderr, "ERROR: wrong format in option %s!\n", eev->va_name);
                fprintf(stderr, "   Aborting...\n");
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
        /* if .probe, rename the current measurement node vcurr_ */
        if (cp_getvar("probe_is_given", CP_BOOL, NULL, 0)) {
            modprobenames(tab);
        }
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
        if (cp_getvar("warn", CP_NUM, &warn, 0))
            ckt->CKTsoaCheck = warn;
        else
            ckt->CKTsoaCheck = 0;

        if (cp_getvar("maxwarns", CP_NUM, &maxwarns, 0))
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
            fflush(stdout);
            do {
                q = strchr(p, '\n');
                if (q)
                    *q = '\0';

                if (p == dd->error) {
                    if (strstr(dd->line, ".model"))
                        fprintf(stderr, "Warning: Model issue on line %d :\n  %.*s ...\n%s\n",
                            dd->linenum_orig, 72, dd->line, dd->error);
                    else if (dd->linenum_orig == 0) {
                        fprintf(stderr, "Error on line:\n  %s\n%s\n",
                                   dd->line, dd->error);
                        have_err = TRUE;
                        return 1;
                    }
                    else {
                        fprintf(stderr, "Error on line %d or its substitute:\n  %s\n%s\n",
                                   dd->linenum_orig, dd->line, dd->error);
                        have_err = TRUE;
                        return 1;
                    }
                    if (ft_stricterror)
                        controlled_exit(EXIT_BAD);
                } else {
                    fprintf(stderr, "%s\n", p);
                }
                if (q)
                    *q++ = '\n';
                p = q;
            } while (p && *p);
            fprintf(stderr, "\n");
        }  /* end  if (dd->error) */

    }   /* for (dd = deck; dd; dd = dd->nextcard) */

    /* Stop here and exit if error occurred in batch mode */
    if (have_err && ft_batchmode) {
        fprintf(stderr, "\nngspice stopped due to error, no simulation run!\n");
        controlled_exit(EXIT_BAD);
    }

    /* Only print out netlist if brief is FALSE */
    if (!cp_getvar("brief", CP_BOOL, NULL, 0)) {
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
    ct->ci_mcdeck = mc_deck;
    ct->ci_options = options;
    if (deck && deck->actualLine)
        ct->ci_origdeck = deck->actualLine;
    else
        ct->ci_origdeck = ct->ci_deck;
    ct->ci_ckt = ckt;             /* attach the input ckt to the list of circuits */
    ct->ci_symtab = tab;
    ct->ci_inprogress = FALSE;
    ct->ci_runonce = FALSE;
    ct->ci_commands = end;
    ct->ci_dicos = nupa_add_dicoslist();
    /* prevent false reads in multi-threaded ngshared */
#ifndef SHARED_MODULE    
    if (reuse)
        tfree(ct->ci_filename);
#endif
    ct->ci_filename = copy(filename);

    /* load the optran data, if provided by .spiceinit or spinit.
       Return immediately, if optran is not selected.*/
    com_optran(NULL);

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
    return 0;
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

    if (!cp_getvar("interactive", CP_BOOL, NULL, 0)) {
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
    if (fgets(buf, BSIZE_SP, stdin) == (char *) NULL || buf[0] != 'n') {
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

    if (!ft_curckt) {
        fprintf(stderr, "Warning: No circuit loaded!\n");
        fprintf(stderr, "    Command 'alterparam' ignored\n");
        return;
    }
    if (!ft_curckt->ci_mcdeck) {
        fprintf(cp_err, "Error: No internal deck available\n");
        fprintf(stderr, "    Command 'alterparam' ignored\n");
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
    for (dd = ft_curckt->ci_mcdeck->nextcard; dd; dd = dd->nextcard) {
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
                        for (xx = ft_curckt->ci_mcdeck->nextcard; xx; xx = xx->nextcard) {
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

    if (cp_getvar("editor", CP_STRING, buf2, sizeof(buf2))) {
        editor = buf2;
    } else {
        if ((editor = getenv("EDITOR")) == NULL) {
            if (Def_Editor && *Def_Editor)
                editor = Def_Editor;
            else
                editor = "/usr/bin/vi";
        }
    }
    int len = snprintf(buf, BSIZE_SP - 1, "%s %s", editor, filename);
    if (len > BSIZE_SP - 1)
        fprintf(stderr, "Error: the filename is probably tuncated\n");
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

    if (wl == NULL)
        return;

    inter = cp_interactive;
    cp_interactive = FALSE;

    firstfile = wl->wl_word;

    if (wl->wl_next) {
        /* There are several files -- put them into a temp file  */
        tempfile = smktemp("sp");
        if ((fp = inp_pathopen(tempfile, "w+")) == NULL) {
            perror(tempfile);
            fprintf(cp_err, "    Simulation interrupted due to error!\n\n");
            cp_interactive = TRUE;
            /* If we cannot open the temporary file, stop all further command execution */
#ifdef SHARED_MODULE
            controlled_exit(1);
#else
            if (cp_getvar("interactive", CP_BOOL, NULL, 0))
                cp_evloop(NULL);
            else
                controlled_exit(1);
#endif
        }
        while (wl) {
            if ((tp = inp_pathopen(wl->wl_word, "r")) == NULL) {
                fprintf(cp_err, "Command 'source' failed:\n");
                perror(wl->wl_word);
                fprintf(cp_err, "    Simulation interrupted due to error!\n\n");
                fclose(fp);
                cp_interactive = TRUE;
                unlink(tempfile);
                /* If we cannot source the file, stop all further command execution */
#ifdef SHARED_MODULE
                controlled_exit(1);
#else
                if (cp_getvar("interactive", CP_BOOL, NULL, 0))
                    cp_evloop(NULL);
                else
                    controlled_exit(1);
#endif
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
        fprintf(cp_err, "Command 'source' failed:\n");
        perror(wl->wl_word);
        fprintf(cp_err, "    Simulation interrupted due to error!\n\n");
        cp_interactive = TRUE;
        /* If we cannot source the file, stop all further command execution */
#ifdef SHARED_MODULE
        controlled_exit(1);
#else
        if (cp_getvar("interactive", CP_BOOL, NULL, 0))
            cp_evloop(NULL);
        else
            controlled_exit(1);
#endif
        return;
    }

    /* Don't print the title if this is a spice initialisation file. */
    if (ft_nutmeg || substring(INITSTR, owl->wl_word) || substring(ALT_INITSTR, owl->wl_word))
        inp_spsource(fp, TRUE, tempfile ? NULL : wl->wl_word, FALSE);
    else {
#ifdef HAS_WINGUI
        /* set the source window */
        SetSource(wl->wl_word);
#endif
        /* Save path name for use in XSPICE fopen_with_path() */
        if (Infile_Path)
            tfree(Infile_Path);
        Infile_Path = ngdirname(firstfile);
        if (inp_spsource(fp, FALSE, tempfile ? NULL : wl->wl_word, FALSE) != 0) {
            fprintf(stderr, "    Simulation interrupted due to error!\n\n");
        }
    }

    cp_interactive = inter;
    if (tempfile)
        unlink(tempfile);
}


void inp_source(const char *file)
{
    /* This wordlist is special in that nothing in it should be freed --
     * the file name word is "borrowed" from the argument to file and
     * the wordlist is allocated on the stack. */
    static struct wordlist wl = { NULL, NULL, NULL };
    wl.wl_word = (char *) file;
    com_source(&wl);
}


/* check the input deck (after inpcom and numparam extensions)
   for linear elements. If only linear elements are found,
   ckt->CKTisLinear is set to 1. Return immediately if a first
   non-linear element is found. */
static void cktislinear(CKTcircuit *ckt, struct card *deck)
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
                case 'k':
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
 * or receiving array from external caller. Array is created whenever
 * a new deck is started. Last line of the array has to get the string ".end" */
char **circarray;


void create_circbyline(char *line, bool reset, bool lastline)
{
    static unsigned int linec = 0;
    static unsigned int n_elem_alloc = 0;

    if (reset) {
        linec = 0;
        n_elem_alloc = 0;
        tfree(circarray);
    }

    /* Ensure up to 2 cards can be added */
    if (n_elem_alloc < linec + 2) {
        n_elem_alloc = n_elem_alloc == 0 ? 256 : 2 * n_elem_alloc;
        circarray = TREALLOC(char *, circarray, n_elem_alloc);
    }

    char *p_src = skip_ws(line);
    /* An empty line may have been received. Skip it. */
    if (*p_src == '\0') {
        return;
    }
    /* Remove any leading whitespace by shifting */
    if (p_src != line) {
        char *p_dst = line;
        char ch_cur;
        do {
            ch_cur = *p_dst++ = *p_src++;
        }
        while (ch_cur != '\0');
    }
    if (ft_ngdebug) {
        if (linec == 0)
            fprintf(stdout, "**** circuit array: circuit netlist sent to shared ngspice ****\n");
        fprintf(stdout, "%d   %s\n", linec, line);
    }
    circarray[linec++] = line; /* add card to deck */

    /* If the card added ended the deck, send it for processing and
     * free the deck. The card allocations themselves will be freed
     * elsewhere */
    if (ciprefix(".end", line) && (line[4] == '\0' || isspace_c(line[4]))) {
        circarray[linec] = NULL; /* terminate the deck */
        inp_spsource((FILE *) NULL, FALSE, NULL, TRUE); /* process */
        tfree(circarray); /* set to empty */
        linec = 0;
        n_elem_alloc = 0;
    }
    /* If the .end statement is missing */
    else if (lastline) {
        fprintf(stderr, "Error: .end statement is missing in netlist!\n");
    }
} /* end of function create_circbyline */



/* fcn called by command 'circbyline' */
void com_circbyline(wordlist *wl)
{
    /* undo the automatic wordline creation.
       wl_flatten allocates memory on the heap for each newline.
       This memory will be released line by line in inp_source(). */

    char *newline = wl_flatten(wl);
    create_circbyline(newline, FALSE, FALSE);
}

/* handle .if('expr') ... .elseif('expr') ... .else ... .endif statements.
   numparam has evaluated .if('boolean expression') to
   .if (   1.000000000e+000  ) or .elseif (   0.000000000e+000  ).
   Evaluation is done recursively, starting with .IF, ending with .ENDIF*/
static void recifeval(struct card *pdeck)
{
    struct card *nd;
    int iftrue = 0, elseiftrue = 0, elsetrue = 0, iffound = 0, elseiffound = 0, elsefound = 0;
    char *t;
    char *s = t = pdeck->line;
    /* get parameter to .if */
    elsefound = 0;
    elseiffound = 0;
    iffound = 1;
    *t = '*';
    s = pdeck->line + 3;
    iftrue = atoi(s);
    nd = pdeck->nextcard;

    while(nd) {
        s = nd->line;
        if (ciprefix(".if", nd->line))
            recifeval(nd);
        else if (ciprefix(".elseif", nd->line) && elseiftrue == 0) {
            elsefound = 0;
            elseiffound = 1;
            iffound = 0;
            *s = '*';
            if (!iftrue) {
                s = nd->line + 7;
                elseiftrue = atoi(s);
            }
        }
        else if (ciprefix(".else", nd->line)) {
            elsefound = 1;
            elseiffound = 0;
            iffound = 0;
            if (!iftrue && !elseiftrue)
                elsetrue = 1;
            *s = '*';
        }
        else if (ciprefix(".endif", nd->line)) {
            elsefound = elseiffound = iffound = 0;
            elsetrue = elseiftrue = iftrue = 0;
            *s = '*';
            return;
        }
        else {
            if (iffound && !iftrue) {
                *s = '*';
            }
            else if (elseiffound && !elseiftrue) {
                *s = '*';
            }
            else if (elsefound && !elsetrue) {
                *s = '*';
            }
        }
        nd = nd->nextcard;
    }
}

/* Scan through all lines of the deck */
static void dotifeval(struct card *deck)
{
    struct card *dd;
    char *dottoken;
    char *s, *t;

    /* skip the first line (title line) */
    for (dd = deck->nextcard; dd; dd = dd->nextcard) {

        s = t = dd->line;

        if (*s == '*')
            continue;

        dottoken = gettok(&t);
        /* find '.if', the starter of any .if --- .endif clause, and call the recursive evaluation.
           recifeval() returns when .endif is found */
        if (cieq(dottoken, ".if")) {
            recifeval(dd);
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

static int inp_parse_temper(struct card *card, struct pt_temper **modtlist_p,
            struct pt_temper **devtlist_p)
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



void inp_evaluate_temper(struct circ *circ)
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

    /* Step through the nodes of the linked list at circ->modtlist */
    for(d = circ->modtlist; d; d = d->next) {
        char *name = d->wl->wl_word;
        INPretrieve(&name, circ->ci_symtab);
        /* only evaluate models which have been entered into the
           hash table ckt->MODnameHash */
        if (ft_sim->findModel (circ->ci_ckt, name) == NULL) {
            continue;
        }

        IFeval((IFparseTree *) d->pt, 1e-12, &result, NULL, NULL);
        if (d->wlend->wl_word)
            tfree(d->wlend->wl_word);
        d->wlend->wl_word = tprintf("%g", result);
        com_altermod(d->wl);
    }
} /* end of funtion inp_evaluate_temper */



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
    if (!p) {
        p = wl_cons(copy(".save all"), NULL);
    }
    else {
        p = NULL;
    }

    /* Scan the deck for devices with their terminals.
     * We currently serve bipolars, resistors, MOS1, capacitors, inductors,
     * controlled current sources. Others may follow.
     */
    for (deck = deck->nextcard; deck; deck = deck->nextcard) {
        char *newline, *devname, *devline = deck->line;

        switch (devline[0]) {
        case 'm':
            devname = gettok(&devline);
            if (strstr(options->line, "savecurrents_bsim3"))
                newline = tprintf(".save @%s[id] @%s[ibd] @%s[ibs]",
                              devname, devname, devname);
            else if (strstr(options->line, "savecurrents_bsim4"))
                newline = tprintf(".save @%s[id] @%s[ibd] @%s[ibs] @%s[isub] @%s[igidl] @%s[igisl] @%s[igs] @%s[igb] @%s[igd] @%s[igcs] @%s[igcd]",
                              devname, devname, devname, devname, devname, devname, devname, devname, devname, devname, devname);
            else if (strstr(options->line, "savecurrents_mos1"))
                newline = tprintf(".save @%s[id] @%s[is] @%s[ig] @%s[ib] @%s[ibd] @%s[ibs]",
                              devname, devname, devname, devname, devname, devname);
            else
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


static double
agauss(double nominal_val, double abs_variation, double sigma)
{
    double stdvar;
    stdvar = abs_variation / sigma;
    return (nominal_val + stdvar * gauss1());
}


static double
gauss(double nominal_val, double rel_variation, double sigma)
{
    double stdvar;
    stdvar = nominal_val * rel_variation / sigma;
    return (nominal_val + stdvar * gauss1());
}


static double
unif(double nominal_val, double rel_variation)
{
    return (nominal_val + nominal_val * rel_variation * drand());
}


static double
aunif(double nominal_val, double abs_variation)
{
    return (nominal_val + abs_variation * drand());
}


static double
limit(double nominal_val, double abs_variation)
{
    return (nominal_val + (drand() > 0 ? abs_variation : -1. * abs_variation));
}


/* Second step to enable functions agauss, gauss, aunif, unif, limit
 * in professional parameter decks:
 * agauss has been preserved by replacement operation of .func
 * (function inp_fix_agauss_in_param() in inpcom.c).
 * After subcircuit expansion, agauss may be still existing in b-lines,
 * however agauss does not exist in the B source parser, and it would
 * not make sense in adding it there, because in each time step a different
 * return from agauss would result.
 * So we have to do the following in each B-line:
 * check for agauss(x,y,z), and replace it by a suitable return value
 * of agauss()
 * agauss  in .param lines has been treated already
 */
static void
eval_agauss(struct card *deck, char *fcn)
{
    struct card *card;
    double x, y, z, val;
    int skip_control = 0;

    card = deck->nextcard; /* skip title line */
    for (; card; card = card->nextcard) {

        char *ap, *curr_line = card->line;

        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", curr_line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", curr_line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }

        if (*curr_line != 'b')
            continue;

        while ((ap = search_identifier(curr_line, fcn, curr_line)) != NULL) {
            char *lparen, *begstr, *contstr = NULL, *new_line, *midstr;
            char *tmp1str, *tmp2str, *delstr;
            int nerror = 0;

            begstr = copy_substring(curr_line, ap);
            lparen = strchr(ap, '(');
            tmp1str = midstr = gettok_char(&lparen, ')', FALSE, TRUE);
            if (!tmp1str) {
                fprintf(cp_err, "ERROR: Incomplete function %s in line %s\n", fcn, curr_line);
                tfree(begstr);
                return;
            }
            contstr = copy(lparen + 1);
            tmp1str++; /* skip '(' */
            /* find the parameters, ignore ( ) , */
            delstr = tmp2str = gettok_np(&tmp1str);
            if (!tmp2str) {
                fprintf(cp_err, "ERROR: Incomplete function %s in line %s\n", fcn, curr_line);
                tfree(begstr);
                tfree(contstr);
                return;
            }
            x = INPevaluate(&tmp2str, &nerror, 1);
            tfree(delstr);
            delstr = tmp2str = gettok_np(&tmp1str);
            if (!tmp2str) {
                fprintf(cp_err, "ERROR: Incomplete function %s in line %s\n", fcn, curr_line);
                tfree(begstr);
                tfree(contstr);
                return;
            }
            y = INPevaluate(&tmp2str, &nerror, 1);
            tfree(delstr);
            if (cieq(fcn, "agauss")) {
                delstr = tmp2str = gettok_np(&tmp1str);
                z = INPevaluate(&tmp2str, &nerror, 1);
                tfree(delstr);
                val = agauss(x, y, z);
            }
            else if (cieq(fcn, "gauss")) {
                delstr = tmp2str = gettok_np(&tmp1str);
                z = INPevaluate(&tmp2str, &nerror, 1);
                tfree(delstr);
                val = gauss(x, y, z);
            }
            else if (cieq(fcn, "aunif")) {
                val = aunif(x, y);
            }
            else if (cieq(fcn, "unif")) {
                val = unif(x, y);
            }
            else if (cieq(fcn, "limit")) {
                val = limit(x, y);
            }
            else {
                fprintf(cp_err, "ERROR: Unknown function %s, cannot evaluate\n", fcn);
                tfree(begstr);
                tfree(contstr);
                tfree(midstr);
                return;
            }

            new_line = tprintf("%s%g%s", begstr, val, contstr);
            tfree(card->line);
            curr_line = card->line = new_line;
            tfree(begstr);
            tfree(contstr);
            tfree(midstr);
        }
    }
}

struct mlist {
    struct card* mod;
    struct card* prevmod;
    struct card* prevcard;
    char* mname;
    float wmin;
    float wmax;
    float lmin;
    float lmax;
    struct mlist* nextm;
    bool used;
    bool checked;
};

/* Finally get rid of unused MOS models */
static void rem_unused_mos_models(struct card* deck) {
    struct card *tmpc, *tmppc = NULL;
    struct mlist* modellist = NULL, *tmplist;
    double scale;
    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;
    /* the old way to remove unused models */
    struct nscope* root = inp_add_levels(deck);
    comment_out_unused_subckt_models(deck);
    inp_rem_unused_models(root, deck);
    inp_rem_levels(root);
    /* remove unused binning models */
    for (tmpc = deck; tmpc; tmppc = tmpc, tmpc = tmpc->nextcard) {
        char* curr_line;
        char* nline = curr_line = tmpc->line;
        if (ciprefix(".model", nline)) {
            float fwmin, fwmax, flmin, flmax;
            char* wmin = strstr(curr_line, " wmin=");
            if (wmin) {
                int err;
                wmin = wmin + 6;
                wmin = skip_ws(wmin);
                fwmin = (float)INPevaluate(&wmin, &err, 0);
                if (err) {
                    continue;
                }
            }
            else {
                continue;
            }
            char* wmax = strstr(curr_line, " wmax=");
            if (wmax) {
                int err;
                wmax = wmax + 6;
                wmax = skip_ws(wmax);
                fwmax = (float)INPevaluate(&wmax, &err, 0);
                if (err) {
                    continue;
                }
            }
            else {
                continue;
            }

            char* lmin = strstr(curr_line, " lmin=");
            if (lmin) {
                int err;
                lmin = lmin + 6;
                lmin = skip_ws(lmin);
                flmin = (float)INPevaluate(&lmin, &err, 0);
                if (err) {
                    continue;
                }
            }
            else {
                continue;
            }
            char* lmax = strstr(curr_line, " lmax=");
            if (lmax) {
                int err;
                lmax = lmax + 6;
                lmax = skip_ws(lmax);
                flmax = (float)INPevaluate(&lmax, &err, 0);
                if (err) {
                    continue;
                }
            }
            else {
                continue;
            }

            nline = nexttok(nline);
            char* modname = gettok(&nline);
            struct mlist* newm = TMALLOC(struct mlist, 1);
            newm->mname = modname;
            newm->mod = tmpc;
            newm->prevmod = tmppc;
            newm->wmin = newm->wmax = newm->lmin = newm->lmax = 0.;
            newm->nextm = NULL;
            newm->used = FALSE;
            newm->checked = FALSE;
            newm->lmax = flmax;
            newm->lmin = flmin;
            newm->wmax = fwmax;
            newm->wmin = fwmin;

            if (!modellist) {
                modellist = newm;
            }
            else {
                struct mlist* tmpl = modellist;
                modellist = newm;
                modellist->nextm = tmpl;
            }
            modellist->prevcard = tmppc;
        }
    }
    for (tmpc = deck; tmpc; tmpc = tmpc->nextcard) {
        char* curr_line = tmpc->line;
        /* We only look for MOS devices and extract W, L, nf, and wnflag */
        if (*curr_line == 'm') {
            float w = 0., l = 0., nf = 1., wnf = 1.;
            int wnflag = 0;
            char* wstr = strstr(curr_line, " w=");
            if (wstr) {
                int err;
                wstr = wstr + 3;
                wstr = skip_ws(wstr);
                w = (float)INPevaluate(&wstr, &err, 0);
                if (err) {
                    continue;
                }
            }
            char* lstr = strstr(curr_line, " l=");
            if (lstr) {
                int err;
                lstr = lstr + 3;
                lstr = skip_ws(lstr);
                l = (float)INPevaluate(&lstr, &err, 0);
                if (err) {
                    continue;
                }
            }
            char* nfstr = strstr(curr_line, " nf=");
            if (nfstr) {
                int err;
                nfstr = nfstr + 4;
                nfstr = skip_ws(nfstr);
                nf = (float)INPevaluate(&nfstr, &err, 0);
                if (err) {
                    continue;
                }
            }
            char* wnstr = strstr(curr_line, " wnflag=");
            if (wnstr) {
                int err;
                wnstr = wnstr + 8;
                wnstr = skip_ws(wnstr);
                wnf = (float)INPevaluate(&wnstr, &err, 0);
                if (err) {
                    continue;
                }
            }
            if (!cp_getvar("wnflag", CP_NUM, &wnflag, 0)) {
                if (newcompat.spe || newcompat.hs)
                    wnflag = 1;
                else
                    wnflag = 0;
            }

            nf = (float)wnflag * wnf > 0.5f ? nf : 1.f;
            w = w / nf;

            /* what is the device's model name? */
            char* mname = nexttok(curr_line);
            int nonodes = 4; /* FIXME: this is a hack! How to really detect the number of nodes? */
            int jj;
            for (jj = 0; jj < nonodes; jj++) {
                mname = nexttok(mname);
            }
            mname = gettok(&mname);
            /* We now check all models */
            for (tmplist = modellist; tmplist; tmplist = tmplist->nextm) {
                if (strstr(tmplist->mname, mname)) {
                    float ls = l * (float)scale;
                    float ws = w * (float)scale;
                    if (tmplist->lmin <= ls && tmplist->lmax >= ls && tmplist->wmin <= ws && tmplist->wmax >= ws)
                        tmplist->used = TRUE;
                    else
                        tmplist->checked = TRUE;
                }
                else {
                    tmplist->checked = TRUE;
                }
            }
            tfree(mname);
        }
    }

    /* Delete the models that have been checked, but are unused */
    for (tmplist = modellist; tmplist; tmplist = tmplist->nextm) {
        if (tmplist->checked && !tmplist->used) {
            if (tmplist->prevcard == NULL) {
                struct card* tmpcard = tmplist->mod;
                tmplist->mod = tmplist->mod->nextcard;
                line_free_x(tmpcard, FALSE);
            }
            else {
                struct card* tmpcard = tmplist->prevcard;
                tmpcard->nextcard = tmplist->mod->nextcard;
                line_free_x(tmplist->mod, FALSE);
            }
        }
    }
    /* Remove modellist */
    while (modellist) {
        struct mlist* tlist = modellist->nextm;
        tfree(modellist->mname);
        tfree(modellist);
        modellist = tlist;
    }
}
