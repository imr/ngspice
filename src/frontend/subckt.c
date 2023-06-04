/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000 AlansFixes
**********/

/*------------------------------------------------------------------------------
 * encapsulated string assembly in translate() and finishLine()
 *   this string facility (bxx_buffer) mainly abstracts away buffer allocation.
 *   this fixes a buffer overflow in finishLine, caused by lengthy descriptions
 *   of the kind:
 *     B1  1 2  I=v(1)+v(2)+v(3)+...
 * Larice, 22nd Aug 2009
 *----------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
 * Added changes supplied by by H.Tanaka with some tidy up of comments, debug
 * statements, and variables. This fixes a problem with nested .subsck elements
 * that accessed .model lines. Code not ideal, but it seems to work okay.
 * Also took opportunity to tidy a few other items (unused variables etc.), plus
 * fix a few spelling errors in the comments, and a memory leak.
 * SJB 25th March 2005
 *----------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
 * re-written by SDB during 4.2003 to enable SPICE2 POLY statements to be processed
 * properly.  This is particularly important for dependent sources, whose argument
 * list changes when POLY is used.
 * Major changes include:
 * -- Added lots of comments which (hopefully) elucidate the steps taken
 *    by the program during its processing.
 * -- Re-wrote translate, which does the processing of each card.
 * Please direct comments/questions/complaints to Stuart Brorson:
 * mailto:sdb@cloud9.net
 *-----------------------------------------------------------------------------*/

/*
 * Expand subcircuits. This is very spice-dependent. Bug fixes by Norbert
 * Jeske on 10/5/85.
 */

/*======================================================================*
 * Expand all subcircuits in the deck. This handles imbedded .subckt
 * definitions. The variables substart, subend, and subinvoke can be used
 * to redefine the controls used. The syntax is invariant though.
 * NOTE: the deck must be passed without the title line.
 * What we do is as follows: first make one pass through the circuit
 * and collect all of the subcircuits. Then, whenever a line that starts
 * with 'x' is found, copy the subcircuit associated with that name and
 * splice it in. A few of the problems: the nodes in the spliced-in
 * stuff must be unique, so when we copy it, append "subcktname:" to
 * each node. If we are in a nested subcircuit, use foo:bar:...:node.
 * Then we have to systematically change all references to the renamed
 * nodes. On top of that, we have to know how many args BJT's have,
 * so we have to keep track of model names.
 *======================================================================*/
/*#define TRACE*/
#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteinp.h"
#include "ngspice/stringskip.h"
#include "ngspice/compatmode.h"

#include <stdarg.h>

#ifdef XSPICE
/* gtri - add - wbk - 11/9/90 - include MIF function prototypes */
#include "ngspice/mifproto.h"
/* gtri - end - wbk - 11/9/90 */
#endif

#include "subckt.h"
#include "variable.h"

#include "numparam/numpaif.h"

extern void line_free_x(struct card *deck, bool recurse);
extern int get_number_terminals(char* c);
extern void tprint(struct card* deck);

#define line_free(line, flag)                   \
    do {                                        \
        line_free_x(line, flag);                \
        line = NULL;                            \
    } while(0)



struct subs;
static struct card *doit(struct card *deck, wordlist *modnames);
static int translate(struct card *deck, char *formal, int flen, char *actual,
		char *scname, const char *subname, struct subs *subs,
		wordlist const *modnames);
struct bxx_buffer;
static void finishLine(struct bxx_buffer *dst, char *src, char *scname);
static int settrans(char *formal, int flen, char *actual, const char *subname);
static char *gettrans(const char *name, const char *name_end);
static int numnodes(const char *line, struct subs *subs, wordlist const *modnames);
static int  numdevs(char *s);
static wordlist *modtranslate(struct card *deck, char *subname, wordlist *new_modnames);
static void devmodtranslate(struct card *deck, char *subname, wordlist * const orig_modnames);
static int inp_numnodes(char c);

#define N_GLOBAL_NODES 1005

/*---------------------------------------------------------------------
 * table is used in settrans and gettrans -- it holds the netnames used
 * in the .subckt definition (t_old), and in the subcircuit invocation
 * (t_new).  The table ends when t_old is NULL.
 *--------------------------------------------------------------------*/
static struct tab {
    char *t_old;
    char *t_new;
} *table;


/*---------------------------------------------------------------------
 *  subs is the linked list which holds the .subckt definitions
 *  found during processing.
 *--------------------------------------------------------------------*/
struct subs {
    char *su_name;          /* The .subckt name. */
    char *su_args;          /* The .subckt arguments, space separated. */
    int su_numargs;
    struct card *su_def;    /* Pointer to the .subckt definition. */
    struct subs *su_next;
};


/* orig_modnames is the list of original model names, modnames is the
 * list of translated names (i.e. after subckt expansion)
 */

/* flag indicating use of the experimental numparams library */
static bool use_numparams = FALSE;

static char start[32], sbend[32], invoke[32], model[32];

static char *global_nodes[N_GLOBAL_NODES];
static int num_global_nodes;


static void
collect_global_nodes(struct card *c)
{
    num_global_nodes = 0;

    global_nodes[num_global_nodes++] = copy("0");

#ifdef XSPICE
    global_nodes[num_global_nodes++] = copy("null");
#endif

    for (; c; c = c->nextcard)
        if (ciprefix(".global", c->line)) {
            char *s = c->line;
            s = nexttok(s);
            while (*s) {
                if (num_global_nodes == N_GLOBAL_NODES) {
                    fprintf(stderr, "ERROR, N_GLOBAL_NODES overflow\n");
                    controlled_exit(EXIT_FAILURE);
                }
                char *t = skip_non_ws(s);
                global_nodes[num_global_nodes++] = copy_substring(s, t);
                s = skip_ws(t);
            }
            c->line[0] = '*'; /* comment it out */
        }

#ifdef TRACE
    {
        int i;
        printf("***Global node option has been found.***\n");
        for (i = 0; i < num_global_nodes; i++)
            printf("***Global node no.%d is %s.***\n", i, global_nodes[i]);
        printf("\n");
    }
#endif

}


static void
free_global_nodes(void)
{
    int i;
    for (i = 0; i < num_global_nodes; i++)
        tfree(global_nodes[i]);
    num_global_nodes = 0;
}


/*-------------------------------------------------------------------
  inp_subcktexpand is the top level function which translates
  .subckts into mainlined code.   Note that there are several things
  we need to do:  1. Find all .subckt definitions & stick them
  into a list.  2. Find all subcircuit invocations (refdes X)
  and replace them with the .subckt definition stored earlier.
  3. Do parameter substitution.

  The algorithm is as follows:
  1.  Define some aliases for .subckt, .ends, etc.
  2.  First numparam pass: substitute paramterized tokens by
  intermediate values 1000000001 etc.
  3.  Make a list global_nodes[] of global nodes
  4.  Clean up parens around netnames
  5.  Call doit, which does the actual translation.
  6.  Second numparam pass: Do final substitution
  7.  Check the results & return.
  inp_subcktexpand takes as argument a pointer to deck, and
  it returns a pointer to the same deck after the new subcircuits
  are spliced in.
  -------------------------------------------------------------------*/
struct card *
inp_subcktexpand(struct card *deck) {
    struct card *c;
    wordlist *modnames = NULL;

    if (!cp_getvar("substart", CP_STRING, start, sizeof(start)))
        strcpy(start, ".subckt");
    if (!cp_getvar("subend", CP_STRING, sbend, sizeof(sbend)))
        strcpy(sbend, ".ends");
    if (!cp_getvar("subinvoke", CP_STRING, invoke, sizeof(invoke)))
        strcpy(invoke, "x");
    if (!cp_getvar("modelcard", CP_STRING, model, sizeof(model)))
        strcpy(model, ".model");
    if (!cp_getvar("modelline", CP_STRING, model, sizeof(model)))
        strcpy(model, ".model");

/*    use_numparams = cp_getvar("numparams", CP_BOOL, NULL, 0); */

    use_numparams = TRUE;

    /*  deck has .control sections already removed, but not comments */
    if (use_numparams) {

#ifdef TRACE
        fprintf(stderr, "Numparams is processing this deck:\n");
        for (c = deck; c; c = c->nextcard)
            fprintf(stderr, "%3d:%s\n", c->linenum, c->line);
#endif

        nupa_signal(NUPADECKCOPY);
        /* get the subckt names from the deck */
        for (c = deck; c; c = c->nextcard) {    /* first Numparam pass */
            if (ciprefix(".subckt", c->line)) {
                nupa_scan(c);
            }
        }

        /* now copy instances */
        for (c = deck; c; c = c->nextcard) {  /* first Numparam pass */
            if (*(c->line) == '*') {
                continue;
            }
            c->line = nupa_copy(c);
        }

#ifdef TRACE
        fprintf(stderr, "Numparams transformed deck:\n");
        for (c = deck; c; c = c->nextcard)
            fprintf(stderr, "%3d:%s\n", c->linenum, c->line);
#endif

    }

    /* Get all the model names so we can deal with BJTs, etc.
     *  Stick all the model names into the doubly-linked wordlist modnames.
     */
    {
        int nest = 0;
        for (c = deck; c; c = c->nextcard) {

            if (ciprefix(".subckt", c->line))
                nest++;
            else if (ciprefix(".ends", c->line))
                nest--;
            else if (nest > 0)
                continue;

            if (ciprefix(model, c->line)) {
                char *s = nexttok(c->line);
                modnames = wl_cons(gettok(&s), modnames);
            }
        }
    }

#ifdef TRACE
    {
        wordlist *w;
        printf("Models found:\n");
        for (w = modnames; w; w = w->wl_next)
            printf("%s\n", w->wl_word);
    }
#endif

    /* Added by H.Tanaka to find global nodes */
    collect_global_nodes(deck);

    /* Let's do a few cleanup things... Get rid of ( ) around node lists... */
    for (c = deck; c; c = c->nextcard) {    /* iterate on lines in deck */

        char *s = c->line;

        if (*s == '*')           /* skip comment */
            continue;

        if (ciprefix(start, s)) {   /* if we find .subckt . . . */
#ifdef TRACE
            /* SDB debug statement */
            printf("In inp_subcktexpand, found a .subckt: %s\n", s);
#endif
            while (*s && *s != '(') /* search opening paren */
                s++;

            if (*s == '(') {
                int level = 0;
                do {
                    /* strip outer parens '(' ')', just the first pair */
                    if (*s == '('  &&  level++ == 0) {
                        *s = ' ';
                    }
                    if (*s == ')'  &&  --level == 0) {
                        *s = ' ';
                        break;
                    }
                } while(*s++);
            }
        } else if  (*s == '.') {
            continue;   /* skip .commands */
        } else {        /* any other line . . . */
            s = skip_non_ws(s);
            s = skip_ws(s);

            if (*s == '(') {
                int level = 0;
                do {
                    /* strip outer parens '(' ')', just the first pair, why ? */
                    if (*s == '('  &&  level++ == 0) {
                        *s = ' ';
                    }
                    if (*s == ')'  &&  --level == 0) {
                        *s = ' ';
                        break;
                    }
                } while(*s++);
            } /* if (*s == '(' . . . */
        } /* any other line */
    }   /*  for (c = deck . . . */

#ifdef TRACE
    /* SDB debug statement */
    printf("In inp_subcktexpand, about to call doit.\n");
#endif

    /* doit does the actual splicing in of the .subckt . . .  */
    deck = doit(deck, modnames);

    free_global_nodes();
    wl_free(modnames);

    /* Count numbers of line in deck after expansion */
    if (deck) {
        dynMaxckt = 0; /* number of lines in deck after expansion */
        for (c = deck; c; c = c->nextcard)
            dynMaxckt++;
    }

    /* Now check to see if there are still subckt instances undefined... */
    for (c = deck; c; c = c->nextcard)
        if (ciprefix(invoke, c->line)) {
            fprintf(cp_err, "Error: unknown subckt: %s\n", c->line);
            if (use_numparams)
                nupa_signal(NUPAEVALDONE);
            return NULL;
        }

    if (use_numparams) {
        /* the NUMPARAM final line translation pass */
        nupa_signal(NUPASUBDONE);
        for (c = deck; c; c = c->nextcard)
            /* 'param' .meas statements can have dependencies on measurement values */
            /* need to skip evaluating here and evaluate after other .meas statements */
            if (ciprefix(".meas", c->line) && strstr(c->line, "param")) {
                ;
            } else {
                nupa_eval(c);
            }

#ifdef TRACE
        fprintf(stderr, "Numparams converted deck:\n");
        for (c = deck; c; c = c->nextcard)
            fprintf(stderr, "%3d:%s\n", c->linenum, c->line);
#endif

        /*nupa_list_params(stdout);*/
        nupa_copy_inst_dico();
        nupa_signal(NUPAEVALDONE);
    }

    return (deck);  /* return the spliced deck.  */
}


static struct card *
find_ends(struct card *l)
{
    int nest = 1;

    while (l->nextcard) {

        if (ciprefix(sbend, l->nextcard->line)) /* found a .ends */
            nest--;
        else if (ciprefix(start, l->nextcard->line))  /* found a .subckt */
            nest++;

        if (!nest)
            break;

        l = l->nextcard;
    }

    return l;
}


#define MAXNEST 21
/*-------------------------------------------------------------------*/
/*  doit does the actual substitution of .subckts.                   */
/*  It takes two passes:  the first extracts .subckts                */
/*  and sticks pointer to them into the linked list sss.  It does    */
/*  the extraction recursively.  Then, it look for subcircuit        */
/*  invocations and substitutes the stored .subckt into              */
/*  the main circuit file.                                           */
/*  It takes as argument a pointer to the deck, and returns a        */
/*  pointer to the deck after the subcircuit has been spliced in.    */
/*-------------------------------------------------------------------*/
static struct card *
doit(struct card *deck, wordlist *modnames) {
    struct subs *sss = NULL;   /*  *sss temporarily hold decks to substitute  */
    int numpasses = MAXNEST;
    bool gotone;
    int error;

    /* Save all the old stuff... */
    struct subs *subs = NULL;
    wordlist *xmodnames = modnames;

#ifdef TRACE
    /* SDB debug statement */
    {
        struct card *c;
        printf("In doit, about to start first pass through deck.\n");
        for (c = deck; c; c = c->nextcard)
            printf("   %s\n", c->line);
    }
#endif

    {
        /* First pass: xtract all the .subckts and stick pointers to them into sss.  */

        struct card *c = deck;
        struct card *prev_of_c = NULL;

        while (c) {
            if (ciprefix(sbend, c->line)) {  /* if line == .ends  */
                fprintf(cp_err, "Error: misplaced %s line: %s\n", sbend,
                        c->line);
                return (NULL);
            }

            if (ciprefix(start, c->line)) {  /* if line == .subckt  */

                struct card *prev_of_ends = find_ends(c);
                struct card *ends = prev_of_ends->nextcard;

                if (!ends) {
                    fprintf(cp_err, "Error: no %s line.\n", sbend);
                    return (NULL);
                }

                /* c     points to the opening .subckt card */
                /* ends  points to the terminating .ends card */


                /*  Now put the .subckt definition found into sss  */

                {
                    char *s = c->line;

                    sss = TMALLOC(struct subs, 1);

                    s = nexttok(s);

                    sss->su_name = gettok(&s);
                    sss->su_args = copy(s);
                    sss->su_def = c->nextcard;

                    /* count the number of args in the .subckt line */
                    sss->su_numargs = 0;
                    for (;;) {
                        s = skip_ws(s);
                        if (*s == '\0')
                            break;
                        s = skip_non_ws(s);
                        sss->su_numargs ++;
                    }
                }

                /* push `sss' onto the `subs' list */
                sss->su_next = subs;
                subs = sss;

                /* cut the whole .subckt ... .ends sequence from the deck chain */

                line_free_x(c, FALSE); /* drop the .subckt card */
                c = ends->nextcard;

                if (prev_of_c)
                    prev_of_c->nextcard = c;
                else
                    deck = c;

                if (use_numparams == FALSE) {
                    line_free_x(ends, FALSE); /* drop the .ends card */
                    prev_of_ends->nextcard = NULL;
                } else {
                    ends->line[0] = '*'; /* comment the .ends card */
                    ends->nextcard = NULL;
                }

            } else {

                prev_of_c = c;
                c = c->nextcard;
            }
        }
    }


    /* At this point, sss holds the .subckt definition found, subs holds
     * all .subckt defs found, including this one
     */

    if (!subs)            /* we have found no subckts.  Just return.  */
        return (deck);

    /* Otherwise, expand sub-subcircuits recursively. */
    for (sss = subs; sss; sss = sss->su_next)  /* iterate through the list of subcircuits */
        if ((sss->su_def = doit(sss->su_def, modnames)) == NULL)
            return (NULL);

#ifdef TRACE
    /* SDB debug statement */
    {
        struct card *c;
        printf("In doit, about to start second pass through deck.\n");
        for (c = deck; c; c = c->nextcard)
            printf("   %s\n", c->line);
    }
#endif

    double scale;
    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    error = 0;
    /* Second pass: do the replacements. */
    do {                    /*  while (!error && numpasses-- && gotone)  */
        struct card *c = deck;
        struct card *prev_of_c = NULL;
        gotone = FALSE;
        for (; c; prev_of_c = c, c = c->nextcard) {
            if (ciprefix(invoke, c->line)) {  /* found reference to .subckt (i.e. component with refdes X)  */

                char *tofree, *tofree2, *s, *t;
                char *scname;

                gotone = TRUE;
                t = tofree = s = copy(c->line);       /*  s & t hold copy of component line  */

                /*  make scname point to first non-whitepace chars after refdes invocation
                 * e.g. if invocation is Xreference, *scname = reference
                 */
                tofree2 = scname = gettok(&s);
                /*scname += strlen(invoke);   */
                while ((*scname == ' ') || (*scname == '\t') || (*scname == ':'))
                    scname++;

                /*  Now set s to point to last non-space chars in line (i.e.
                 *   the name of the model invoked
                 */
                while (*s)
                    s++;
                s--;
                while ((*s == ' ') || (*s == '\t'))
                    *s-- = '\0';
                while ((*s != ' ') && (*s != '\t'))
                    s--;
                s++;

                /* iterate through .subckt list and look for .subckt name invoked */
                for (sss = subs; sss; sss = sss->su_next)
                    if (eq(sss->su_name, s))
                        break;


                /* At this point, sss points to the .subckt invoked,
                 * and scname points to the netnames
                 * involved.
                 */


                /* If no .subckt is found, don't complain -- this might be an
                 * instance of a subckt that is defined above at higher level.
                 */
                if (sss) {
//                    tprint(sss->su_def);
                    struct card *su_deck = inp_deckcopy(sss->su_def);
                    /* If we have modern PDKs, we have to reduce the amount of memory required.
                       We try to reduce the models to the one really used.
                       Otherwise su_deck is full of unused binning models.*/
                    if ((newcompat.hs || newcompat.spe) && c->w > 0 && c->l > 0) {
                        /* extract wmin, wmax, lmin, lmax */
                        struct card* new_deck = su_deck;
                        struct card* prev = NULL;
                        while (su_deck) {
                            if (!ciprefix(".model", su_deck->line)) {
                                prev = su_deck;
                                su_deck = su_deck->nextcard;
                                    continue;
                            }

                            char* curr_line = su_deck->line;
                            float fwmin, fwmax, flmin, flmax;
                            char *wmin = strstr(curr_line, " wmin=");
                            if (wmin) {
                                int err;
                                wmin = wmin + 6;
                                fwmin = (float)INPevaluate(&wmin, &err, 0);
                                if (err) {
                                    prev = su_deck;
                                    su_deck = su_deck->nextcard;
                                    continue;
                                }
                            }
                            else {
                                prev = su_deck;
                                su_deck = su_deck->nextcard;
                                continue;
                            }
                            char *wmax = strstr(curr_line, " wmax=");
                            if (wmax) {
                                int err;
                                wmax = wmax + 6;
                                fwmax = (float)INPevaluate(&wmax, &err, 0);
                                if (err) {
                                    prev = su_deck;
                                    su_deck = su_deck->nextcard;
                                    continue;
                                }
                            }
                            else {
                                prev = su_deck;
                                su_deck = su_deck->nextcard;
                                continue;
                            }

                            char* lmin = strstr(curr_line, " lmin=");
                            if (lmin) {
                                int err;
                                lmin = lmin + 6;
                                flmin = (float)INPevaluate(&lmin, &err, 0);
                                if (err) {
                                    prev = su_deck;
                                    su_deck = su_deck->nextcard;
                                    continue;
                                }
                            }
                            else {
                                prev = su_deck;
                                su_deck = su_deck->nextcard;
                                continue;
                            }
                            char* lmax = strstr(curr_line, " lmax=");
                            if (lmax) {
                                int err;
                                lmax = lmax + 6;
                                flmax = (float)INPevaluate(&lmax, &err, 0);
                                if (err) {
                                    prev = su_deck;
                                    su_deck = su_deck->nextcard;
                                    continue;
                                }
                            }
                            else {
                                prev = su_deck;
                                su_deck = su_deck->nextcard;
                                continue;
                            }

                            float csl = (float)scale * c->l;
                            /* scale by nf */
                            float csw = (float)scale * c->w / c->nf;
                            /*fprintf(stdout, "Debug: nf = %f\n", c->nf);*/
                            if (csl >= flmin && csl < flmax && csw >= fwmin && csw < fwmax) {
                                /* use the current .model card */
                                prev = su_deck;
                                su_deck = su_deck->nextcard;
                                continue;
                            }
                            else {
                                struct card* tmpcard = su_deck->nextcard;
                                line_free_x(prev->nextcard, FALSE);
                                su_deck = prev->nextcard = tmpcard;
                            }
                        }
                        su_deck = new_deck;
                    }

                    if (!su_deck) {
                        fprintf(stderr, "\nError: Could not find a model for device %s in subcircuit %s\n",
                            scname, sss->su_name);
                        controlled_exit(1);
                    }

                    struct card *rest_of_c = c->nextcard;

                    /* Now we have to replace this line with the
                     * macro definition.
                     */

                    /* Change the names of .models found in .subckts . . .  */
                    /* prepend the translated model names to the list `modnames' */
                    modnames = modtranslate(su_deck, scname, modnames);

                    t = nexttok(t);  /* Throw out the subcircuit refdes */

                    /* now invoke translate, which handles the remainder of the
                     * translation.
                     */
                    if (!translate(su_deck, sss->su_args, sss->su_numargs, t, scname, sss->su_name, subs, modnames))
                        error = 1;

                    /* Now splice the decks together. */

                    if (use_numparams == FALSE) {
                        line_free_x(c, FALSE); /* drop the invocation */
                        if (prev_of_c)
                            prev_of_c->nextcard = su_deck;
                        else
                            deck = su_deck;
                    } else {
                        c->line[0] = '*'; /* comment the invocation */
                        c->nextcard = su_deck;
                    }

                    c = su_deck;
                    while (c->nextcard)
                        c = c->nextcard;

                    c->nextcard = rest_of_c;
                }

                tfree(tofree);
                tfree(tofree2);
            }
        }
    } while (!error && numpasses-- && gotone);


    if (!numpasses) {
        fprintf(cp_err, "Error: infinite subckt recursion\n");
        error = 1;
    }

#ifdef TRACE
    /* Added by H.Tanaka to display converted deck */
    {
        struct card *c = deck;
        printf("Converted deck\n");
        for (; c; c = c->nextcard)
            printf("%s\n", c->line);
    }
    {
        wordlist *w = modnames;
        printf("Models:\n");
        for (; w; w = w->wl_next)
            printf("%s\n", w->wl_word);
    }
#endif

    wl_delete_slice(modnames, xmodnames);

    if (error)
        return NULL;    /* error message already reported; should free() */


    while (subs) {
        struct subs *rest = subs->su_next;

        tfree(subs->su_name);
        tfree(subs->su_args);
        line_free(subs->su_def, TRUE);
        tfree(subs);

        subs = rest;
    }

    return (deck);
}


/*-------------------------------------------------------------------*/
/* Copy a deck, including the actual lines.                          */
/*-------------------------------------------------------------------*/
struct card * inp_deckcopy(struct card *deck) {
    struct card *d = NULL, *nd = NULL;

    while (deck) {
        if (nd) {
            d->nextcard = TMALLOC(struct card, 1);
            d = d->nextcard;
        } else {
            nd = d = TMALLOC(struct card, 1);
        }
        d->linenum = deck->linenum;
        d->w = deck->w;
        d->l = deck->l;
        d->nf = deck->nf;
        d->line = copy(deck->line);
        if (deck->error)
            d->error = copy(deck->error);
        d->actualLine = inp_deckcopy(deck->actualLine);
        deck = deck->nextcard;
    }
    return (nd);
}

/*
 * Copy a deck, without the ->actualLine lines, without comment lines, and
 * without .control section(s).
 * First line is always copied (except being .control).
 */
struct card *inp_deckcopy_oc(struct card * deck)
{
    struct card *d = NULL, *nd = NULL;
    int skip_control = 0, i = 0;

    while (deck) {
        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", deck->line)) {
            skip_control++;
            deck = deck->nextcard;
            continue;
        }
        else if (ciprefix(".endc", deck->line)) {
            skip_control--;
            deck = deck->nextcard;
            continue;
        }
        else if (skip_control > 0) {
            deck = deck->nextcard;
            continue;
        }
        if (nd) { /* First card already found */
            /* d is the card at the end of the deck */
            d = d->nextcard = TMALLOC(struct card, 1);
        }
        else { /* This is the first card */
            nd = d = TMALLOC(struct card, 1);
        }
        d->w = deck->w;
        d->l = deck->l;
        d->nf = deck->nf;
        d->linenum_orig = deck->linenum;
        d->linenum = i++;
        d->line = copy(deck->line);
        if (deck->error) {
            d->error = copy(deck->error);
        }
        d->actualLine = NULL;
        deck = deck->nextcard;
        while (deck && *(deck->line) == '*') { /* skip comments */
            deck = deck->nextcard;
        }
    } /* end of loop over cards in the source deck */

    return nd;
} /* end of function inp_deckcopy_oc */

/*
 * Copy a deck, without the ->actualLine lines, without comment lines, and
 * without .control section(s).
 * Keep the line numbers.
 */
struct card* inp_deckcopy_ln(struct card* deck)
{
    struct card* d = NULL, * nd = NULL;
    int skip_control = 0;

    while (deck) {
        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", deck->line)) {
            skip_control++;
            deck = deck->nextcard;
            continue;
        }
        else if (ciprefix(".endc", deck->line)) {
            skip_control--;
            deck = deck->nextcard;
            continue;
        }
        else if (skip_control > 0) {
            deck = deck->nextcard;
            continue;
        }
        else if (*(deck->line) == '*') {
            deck = deck->nextcard;
            continue;
        }

        if (nd) { /* First card already found */
            /* d is the card at the end of the deck */
            d = d->nextcard = TMALLOC(struct card, 1);
        }
        else { /* This is the first card */
            nd = d = TMALLOC(struct card, 1);
        }
        d->w = deck->w;
        d->l = deck->l;
        d->nf = deck->nf;
        d->linenum_orig = deck->linenum_orig;
        d->linenum = deck->linenum;
        d->line = copy(deck->line);
        if (deck->error) {
            d->error = copy(deck->error);
        }
        d->actualLine = NULL;
        deck = deck->nextcard;
    } /* end of loop over cards in the source deck */

    return nd;
} /* end of function inp_deckcopy_ln */


/*-------------------------------------------------------------------
 * struct bxx_buffer,
 *   a string assembly facility.
 *
 * usage:
 *
 *   struct bxx_buffer thing;
 *   bxx_init(&thing);
 *   ...
 *   while (...) {
 *     bxx_rewind(&thing);
 *     ...
 *     bxx_putc(&thing, ...)
 *     bxx_printf(&thing, ...)
 *     bxx_put_cstring(&thing, ...)
 *     bxx_put_substring(&thing, ...)
 *     ...
 *     strcpy(bxx_buffer(&thing)
 *   }
 *   ..
 *   bxx_free(&thing)
 *
 * main aspect:
 *   reallocates/extends its buffer itself.
 *
 * note:
 *   during asssembly the internal buffer is
 *   not necessarily '\0' terminated.
 *   but will be when bxx_buffer() is invoked
 */

struct bxx_buffer {
    char *dst;
    char *limit;
    char *buffer;
};

/* must be a power of 2 */
static const int bxx_chunksize = 1024;

static void
bxx_init(struct bxx_buffer *t)
{
    /* assert(0 == (bxx_chunksize & (bxx_chunksize - 1))); */

    t->buffer = TMALLOC(char, bxx_chunksize);

    t->dst   = t->buffer;
    t->limit = t->buffer + bxx_chunksize;
}


static void
bxx_free(struct bxx_buffer *t)
{
    tfree(t->buffer);
}


static void
bxx_rewind(struct bxx_buffer *t)
{
    t->dst = t->buffer;
}


static void
bxx_extend(struct bxx_buffer *t, int howmuch)
{
    int pos = (int)(t->dst   - t->buffer);
    int len = (int)(t->limit - t->buffer);

    /* round up */
    howmuch +=  (bxx_chunksize - 1);
    howmuch &= ~(bxx_chunksize - 1);

    len += howmuch;

    t->buffer = TREALLOC(char, t->buffer, len);

    t->dst   = t->buffer + pos;
    t->limit = t->buffer + len;
}


static void
bxx_printf(struct bxx_buffer *t, const char *fmt, ...)
{
    va_list ap;

    for (;;) {
        int ret;
        int size = (int)(t->limit - t->dst);
        va_start(ap, fmt);
        ret = vsnprintf(t->dst, (size_t) size, fmt, ap);
        va_end(ap);
        if (ret == -1) {
            bxx_extend(t, bxx_chunksize);
        } else if (ret >= size) {
            bxx_extend(t, ret - size + 1);
        } else {
            t->dst += ret;
            break;
        }
    }

    va_end(ap);
}


static inline char
bxx_putc(struct bxx_buffer *t, char c)
{
    if (t->dst >= t->limit)
        bxx_extend(t, 1);
    return *(t->dst)++ = c;
}


static void
bxx_put_cstring(struct bxx_buffer *t, const char *cstring)
{
    while (*cstring)
        bxx_putc(t, *cstring++);
}


static void
bxx_put_substring(struct bxx_buffer *t, const char *str, const char *end)
{
    while (str < end)
        bxx_putc(t, *str++);
}


static char *
bxx_buffer(struct bxx_buffer *t)
{
    if ((t->dst == t->buffer) || (t->dst[-1] != '\0'))
        bxx_putc(t, '\0');
    return t->buffer;
}


/*------------------------------------------------------------------------------------------*
 * Translate all of the device names and node names in the .subckt deck. They are
 * pre-pended with subname:, unless they are in the formal list, in which case
 * they are replaced with the corresponding entry in the actual list.
 * The one special case is node 0 -- this is always ground and we don't
 * touch it.
 *
 * Variable name meanings:
 * *deck = pointer to subcircuit definition (lcc) (struct card)
 * formal = copy of the .subckt definition line (e.g. ".subckt subcircuitname 1 2 3") (string)
 * actual = copy of the .subcircuit invocation line (e.g. "Xexample 4 5 6 subcircuitname") (string)
 * scname = refdes (- first letter) used at invocation (e.g. "example") (string)
 * subname = copy of the subcircuit name
 *-------------------------------------------------------------------------------------------*/

static void
translate_node_name(struct bxx_buffer *buffer, const char *scname, const char *name, const char *name_e)
{

    const char *t;
    if (!name_e)
        name_e = strchr(name, '\0');

    t = gettrans(name, name_e);
    if (t) {
        bxx_put_cstring(buffer, t);
    } else {
        bxx_put_cstring(buffer, scname);
        bxx_putc(buffer, '.');
        bxx_put_substring(buffer, name, name_e);
    }
}


static void
translate_inst_name(struct bxx_buffer *buffer, const char *scname, const char *name, const char *name_e)
{
    if (!name_e)
        name_e = strchr(name, '\0');

    if (tolower_c(*name) != 'x') {
        bxx_putc(buffer, *name);
        bxx_putc(buffer, '.');
    }
    bxx_put_cstring(buffer, scname);
    bxx_putc(buffer, '.');
    bxx_put_substring(buffer, name, name_e);
}


static int
translate(struct card *deck, char *formal, int flen, char *actual, char *scname, const char *subname, struct subs *subs, wordlist const *modnames)
{
    struct card *c;
    struct bxx_buffer buffer;
    char *next_name, *name, *t, *nametofree, *paren_ptr;
    int nnodes, i, dim;
    int rtn = 0;
#ifdef XSPICE
    bool got_vnam = FALSE;
#endif
    bxx_init(&buffer);

    /* settrans builds the table holding the translated netnames.  */
    i = settrans(formal, flen, actual, subname);
    if (i < 0) {
        fprintf(stderr,
                "Too few parameters for subcircuit type \"%s\" (instance: x%s)\n",
                subname, scname);
        goto quit;
    } else if (i > 0) {
        fprintf(stderr,
                "Too many parameters for subcircuit type \"%s\" (instance: x%s)\n",
                subname, scname);
        goto quit;
    }

    for (c = deck; c; c = c->nextcard) {
        char *s = c->line;
        char dev_type = tolower_c(s[0]);

        bxx_rewind(&buffer);

#ifdef TRACE
        printf("\nIn translate, examining line (dev_type: %c, subname: %s, instance: %s) %s \n", dev_type, subname, scname, s);
#endif

        switch (dev_type) {

        case '.':
            if (ciprefix(".save", s)) {
                while ((paren_ptr = strchr(s, '(')) != NULL) {
                    bool curr = FALSE;
                    char* comma_ptr = NULL;

                    if (ciprefix(" i(", paren_ptr - 2))
                        curr = TRUE;

                    name = paren_ptr + 1;

                    if ((paren_ptr = strchr(name, ')')) == NULL) {
                        fprintf(cp_err, "Error: missing closing ')' for .save statement %s\n", c->line);
                        goto quit;
                    }

                    comma_ptr = strchr(s, ',');

                    bxx_put_substring(&buffer, s, name);
                    /* i(Vxx) */
                    if (curr) {
                        translate_inst_name(&buffer, scname, name, paren_ptr);
                        s = paren_ptr;
                    }
                    /* V(a,b) */
                    else if (comma_ptr && comma_ptr < paren_ptr) {
                        translate_node_name(&buffer, scname, name, comma_ptr);
                        bxx_putc(&buffer, ',');
                        name = comma_ptr + 1;
                        translate_node_name(&buffer, scname, name, paren_ptr);
                        s = paren_ptr;
                    }
                    /* V(a) */
                    else {
                        translate_node_name(&buffer, scname, name, paren_ptr);
                        s = paren_ptr;
                    }
                }
                bxx_put_cstring(&buffer, s); /* rest of line */
                break;
            }
            else if (ciprefix(".ic", s) || ciprefix(".nodeset", s)) {
                while ((paren_ptr = strchr(s, '(')) != NULL) {
                    name = paren_ptr + 1;

                    if ((paren_ptr = strchr(name, ')')) == NULL) {
                        fprintf(cp_err, "Error: missing closing ')' for .ic|.nodeset statement %s\n", c->line);
                        goto quit;
                    }

                    bxx_put_substring(&buffer, s, name);
                    translate_node_name(&buffer, scname, name, paren_ptr);

                    s = paren_ptr;
                }
                bxx_put_cstring(&buffer, s); /* rest of line */
                break;
            } else {
                continue;
            }

        case '\0':
        case '*':
        case '$':
            continue;


#ifdef XSPICE
            /*===================  case A  ====================*/
            /* gtri - add - wbk - 10/23/90 - process A devices specially */
            /* since they have a more involved and variable length node syntax */

        case 'a':

            /* translate the instance name according to normal rules */
            name = MIFgettok(&s);

            translate_inst_name(&buffer, scname, name, NULL);
            bxx_putc(&buffer, ' ');

            /* Now translate the nodes, looking ahead one token to recognize */
            /* when we reach the model name which should not be translated   */
            /* here.                                                         */

            next_name = MIFgettok(&s);

            for (;;) {
                /* rotate the tokens and get the the next one */
                if (name)
                    tfree(name);
                name = next_name;
                next_name = MIFgettok(&s);

                /* if next token is NULL, name holds the model name, so exit */
                if (next_name == NULL)
                    break;

                /* Process the token in name.  If it is special, then don't */
                /* translate it.                                            */
                switch (*name) {
                case '[':
                case ']':
                case '~':
                    bxx_put_cstring(&buffer, name);
                    break;

                case '%':
                    bxx_putc(&buffer, '%');
                    /* don't translate the port type identifier */
                    if (name)
                        tfree(name);
                    name = next_name;
                    /* vname requires instance translation of token following */
                    if (eq(name, "vnam"))
                        got_vnam = TRUE;
                    next_name = MIFgettok(&s);
                    bxx_put_cstring(&buffer, name);
                    break;

                default:
                    if (got_vnam) {
                        /* after %vnam an instance name is following */
                        translate_inst_name(&buffer, scname, name, NULL);
                        got_vnam = FALSE;
                    }
                    else {
                        /* must be a node name at this point, so translate it */
                        translate_node_name(&buffer, scname, name, NULL);
                    }
                    break;

                }
                bxx_putc(&buffer, ' ');
            }

            /* copy in the last token, which is the model name */
            if (name) {
                bxx_put_cstring(&buffer, name);
                tfree(name);
            }

            break; /* case 'a' */

            /* gtri - end - wbk - 10/23/90 */
#endif

            /*================   case E, F, G, H  ================*/
            /* This section handles controlled sources and allows for SPICE2 POLY attributes.
             * This is a new section, added by SDB to handle POLYs in sources.  Significant
             * changes were made in here.
             * 4.21.2003 -- SDB.  mailto:sdb@cloud9.net
             */
        case 'e':
        case 'f':
        case 'g':
        case 'h':

            name = gettok(&s);    /* name points to the refdes  */
            if (!name)
                continue;
            if (!*name) {
                tfree(name);
                continue;
            }

            /* Here's where we translate the refdes to e.g. F:subcircuitname:57
             * and stick the translated name into buffer.
             */
            translate_inst_name(&buffer, scname, name, NULL);
            tfree(name);
            bxx_putc(&buffer, ' ');

            /* Next iterate over all nodes (netnames) found and translate them. */
            nnodes = numnodes(c->line, subs, modnames);

            while (--nnodes >= 0) {
                name = gettok_node(&s);
                if (name == NULL) {
                    fprintf(cp_err, "Error: too few nodes: %s\n",
                            c->line);
                    goto quit;
                }

                translate_node_name(&buffer, scname, name, NULL);
                tfree(name);
                bxx_putc(&buffer, ' ');
            }

            /* Next we handle the POLY (if any) */
            /* get next token */
            t = s;
            next_name = gettok_noparens(&t);
            if ((strcmp(next_name, "POLY") == 0) ||
                (strcmp(next_name, "poly") == 0)) {

#ifdef TRACE
                printf("In translate, looking at e, f, g, h found poly\n");
#endif

                /* move pointer ahead of '(' */
                if (get_l_paren(&s) == 1) {
                    fprintf(cp_err, "Error: no left paren after POLY %s\n",
                            c->line);
                    tfree(next_name);
                    goto quit;
                }

                nametofree = gettok_noparens(&s);
                dim = atoi(nametofree);  /* convert returned string to int */
                tfree(nametofree);

                /* move pointer ahead of ')' */
                if (get_r_paren(&s) == 1) {
                    fprintf(cp_err, "Error: no right paren after POLY %s\n",
                            c->line);
                    tfree(next_name);
                    goto quit;
                }

                /* Write POLY(dim) into buffer */
                bxx_printf(&buffer, "POLY( %d ) ", dim);
            }
            else
                dim = 1;    /* only one controlling source . . . */
            tfree(next_name);

            /* Now translate the controlling source/nodes */
            nnodes = dim * numdevs(c->line);
            while (--nnodes >= 0) {
                name = gettok_node(&s);   /* name points to the returned token */
                if (name == NULL) {
                    fprintf(cp_err, "Error: too few devs: %s\n", c->line);
                    goto quit;
                }

                if ((dev_type == 'f') || (dev_type == 'h'))
                    translate_inst_name(&buffer, scname, name, NULL);
                else
                    translate_node_name(&buffer, scname, name, NULL);
                tfree(name);
                bxx_putc(&buffer, ' ');
            }

            /* Now write out remainder of line (polynomial coeffs) */
            finishLine(&buffer, s, scname);
            break;

        default:            /* this section handles ordinary components */
            name = gettok_node(&s);  /* changed to gettok_node to handle netlists with ( , ) */
            if (!name)
                continue;
            if (!*name) {
                tfree(name);
                continue;
            }

            translate_inst_name(&buffer, scname, name, NULL);
            tfree(name);
            bxx_putc(&buffer, ' ');

            /* FIXME anothet hack: if no models found for m devices, set number of nodes to 4 */
            if (!modnames && *(c->line) == 'm')
                nnodes = get_number_terminals(c->line);
            else if (*(c->line) == 'n')
                nnodes = get_number_terminals(c->line);
            else
                nnodes = numnodes(c->line, subs, modnames);
            while (--nnodes >= 0) {
                name = gettok_node(&s);
                if (name == NULL) {
                    fprintf(cp_err, "Error: too few nodes: %s\n", c->line);
                    goto quit;
                }

                translate_node_name(&buffer, scname, name, NULL);
                tfree(name);
                bxx_putc(&buffer, ' ');
            }

            /* Now translate any devices (i.e. controlling sources).
             * This may be superfluous because we handle dependent
             * source devices above . . . .
             */
            nnodes = numdevs(c->line);
            while (--nnodes >= 0) {
                name = gettok_node(&s);
                if (name == NULL) {
                    fprintf(cp_err, "Error: too few devs: %s\n", c->line);
                    goto quit;
                }

                translate_inst_name(&buffer, scname, name, NULL);
                tfree(name);
                bxx_putc(&buffer, ' ');
            }

            /* Now we finish off the line.  For most components (R, C, etc),
             * this involves adding the component value to the buffer.
             * We also scan through the line for v(something) and
             * i(something)...
             */
            finishLine(&buffer, s, scname);
            break;
        }

        tfree(c->line);
        c->line = copy(bxx_buffer(&buffer));

#ifdef TRACE
        printf("In translate, translated line = %s \n", c->line);
#endif
    }
    rtn = 1;
 quit:
    for (i = 0; ; i++) {
        if (!table[i].t_old && !table[i].t_new)
            break;
        FREE(table[i].t_old);
        FREE(table[i].t_new);
    }
    FREE(table);
    table = (struct tab *)NULL;

    bxx_free(&buffer);
    return rtn;
}


/*-------------------------------------------------------------------*
 * finishLine now doesn't handle current or voltage sources.
 * Therefore, it just writes out the final netnames, if required.
 * Changes made by SDB on 4.29.2003.
 *-------------------------------------------------------------------*/
static void
finishLine(struct bxx_buffer *t, char *src, char *scname)
{
    char *buf, *buf_end, which;
    char *s;
    int lastwasalpha;

    lastwasalpha = 0;
    while (*src) {
        /* Find the next instance of "<non-alpha>[vi]<opt spaces>(" in
         * this string.
         */
        if (((*src != 'v') && (*src != 'V') &&
             (*src != 'i') && (*src != 'I')) ||
            lastwasalpha) {
            lastwasalpha = isalpha_c(*src);
            bxx_putc(t, *src++);
            continue;
        }
        which = *src;
        s = skip_ws(src + 1);
        if (*s != '(') {
            lastwasalpha = isalpha_c(*src);
            bxx_putc(t, *src++);
            continue;
        }
        src = skip_ws(s + 1);
        lastwasalpha = 0;
        bxx_putc(t, which);
        bxx_putc(t, '(');
        for (buf = src; *src && !isspace_c(*src) && *src != ',' && *src != ')'; )
            src++;
        buf_end = src;

        if ((which == 'v') || (which == 'V')) {
            translate_node_name(t, scname, buf, buf_end);

            /* translate the reference node, as in the "2" in "v(4,2)" */
            while (*src && (isspace_c(*src) || *src == ','))
                src++;

            if (*src && *src != ')') {
                for (buf = src; *src && !isspace_c(*src) && (*src != ')'); )
                    src++;
                bxx_putc(t, ',');
                translate_node_name(t, scname, buf, buf_end = src);
            }
        } else {
            /*
             * i(instance_name) --> i(instance_name[0].subckt.instance_name)
             */
            translate_inst_name(t, scname, buf, buf_end);
        }
    }
}


/*------------------------------------------------------------------------------*
 * settrans builds the table which holds the old and new netnames.
 * it also compares the number of nets present in the .subckt definition against
 * the number of nets present in the subcircuit invocation.  It returns 0 if they
 * match, otherwise, it returns an error.
 *
 * Variable definitions:
 * formal = copy of the .subckt definition line (e.g. ".subckt subcircuitname 1 2 3") (string)
 * actual = copy of the .subcircuit invocation line (e.g. "Xexample 4 5 6 subcircuitname") (string)
 * subname = copy of the subcircuit name
 *------------------------------------------------------------------------------*/
static int
settrans(char *formal, int flen, char *actual, const char *subname)
{
    int i;

    table = TMALLOC(struct tab, flen + 1);
    memset(table, 0, (size_t)(flen + 1) * sizeof(struct tab));

    for (i = 0; i < flen; i++) {
        table[i].t_old = gettok(&formal);
        table[i].t_new = gettok(&actual);

        if (table[i].t_new == NULL) {
            return -1;          /* Too few actual / too many formal */
        } else if (table[i].t_old == NULL) {
            if (eq(table[i].t_new, subname))
                break;
            else
                return 1;       /* Too many actual / too few formal */
        }
    }

    return 0;
}


/* compare a substring, with a '\0' terminated string
 *   the substring itself is required to be free of a '\0'
 */

static int
eq_substr(const char *str, const char *end, const char *cstring)
{
    while (str < end)
        if (*str++ != *cstring++)
            return 0;
    return (*cstring == '\0');
}


/*------------------------------------------------------------------------------*
 * gettrans returns the name of the top level net if it is in the list,
 * otherwise it returns NULL.
 *------------------------------------------------------------------------------*/
static char *
gettrans(const char *name, const char *name_end)
{
    int i;

    if (!name_end)
        name_end = strchr(name, '\0');

    /* Added by H.Tanaka to translate global nodes */
    for (i = 0; i<num_global_nodes; i++)
        if (eq_substr(name, name_end, global_nodes[i]))
            return (global_nodes[i]);

    for (i = 0; table[i].t_old; i++)
        if (eq_substr(name, name_end, table[i].t_old))
            return (table[i].t_new);

    return (NULL);
}


/*-------------------------------------------------------------------*/
/*-------------------------------------------------------------------*/
static int
numnodes(const char *line, struct subs *subs, wordlist const *modnames)
{
    /* gtri - comment - wbk - 10/23/90 - Do not modify this routine for */
    /* 'A' type devices since the callers will not know how to find the */
    /* nodes even if they know how many there are.  Modify the callers  */
    /* instead.                                                         */
    /* gtri - end - wbk - 10/23/90 */
    char c;
    int n;

    line = skip_ws(line);

    c = tolower_c(*line);

    if (c == 'x') {     /* Handle this ourselves. */
        const char *xname_e = skip_back_ws(strchr(line, '\0'), line);
        const char *xname = skip_back_non_ws(xname_e, line);
        for (; subs; subs = subs->su_next)
            if (eq_substr(xname, xname_e, subs->su_name))
                return subs->su_numargs;
        /*
         * number of nodes not known so far.
         * lets count the nodes ourselves,
         * assuming `buf' looks like this:
         *    xname n1 n2 ... nn subname
         */
        {
            int nodes = -2;
            while (*line) {
                nodes++;
                line = skip_ws(skip_non_ws(line));
            }
            return (nodes);
        }
    }

    n = inp_numnodes(c);

    /* Added this code for variable number of nodes on certain devices.  */
    /* The consequence of this code is that the value returned by the    */
    /* inp_numnodes(c) call must be regarded as "maximum number of nodes */
    /* for a given device type.                                          */
    /* Paolo Nenzi Jan-2001                                              */

    /* If model names equal node names, this code will fail! */
    if ((c == 'm') || (c == 'p') || (c == 'q') || (c == 'd')) { /* IF this is a mos, cpl, bjt or diode */
        char *s = nexttok(line);       /* Skip the instance name */
        int gotit = 0;
        int i = 0;

        while ((i <= n) && (*s) && !gotit) {
            char *t = gettok_node(&s);       /* get nodenames . . .  */
            const wordlist *wl;
            for (wl = modnames; wl; wl = wl->wl_next)
                if (model_name_match(t, wl->wl_word)) {
                    gotit = 1;
                    break;
                }
            i++;
            tfree(t);
        }

        /* Note: node checks must be done on #_of_node-1 because the */
        /* "while" cycle increments the counter even when a model is */
        /* recognized. This code may be better!                      */

        if ((i < 4) && ((c == 'm') || (c == 'q'))) {
            fprintf(cp_err, "Error: too few nodes for MOS or BJT: %s\n", line);
            return (0);
        }
        if ((i < 5) && (c == 'p')) {
            fprintf(cp_err, "Error: too few nodes for CPL: %s\n", line);
            return (0);
        }
        return (i-1); /* compensate the unnecessary increment in the while cycle */
    } else {
        /* for all other elements */
        return (n);
    }
}


/*-------------------------------------------------------------------*
 *  This function returns the number of controlling voltage sources
 *  (for F, H) or controlling nodes (for G, E)  attached to a dependent
 *  source.
 *-------------------------------------------------------------------*/
static int
numdevs(char *s)
{

    s = skip_ws(s);
    switch (*s) {
    case 'K':
    case 'k':
        return (2);

        /* two nodes per voltage controlled source */
    case 'G':
    case 'g':
    case 'E':
    case 'e':
        return (2);

        /* one source per current controlled source */
    case 'F':
    case 'f':
    case 'H':
    case 'h':
        /* 2 lines here added to fix w bug, NCF 1/31/95 */
    case 'W':
    case 'w':
        return (1);

    default:
        return (0);
    }
}


/*----------------------------------------------------------------------*
 *  modtranslate --  translates .model lines found in subckt definitions.
 *  Calling arguments are:
 *  *c = pointer to the .subckt definition (linked list)
 *  *subname = pointer to the subcircuit name used at the subcircuit invocation (string)
 *  modtranslate returns the list of model names which have been translated
 *----------------------------------------------------------------------*/
static wordlist *
modtranslate(struct card *c, char *subname, wordlist *new_modnames)
{
    wordlist *orig_modnames = NULL;
    struct card *lcc = c;

    for (; c; c = c->nextcard)
        if (ciprefix(".model", c->line)) {
            char *model_name, *new_model_name;
            char *t = c->line;

#ifdef TRACE
            printf("modtranslate(), translating:\n"
                   "  \"%s\" -->\n", t);
#endif

            /* swallow ".model" */
            t = nexttok(t);

            model_name = gettok(&t);

            new_model_name = tprintf("%s:%s", subname, model_name);

            /* remember the translation */
            orig_modnames = wl_cons(model_name, orig_modnames);
            new_modnames = wl_cons(new_model_name, new_modnames);

            /* perform the actual translation of this .model line */
            t = tprintf(".model %s %s", new_model_name, t);
            tfree(c->line);
            c->line = t;

#ifdef TRACE
            printf("  \"%s\"\n", t);
            printf("  mapped modelname \"%s\" --> \"%s\"\n",
                   model_name, new_model_name);
#endif

        }

    if (orig_modnames) {
        devmodtranslate(lcc, subname, orig_modnames);
        wl_free(orig_modnames);
    }

    return new_modnames;
}


/*-------------------------------------------------------------------*
 *  Devmodtranslate scans through the deck, and translates the
 *  name of the model in a line held in a .subckt.  For example:
 *  before:   .subckt U1 . . . .
 *            Q1 c b e 2N3904
 *  after:    Q1 c b e U1:2N3904
 *-------------------------------------------------------------------*/

static void
translate_mod_name(struct bxx_buffer *buffer, char *modname, char *subname, struct wordlist *orig_modnames)
{
    /*
     *  Note that we compare against orig_modnames,
     *    which is the list of untranslated names of models.
     */
    wordlist *wlsub = wl_find(modname, orig_modnames);

    if (!wlsub)
        bxx_printf(buffer, "%s", modname);
    else
        bxx_printf(buffer, "%s:%s", subname, modname);
}


static void
devmodtranslate(struct card *s, char *subname, wordlist * const orig_modnames)
{
    int found;

    struct bxx_buffer buffer;
    bxx_init(&buffer);


    for (; s; s = s->nextcard) {

        char *t, c, *name, *next_name;
        wordlist *wlsub;

        bxx_rewind(&buffer);

        t = s->line;

#ifdef TRACE
        /* SDB debug stuff */
        printf("In devmodtranslate, examining line %s.\n", t);
#endif

        t = skip_ws(t);
        c = *t;                           /* set c to first char in line. . . . */
        if (isupper_c(c))
            c = tolower_c(c);

        switch (c) {

#ifdef XSPICE

        case 'a':
            /*  Code for codemodels (dev prefix "A") added by SDB on 6.10.2004.
             *  The algorithm is simple.  We don't know how many nodes or sources are attached,
             *  but the name of the model is always last.  Therefore, just iterate through all
             *  tokens until the last one is reached.  Then translate it.
             */

#ifdef TRACE
            /* SDB debug statement */
            printf("In devmodtranslate, found codemodel, line= %s\n", t);
#endif

            /* first do refdes. */
            name = gettok(&t);  /* get refdes */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);

            /* now do remainder of line. */
            next_name = gettok(&t);
            for (;;) {
                name = next_name;
                next_name = gettok(&t);

                if (next_name == NULL) {
                    /* if next_name is NULL, we are at the line end.
                     * name holds the model name.  Therefore, break */
                    break;

                } else {
                    /* next_name holds something.  Write name into the buffer and continue. */
                    bxx_printf(&buffer, "%s ", name);
                    tfree(name);
                }
            }  /* while  */

            translate_mod_name(&buffer, name, subname, orig_modnames);
            tfree(name);
            bxx_putc(&buffer, ' ');

#ifdef TRACE
            /* SDB debug statement */
            printf("In devmodtranslate, translated codemodel line= %s\n", buffer);
#endif

            bxx_put_cstring(&buffer, t);
            tfree(s->line);
            s->line = copy(bxx_buffer(&buffer));
            break;

#endif /* XSPICE */

        case 'r':
        case 'c':
        case 'l':
            name = gettok(&t);  /* get refdes */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get first netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get second netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);

            if (*t) {    /* if there is a model, process it. . . . */
                name = gettok(&t);
                translate_mod_name(&buffer, name, subname, orig_modnames);
                tfree(name);
                bxx_putc(&buffer, ' ');
            }

            if (*t) {
                name = gettok(&t);
                translate_mod_name(&buffer, name, subname, orig_modnames);
                tfree(name);
                bxx_putc(&buffer, ' ');
            }

            bxx_put_cstring(&buffer, t);
            tfree(s->line);
            s->line = copy(bxx_buffer(&buffer));
            break;

           /* 2 or 3 (temp) terminals for diode d, 2 or more for OSDI devices */
        case 'd':
#ifdef OSDI
        case 'n':
#endif
            name = gettok(&t);  /* get refdes */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get first attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get second attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* this can be either a model name or a node name. */
            if (name == NULL) {
                name = copy(""); /* allow 'tfree' */
            } else {
                for (;;) {
                    wlsub = wl_find(name, orig_modnames);
                    if (wlsub) {
                        break;
                    } else {
                        bxx_printf(&buffer, "%s ", name);
                        tfree(name);
                        name = gettok(&t);
                        if (name == NULL) {  /* No token anymore - leave */
                            name = copy(""); /* allow 'tfree' */
                            break;
                        }
                    }
                }  /* while  */
            }

            translate_mod_name(&buffer, name, subname, orig_modnames);

            tfree(name);
            bxx_putc(&buffer, ' ');
            bxx_put_cstring(&buffer, t);
            tfree(s->line);
            s->line = copy(bxx_buffer(&buffer));
            break;

        case 'u': /* urc transmissionline */
            /* 3 terminal devices */
        case 'w': /* current controlled switch */
        case 'j': /* jfet */
        case 'z': /* hfet, mesa */
            name = gettok(&t);
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok(&t);
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok(&t);
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok(&t);
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok(&t);

            translate_mod_name(&buffer, name, subname, orig_modnames);
            tfree(name);
            bxx_putc(&buffer, ' ');
            bxx_put_cstring(&buffer, t);
            tfree(s->line);
            s->line = copy(bxx_buffer(&buffer));
            break;

            /* 4 terminal devices */
        case 'o':    /* ltra */
        case 's':    /* vc switch */
        case 'y':    /* txl */
            /*  Changed gettok() to gettok_node() on 12.2.2003 by SDB
                to enable parsing lines like "S1 10 11 (80,51) SLATCH1"
                which occur in real Analog Devices SPICE models.
            */
            name = gettok(&t);  /* get refdes */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get first attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get second attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get third attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get fourth attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok(&t);

            translate_mod_name(&buffer, name, subname, orig_modnames);
            bxx_putc(&buffer, ' ');
            bxx_put_cstring(&buffer, t);
            tfree(s->line);
            s->line = copy(bxx_buffer(&buffer));
            tfree(name);
            break;

            /* 3-7 terminal mos devices */
        case 'm':
            name = gettok(&t);  /* get refdes */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get first attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get second attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get third attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);

            if (!name) {
                break;
            }

            found = 0;
            while (!found) {
                /* Now, is this a subcircuit model? */
                for (wlsub = orig_modnames; wlsub; wlsub = wlsub->wl_next)
                    if (model_name_match(name, wlsub->wl_word)) {
                        found = 1;
                        break;
                    }
                if (!found) { /* name was not a model - was a netname */
                    bxx_printf(&buffer, "%s ", name);
                    tfree(name);
                    name = gettok_node(&t);
                    if (name == NULL) {
                        name = copy(""); /* allow 'tfree' */
                        break;
                    }
                }
            }  /* while  */

            if (!found)
                bxx_printf(&buffer, "%s", name);
            else
                bxx_printf(&buffer, "%s:%s", subname, name);
            bxx_putc(&buffer, ' ');

            bxx_put_cstring(&buffer, t);
            tfree(s->line);
            s->line = copy(bxx_buffer(&buffer));
            tfree(name);
            break;

            /* 3-5 terminal bjt devices */
        case 'q':
            name = gettok(&t);  /* get refdes */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get first attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get second attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* get third attached netname */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);
            name = gettok_node(&t);  /* this can be either a model name or a node name. */

            if (name == NULL) {
                name = copy(""); /* allow 'tfree' */
            } else {
                for (;;) {
                    wlsub = wl_find(name, orig_modnames);
                    if (wlsub) {
                        break;
                    } else {
                        bxx_printf(&buffer, "%s ", name);
                        tfree(name);
                        name = gettok(&t);
                        if (name == NULL) {  /* No token anymore - leave */
                            name = copy(""); /* allow 'tfree' */
                            break;
                        }
                    }
                }  /* while  */
            }

            translate_mod_name(&buffer, name, subname, orig_modnames);
            tfree(name);
            bxx_putc(&buffer, ' ');

            bxx_put_cstring(&buffer, t);
            tfree(s->line);
            s->line = copy(bxx_buffer(&buffer));
            break;

            /* 4-18 terminal devices */
        case 'p': /* cpl */
            name = gettok(&t);  /* get refdes */
            bxx_printf(&buffer, "%s ", name);
            tfree(name);

            /* now do remainder of line. */
            next_name = gettok(&t);
            for (;;) {
                name = next_name;
                next_name = gettok(&t);
                if (!next_name || strstr(next_name, "len")) {
                    /* if next_name is NULL or len or length, we are at the line end.
                     * name holds the model name.  Therefore, break */
                    break;
                } else {
                    /* next_name holds something.  Write name into the buffer and continue. */
                    bxx_printf(&buffer, "%s ", name);
                    tfree(name);
                }
            }  /* while  */

            translate_mod_name(&buffer, name, subname, orig_modnames);
            tfree(name);
            bxx_putc(&buffer, ' ');

            bxx_put_cstring(&buffer, t);
            tfree(s->line);
            s->line = copy(bxx_buffer(&buffer));
            break;

        default:
            break;
        }
    }

    bxx_free(&buffer);
}


/*----------------------------------------------------------------------*
 * inp_numnodes returns the maximum number of nodes (netnames) attached
 * to the component.
 * This is a spice-dependent thing.  It should probably go somewhere
 * else, but...  Note that we pretend that dependent sources and mutual
 * inductors have more nodes than they really do...
 *----------------------------------------------------------------------*/
static int
inp_numnodes(char c)
{
    if (isupper_c(c))
        c = tolower_c(c);
    switch (c) {
    case ' ':
    case '\t':
    case '.':
    case 'x':
    case '*':
    case '$':
        return (0);

    case 'b':
        return (2);
    case 'c':
        return (2);
    case 'd':
        return (3);
    case 'e':
        return (2); /* changed from 4 to 2 by SDB on 4.22.2003 to enable POLY */
    case 'f':
        return (2);
    case 'g':
        return (2); /* changed from 4 to 2 by SDB on 4.22.2003 to enable POLY */
    case 'h':
        return (2);
    case 'i':
        return (2);
    case 'j':
        return (3);
    case 'k':
        return (0);
    case 'l':
        return (2);
    case 'm':
        return (7); /* This means that 7 is the maximun number of nodes */
    case 'o':
        return (4);
    case 'p':
        return (18);/* 16 lines + 2 gnd is the maximum number of nodes for CPL */
    case 'q':
        return (5);
    case 'r':
        return (2);
    case 's':
        return (4);
    case 't':
        return (4);
    case 'u':
        return (3);
    case 'v':
        return (2);
    case 'w':
        return (2); /* change 3 to 2 here to fix w bug, NCF 1/31/95 */
    case 'y':
        return (4);
    case 'z':
        return (3);

    default:
        fprintf(cp_err, "Warning: unknown device type: %c\n", c);
        return (2);
    }
}
