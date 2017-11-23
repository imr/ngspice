/*       spicenum.c                Copyright (C)  2002    Georg Post
 *
 *  This file is part of Numparam, see:  readme.txt
 *  Free software under the terms of the GNU Lesser General Public License
 */

/* number parameter add-on for Spice.
   to link with mystring.o, xpressn.o (math formula interpreter),
   and with Spice frontend src/lib/fte.a .
   Interface function nupa_signal to tell us about automaton states.
Buglist (some are 'features'):
  blank lines get category '*'
  inserts conditional blanks before or after  braces
  between .control and .endc, flags all lines as 'category C', dont touch.
  there are reserved magic numbers (1e9 + n) as placeholders
  control lines must not contain {} .
  ignores the '.option numparam' line planned to trigger the actions
  operation of .include certainly doesnt work
  there are frozen maxima for source and expanded circuit size.
Todo:
  add support for nested .if .elsif .else .endif controls.
*/

#include "ngspice/ngspice.h"

#include "general.h"
#include "numparam.h"

#include "ngspice/fteext.h"
#include "ngspice/stringskip.h"

#ifdef SHARED_MODULE
extern ATTRIBUTE_NORETURN void shared_exit(int status);
#endif

extern bool ft_batchmode;

/* number of parameter substitutions, available only after the substitution */
extern long dynsubst; /* spicenum.c:144 */

/* number of lines in input deck */
extern int dynmaxline; /* inpcom.c:1529 */

/* Uncomment this line to allow debug tracing */
/* #define TRACE_NUMPARAMS */

/*  the nupa_signal arguments sent from Spice:

    sig=1: Start of the subckt expansion.
    sig=2: Stop of the subckt expansion.
    sig=3: Stop of the evaluation phase.
    sig=0: Start of a deck copy operation

    After sig=1 until sig=2, nupa_copy does no transformations.
    At sig=2, we prepare for nupa_eval loop.
    After sig=3, we assume the initial state (clean).

    In Clean state, a lot of deckcopy operations come in and we
    overwrite any line pointers, or we start a new set after each sig=0 ?
    Anyway, we neutralize all & and .param lines  (category[] array!)
    and we substitute all {} &() and &id placeholders by dummy identifiers.
    those look like numparm__________XXXXXXXX (8 hexadecimal digits)

*/
/**********  string handling ***********/

static long placeholder = 0;


static void
stripsomespace(SPICE_DSTRINGPTR dstr_p, bool incontrol)
{
    /* if s starts with one of some markers, strip leading space */

    const char *markers =
        incontrol
        ? "*.&+#$"
        : "*.&+#$" "xX";

    char *s = spice_dstring_value(dstr_p);

    int i = 0;
    while (s[i] && (s[i] <= ' '))
        i++;

    if ((i > 0) && s[i] && strchr(markers, s[i]))
        pscopy(dstr_p, s + i, 0, (int) strlen(s + i));
}


static int
stripbraces(SPICE_DSTRINGPTR dstr_p)
/* puts the funny placeholders. returns the number of {...} substitutions */
{
    int n = 0;
    char *s = spice_dstring_value(dstr_p);
    char *p, *brace;
    SPICE_DSTRING tstr;         /* temporary dynamic string */

    spice_dstring_init(&tstr);
    p = s;

    while ((brace = strchr(p, '{')) != NULL) {

        /* something to strip */
        const char *j_ptr = brace + 1;
        int nest = 1;
        n++;

        while ((nest > 0) && *j_ptr) {
            if (*j_ptr == '{')
                nest++;
            else if (*j_ptr == '}')
                nest--;
            j_ptr++;
        }

        pscopy(&tstr, s, 0, (int) (brace - s));

        if (brace[-1] > ' ')
            cadd(&tstr, ' ');

        cadd(&tstr, ' ');
        {
            char buf[25+1];
            sprintf(buf, "numparm__________%08lx", ++placeholder);
            sadd(&tstr, buf);
        }
        cadd(&tstr, ' ');

        if (*j_ptr >= ' ')
            cadd(&tstr, ' ');

        int ilen = spice_dstring_length(&tstr);
        sadd(&tstr, j_ptr);
        scopyd(dstr_p, &tstr);
        s = spice_dstring_value(dstr_p);
        p = s + ilen;
    }

    dynsubst = placeholder;
    spice_dstring_free(&tstr);

    return n;
}


static void
findsubname(dico_t *dico, SPICE_DSTRINGPTR dstr_p)
/* truncate the parameterized subckt call to regular old Spice */
/* scan a string from the end, skipping non-idents and {expressions} */
/* then truncate s after the last subckt(?) identifier */
{
    SPICE_DSTRING name;         /* extract a name */
    char * const s = spice_dstring_value(dstr_p);
    char * const ls_ptr = s + spice_dstring_length(dstr_p);

    int (p - s - 1) = (int) (ls_ptr - s) - 1;
    spice_dstring_init(&name);

    while ((p - s - 1) >= 0) {

        /* skip space, then non-space */
        while (((p - s - 1) >= 0) && (s[(p - s - 1)] <= ' '))
            (p - s - 1)--;

        int h = (int) (p - s - 1) + 1;              /* at h: space */
        while (((p - s - 1) >= 0) && (s[(p - s - 1)] > ' ')) {

            if (s[(p - s - 1)] == '}') {
                int nest = 1;
                (p - s - 1)--;

                while ((nest > 0) && ((p - s - 1) >= 0)) {
                    if (s[(p - s - 1)] == '{')
                        nest--;
                    else if (s[(p - s - 1)] == '}')
                        nest++;

                    (p - s - 1)--;
                }
                h = (int) (p - s - 1) + 1;      /* h points to '{' */

            } else {
                (p - s - 1)--;
            }
        }

        if (((p - s - 1) >= 0) && alfanum(s[(p - s - 1) + 1])) { /* suppose an identifier */
            entry_t *entry;
            /* check for known subckt name */
            spice_dstring_reinit(&name);
            int j = (int) (p - s - 1) + 1;
            while (alfanum(s[j])) {
                cadd(&name, toupper_c(s[j]));
                j++;
            }
            entry = entrynb(dico, spice_dstring_value(&name));
            if (entry && (entry->tp == NUPA_SUBCKT)) {
                if (h < (int) (ls_ptr - s))
                    pscopy(dstr_p, s, 0, h);
                return;
            }
        }
    }
}


static void
modernizeex(SPICE_DSTRINGPTR dstr_p)
/* old style expressions &(..) and &id --> new style with braces. */
{
    int i, state, ls;
    char c, d;
    char *s;                    /* current string */
    SPICE_DSTRING t;            /* temporary dyna string */

    i = 0;
    state = 0;
    ls = spice_dstring_length(dstr_p);
    s = spice_dstring_value(dstr_p);

    /* check if string might need modernizing */
    if (!memchr(s, Intro, (size_t) ls))
        return;

    spice_dstring_init(&t);

    while (i < ls) {
        c = s[i];
        d = s[i + 1];
        if ((!state) && (c == Intro) && (i > 0)) {
            if (d == '(') {
                state = 1;
                i++;
                c = '{';
            } else if (alfa(d)) {
                cadd(&t, '{');
                i++;
                while (alfanum(s[i])) {
                    cadd(&t, s[i]);
                    i++;
                }
                c = '}';
                i--;
            }
        } else if (state) {
            if (c == '(')
                state++;
            else if (c == ')')
                state--;

            if (!state)         /* replace--) by terminator */
                c = '}';
        }

        cadd(&t, c);
        i++;
    }

    scopyd(dstr_p, &t);
    spice_dstring_free(&t);
}


static char
transform(dico_t *dico, SPICE_DSTRINGPTR dstr_p, bool incontrol)
/*         line s is categorized and crippled down to basic Spice
 *         returns in u control word following dot, if any
 *
 * any + line is copied as-is.
 * any & or .param line is commented-out.
 * any .subckt line has params section stripped off
 * any X line loses its arguments after sub-circuit name
 * any &id or &() or {} inside line gets a 10-digit substitute.
 *
 * strip  the new syntax off the codeline s, and
 * return the line category as follows:
 *   '*'  comment line
 *   '+'  continuation line
 *   ' '  other untouched netlist or command line
 *   'P'  parameter line, commented-out; (name,linenr)-> symbol table.
 *   'S'  subckt entry line, stripped;   (name,linenr)-> symbol table.
 *   'U'  subckt exit line
 *   'X'  subckt call line, stripped
 *   'C'  control entry line
 *   'E'  control exit line
 *   '.'  any other dot line
 *   'B'  netlist (or .model ?) line that had Braces killed
 */
{
    char *s;                    /* dstring value of dstr_p */
    char category;
    stripsomespace(dstr_p, incontrol);
    modernizeex(dstr_p);        /* required for stripbraces count */

    s = spice_dstring_value(dstr_p);

    if (s[0] == '.') {
        /* check PS parameter format */
        if (ci_prefix(".PARAM", s)) {
            /* comment it out */
            /* s[0] = '*'; */
            category = 'P';
        } else if (ci_prefix(".SUBCKT", s)) {
            char *params, *t;
            SPICE_DSTRING tstr;
            spice_dstring_init(&tstr);
            scopy_up(&tstr, s);
            t = spice_dstring_value(&tstr);
            /* split off any "params" tail */
            params = strstr(t, "PARAMS:");
            if (params)
                pscopy(dstr_p, s, 0, (int)(params - t));
            spice_dstring_free(&tstr);
            category = 'S';
        } else if (ci_prefix(".CONTROL", s)) {
            category = 'C';
        } else if (ci_prefix(".ENDC", s)) {
            category = 'E';
        } else if (ci_prefix(".ENDS", s)) {
            category = 'U';
        } else {
            category = '.';
            if (stripbraces(dstr_p) > 0)
                category = 'B'; /* priority category ! */
        }
    } else if (s[0] == Intro) {
        /* private style preprocessor line */
        s[0] = '*';
        category = 'P';
    } else if (toupper_c(s[0]) == 'X') {
        /* strip actual parameters */
        findsubname(dico, dstr_p);
        category = 'X';
    } else if (s[0] == '+') {   /* continuation line */
        category = '+';
    } else if (!strchr("*$#", s[0])) {
        /* not a comment line! */
        if (stripbraces(dstr_p) > 0)
            category = 'B';     /* line that uses braces */
        else
            category = ' ';     /* ordinary code line */
    } else {
        category = '*';
    }

    return category;
}


/************ core of numparam **************/

/* some day, all these nasty globals will go into the dico_t structure
   and everything will get hidden behind some "handle" ...
   For the time being we will rename this variable to end in S so we know
   they are statics within this file for easier reading of the code.
*/

static int linecountS = 0;      /* global: number of lines received via nupa_copy */
static int evalcountS = 0;      /* number of lines through nupa_eval() */
static int nblogS = 0;          /* serial number of (debug) logfile */
static bool inexpansionS = 0;   /* flag subckt expansion phase */
static bool incontrolS = 0;     /* flag control code sections */
static bool dologfileS = 0;     /* for debugging */
static bool firstsignalS = 1;
static FILE *logfileS = NULL;
static dico_t *dicoS = NULL;
static dico_t *dicos_list[100];


/*  already part of dico : */
/*
  Open ouput to a log file.
  takes no action if logging is disabled.
  Open the log if not already open.
*/
static void
putlogfile(char c, int num, char *t)
{
    if (!dologfileS)
        return;

    if (!logfileS) {
        char *fname = tprintf("logfile.%d", ++nblogS);
        /* add user defined path (nname has to be freed after usage) */
        char *nname = set_output_path(fname);
        logfileS = fopen(nname, "w");
        tfree(fname);
        tfree(nname);
    }

    if (logfileS)
        fprintf(logfileS, "%c%d: %s\n", c, num, t);
}


static void
nupa_init(void)
{
    int i;

    /* init the symbol table and so on, before the first  nupa_copy. */
    evalcountS = 0;
    linecountS = 0;
    incontrolS = 0;
    placeholder = 0;
    dicoS = TMALLOC(dico_t, 1);
    initdico(dicoS);

    dicoS->dynrefptr = TMALLOC(char*, dynmaxline + 1);
    dicoS->dyncategory = TMALLOC(char, dynmaxline + 1);

    for (i = 0; i <= dynmaxline; i++) {
        dicoS->dynrefptr[i] = NULL;
        dicoS->dyncategory[i] = '?';
    }

    dicoS->linecount = dynmaxline;
}


/* free dicoS (called from com_remcirc()) */
void
nupa_del_dicoS(void)
{
    int i;

    if(!dicoS)
        return;

    for (i = dicoS->linecount; i >= 0; i--)
        txfree(dicoS->dynrefptr[i]);

    txfree(dicoS->dynrefptr);
    txfree(dicoS->dyncategory);
    txfree(dicoS->inst_name);
    nghash_free(dicoS->symbols[0], del_attrib, NULL);
    txfree(dicoS->symbols);
    txfree(dicoS);
    dicoS = NULL;
}


static void
nupa_done(void)
{
    int nerrors = dicoS->errcount;
    int dictsize = donedico(dicoS);

    if (logfileS) {
        fclose(logfileS);
        logfileS = NULL;
    }

    /* We cannot remove dicoS here because numparam is used by
       the .measure statements, which are invoked only after the
       simulation has finished. */

    if (nerrors) {
        bool is_interactive = FALSE;
        if (cp_getvar("interactive", CP_BOOL, NULL))
            is_interactive = TRUE;
        printf(" Copies=%d Evals=%d Placeholders=%ld Symbols=%d Errors=%d\n",
               linecountS, evalcountS, placeholder, dictsize, nerrors);
        /* debug: ask if spice run really wanted */
        if (ft_batchmode)
            controlled_exit(EXIT_FAILURE);
        if (!is_interactive) {
            fprintf(cp_err, "Numparam expansion errors: Problem with input file.\n");
            controlled_exit(EXIT_FAILURE);
        }
        for (;;) {
            int c;
            printf("Numparam expansion errors: Run Spice anyway? y/n ?\n");
            c = yes_or_no();
            if (c == 'n' || c == EOF)
                controlled_exit(EXIT_FAILURE);
            if (c == 'y')
                break;
        }
    }

    linecountS = 0;
    evalcountS = 0;
    placeholder = 0;
    /* release symbol table data */
}


/* SJB - Scan the line for subcircuits */
void
nupa_scan(struct card *card)
{
    defsubckt(dicoS, card);
}


/* -----------------------------------------------------------------
 * Dump the contents of a symbol table.
 * ----------------------------------------------------------------- */
static void
dump_symbol_table(dico_t *dico, NGHASHPTR htable_p, FILE *fp)
{
    char *name;                 /* current symbol */
    entry_t *entry;             /* current entry */
    NGHASHITER iter;            /* hash iterator - thread safe */

    NGHASH_FIRST(&iter);
    for (entry = (entry_t *) nghash_enumerateRE(htable_p, &iter);
         entry;
         entry = (entry_t *) nghash_enumerateRE(htable_p, &iter))
    {
        if (entry->tp == NUPA_REAL) {
            spice_dstring_reinit(& dico->lookup_buf);
            scopy_lower(& dico->lookup_buf, entry->symbol);
            name = spice_dstring_value(& dico->lookup_buf);
            fprintf(fp, "       ---> %s = %g\n", name, entry->vl);
            spice_dstring_free(& dico->lookup_buf);
        }
    }
}


/* -----------------------------------------------------------------
 * Dump the contents of the symbol table.
 * ----------------------------------------------------------------- */
void
nupa_list_params(FILE *fp)
{
    int depth;                  /* nested subcircit depth */
    dico_t *dico;               /* local copy for speed */

    dico = dicoS;
    if (dico == NULL) {
        fprintf(cp_err, "\nWarning: No symbol table available for 'listing param'\n");
        return;
    }

    fprintf(fp, "\n\n");

    for (depth = dico->stack_depth; depth >= 0; depth--) {
        NGHASHPTR htable_p = dico->symbols[depth];
        if (htable_p) {
            if (depth > 0)
                fprintf(fp, " local symbol definitions for: %s\n", dico->inst_name[depth]);
            else
                fprintf(fp, " global symbol definitions:\n");
            dump_symbol_table(dico, htable_p, fp);
        }
    }
}


/* -----------------------------------------------------------------
 * Lookup a parameter value in the symbol tables.   This involves
 * multiple lookups in various hash tables in order to get the scope
 * correct.  Each subcircuit instance will have its own local hash
 * table if it has parameters.   We can return whenever we get a hit.
 * Otherwise, we have to exhaust all of the tables including the global
 * table.
 * ----------------------------------------------------------------- */
double
nupa_get_param(char *param_name, int *found)
{
    int depth;                  /* nested subcircit depth */
    char *up_name;              /* current parameter upper case */
    entry_t *entry;             /* current entry */
    dico_t *dico;               /* local copy for speed */
    double result = 0;          /* parameter value */

    dico = dicoS;
    spice_dstring_reinit(& dico->lookup_buf);
    scopy_up(& dico->lookup_buf, param_name);
    up_name = spice_dstring_value(& dico->lookup_buf);

    *found = 0;
    for (depth = dico->stack_depth; depth >= 0; depth--) {
        NGHASHPTR htable_p = dico->symbols[depth];
        if (htable_p) {
            entry = (entry_t *) nghash_find(htable_p, up_name);
            if (entry) {
                result = entry->vl;
                *found = 1;
                break;
            }
        }
    }

    spice_dstring_free(& dico->lookup_buf);
    return result;
}


void
nupa_add_param(char *param_name, double value)
{
    char *up_name;              /* current parameter upper case */
    entry_t *entry;             /* current entry */
    dico_t *dico;               /* local copy for speed */
    NGHASHPTR htable_p;         /* hash table of interest */

    dico = dicoS;
    /* -----------------------------------------------------------------
     * We use a dynamic string here because most of the time we will
     * be using short names and no memory allocation will occur.
     * ----------------------------------------------------------------- */
    spice_dstring_reinit(& dico->lookup_buf);
    scopy_up(& dico->lookup_buf, param_name);
    up_name = spice_dstring_value(& dico->lookup_buf);

    /* can't be lazy anymore */
    if (!(dico->symbols[dico->stack_depth]))
        dico->symbols[dico->stack_depth] = nghash_init(NGHASH_MIN_SIZE);

    htable_p = dico->symbols[dico->stack_depth];

    entry = attrib(dico, htable_p, up_name, 'N', NULL);
    if (entry) {
        entry->vl = value;
        entry->tp = NUPA_REAL;
        entry->ivl = 0;
        entry->sbbase = NULL;
    }

    spice_dstring_free(& dico->lookup_buf);
}


void
nupa_add_inst_param(char *param_name, double value)
{
    char *up_name;              /* current parameter upper case */
    entry_t *entry;             /* current entry */
    dico_t *dico;               /* local copy for speed */

    dico = dicoS;
    spice_dstring_reinit(& dico->lookup_buf);
    scopy_up(& dico->lookup_buf, param_name);
    up_name = spice_dstring_value(& dico->lookup_buf);

    if (!(dico->inst_symbols))
        dico->inst_symbols = nghash_init(NGHASH_MIN_SIZE);

    entry = attrib(dico, dico->inst_symbols, up_name, 'N', NULL);
    if (entry) {
        entry->vl = value;
        entry->tp = NUPA_REAL;
        entry->ivl = 0;
        entry->sbbase = NULL;
    }

    spice_dstring_free(& dico->lookup_buf);
}


/* -----------------------------------------------------------------
 * This function copies any definitions in the inst_symbols hash
 * table which are qualified symbols and makes them available at
 * the global level.  Afterwards, the inst_symbols table is freed.
 * ----------------------------------------------------------------- */
void
nupa_copy_inst_dico(void)
{
    entry_t *entry;             /* current entry */
    dico_t *dico;               /* local copy for speed */
    NGHASHITER iter;            /* hash iterator - thread safe */

    dico = dicoS;
    if (dico->inst_symbols) {
        /* We we perform this operation we should be in global scope */
        if (dico->stack_depth > 0)
            fprintf(stderr, "stack depth should be zero.\n");

        NGHASH_FIRST(&iter);
        for (entry = (entry_t *) nghash_enumerateRE(dico->inst_symbols, &iter);
             entry;
             entry = (entry_t *) nghash_enumerateRE(dico->inst_symbols, &iter))
        {
            nupa_add_param(entry->symbol, entry->vl);
            dico_free_entry(entry);
        }

        nghash_free(dico->inst_symbols, NULL, NULL);
        dico->inst_symbols = NULL;
    }
}


char *
nupa_copy(struct card *deck)
/* returns a copy (not quite) of s in freshly allocated memory.
   linenum, for info only, is the source line number.
   origin pointer s is kept, memory is freed later in nupa_done.
   must abort all Spice if malloc() fails.
   :{ called for the first time sequentially for all spice deck lines.
   :{ then called again for all X invocation lines, top-down for
   subckts defined at the outer level, but bottom-up for local
   subcircuit expansion, but has no effect in that phase.
   we steal a copy of the source line pointer.
   - comment-out a .param or & line
   - substitute placeholders for all {..} --> 10-digit numeric values.
*/
{
    char * const s = deck->line;
    const int linenum = deck->linenum;

    char *t;
    int ls;
    char c, d;
    SPICE_DSTRING u;

    spice_dstring_init(&u);
    ls = (int) strlen(s);

    while ((ls > 0) && (s[ls - 1] <= ' '))
        ls--;

    pscopy(&u, s, 0, ls);       /* strip trailing space, CrLf and so on */
    dicoS->srcline = linenum;

    if ((!inexpansionS) && (linenum >= 0) && (linenum <= dynmaxline)) {
        linecountS++;
        dicoS->dynrefptr[linenum] = deck->line;
        c = transform(dicoS, &u, incontrolS);
        if (c == 'C')
            incontrolS = 1;
        else if (c == 'E')
            incontrolS = 0;

        if (incontrolS)
            c = 'C';            /* force it */

        d = dicoS->dyncategory[linenum]; /* warning if already some strategic line! */

        if ((d == 'P') || (d == 'S') || (d == 'X'))
            fprintf(stderr,
                    " Numparam warning: overwriting P,S or X line (linenum == %d).\n",
                    linenum);
        dicoS->dyncategory[linenum] = c;
    } /* keep a local copy and mangle the string */

    t = strdup(spice_dstring_value(&u));

    if (t == NULL) {
        fputs("Fatal: String malloc crash in nupa_copy()\n", stderr);
        controlled_exit(EXIT_FAILURE);
    } else {
        if (!inexpansionS)
            putlogfile(dicoS->dyncategory[linenum], linenum, t);
    }

    spice_dstring_free(&u);
    return t;
}


int
nupa_eval(struct card *card)
/* s points to a partially transformed line.
   compute variables if linenum points to a & or .param line.
   if ( the original is an X line,  compute actual params.;
   } else {  substitute any &(expr) with the current values.
   All the X lines are preserved (commented out) in the expanded circuit.
*/
{
    char *s = card->line;
    int linenum = card->linenum;
    int orig_linenum = card->linenum_orig;

    int idef;                   /* subckt definition line */
    char c;
    bool err = 1;

    dicoS->srcline = linenum;
    dicoS->oldline = orig_linenum;

    c = dicoS->dyncategory[linenum];

#ifdef TRACE_NUMPARAMS
    fprintf(stderr, "** SJB - in nupa_eval()\n");
    fprintf(stderr, "** SJB - processing line %3d: %s\n", linenum, s);
    fprintf(stderr, "** SJB - category '%c'\n", c);
#endif

    if (c == 'P') {                     /* evaluate parameters */
        // err = nupa_substitute(dico, dico->dynrefptr[linenum], s);
        nupa_assignment(dicoS, dicoS->dynrefptr[linenum], 'N');
    } else if (c == 'B') {              /* substitute braces line */
        err = nupa_substitute(dicoS, dicoS->dynrefptr[linenum], s);
    } else if (c == 'X') {
        /* compute args of subcircuit, if required */
        char *inst_name = copy_substring(s, skip_non_ws(s));
        *inst_name = 'x';

        strtoupper(inst_name);

        idef = findsubckt(dicoS, s);
        if (idef > 0)
            nupa_subcktcall(dicoS, dicoS->dynrefptr[idef], dicoS->dynrefptr[linenum], inst_name);
        else
            putlogfile('?', linenum, "  illegal subckt call.");
    } else if (c == 'U') {              /*  release local symbols = parameters */
        nupa_subcktexit(dicoS);
    }

    putlogfile('e', linenum, s);
    evalcountS++;

#ifdef TRACE_NUMPARAMS
    fprintf(stderr, "** SJB - leaving nupa_eval(): %s   %d\n", s, err);
    printf("** SJB -                  --> %s\n", s);
    printf("** SJB - leaving nupa_eval()\n\n");
#endif

    if (err)
        return 0;
    else
        return 1;
}


void
nupa_signal(int sig)
/* warning: deckcopy may come inside a recursion ! substart no! */
/* info is context-dependent string data */
{
    putlogfile('!', sig, " Nupa Signal");

    if (sig == NUPADECKCOPY) {
        if (firstsignalS) {
            nupa_init();
            firstsignalS = 0;
        }
    } else if (sig == NUPASUBSTART) {
        inexpansionS = 1;
    } else if (sig == NUPASUBDONE) {
        inexpansionS = 0;
    } else if (sig == NUPAEVALDONE) {
        nupa_done();
        firstsignalS = 1;
    }
}


/* Store dicoS for each circuit loaded.
   The return value will be stored in ft_curckt->ci_dicos.
   We need to keep dicoS because it may be used by measure. */
int
nupa_add_dicoslist(void)
{
    int i;
    for (i = 0; i < 100; i++)
        if (dicos_list[i] == NULL) {
            dicos_list[i] = dicoS;
            break;
        }

    return (i);
}


/* remove dicoS from list if circuit is removed */
void
nupa_rem_dicoslist(int ir)
{
    dicos_list[ir] = NULL;
}


/* change dicoS to the active circuit */
void
nupa_set_dicoslist(int ir)
{
    dicoS = dicos_list[ir];
}
