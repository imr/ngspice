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
#include "ngspice/compatmode.h"

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
stripsomespace(DSTRINGPTR dstr_p, bool incontrol)
{
    /* if s starts with one of some markers, strip leading space */

    const char *markers =
        incontrol
        ? "*.&+#$"
        : "*.&+#$" "xX";

    char *s = ds_get_buf(dstr_p);

    int i = 0;
    while (s[i] && ((unsigned char)s[i] <= ' '))
        i++;

    if ((i > 0) && s[i] && strchr(markers, s[i]))
        pscopy(dstr_p, s + i, NULL);
}


static int
stripbraces(DSTRINGPTR dstr_p)
/* puts the funny placeholders. returns the number of {...} substitutions */
{
    int n = 0;
    char *s = ds_get_buf(dstr_p);
    char *p, *brace;
    DS_CREATE(tstr, 200);
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

        pscopy(&tstr, s, brace);

        if ((unsigned char)brace[-1] > ' ')
            cadd(&tstr, ' ');

        cadd(&tstr, ' ');
        {
            char buf[25+1];
            sprintf(buf, "numparm__________%08lx", ++placeholder);
            sadd(&tstr, buf);
        }
        cadd(&tstr, ' ');

        if ((unsigned char)(* j_ptr) >= ' ')
            cadd(&tstr, ' ');

        int ilen = (int) ds_get_length(&tstr);
        sadd(&tstr, j_ptr);
        scopyd(dstr_p, &tstr);
        s = ds_get_buf(dstr_p);
        p = s + ilen;
    }

    dynsubst = placeholder;
    ds_free(&tstr);

    return n;
}


static void
findsubname(dico_t *dico, DSTRINGPTR dstr_p)
/* truncate the parameterized subckt call to regular old Spice */
/* scan a string from the end, skipping non-idents and {expressions} */
/* then truncate s after the last subckt(?) identifier */
{
    char * const s = ds_get_buf(dstr_p);
    char *p = s + ds_get_length(dstr_p);

    DS_CREATE(name, 200); /* extract a name */

    while (p > s) {

        /* skip space, then non-space */
        char *p_end = p = skip_back_ws(p, s); /* at p_end: space */

        while ((p > s) && !isspace_c(p[-1]))
            if (p[-1] == '}') {
                int nest = 1;
                while (--p > s) {
                    if (p[-1] == '{')
                        nest--;
                    else if (p[-1] == '}')
                        nest++;
                    if (nest <= 0) {
                        p--;
                        break;
                    }
                }
                p_end = p;      /* p_end points to '{' */
            } else {
                p--;
            }

        if ((p > s) && alfanum(*p)) { /* suppose an identifier */
            char *t;
            entry_t *entry;
            /* check for known subckt name */
            if (newcompat.ps)
                for (t = p; alfanumps(*t); t++)
                    ;
            else
                for (t = p; alfanum(*t); t++)
                    ;
            ds_clear(&name);
            pscopy(&name, p, t);
            entry = entrynb(dico, ds_get_buf(&name));
            if (entry && (entry->tp == NUPA_SUBCKT)) {
                (void) ds_set_length(dstr_p, (size_t) (p_end - s));
                ds_free(&name);
                return;
            }
        }
    }

    ds_free(&name);
}


static char
transform(dico_t *dico, DSTRINGPTR dstr_p, bool incontrol)
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

    s = ds_get_buf(dstr_p);

    if (s[0] == '.') {
        /* check PS parameter format */
        if (prefix(".param", s)) {
            /* comment it out */
            /* s[0] = '*'; */
            category = 'P';
        } else if (prefix(".subckt", s)) {
            char *params;
            /* split off any "params" tail */
            params = strstr(s, "params:");
            if (params) {
                ds_set_length(dstr_p, (size_t) (params - s));
            }
            category = 'S';
        } else if (prefix(".control", s)) {
            category = 'C';
        } else if (prefix(".endc", s)) {
            category = 'E';
        } else if (prefix(".ends", s)) {
            category = 'U';
        } else {
            category = '.';
            if (stripbraces(dstr_p) > 0)
                category = 'B'; /* priority category ! */
        }
    } else if (s[0] == 'x') {
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
static bool inexpansionS = 0;   /* flag subckt expansion phase */
static bool incontrolS = 0;     /* flag control code sections */
static bool firstsignalS = 1;
static dico_t *dicoS = NULL;
static dico_t *dicos_list[100];


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

    /* We cannot remove dicoS here because numparam is used by
       the .measure statements, which are invoked only after the
       simulation has finished. */

    if (nerrors) {
        bool is_interactive = FALSE;
        if (cp_getvar("interactive", CP_BOOL, NULL, 0))
            is_interactive = TRUE;
        if (ft_ngdebug)
            printf(" Copies=%d Evals=%d Placeholders=%ld Symbols=%d Errors=%d\n",
                linecountS, evalcountS, placeholder, dictsize, nerrors);
        /* debug: ask if spice run really wanted */
        if (ft_batchmode)
            controlled_exit(EXIT_FAILURE);
        if (!is_interactive) {
            if (ft_ngdebug) {
                fprintf(cp_err, "Numparam expansion errors: Problem with the input netlist.\n");
            }
            else {
                fprintf(cp_err, "    Please check your input netlist.\n");
            }
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
nupa_scan(const struct card *card)
{
    defsubckt(dicoS, card);
}


/* -----------------------------------------------------------------
 * Dump the contents of a symbol table.
 * ----------------------------------------------------------------- */
static void
dump_symbol_table(NGHASHPTR htable_p, FILE *fp)
{
    entry_t *entry;             /* current entry */
    NGHASHITER iter;            /* hash iterator - thread safe */

    NGHASH_FIRST(&iter);
    for (entry = (entry_t *) nghash_enumerateRE(htable_p, &iter);
         entry;
         entry = (entry_t *) nghash_enumerateRE(htable_p, &iter))
    {
        if (entry->tp == NUPA_REAL)
            fprintf(fp, "       ---> %s = %g\n", entry->symbol, entry->vl);
        else if (entry->tp == NUPA_STRING)
            fprintf(fp, "       ---> %s = \"%s\"\n",
                    entry->symbol, entry->sbbase);
    }
}


/* -----------------------------------------------------------------
 * Dump the contents of the symbol table.
 * ----------------------------------------------------------------- */
void
nupa_list_params(FILE *fp)
{
    dico_t *dico = dicoS;       /* local copy for speed */
    int depth;                  /* nested subcircit depth */

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
            dump_symbol_table(htable_p, fp);
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
static entry_t *nupa_get_entry(const char *param_name)
{
    dico_t *dico = dicoS;       /* local copy for speed */
    int depth;                  /* nested subcircit depth */

    for (depth = dico->stack_depth; depth >= 0; depth--) {
        NGHASHPTR htable_p = dico->symbols[depth];
        if (htable_p) {
            entry_t *entry;

            entry = (entry_t *)nghash_find(htable_p, (void *)param_name);
            if (entry)
                return entry;
        }
    }
    return NULL;
}

double
nupa_get_param(const char *param_name, int *found)
{
    entry_t *entry = nupa_get_entry(param_name);
    if (entry && entry->tp == NUPA_REAL) {
        *found = 1;
        return entry->vl;
    }
    *found = 0;
    return 0;
}

const char *
nupa_get_string_param(const char *param_name)
{
    entry_t *entry = nupa_get_entry(param_name);
    if (entry && entry->tp == NUPA_STRING)
        return entry->sbbase;
    return NULL;
}


static void
nupa_copy_entry(entry_t *proto)
{
    dico_t *dico = dicoS;       /* local copy for speed */
    entry_t *entry;             /* current entry */
    NGHASHPTR htable_p;         /* hash table of interest */

    /* can't be lazy anymore */
    if (!(dico->symbols[dico->stack_depth]))
        dico->symbols[dico->stack_depth] = nghash_init(NGHASH_MIN_SIZE);

    htable_p = dico->symbols[dico->stack_depth];

    entry = attrib(dico, htable_p, proto->symbol, 'N');
    if (entry) {
        entry->vl = proto->vl;
        entry->tp = proto->tp;
        entry->ivl = proto->ivl;
        entry->sbbase = proto->sbbase;
    }
}


void
nupa_add_param(char *param_name, double value)
{
    entry_t entry;

    entry.symbol = param_name;
    entry.vl = value;
    entry.tp = NUPA_REAL;
    entry.ivl = 0;
    entry.sbbase = NULL;
    nupa_copy_entry(&entry);
}


void
nupa_copy_inst_entry(char *param_name, entry_t *proto)
{
    dico_t *dico = dicoS;       /* local copy for speed */
    entry_t *entry;             /* current entry */

    if (!(dico->inst_symbols))
        dico->inst_symbols = nghash_init(NGHASH_MIN_SIZE);

    entry = attrib(dico, dico->inst_symbols, param_name, 'N');
    if (entry) {
        entry->vl = proto->vl;
        entry->tp = proto->tp;
        entry->ivl = proto->ivl;
        entry->sbbase = proto->sbbase;
    }
}


/* -----------------------------------------------------------------
 * This function copies any definitions in the inst_symbols hash
 * table which are qualified symbols and makes them available at
 * the global level.  Afterwards, the inst_symbols table is freed.
 * ----------------------------------------------------------------- */
void
nupa_copy_inst_dico(void)
{
    dico_t *dico = dicoS;       /* local copy for speed */
    entry_t *entry;             /* current entry */
    NGHASHITER iter;            /* hash iterator - thread safe */

    if (dico->inst_symbols) {
        /* We we perform this operation we should be in global scope */
        if (dico->stack_depth > 0)
            fprintf(stderr, "stack depth should be zero.\n");

        NGHASH_FIRST(&iter);
        for (entry = (entry_t *) nghash_enumerateRE(dico->inst_symbols, &iter);
             entry;
             entry = (entry_t *) nghash_enumerateRE(dico->inst_symbols, &iter))
        {
            nupa_copy_entry(entry);
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
    char * const s_end = skip_back_ws(s + strlen(s), s);
    const int linenum = deck->linenum;

    char *t;
    char c, d;

    DS_CREATE(u, 200);

    pscopy(&u, s, s_end);       /* strip trailing space, CrLf and so on */
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

    t = copy(ds_get_buf(&u));

    if (!t) {
        fputs("Fatal: String malloc crash in nupa_copy()\n", stderr);
        controlled_exit(EXIT_FAILURE);
    }

    ds_free(&u);
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
        nupa_assignment(dicoS, dicoS->dynrefptr[linenum], 'N');
    } else if (c == 'B') {              /* substitute braces line */
        /* nupa_substitute() may reallocate line buffer. */

        err = nupa_substitute(dicoS, dicoS->dynrefptr[linenum], &card->line);
        s = card->line;
    } else if (c == 'X') {
        /* compute args of subcircuit, if required */
        char *inst_name = copy_substring(s, skip_non_ws(s));
        *inst_name = 'x';

        idef = findsubckt(dicoS, s);
        if (idef > 0)
            nupa_subcktcall(dicoS, dicoS->dynrefptr[idef], dicoS->dynrefptr[linenum], inst_name);
        else
            fprintf(stderr, "Error, illegal subckt call.\n  %s\n", s);
    } else if (c == 'U') {              /*  release local symbols = parameters */
        nupa_subcktexit(dicoS);
    }

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
