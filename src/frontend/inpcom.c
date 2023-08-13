/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher
**********/

/*
  For dealing with spice input decks and command scripts

  Central function is inp_readall()
*/

/* Note: Must include shlwapi.h before ngspice header defining BOOL due
 * to conflict */
#include <stdio.h>
#ifdef _WIN32
#include <shlwapi.h> /* for definition of PathIsRelativeA() */
#pragma comment(lib, "Shlwapi.lib")
#endif

#include "ngspice/ngspice.h"

#include "ngspice/compatmode.h"
#include "ngspice/cpdefs.h"
#include "ngspice/dstring.h"
#include "ngspice/dvec.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteext.h"
#include "ngspice/fteinp.h"
#include "numparam/general.h"

#include <limits.h>
#include <stdlib.h>

#include <sys/stat.h>
#include <sys/types.h>

#if !defined(__MINGW32__) && !defined(_MSC_VER)
#include <unistd.h>
#endif

#include "../misc/util.h" /* ngdirname() */
#include "inpcom.h"
#include "ngspice/stringskip.h"
#include "ngspice/stringutil.h"
#include "ngspice/wordlist.h"
#include "subckt.h"
#include "variable.h"

#include "inpcompat.h"

#ifdef XSPICE
/* gtri - add - 12/12/90 - wbk - include new stuff */
#include "ngspice/enh.h"
#include "ngspice/ipctiein.h"
/* gtri - end - 12/12/90 */
#endif

/* SJB - Uncomment this line for debug tracing */
/*#define TRACE*/

/* globals -- wanted to avoid complicating inp_readall interface */
#define N_LIBRARIES       1000
#define N_PARAMS          1000
#define N_SUBCKT_W_PARAMS 4000

#define NPARAMS 10000
#define FCN_PARAMS 1000

#define DEPENDSON 200

#define VALIDCHARS "!$%_#?@.[]&"

static struct library {
    char *realpath;
    char *habitat;
    struct card *deck;
} libraries[N_LIBRARIES];

static int num_libraries;

struct names {
    char *names[N_SUBCKT_W_PARAMS];
    int num_names;
};

struct function_env
{
    struct function_env *up;

    struct function {
        struct function *next;
        char *name;
        char *body;
        char *params[N_PARAMS];
        int num_parameters;
        const char *accept;
    } *functions;
};

struct func_temper
{
    char *funcname;
    int subckt_depth;
    int subckt_count;
    struct func_temper *next;
};

extern void line_free_x(struct card *deck, bool recurse);

/* Collect information for dynamic allocation of numparam arrays */
/* number of lines in input deck */
int dynmaxline; /* inpcom.c 1529 */
/* number of lines in deck after expansion */
int dynMaxckt = 0; /* subckt.c 307 */
/* number of parameter substitutions */
long dynsubst; /* spicenum.c 221 */

static bool has_if = FALSE; /* if we have an .if ... .endif pair */

static char *readline(FILE *fd);
int get_number_terminals(char *c);
static void inp_stripcomments_deck(struct card *deck, bool cs);
static void inp_stripcomments_line(char *s, bool cs);
static void inp_fix_for_numparam(
        struct names *subckt_w_params, struct card *deck);
static void inp_remove_excess_ws(struct card *deck);
static void expand_section_references(struct card *deck,
        const char *dir_name);
static void inp_grab_func(struct function_env *, struct card *deck);
static void inp_fix_inst_calls_for_numparam(
        struct names *subckt_w_params, struct card *deck);
static void inp_expand_macros_in_func(struct function_env *);
static struct card *inp_expand_macros_in_deck(
        struct function_env *, struct card *deck);
static void inp_fix_param_values(struct card *deck);
static void inp_reorder_params(
        struct names *subckt_w_params, struct card *list_head);
static int inp_split_multi_param_lines(struct card *deck, int line_number);
static void inp_sort_params(struct card *param_cards,
        struct card *card_bf_start, struct card *s_c, struct card *e_c);
static void inp_compat(struct card *deck);
static void inp_bsource_compat(struct card *deck);
static bool inp_temper_compat(struct card *card);
static void inp_meas_current(struct card *card);
static void inp_dot_if(struct card *deck);
static char *inp_modify_exp(char *expression);
static struct func_temper *inp_new_func(char *funcname, char *funcbody,
        struct card *card, int *sub_count, int subckt_depth);
static void inp_delete_funcs(struct func_temper *funcs);

static bool chk_for_line_continuation(char *line);
void comment_out_unused_subckt_models(struct card *start_card);
static char inp_get_elem_ident(char *type);
static void rem_mfg_from_models(struct card *start_card);
static void inp_fix_macro_param_func_paren_io(struct card *begin_card);
static void inp_fix_gnd_name(struct card *deck);
static void inp_chk_for_e_source_to_xspice(struct card *deck, int *line_number);
static void inp_add_control_section(struct card *deck, int *line_number);
static char *get_quoted_token(char *string, char **token);
static void replace_token(char *string, char *token, int where, int total);
static void inp_add_series_resistor(struct card *deck);
static void subckt_params_to_param(struct card *deck);
static void inp_fix_temper_in_param(struct card *deck);
static void inp_fix_agauss_in_param(struct card *deck, char *fcn);
static int inp_vdmos_model(struct card *deck);
static void inp_check_syntax(struct card *deck);

static char *inp_spawn_brace(char *s);

static char *inp_pathresolve_at(const char *name, const char *dir);

struct nscope *inp_add_levels(struct card *deck);
static struct card_assoc *find_subckt(struct nscope *scope, const char *name);
void inp_rem_levels(struct nscope *root);
void inp_rem_unused_models(struct nscope *root, struct card *deck);
static struct modellist *inp_find_model(
        struct nscope *scope, const char *name);

void tprint(struct card *deck);
static char* libprint(struct card* t, const char *dir);

static void inp_repair_dc_ps(struct card* oldcard);
static void inp_get_w_l_x(struct card* oldcard);

static char* eval_m(char* line, char* tline);
static char* eval_tc(char* line, char* tline);
static char* eval_mvalue(char* line, char* tline);

extern void inp_probe(struct card* card);
#ifndef EXT_ASC
static void utf8_syntax_check(struct card *deck);
#endif

struct inp_read_t {
    struct card *cc;
    int line_number;
};

struct inp_read_t inp_read( FILE *fp, int call_depth, const char *dir_name,
        bool comfile, bool intfile);


#ifdef XSPICE
static int inp_poly_2g6_compat(struct card* deck);
#else
static void inp_poly_err(struct card *deck);
#endif

#ifdef CIDER
static char *keep_case_of_cider_param(char *buffer)
{
    int numq = 0, keep_case = 0;
    char *s = 0;
    /* Retain the case of strings enclosed in double quotes for
       output rootfile and doping infile params within Cider .model
       statements. Also for the ic.file filename param in an element
       instantiation statement.
       No nested double quotes.
    */
    for (s = buffer; *s && (*s != '\n'); s++) {
        if (*s == '\"') {
            numq++; 
        }
    }
    if (numq == 2) {
        /* One pair of double quotes */
        for (s = buffer; *s && (*s != '\n'); s++) {
            if (*s == '\"') {
                keep_case = (keep_case == 0 ? 1 : 0); 
            }
            if (!keep_case) {
                *s = tolower_c(*s);
            }
        }
    } else {
        for (s = buffer; *s && (*s != '\n'); s++) {
            *s = tolower_c(*s);
        }
    }
    return s;
}

static int is_comment_or_blank(char *buffer)
{
    /* Assume line buffers have initial whitespace removed */
    switch (buffer[0]) {
        case '*':
        case '$':
        case '#':
        case '\n':
        case '\0':
            return 1;
        default:
            return 0;
    }
}

static int turn_off_case_retention(char *buffer)
{
    if (!buffer) {
        return 1;
    }
    if (buffer[0] == '.') {
        if (ciprefix(".model", buffer)) {
            return 0;
        } else {
            return 1;
        }
    } else if (is_comment_or_blank(buffer)) {
        return 0;
    } else if (buffer[0] == '+') {
        return 0;
    } else {
        return 1;
    }
}

static char *make_lower_case_copy(char *inbuf)
{
    char *s = NULL;
    char *rets = NULL;
    size_t lenb = 0;

    if (!inbuf) {
        return NULL;
    }
    lenb = strlen(inbuf);
    if (lenb < 1) {
        return NULL;
    }
    rets = dup_string(inbuf, lenb);
    if (!rets) {
        return NULL;
    }
    for (s = rets; *s; s++) {
        *s = tolower_c(*s);
    }
    return rets;
}

static int ignore_line(char *buf)
{
    /* Can the line in buf be ignored for ic.file checking?
       Expect to examine only diode, mos, bipolar instance lines.
       If the ic.file param is on a continuation line, it will be missed.
       This should be rare.
    */
    if (!buf) {
        return 1;
    }
    if (buf[0] == '.') {
        return 1;
    }
    if (is_comment_or_blank(buf)) {
        return 1;
    }
    /* Interpreter d.., q.., m.. */
    switch (buf[0]) {
        case 'D':
        case 'd':
            if (ciprefix("dc", buf)
             || ciprefix("dowhile", buf) || ciprefix("define", buf)
             || ciprefix("deftype", buf) || ciprefix("delete", buf)
             || ciprefix("destroy", buf) || ciprefix("devhelp", buf)
             || ciprefix("diff", buf)    || ciprefix("display", buf)
            ) {
                return 1;
            } else {
                return 0;
            }
            break;
        case 'M':
        case 'm':
            if (ciprefix("mc_source", buf)  || ciprefix("meas", buf)
             || ciprefix("mdump", buf)      || ciprefix("mrdump", buf)
            ) {
                return 1;
            } else {
                return 0;
            }
            break;
        case 'Q':
        case 'q':
            if (ciprefix("quit", buf)) {
                return 1;
            } else {
                return 0;
            }
            break;
        default:
            break;
    }
    return 1;
}

static int line_contains_icfile(char *buf)
{
    /* Find "ic.file" in a lower cased copy of buf. */
    char str[] = "ic.file";
    char *s = NULL;

    if (ignore_line(buf)) {
        return 0;
    }
    /* make_lower_case_copy checks its input string */
    s = make_lower_case_copy(buf);
    if (!s) {
        return 0;
    }
    if (strstr(s, str)) {
        tfree(s);
        return 1;
    } else {
        tfree(s);
        return 0;
    }
}

static int is_cider_model(char *buf)
{
    /* Expect numos, numd, nbjt to be on the same line as the .model.
       Otherwise it will be missed if on a continuation line.
       This should be rare.
    */
    char *s;
    if (!ciprefix(".model", buf)) {
        return 0;
    }
    s = make_lower_case_copy(buf);
    if (!s) return 0;
    if (strstr(s, "numos") || strstr(s, "numd") || strstr(s, "nbjt")) {
        tfree(s);
        return 1;
    } else {
        tfree(s);
        return 0;
    }
}
#endif

/* Insert a new card, just behind the given card.
 * The new card takes ownership of the memory pointed to by "line".
 */

struct card *insert_new_line(
        struct card *card, char *line, int linenum, int linenum_orig)
{
    struct card *x = TMALLOC(struct card, 1);

    x->nextcard = card ? card->nextcard : NULL;
    x->error = NULL;
    x->actualLine = NULL;
    x->line = line;
    x->linenum = linenum;
    x->linenum_orig = linenum_orig;
    x->level = card ? card->level : NULL;

    if (card)
        card->nextcard = x;

    return x;
}


/* insert new_card, just behind the given card */
static struct card *insert_deck(struct card *card, struct card *new_card)
{
    if (card) {
        new_card->nextcard = card->nextcard;
        card->nextcard = new_card;
    }
    else {
        new_card->nextcard = NULL;
    }
    return new_card;
}


static struct library *new_lib(void)
{
    if (num_libraries >= N_LIBRARIES) {
        fprintf(stderr, "ERROR, N_LIBRARIES overflow\n");
        controlled_exit(EXIT_FAILURE);
    }

    return &libraries[num_libraries++];
}


static void delete_libs(void)
{
    int i;

    for (i = 0; i < num_libraries; i++) {
        tfree(libraries[i].realpath);
        tfree(libraries[i].habitat);
        line_free_x(libraries[i].deck, TRUE);
    }
}


static struct library *find_lib(char *name)
{
    int i;

    for (i = 0; i < num_libraries; i++)
        if (cieq(libraries[i].realpath, name))
            return &libraries[i];

    return NULL;
}


static struct card *find_section_definition(struct card *c, char *name)
{
    for (; c; c = c->nextcard) {

        char *line = c->line;

        if (ciprefix(".lib", line)) {

            char *s, *t, *y;

            s = skip_non_ws(line);
            while (isspace_c(*s) || isquote(*s))
                s++;
            for (t = s; *t && !isspace_c(*t) && !isquote(*t); t++)
                ;
            y = t;
            while (isspace_c(*y) || isquote(*y))
                y++;

            if (!*y) {
                /* library section definition: `.lib <section-name>' ..
                 * `.endl' */

                char keep_char = *t;
                *t = '\0';

                if (strcasecmp(name, s) == 0) {
                    *t = keep_char;
                    return c;
                }

                *t = keep_char;
            }
        }
    }

    return NULL;
}

static struct library *read_a_lib(const char *y, const char *dir_name)
{
    char *yy, *y_resolved;

    struct library *lib;

    y_resolved = inp_pathresolve_at(y, dir_name);

    if (!y_resolved) {
        fprintf(cp_err, "Error: Could not find library file %s\n", y);
        return NULL;
    }

#if defined(_WIN32)
    yy = _fullpath(NULL, y_resolved, 0);
#else
    yy = realpath(y_resolved, NULL);
#endif

    if (!yy) {
        fprintf(cp_err, "Error: Could not `realpath' library file %s\n", y);
        controlled_exit(EXIT_FAILURE);
    }

    lib = find_lib(yy);

    if (!lib) {

        FILE *newfp = fopen(y_resolved, "r");

        if (!newfp) {
            fprintf(cp_err, "Error: Could not open library file %s\n", y);
            return NULL;
        }

        /* lib points to a new entry in global lib array
         * libraries[N_LIBRARIES] */
        lib = new_lib();

        lib->realpath = copy(yy);
        lib->habitat = ngdirname(yy);

        lib->deck =
                inp_read(newfp, 1 /*dummy*/, lib->habitat, FALSE, FALSE).cc;

        fclose(newfp);
    }

    txfree(yy);
    txfree(y_resolved);

    return lib;
} /* end of function read_a_lib */



static struct names *new_names(void)
{
    struct names *p = TMALLOC(struct names, 1);
    p->num_names = 0;

    return p;
}


static void delete_names(struct names *p)
{
    int i;
    for (i = 0; i < p->num_names; i++)
        tfree(p->names[i]);
    tfree(p);
}

#ifndef _MSC_VER
/* concatenate 2 strings, with space if spa == TRUE,
   return malloced string (replacement for tprintf,
   which is not efficient enough when reading PDKs
   under Linux) */
static char *cat2strings(char *s1, char *s2, bool spa)
{
   if (s2 == NULL || *s2 == '\0') {
        return copy(s1);
    }
    else if (s1 == NULL || *s1 == '\0') {
        return copy(s2);
    }
    size_t l1 = strlen(s1);
    size_t l2 = strlen(s2);
    char *strsum = TMALLOC(char, l1 + l2 + 2);
    if (spa) {
        memcpy(strsum, s1, l1);
        memcpy(strsum + l1 + 1, s2, l2);
        strsum[l1] = ' ';
        strsum[l1 + l2 + 1] = '\0';
    }
    else {
        memcpy(strsum, s1, l1);
        memcpy(strsum + l1, s2, l2);
        strsum[l1 + l2] = '\0';
    }
    return strsum;
}
#endif


/* line1
   + line2
   ---->
   line1 line2
   Proccedure: store regular card in prev, skip comment lines (*..) and some
   others, add tokens from + lines to prev using dstring.
   */
static void inp_stitch_continuation_lines(struct card* working)
{
    struct card* prev = NULL;
    bool firsttime = TRUE;

    DS_CREATE(newline, 200);

    while (working) {
        char* s, c;

        for (s = working->line; (c = *s) != '\0' && c <= ' '; s++)
            ;

#ifdef TRACE
        /* SDB debug statement */
        printf("In inp_read, processing linked list element line = %d, s = "
            "%s . . . \n",
            working->linenum, s);
#endif

        switch (c) {
        case '#':
        case '$':
        case '*':
        case '\0':
            /* skip these cards, and keep prev as the last regular card */
            working = working->nextcard; /* for these chars, go to next
                                            card */
            break;

        case '+': /* handle continuation */
            if (!prev) {
                working->error =
                    copy("Illegal continuation line: ignored.");
                working = working->nextcard;
                break;
            }

            /* We now may have lept over some comment lines, which are
            located among the continuation lines. We have to delete them
            here to prevent a memory leak */
            while (prev->nextcard != working) {
                struct card* tmpl = prev->nextcard->nextcard;
                line_free_x(prev->nextcard, FALSE);
                prev->nextcard = tmpl;
            }

            if (firsttime) {
                sadd(&newline, prev->line);
                firsttime = FALSE;
            }
            else {
                /* replace '+' by space */
                *s = ' ';
                sadd(&newline, s);
                /* mark for later removal */
                *s = '*';
            }

            break;

        default: /* regular one-line card */
            if (!firsttime) {
                tfree(prev->line);
                prev->line = copy(ds_get_buf(&newline));
                ds_clear(&newline);
                firsttime = TRUE;
                /* remove final used '+' line, if regular line is following */
                struct card* tmpl = prev->nextcard->nextcard;
                line_free_x(prev->nextcard, FALSE);
                prev->nextcard = tmpl;
            }
            prev = working;
            working = working->nextcard;
            break;
        }
    }
    /* remove final used '+' line when no regular line is following */
    if (!firsttime) {
        tfree(prev->line);
        prev->line = copy(ds_get_buf(&newline));
    }
    ds_free(&newline);
}

#ifdef CIDER
/* Only if we have a CIDER .model line with regular structure
'.model modname modtype level',
with modtype being one of numos, numd, nbjt:
Concatenate lines
line1
   + line2
   ---->
   line1 line 2
Store the original lines in card->actualLine, to be used for
CIDER model parameter parsing in INPparseNumMod() of inpgmod.c
   */
static void inp_cider_models(struct card* working)
{
    struct card* prev = NULL;
    bool iscmod = FALSE;

    while (working) {
        char *s, c, *buffer;

        for (s = working->line; (c = *s) != '\0' && c <= ' '; s++)
            ;

        if(!iscmod)
            iscmod = is_cider_model(s);

#ifdef TRACE
        /* SDB debug statement */
        printf("In inp_read, processing linked list element line = %d, s = "
            "%s . . . \n",
            working->linenum, s);
#endif

        if (iscmod) {
            switch (c) {
            case '#':
            case '$':
            case '*':
            case '\0':
                /* skip these cards, and keep prev as the last regular card */
                working = working->nextcard; /* for these chars, go to next
                                                card */
                break;

            case '+': /* handle continuation */
                if (!prev) {
                    working->error =
                        copy("Illegal continuation line: ignored.");
                    working = working->nextcard;
                    break;
                }

                /* We now may have lept over some comment lines, which are
                located among the continuation lines. We have to delete them
                here to prevent a memory leak */
                while (prev->nextcard != working) {
                    struct card* tmpl = prev->nextcard->nextcard;
                    line_free_x(prev->nextcard, FALSE);
                    prev->nextcard = tmpl;
                }

                /* create buffer and write last and current line into it.
                   When reading a PDK, the following may be called more than 1e6 times. */
#if defined (_MSC_VER)
                   /* vsnprintf (used by tprintf) in Windows is efficient, VS2019 arb. referencevalue 7,
                      cat2strings() yields ref. speed value 12 only, CYGWIN is 12 in both cases,
                      MINGW is 36. */
                buffer = tprintf("%s %s", prev->line, s + 1);
#else
                   /* vsnprintf in Linux is very inefficient, ref. value 24
                      cat2strings() is efficient with  ref. speed value 6,
                      MINGW is 12 */
                buffer = cat2strings(prev->line, s + 1, TRUE);
#endif
                /* replace prev->line by buffer */
                s = prev->line;
                prev->line = buffer;
                prev->nextcard = working->nextcard;
                working->nextcard = NULL;
                /* add original line to prev->actualLine */
                if (prev->actualLine) {
                    struct card* end;
                    for (end = prev->actualLine; end->nextcard;
                        end = end->nextcard)
                        ;
                    end->nextcard = working;
                    tfree(s);
                }
                else {
                    prev->actualLine =
                        insert_new_line(NULL, s, prev->linenum, 0);
                    prev->actualLine->level = prev->level;
                    prev->actualLine->nextcard = working;
                }
                working = prev->nextcard;
                break;

            default: /* regular one-line card */
                prev = working;
                working = working->nextcard;
                iscmod = is_cider_model(s);
                break;
            }
        }
        else
            working = working->nextcard;
    }
}
#endif

/*
 * search for `=' assignment operator
 *   take care of `!=' `<=' `==' and `>='
 */

char *find_assignment(const char *str)
{
    const char *p = str;

    while ((p = strchr(p, '=')) != NULL) {

        // check for equality '=='
        if (p[1] == '=') {
            p += 2;
            continue;
        }

        // check for '!=', '<=', '>='
        if (p > str)
            if (p[-1] == '!' || p[-1] == '<' || p[-1] == '>') {
                p += 1;
                continue;
            }

        return (char *) p;
    }

    return NULL;
}


/*
 * backward search for an assignment
 *   fixme, doesn't honour neither " nor ' quotes
 */

char *find_back_assignment(const char *p, const char *start)
{
    while (--p >= start) {
        if (*p != '=')
            continue;
        // check for '!=', '<=', '>=', '=='
        if (p <= start || !strchr("!<=>", p[-1]))
            return (char *) p;
        p--;
    }

    return NULL;
}


/* We check x lines for nf=, w= and l= and fill in their values.
   To be used when expanding subcircuits with binned model cards. 
   
   In subckt.c, function doit(), lines 621ff. the unsused models
   are filtered out. 'nf' given on an x line (subcircuit invocation)
   is aknowledged. The option 'wnflag', if set to 0 in .spiceinit,
   will set 'nf' to 1 and thus suppress its usage.
   
   In inp.c, function rem_unused_mos_models, another trial to removing
   unused MOS models is given, this time on the expanded m lines and
   its models.*/
void inp_get_w_l_x(struct card* card) {
    int wnflag;
    if (!cp_getvar("wnflag", CP_NUM, &wnflag, 0)) {
        if (newcompat.spe || newcompat.hs)
            wnflag = 1;
        else
            wnflag = 0;
    }
    for (; card; card = card->nextcard) {
        char* curr_line = card->line;
        int skip_control = 0;
        char* w = NULL, * l = NULL, * nf = NULL;

        card->w = card->l = 0;
        card->nf = 1.;

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
        /* only subcircuit invocations */
        if (*curr_line != 'x' || (!newcompat.hs && !newcompat.spe)) {
            continue;
        }

        w = strstr(curr_line, " w=");
        if (w) {
            int err;
            w = w + 3;
            card->w = (float)INPevaluate(&w, &err, 0);
            if(err) { 
                card->w = 0;
                continue;
            }
        }
        else {
            card->w = card->l = 0;
            continue;
        }

        l = strstr(curr_line, " l=");
        if (l) {
            int err;
            l = l + 3;
            card->l = (float)INPevaluate(&l, &err, 0);
            if(err) { 
                card->l = 0;
                continue;
            }
        }
        else {
            card->w = card->l = 0;
            continue;
        }
        nf = strstr(curr_line, " nf=");
        if (nf) {
            int err;
            nf = nf + 4;
            card->nf = (float)INPevaluate(&nf, &err, 0);
            if (err) {
                card->w = card->l = 0;
                card->nf = 1.;
                continue;
            }
        }
        else {
            continue;
        }
    }
}


/*-------------------------------------------------------------------------
  Read the entire input file and return  a pointer to the first line of
  the linked list of 'card' records in data.  The pointer is stored in
  *data.
  Called from fcn inp_spsource() in inp.c to load circuit or command files.
  Called from fcn com_alter_mod() in device.c to load model files.
  Called from here to load .library or .include files.

  Procedure:
  read in all lines & put them in the struct cc
  read next line
  process .TITLE line
  store contents in string new_title
  process .lib lines
  read file and library name, open file using fcn inp_pathopen()
  read file contents and put into struct libraries[].deck, one entry per .lib line
  process .inc lines
  read file and library name, open file using fcn inp_pathopen()
  read file contents and add lines to cc
  make line entry lower case
  allow for shell end of line continuation (\\)
  add '+' to beginning of next line
  add line entry to list cc
  add '.global gnd'
  add libraries
  find library section
  add lines
  add .end card
  strip end-of-line comments
  make continuation lines a single line
  *** end of processing for command files ***
  start preparation of input deck for numparam
  ...
  debug printout to debug-out.txt
  remove the 'level' entries from each card
  *-------------------------------------------------------------------------*/

struct card *inp_readall(FILE *fp, const char *dir_name,
        bool comfile, bool intfile, bool *expr_w_temper_p)
{
    struct card *cc;
    struct inp_read_t rv;

    num_libraries = 0;
    /* set the members of the compatibility structure */
    set_compat_mode();

    rv = inp_read(fp, 0, dir_name, comfile, intfile);
    cc = rv.cc;

    /* files starting with *ng_script are user supplied command files */
    if (cc && ciprefix("*ng_script", cc->line))
        comfile = TRUE;

    /* The following processing of an input file is not required for command
       files like spinit or .spiceinit, so return command files here. */

    if (!comfile && cc) {

        unsigned int no_braces; /* number of '{' */
        size_t max_line_length; /* max. line length in input deck */
        struct card *tmp_ptr1;
        struct names *subckt_w_params = new_names();

        /* skip title line */
        struct card *working = cc->nextcard;

        print_compat_mode();

        delete_libs();

#ifndef EXT_ASC
        utf8_syntax_check(working);
#endif		

        /* some syntax checks, excluding title line */
        inp_check_syntax(working);

        if (newcompat.lt && newcompat.a)
            ltspice_compat_a(working);
        if (newcompat.ps && newcompat.a)
            pspice_compat_a(working);

        struct nscope *root = inp_add_levels(working);

        inp_probe(working);

        inp_fix_for_numparam(subckt_w_params, working);

        inp_remove_excess_ws(working);

        if(inp_vdmos_model(working)) {
            line_free_x(cc, TRUE);
            inp_rem_levels(root);
            return NULL;
        }

        /* don't remove unused model if we have an .if clause, because we
           cannot yet decide here which model we finally will need.
           There is another trial using these functions in inpc,
           when the netlist is expanded and flattened.*/
        if (!has_if) {
            comment_out_unused_subckt_models(working);
            inp_rem_unused_models(root, working);
        }

        if (newcompat.lt || newcompat.ps)
            rem_mfg_from_models(working);

        subckt_params_to_param(working);

        rv.line_number = inp_split_multi_param_lines(working, rv.line_number);

        inp_fix_macro_param_func_paren_io(working);

        static char *statfcn[] = {
                "agauss", "gauss", "aunif", "unif", "limit"};
        int ii;
        for (ii = 0; ii < 5; ii++)
            inp_fix_agauss_in_param(working, statfcn[ii]);

        inp_fix_temper_in_param(working);

        inp_expand_macros_in_deck(NULL, working);
        inp_fix_param_values(working);

        inp_reorder_params(subckt_w_params, cc);
//        tprint(working);
        /* Special handling for large PDKs: We need to know W and L of
           transistor subcircuits by checking their x invocation */
        inp_get_w_l_x(working);

        inp_fix_inst_calls_for_numparam(subckt_w_params, working);

        delete_names(subckt_w_params);
        subckt_w_params = NULL;
        if (!cp_getvar("no_auto_gnd", CP_BOOL, NULL, 0))
            inp_fix_gnd_name(working);
        inp_chk_for_e_source_to_xspice(working, &rv.line_number);

        /* "addcontrol" variable is set if "ngspice -a file" was used. */

        if (cp_getvar("addcontrol", CP_BOOL, NULL, 0)) {
            inp_add_control_section(working, &rv.line_number);
            cp_remvar("addcontrol"); // Use only for initial netlist
        }
#ifdef XSPICE
        if (inp_poly_2g6_compat(working)) {
            inp_rem_levels(root);
            line_free_x(cc, TRUE);
            return NULL;
        }
#else
        inp_poly_err(working);
#endif
        /* a preliminary fix: if ps is enabled, .dc TEMP -15 75 5 will
        have been modified to .dc (TEMPER) -15 75 5. So we repair it here. */
        if (newcompat.ps) {
            inp_repair_dc_ps(working);
        }
        bool expr_w_temper = FALSE;
        if (!newcompat.s3) {
            /* Do all the compatibility stuff here */
            working = cc->nextcard;
            inp_meas_current(working);
            /* E, G, L, R, C compatibility transformations */
            inp_compat(working);
            working = cc->nextcard;
            /* B source numparam compatibility transformation */
            inp_bsource_compat(working);
            inp_dot_if(working);
            expr_w_temper = inp_temper_compat(working);
        }
        if (expr_w_temper_p)
            *expr_w_temper_p = expr_w_temper;

        inp_add_series_resistor(working);

        /* get max. line length and number of lines in input deck,
           and renumber the lines,
           count the number of '{' per line as an upper estimate of the number
           of parameter substitutions in a line */
        dynmaxline = 0;
        max_line_length = 0;
        no_braces = 0;
        for (tmp_ptr1 = cc; tmp_ptr1; tmp_ptr1 = tmp_ptr1->nextcard) {
            char *s;
            unsigned int braces_per_line = 0;
            /* count number of lines */
            dynmaxline++;
            /* renumber the lines of the processed input deck */
            tmp_ptr1->linenum = dynmaxline;
            if (max_line_length < strlen(tmp_ptr1->line))
                max_line_length = strlen(tmp_ptr1->line);
            /* count '{' */
            for (s = tmp_ptr1->line; *s; s++)
                if (*s == '{')
                    braces_per_line++;
            if (no_braces < braces_per_line)
                no_braces = braces_per_line;
        }

        if (ft_ngdebug) {
            FILE *fd = fopen("debug-out.txt", "w");
            if (fd) {
                /*debug: print into file*/
                struct card *t;
                fprintf(fd,
                        "**************** uncommented deck "
                        "**************\n\n");
                /* always print first line */
                fprintf(fd, "%6d  %6d  %s\n", cc->linenum_orig, cc->linenum,
                        cc->line);
                /* here without out-commented lines */
                for (t = cc->nextcard; t; t = t->nextcard) {
                    if (*(t->line) == '*')
                        continue;
                    fprintf(fd, "%6d  %6d  %s\n",
                            t->linenum_orig, t->linenum, t->line);
                }
                fprintf(fd,
                        "\n****************** complete deck "
                        "***************\n\n");
                /* now completely */
                for (t = cc; t; t = t->nextcard)
                    fprintf(fd, "%6d  %6d  %s\n",
                            t->linenum_orig, t->linenum, t->line);
                fclose(fd);

                fprintf(stdout,
                        "max line length %d, max subst. per line %d, number "
                        "of lines %d\n",
                        (int) max_line_length, no_braces, dynmaxline);
            }
            else
                fprintf(stderr,
                        "Warning: Cannot open file debug-out.txt for saving "
                        "debug info\n");
        }
        inp_rem_levels(root);
    }

    return cc;
}


struct inp_read_t inp_read( FILE *fp, int call_depth, const char *dir_name,
        bool comfile, bool intfile)
/* fp: in, pointer to file to be read,
   call_depth: in, nested call to fcn
   dir_name: in, name of directory of file to be read
   comfile: in, TRUE if command file (e.g. spinit, .spiceinit)
   intfile: in, TRUE if deck is generated from internal circarray
*/
{
    struct inp_read_t rv;
    struct card *end = NULL, *cc = NULL;
    char *buffer = NULL;
    /* segfault fix */
#ifdef XSPICE
    char big_buff[5000];
    int line_count = 0;
#endif
    char *new_title = NULL;
    int line_number = 1;
            /* sjb - renamed to avoid confusion with struct card */
    int line_number_orig = 1;
    int cirlinecount = 0; /* length of circarray */
    static int is_control = 0; /* We are reading from a .control section */

    bool found_end = FALSE, shell_eol_continuation = FALSE;
#ifdef CIDER
    static int in_cider_model = 0;
#endif

    /* First read in all lines & put them in the struct cc */
    for (;;) {
        /* derive lines from circarray */
        if (intfile) {
            buffer = circarray[cirlinecount++];
            if (!buffer) {
                tfree(circarray);
                break;
            }
        }
        /* read lines from file fp */
        else {

#ifdef XSPICE
            /* gtri - modify - 12/12/90 - wbk - read from mailbox if ipc
             * enabled */

            /* If IPC is not enabled, do equivalent of what SPICE did before
             */
            if (!g_ipc.enabled) {
                if (call_depth == 0 && line_count == 0) {
                    line_count++;
                    if (fgets(big_buff, 5000, fp))
                        buffer = copy(big_buff);
                }
                else {
                    buffer = readline(fp);
                    if (!buffer)
                        break;
                }
            }
            else {
                /* else, get the line from the ipc channel. */
                /* We assume that newlines are not sent by the client */
                /* so we add them here */
                char ipc_buffer[1025]; /* Had better be big enough */
                int ipc_len;
                Ipc_Status_t ipc_status =
                        ipc_get_line(ipc_buffer, &ipc_len, IPC_WAIT);
                if (ipc_status == IPC_STATUS_END_OF_DECK) {
                    buffer = NULL;
                    break;
                }
                else if (ipc_status == IPC_STATUS_OK) {
                    buffer = TMALLOC(char, strlen(ipc_buffer) + 3);
                    strcpy(buffer, ipc_buffer);
                    strcat(buffer, "\n");
                }
                else { /* No good way to report this so just die */
                    fprintf(stderr, "Error: IPC status not o.k.\n");
                    controlled_exit(EXIT_FAILURE);
                }
            }

            /* gtri - end - 12/12/90 */
#else

            buffer = readline(fp);
            if (!buffer) {
                break;
            }

#endif
        }

#ifdef TRACE
        /* SDB debug statement */
        printf("in inp_read, just read   %s", buffer);
#endif

        if (!buffer) {
            continue;
        }

        /* OK -- now we have loaded the next line into 'buffer'.  Process it.
         */
        /* If input line is blank, ignore it & continue looping.  */
        if ((strcmp(buffer, "\n") == 0) || (strcmp(buffer, "\r\n") == 0))
            if (call_depth != 0 || (call_depth == 0 && cc != NULL)) {
                line_number_orig++;
                tfree(buffer); /* was allocated by readline() */
                continue;
            }

        if (*buffer == '@') {
            tfree(buffer); /* was allocated by readline() */
            break;
        }

        /* now check if we are in a .control section */
        if (ciprefix(".control", buffer))
            is_control++;
        else if (ciprefix(".endc", buffer))
            is_control--;

        /* now handle .title statement */
        if (ciprefix(".title", buffer)) {
            char *s;
            s = skip_non_ws(buffer); /* skip over .title */
            s = skip_ws(s); /* advance past space chars */

            /* only the last title line remains valid */
            tfree(new_title);
            new_title = copy(s);
            if ((s = strchr(new_title, '\n')) != NULL)
                *s = '\0';
            if ((s = strchr(new_title, '\r')) != NULL)
                *s = '\0';
            *buffer = '*'; /* change .TITLE line to comment line */
        }

        /* now handle old style .lib entries */
        /* new style .lib entries handling is in expand_section_references()
         */
        if (ciprefix(".lib", buffer))
            if (newcompat.lt || newcompat.ps) {
                /* In lt or ps there is no library section definition defined,
                 * so .lib is interpreted as old style .lib <file name> (no lib
                 * name given, .lib replaced by .include).
                 */
                char *s = skip_non_ws(buffer); /* skip over .lib */
                fprintf(cp_err, "  File included as:   .inc %s\n", s);
                memcpy(buffer, ".inc", 4);
            }

        /* now handle .include statements */
        if (ciprefix(".include", buffer) || ciprefix(".inc", buffer)) {

            char *y = NULL;
            char *s;

            struct card *newcard;

            inp_stripcomments_line(buffer, FALSE);

            s = skip_non_ws(buffer); /* advance past non-space chars */

            s = get_quoted_token(s, &y);

            if (!y) {
                fprintf(cp_err, "Error: .include filename missing\n");
                tfree(buffer); /* was allocated by readline() */
                controlled_exit(EXIT_FAILURE);
            }

            {
                char *y_resolved = inp_pathresolve_at(y, dir_name);
                char *y_dir_name;
                FILE *newfp;

                if (!y_resolved) {
                    fprintf(cp_err, "Error: Could not find include file %s\n",
                            y);
                    rv.line_number = line_number;
                    rv.cc = NULL;
                    return rv;
                }

                newfp = fopen(y_resolved, "r");

                if (!newfp) {
                    fprintf(cp_err, "Error: .include statement failed.\n");
                    tfree(buffer); /* allocated by readline() above */
                    controlled_exit(EXIT_FAILURE);
                }

                y_dir_name = ngdirname(y_resolved);

                newcard = inp_read(
                        newfp, call_depth + 1, y_dir_name, FALSE, FALSE)
                                  .cc; /* read stuff in include file into
                                          netlist */

                tfree(y_dir_name);
                tfree(y_resolved);

                (void) fclose(newfp);
            }

            /* Make the .include a comment */
            *buffer = '*';

            /* append `buffer' to the (cc, end) chain of decks */
            {
                end = insert_new_line(
                        end, copy(buffer), line_number, line_number);

                if (!cc)
                    cc = end;

                line_number++;
            }

            if (newcard) {
                if (newcompat.lt && !newcompat.a)
                    newcard = ltspice_compat(newcard);
                if (newcompat.ps && !newcompat.a)
                    newcard = pspice_compat(newcard);

                int line_number_inc = 1;
                end->nextcard = newcard;
                /* Renumber the lines */
                for (end = newcard; end && end->nextcard;
                        end = end->nextcard) {
                    end->linenum = line_number++;
                    end->linenum_orig = line_number_inc++;
                }
                end->linenum = line_number++; /* SJB - renumber last line */
                end->linenum_orig = line_number_inc++;
                                         /* SJB - renumber the last line */
            }

            /* Fix the buffer up a bit. */
            (void) memcpy(buffer + 1, "end of: ", 8);
        } /*  end of .include handling  */

        /* loop through 'buffer' until end is reached. Make all letters lower
         * case except for the commands given below. Special treatment for
         * commands 'hardcopy' and 'plot', where all letters are made lower
         * case except for the tokens following xlabel, ylabel and title.
         * These tokens may contain spaces, if they are enclosed in single or
         * double quotes. Single quotes are later on swallowed and disappear,
         * double quotes are printed. */
        {
            char *s;
#ifdef CIDER
            if (ciprefix(".model", buffer)) {
                in_cider_model = is_cider_model(buffer);
#ifdef TRACE
                printf("Found .model Cider model is %s\n",
                    (in_cider_model ? "ON" : "OFF"));
#endif
            }
            if (in_cider_model && turn_off_case_retention(buffer)) {
                in_cider_model = 0;
#ifdef TRACE
                printf("Cider model is OFF\n");
#endif
            }
#endif
            if (ciprefix("plot", buffer) || ciprefix("gnuplot", buffer) ||
                    ciprefix("hardcopy", buffer)) {
                /* lower case excluded for tokens following title, xlabel,
                 * ylabel. tokens may contain spaces, then they have to be
                 * enclosed in quotes. keywords and tokens have to be
                 * separated by spaces. */
                int j;
                char t = ' ';
                for (s = buffer; *s && (*s != '\n'); s++) {
                    *s = tolower_c(*s);
                    if (ciprefix("title", s)) {
                        /* jump beyond title */
                        for (j = 0; j < 5; j++) {
                            s++;
                            *s = tolower_c(*s);
                        }
                        while (*s == ' ')
                            s++;
                        if (!s || (*s == '\n'))
                            break;
                        /* check if single quote is at start of token */
                        else if (*s == '\'') {
                            s++;
                            t = '\'';
                        }
                        /* check if double quote is at start of token */
                        else if (*s == '\"') {
                            s++;
                            t = '\"';
                        }
                        else
                            t = ' ';
                        /* jump beyond token without lower casing */
                        while ((*s != '\n') && (*s != t))
                            s++;
                    }
                    else if (ciprefix("xlabel", s) || ciprefix("ylabel", s)) {
                        /* jump beyond xlabel, ylabel */
                        for (j = 0; j < 6; j++) {
                            s++;
                            *s = tolower_c(*s);
                        }
                        while (*s == ' ')
                            s++;
                        if (!s || (*s == '\n'))
                            break;
                        /* check if single quote is at start of token */
                        else if (*s == '\'') {
                            s++;
                            t = '\'';
                        }
                        /* check if double quote is at start of token */
                        else if (*s == '\"') {
                            s++;
                            t = '\"';
                        }
                        else
                            t = ' ';
                        /* jump beyond token without lower casing */
                        while ((*s != '\n') && (*s != t))
                            s++;
                    }
                }
            }
            else if (ciprefix("print", buffer) ||
                    ciprefix("eprint", buffer) ||
                    ciprefix("eprvcd", buffer) ||
                    ciprefix("asciiplot", buffer)) {
                /* lower case excluded for tokens following output redirection
                 * '>' */
                bool redir = FALSE;
                for (s = buffer; *s && (*s != '\n'); s++) {
                    if (*s == '>')
                        redir = TRUE; /* do not lower, but move to end of
                                         string */
                    if (!redir)
                        *s = tolower_c(*s);
                }
            }
#ifdef CIDER
            else if (in_cider_model && !is_comment_or_blank(buffer) &&
                    (ciprefix(".model", buffer) || buffer[0] == '+')) {
                s = keep_case_of_cider_param(buffer);
            }
            else if (line_contains_icfile(buffer)) {
                s = keep_case_of_cider_param(buffer);
            }
#endif
            /* no lower case letters for lines beginning with: */
            else if (!ciprefix("write", buffer) &&
                    !ciprefix("wrdata", buffer) &&
                    !ciprefix(".lib", buffer) && !ciprefix(".inc", buffer) &&
                    !ciprefix("codemodel", buffer) &&
                    !ciprefix("osdi", buffer) &&
                    !ciprefix("pre_osdi", buffer) &&
                    !ciprefix("echo", buffer) && !ciprefix("shell", buffer) &&
                    !ciprefix("source", buffer) && !ciprefix("cd ", buffer) &&
                    !ciprefix("load", buffer) && !ciprefix("setcs", buffer)) {
                /* lower case for all other lines */
                for (s = buffer; *s && (*s != '\n'); s++)
                    *s = tolower_c(*s);
            }
            else {
                /* s points to end of buffer for all cases not treated so far
                 */
                for (s = buffer; *s && (*s != '\n'); s++)
                    ;
            }

            /* add Inp_Path to buffer while keeping the sourcepath variable contents */
            if (ciprefix("set", buffer)) {
                char *p = strstr(buffer, "sourcepath");
                if (p) {
                    p = strchr(buffer, ')');
                    if (p) {
                        *p = 0; // clear ) and insert Inp_Path in between
                        p = tprintf("%s %s ) %s", buffer,
                                Inp_Path ? Inp_Path : "", p + 1);
                        tfree(buffer);
                        buffer = p;
                        /* s points to end of buffer */
                        for (s = buffer; *s && (*s != '\n'); s++)
                            ;
                    }
                    else {
                        fprintf(stderr, "Warning: no closing parens found in 'set sourcepath' statement\n");
                    }
                }
            }

            if (!*s) {
                // fprintf(cp_err, "Warning: premature EOF\n");
            }
            *s = '\0'; /* Zap the newline. */

            if ((s - 1) >= buffer && s[- 1] == '\r') {
                /* Zap the carriage return under windows */
                s[- 1] = '\0';
            }
        }

        /* find the true .end command out of .endc, .ends, .endl, .end
         * (comments may follow) */
        if (ciprefix(".end", buffer))
            if ((buffer[4] == '\0') || isspace_c(buffer[4])) {
                found_end = TRUE;
                *buffer = '*';
            }

        if (shell_eol_continuation) {
            char *new_buffer = tprintf("+%s", buffer);

            tfree(buffer);
            buffer = new_buffer;
        }

        /* If \\ at end of line is found, next line in loop will get + (see
         * code above) */
        shell_eol_continuation = chk_for_line_continuation(buffer);

        {
            end = insert_new_line(
                    end, copy(buffer), line_number++, line_number_orig++);

            if (!cc)
                cc = end;
        }

        tfree(buffer);
    } /* end while ((buffer = readline(fp)) != NULL) */

    if (!cc) /* No stuff here */
    {
        rv.line_number = line_number;
        rv.cc = cc;
        return rv;
    }

    /* files starting with *ng_script are user supplied command files */
    if (call_depth == 0 && ciprefix("*ng_script", cc->line))
        comfile = TRUE;

    if (call_depth == 0 && !comfile) {
        if (!cp_getvar("no_auto_gnd", CP_BOOL, NULL, 0))
            insert_new_line(cc, copy(".global gnd"), 1, 0);
        else
            insert_new_line(
                    cc, copy("* gnd is not set to 0 automatically "), 1, 0);

        if (!newcompat.lt && !newcompat.ps && !newcompat.s3) {
            /* process all library section references */
            expand_section_references(cc, dir_name);
        }
    }

    /*
      add a terminal ".end" card
    */

    if (call_depth == 0 && !comfile)
        if (found_end == TRUE)
            end = insert_new_line(
                    end, copy(".end"), line_number++, line_number_orig++);

    /* Replace first line with the new title, if available */
    if (call_depth == 0 && !comfile && new_title) {
        tfree(cc->line);
        cc->line = new_title;
    }

    /* Strip or convert end-of-line comments.
       Afterwards stitch the continuation lines.
       If the line only contains an end-of-line comment then it is converted
       into a normal comment with a '*' at the start.  Some special handling
       if this is a command file or called from within a .control section. */
    inp_stripcomments_deck(cc->nextcard, comfile || is_control);

#ifdef CIDER
    inp_cider_models(cc->nextcard);
#endif

    inp_stitch_continuation_lines(cc->nextcard);

    rv.line_number = line_number;
    rv.cc = cc;
    return rv;
}



/* Returns true if path is an absolute path and false if it is a
 * relative path. No check is done for the existance of the path. */
inline static bool is_absolute_pathname(const char *path)
{
#ifdef _WIN32
    return !PathIsRelativeA(path);
#else
    return path[0] == DIR_TERM;
#endif
} /* end of funciton is_absolute_pathname */



#if 0

static bool
is_plain_filename(const char *p)
{
#if defined(_WIN32)
    return
        !strchr(p, DIR_TERM) &&
        !strchr(p, DIR_TERM_LINUX);
#else
    return
        !strchr(p, DIR_TERM);
#endif
}

#endif


FILE *inp_pathopen(const char *name, const char *mode)
{
    char * const path = inp_pathresolve(name);

    if (path) {
        FILE *fp = fopen(path, mode);
        txfree(path);
        return fp;
    }

    return (FILE *) NULL;
} /* end of function inp_pathopen */


/* for MultiByteToWideChar */
#if defined(__MINGW32__) || defined(_MSC_VER)
#ifndef EXT_ASC
#undef BOOLEAN
#include <windows.h>
#endif
#endif

/*-------------------------------------------------------------------------*
  Look up the variable sourcepath and try everything in the list in order
  if the file isn't in . and it isn't an abs path name.
  *-------------------------------------------------------------------------*/

char *inp_pathresolve(const char *name)
{
    struct variable *v;
    struct stat st;

#if defined(_WIN32)

    /* If variable 'mingwpath' is set: convert mingw /d/... to d:/... */
    if (cp_getvar("mingwpath", CP_BOOL, NULL, 0) &&
            name[0] == DIR_TERM_LINUX && isalpha_c(name[1]) &&
            name[2] == DIR_TERM_LINUX) {
        DS_CREATE(ds, 100);
        if (ds_cat_str(&ds, name) != 0) {
            fprintf(stderr, "Error: Unable to copy string while resolving path");
            controlled_exit(EXIT_FAILURE);
        }
        char *const buf = ds_get_buf(&ds);
        buf[0] = buf[1];
        buf[1] = ':';
        char * const resolved_path = inp_pathresolve(buf);
        ds_free(&ds);
        return resolved_path;
    }

#endif

    /* just try it */
    if (stat(name, &st) == 0)
        return copy(name);
	
#if !defined(EXT_ASC) && (defined(__MINGW32__) || defined(_MSC_VER))
    wchar_t wname[BSIZE_SP];
    if (MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, 2 * (int)strlen(name) + 1) == 0) {
        fprintf(stderr, "UTF-8 to UTF-16 conversion failed with 0x%x\n", GetLastError());
        fprintf(stderr, "%s could not be converted\n", name);
        return NULL;
    }
    if (_waccess(wname, 0) == 0)
        return copy(name);
#endif	

    /* fail if this was an absolute filename or if there is no sourcepath var
     */
    if (is_absolute_pathname(name) ||
            !cp_getvar("sourcepath", CP_LIST, &v, 0)) {
        return (char *) NULL;
    }

    {
        DS_CREATE(ds, 100);
        for (; v; v = v->va_next) {
            int rc_ds;
            ds_clear(&ds); /* empty buffer */

            switch (v->va_type) {
                case CP_STRING:
                    rc_ds = ds_cat_printf(&ds, "%s%s%s",
                            v->va_string, DIR_PATHSEP, name);
                    break;
                case CP_NUM:
                    rc_ds = ds_cat_printf(&ds, "%d%s%s",
                            v->va_num, DIR_PATHSEP, name);
                    break;
                case CP_REAL: /* This is foolish */
                    rc_ds = ds_cat_printf(&ds, "%g%s%s",
                            v->va_real, DIR_PATHSEP, name);
                    break;
                default:
                    fprintf(stderr,
                            "ERROR: enumeration value `CP_BOOL' or `CP_LIST' "
                            "not handled in inp_pathresolve\nAborting...\n");
                    controlled_exit(EXIT_FAILURE);
            }

            if (rc_ds != 0) { /* unable to build string */
                (void) fprintf(cp_err,
                        "Error: Unable to build path name in inp_pathresolve");
                controlled_exit(EXIT_FAILURE);
            }

            /* Test if the file is found */
            {
                const char * const buf = ds_get_buf(&ds);
                if (stat(buf, &st) == 0) {
                    char * const buf_cpy = dup_string(
                            buf, ds_get_length(&ds));
                    ds_free(&ds);
                    return buf_cpy;
                }
                /* Else contiue with next attempt */
            }
        } /* end of loop over linked variables */
        ds_free(&ds);
    } /* end of block trying to find a valid name */

    return (char *) NULL;
} /* end of function inp_pathresolve */



static char *inp_pathresolve_at(const char *name, const char *dir)
{
    /* if name is an absolute path name,
     *   or if we haven't anything to prepend anyway
     */
    if (is_absolute_pathname(name) || !dir || !dir[0]) {
        return inp_pathresolve(name);
    }

    if (name[0] == '~' && name[1] == '/') {
        char * const y = cp_tildexpand(name);
        if (y) {
            char * const r = inp_pathresolve(y);
            txfree(y);
            return r;
        }
    }

    /*
     * Try in current dir and then in the actual dir the file was read.
     * Current dir . is needed to correctly support absolute paths in
     * sourcepath
     */

    {
        DS_CREATE(ds, 100);
        if (ds_cat_printf(&ds, ".%c%s", DIR_TERM, name) != 0) {
            (void) fprintf(cp_err,
                    "Error: Unable to build \".\" path name in inp_pathresolve_at");
            controlled_exit(EXIT_FAILURE);
        }
        char * const r = inp_pathresolve(ds_get_buf(&ds));
        ds_free(&ds);
        if (r != (char *) NULL) {
            return r;
        }
    }

    {
        DS_CREATE(ds, 100);
        int rc_ds = 0;
        rc_ds |= ds_cat_str(&ds, dir); /* copy the dir name */
        const size_t n = ds_get_length(&ds); /* end of copied dir name */

        /* Append a directory separator if not present already */
        const char ch_last = n > 0 ? dir[n - 1] : '\0';
        if (ch_last != DIR_TERM
#ifdef _WIN32
                && ch_last != DIR_TERM_LINUX
#endif
                ) {
            rc_ds |= ds_cat_char(&ds, DIR_TERM);
        }
        rc_ds |= ds_cat_str(&ds, name); /* append the file name */

        if (rc_ds != 0) {
            (void) fprintf(cp_err, "Error: Unable to build \"dir\" path name "
                    "in inp_pathresolve_at");
            controlled_exit(EXIT_FAILURE);
        }

        char * const r = inp_pathresolve(ds_get_buf(&ds));
        ds_free(&ds);
        return r;
    }
} /* end of function inp_pathresolve_at */



/*-------------------------------------------------------------------------*
 *  This routine reads a line (of arbitrary length), up to a '\n' or 'EOF' *
 *  and returns a pointer to the resulting null terminated string.         *
 *  The '\n' if found, is included in the returned string.                 *
 *  From: jason@ucbopal.BERKELEY.EDU (Jason Venner)                        *
 *  Newsgroups: net.sources                                                *
 *-------------------------------------------------------------------------*/

#define STRGROW 256

static char *readline(FILE *fd)
{
    int c;
    int memlen;
    char *strptr;
    int strlen;

    strlen = 0;
    memlen = STRGROW;
    strptr = TMALLOC(char, memlen);
    memlen -= 1; /* Save constant -1's in while loop */

    while ((c = getc(fd)) != EOF) {

        if (strlen == 0 && (c == '\t' || c == ' ')) /* Leading spaces away */
            continue;

        if (c == '\r')
            continue;
        strptr[strlen++] = (char) c;

        if (strlen >= memlen) {
            memlen += STRGROW;
            if ((strptr = TREALLOC(char, strptr, memlen + 1)) == NULL)
                return (NULL);
        }

        if (c == '\n')
            break;
    }

    if (!strlen) {
        tfree(strptr);
        return (NULL);
    }

    // strptr[strlen] = '\0';
    /* Trim the string */
    strptr = TREALLOC(char, strptr, strlen + 1);
    strptr[strlen] = '\0';

    return (strptr);
}


/* Replace "gnd" by " 0 "
   Delimiters of gnd may be ' ' or ',' or '(' or ')',
   may be disabled by setting variable no_auto_gnd */

static void inp_fix_gnd_name(struct card *c)
{
    for (; c; c = c->nextcard) {

        char *gnd = c->line;

        // if there is a comment or no gnd, go to next line
        if ((*gnd == '*') || !strstr(gnd, "gnd"))
            continue;

        // replace "?gnd?" by "? 0 ?", ? being a ' '  ','  '('  ')'.
        while ((gnd = strstr(gnd, "gnd")) != NULL) {
            if ((isspace_c(gnd[-1]) || gnd[-1] == '(' || gnd[-1] == ',') &&
                    (isspace_c(gnd[3]) || gnd[3] == ')' || gnd[3] == ',')) {
                memcpy(gnd, " 0 ", 3);
            }
            gnd += 3;
        }

        // now remove the extra white spaces around 0
        c->line = inp_remove_ws(c->line);
    }
}

/*
 * transform a VCVS "gate" instance into a XSPICE instance
 *
 *   Exx  out+ out-  <VCVS>  {nand|nor|and|or}(n)
 *   +  in[1]+ in[1]- ... in[n]+ in[n]-
 *   +  x1,y1 x2,y2
 * ==>
 *   Axx  %vd[ in[1]+ in[1]- ... in[n]+ in[n]- ]
 *   +    %vd( out+ out- )  Exx
 *   .model Exx multi_input_pwd ( x = [x1 x2] x = [y1 y2] model =
 * {nand|nor|and|or} )
 *
 * fixme,
 *   `n' is not checked
 *   the x,y list is fixed to length 2
 */

static int inp_chk_for_multi_in_vcvs(struct card *c, int *line_number)
{
    char *fcn_b, *line;

    line = c->line;
    if (((fcn_b = strstr(line, "nand(")) != NULL ||
         (fcn_b = strstr(line, "and(")) != NULL ||
         (fcn_b = strstr(line, "nor(")) != NULL ||
         (fcn_b = strstr(line, "or(")) != NULL) &&
        isspace_c(fcn_b[-1])) {
#ifndef XSPICE
        fprintf(stderr,
                "\n"
                "Error: XSPICE is required to run the 'multi-input "
                "pwl' option in line %d\n"
                "  %s\n"
                "\n"
                "See manual chapt. 31 for installation "
                "instructions\n",
                *line_number, line);
        controlled_exit(EXIT_BAD);
#else
        char keep, *comma_ptr, *xy_values1[5], *xy_values2[5];
        char *out_str, *ctrl_nodes_str,
             *xy_values1_b = NULL, *ref_str, *fcn_name,
             *fcn_e = NULL, *out_b, *out_e, *ref_e;
        char *m_instance, *m_model;
        char *xy_values2_b = NULL, *xy_values1_e = NULL,
             *ctrl_nodes_b = NULL, *ctrl_nodes_e = NULL;
        int xy_count1, xy_count2;
        bool ok = FALSE;

        do {
            ref_e = skip_non_ws(line);

            out_b = skip_ws(ref_e);

            out_e = skip_back_ws(fcn_b, out_b);
            if (out_e <= out_b)
                break;

            fcn_e = strchr(fcn_b, '(');

            ctrl_nodes_b = strchr(fcn_e, ')');
            if (!ctrl_nodes_b)
                break;
            ctrl_nodes_b = skip_ws(ctrl_nodes_b + 1);

            comma_ptr = strchr(ctrl_nodes_b, ',');
            if (!comma_ptr)
                break;

            xy_values1_b = skip_back_ws(comma_ptr, ctrl_nodes_b);
            if (xy_values1_b[-1] == '}') {
                while (--xy_values1_b >= ctrl_nodes_b)
                    if (*xy_values1_b == '{')
                        break;
            } else {
                xy_values1_b = skip_back_non_ws(xy_values1_b, ctrl_nodes_b);
            }
            if (xy_values1_b <= ctrl_nodes_b)
                break;

            ctrl_nodes_e = skip_back_ws(xy_values1_b, ctrl_nodes_b);
            if (ctrl_nodes_e <= ctrl_nodes_b)
                break;

            xy_values1_e = skip_ws(comma_ptr + 1);
            if (*xy_values1_e == '{') {
                xy_values1_e = inp_spawn_brace(xy_values1_e);
            } else {
                xy_values1_e = skip_non_ws(xy_values1_e);
            }
            if (!xy_values1_e)
                break;

            xy_values2_b = skip_ws(xy_values1_e);

            ok = TRUE;
        } while (0);

        if (!ok) {
            fprintf(stderr, "ERROR: malformed line: %s\n", line);
            controlled_exit(EXIT_FAILURE);
        }

        ref_str = copy_substring(line, ref_e);
        out_str = copy_substring(out_b, out_e);
        fcn_name = copy_substring(fcn_b, fcn_e);
        ctrl_nodes_str = copy_substring(ctrl_nodes_b, ctrl_nodes_e);

        keep = *xy_values1_e;
        *xy_values1_e = '\0';
        xy_count1 =
            get_comma_separated_values(xy_values1, xy_values1_b);
        *xy_values1_e = keep;

        xy_count2 = get_comma_separated_values(xy_values2, xy_values2_b);

        // place restrictions on only having 2 point values; this can
        // change later
        if (xy_count1 != 2 && xy_count2 != 2)
            fprintf(stderr,
                    "ERROR: only expecting 2 pair values for "
                    "multi-input vcvs!\n");

        m_instance = tprintf("%s %%vd[ %s ] %%vd( %s ) %s", ref_str,
                             ctrl_nodes_str, out_str, ref_str);
        m_instance[0] = 'a';

        m_model = tprintf(".model %s multi_input_pwl ( x = [%s %s] y "
                          "= [%s %s] model = \"%s\" )",
                          ref_str, xy_values1[0], xy_values2[0], xy_values1[1],
                          xy_values2[1], fcn_name);

        tfree(ref_str);
        tfree(out_str);
        tfree(fcn_name);
        tfree(ctrl_nodes_str);
        tfree(xy_values1[0]);
        tfree(xy_values1[1]);
        tfree(xy_values2[0]);
        tfree(xy_values2[1]);

        *c->line = '*';
        c = insert_new_line(c, m_instance, (*line_number)++, c->linenum_orig);
        c = insert_new_line(c, m_model, (*line_number)++, c->linenum_orig);
#endif
        return 1;
    } else {
        return 0;       // No keyword match. */
    }
}

/* replace the E and G source FREQ function by an XSPICE xfer instance
 * (used by Touchstone to netlist converter programs).
 * E1 n1 n2 FREQ {expression} = DB values ...
 * will become
 * B1_gen 1_gen 0 v = expression
 * A1_gen 1_gen %d(n1 n2) 1_gen
 * .model 1_gen xfer db=true table=[ values ]
 */

static void replace_freq(struct card *c, int *line_number)
{
#ifdef XSPICE
    char *line, *e, *e_e, *n1, *n1_e, *n2, *n2_e=NULL, *freq;
    char *expr, *expr_e, *in, *in_e=NULL, *keywd, *cp, *list, *list_e;
    int   db, ri, rad, got_key, diff;
    char  pt, key[4];

    line = c->line;

    /* First token is a node name. */

    e = line + 1;
    e_e = skip_non_ws(e);
    n1 = skip_ws(e_e);
    n1_e = skip_non_ws(n1);
    freq = strstr(n1_e, "freq");
    if (!freq || !isspace_c(freq[-1]) || !isspace_c(freq[4]))
        return;
    n2 = skip_ws(n1_e);
    if (n2 == freq) {
        n2 = NULL;
    } else {
        n2_e = skip_non_ws(n2);
        if (freq != skip_ws(n2_e)) // Three nodes or another keyword.
            return;
    }

    /* Isolate the input expression. */

    expr = skip_ws(freq + 4);
    if (*expr != '{')
        return;
    expr = skip_ws(expr + 1);
    expr_e = strchr(expr, '}');
    if (!expr_e)
        return;
    skip_back_ws(expr_e, expr);

    /* Is the expression just a node name, or v(node) or v(node1, node2)? */

    in = NULL;
    diff = 0;
    if (*expr < '0' || *expr > '9') {
        for (in_e = expr; in_e < expr_e; ++in_e) {
            if ((*in_e < '0' || *in_e > '9') && (*in_e < 'a' || *in_e > 'z') &&
                *in_e != '_')
            break;
        }
        if (in_e == expr_e) {
            /* A simple identifier. */

            in = expr;
        }
    }
    if (expr[0] == 'v' && expr[1] == '(' && expr_e[-1] == ')') {
        in = expr + 2;
        in_e = expr_e - 1;
        cp = strchr(in, ',');
        diff =  (cp && cp < in_e); // Assume v(n1, n2)
    }

    /* Look for a keyword.  Previous processing may put braces around it. */

    keywd = skip_ws(expr_e + 1);
    if (*keywd == '=')
        keywd = skip_ws(keywd + 1);

    db = 1;
    rad = 0;
    ri = 0;
    do {
        if (!keywd)
            return;
        list = keywd; // Perhaps not keyword
        if (*keywd == '{')
            ++keywd;
        cp = key;
        while (*keywd && !isspace_c(*keywd) && *keywd != '}' &&
               cp - key < sizeof key - 1) {
            *cp++ = *keywd++;
        }
        *cp = 0;
        if (*keywd == '}')
            ++keywd;
        if (!isspace_c(*keywd))
            return;

        /* Parse the format keyword, if any. */

        got_key = 0;
        if (!strcmp(key, "mag")) {
            db = 0;
            got_key = 1;
        } else if (!strcmp(key, "db")) {
            db = 1;
            got_key = 1;
        } else if (!strcmp(key, "rad")) {
            rad = 1;
            got_key = 1;
        } else if (!strcmp(key, "deg")) {
            rad = 0;
            got_key = 1;
        } else if (!strcmp(key, "r_i")) {
            ri = 1;
            got_key = 1;
        }

        /* Get the list of values. */

        if (got_key)
            list = skip_ws(keywd);
        if (!list)
            return;
        keywd = list;
    } while(got_key);

    list_e = list + strlen(list) - 1;
    skip_back_ws(list_e, list);
    if (list >= list_e)
        return;

    /* All good, rewrite the line.
     * Macro BSTR is used to pass counted string arguments to tprintf().
     */

#define BSTR(s) (int)(s##_e - s), s

    pt = (*line == 'e') ? 'v' : 'i';
    *line = '*';    // Make a comment
    if (in) {
        /* Connect input nodes directly. */

        if (diff) {
            /* Differential input. */

            if (n2) {
                line = tprintf("a_gen_%.*s %%vd(%.*s) %%%cd(%.*s %.*s) "
                           "gen_model_%.*s",
                           BSTR(e), BSTR(in), pt, BSTR(n1), BSTR(n2), BSTR(e));
            } else {
                line = tprintf("a_gen_%.*s %%vd(%.*s) %%%c(%.*s) "
                               "gen_model_%.*s",
                               BSTR(e), BSTR(in), pt, BSTR(n1), BSTR(e));
            }
        } else {
            /* Single node input. */

            if (n2) {
                line = tprintf("a_gen_%.*s %.*s  %%%cd(%.*s %.*s) "
                               "gen_model_%.*s",
                               BSTR(e), BSTR(in), pt, BSTR(n1), BSTR(n2),
                               BSTR(e));
            } else {
                line = tprintf("a_gen_%.*s %.*s %%%c(%.*s) gen_model_%.*s",
                               BSTR(e), BSTR(in), pt, BSTR(n1), BSTR(e));
            }
        }
    } else {
        /* Use a B-source for input. */

        line = tprintf("b_gen_%.*s gen_node_%.*s 0 v=%.*s",
                       BSTR(e), BSTR(e), BSTR(expr));
        c = insert_new_line(c, line, (*line_number)++, c->linenum_orig);
        if (n2) {
            line = tprintf("a_gen_%.*s gen_node_%.*s  %%%cd(%.*s %.*s) "
                           "gen_model_%.*s",
                           BSTR(e), BSTR(e), pt, BSTR(n1), BSTR(n2), BSTR(e));
        } else {
            line = tprintf("a_gen_%.*s gen_node_%.*s %%%c(%.*s) "
                           "gen_model_%.*s",
                           BSTR(e), BSTR(e), pt, BSTR(n1), BSTR(e));
        }
    }
    c = insert_new_line(c, line, (*line_number)++, c->linenum_orig);

    line = tprintf(".model gen_model_%.*s xfer %s table = [%.*s]",
                   BSTR(e),
                   ri ? "r_i=true" : rad ? "rad=true" : !db ? "db=false" : "",
                   BSTR(list));
     c = insert_new_line(c, line, (*line_number)++, c->linenum_orig);
#endif
}

/* Convert some E and G-source variants to XSPICE code models. */

static void inp_chk_for_e_source_to_xspice(struct card *c, int *line_number)
{
    int skip_control = 0;

    for (; c; c = c->nextcard) {

        char *line = c->line;

        /* there is no e source inside .control ... .endc */
        if (ciprefix(".control", line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }

        if (*line == 'e' && inp_chk_for_multi_in_vcvs(c, line_number))
            continue;
        if (*line != 'e' && *line != 'g')
            continue;

        /* Is it the FREQ form with S-parameter table? */

        replace_freq(c, line_number);
    }
}

/* If ngspice is started with option -a, then variable 'autorun'
 * will be set and a control section is inserted to try and ensure
 * some analysis is done;
 *
 *   .control
 *   strcmp __flag $curplot const
 *   if $__flag eq 0
 *     run
 *   end
 *   write rawfile   ; if rawfile given
 *   .endc
 *
 * The effect is that "run" is executed if there was no previous
 * analysis.
 */
static void inp_add_control_section(struct card *deck, int *line_number)
{
    static const char * const cards[] =
        {".control", "strcmp __flag $curplot const", "if $__flag eq 0",
         "  run", "end", NULL};
    const char   * const *lp;
    struct card   *c, *prev_card = NULL, *last_end = NULL;
    char           rawfile[1000], *line;

    for (c = deck; c; c = c->nextcard) {
        if (ciprefix(".end", c->line))
            last_end = prev_card;
        prev_card = c;
    }

    if (last_end)
        prev_card = last_end;
    for (lp = cards; *lp; ++lp)
        prev_card = insert_new_line(prev_card, copy(*lp), (*line_number)++, 0);
    if (cp_getvar("rawfile", CP_STRING, rawfile, sizeof(rawfile))) {
        line = tprintf("write %s", rawfile);
        prev_card = insert_new_line(prev_card, line, (*line_number)++, 0);
    }
    insert_new_line(prev_card, copy(".endc"), (*line_number)++, 0);
}


/* overwrite shell-style end-of-line continuation '\\' with spaces,
 *   and return TRUE when found */
static bool chk_for_line_continuation(char *line)
{
    if (*line != '*' && *line != '$') {

        char *ptr = skip_back_ws(strchr(line, '\0'), line);

        if ((ptr - 2 >= line) && (ptr[-1] == '\\') && (ptr[-2] == '\\')) {
            ptr[-1] = ' ';
            ptr[-2] = ' ';
            return TRUE;
        }
    }

    return FALSE;
}


//
// change .macro --> .subckt
//        .eom   --> .ends
//        .subckt name 1 2 3 params: w=9u l=180n -->
//                               .subckt name 1 2 3 w=9u l=180n
//        .subckt name (1 2 3) --> .subckt name 1 2 3
//        x1 (1 2 3)      --> x1 1 2 3
//        .param func1(x,y) = {x*y} --> .func func1(x,y) {x*y}

static void inp_fix_macro_param_func_paren_io(struct card *card)
{
    char *str_ptr, *new_str;

    for (; card; card = card->nextcard) {

        if (*card->line == '*')
            continue;

        if (ciprefix(".macro", card->line) || ciprefix(".eom", card->line)) {
            str_ptr = skip_non_ws(card->line);

            if (ciprefix(".macro", card->line)) {
                new_str = tprintf(".subckt%s", str_ptr);
            }
            else {
                new_str = tprintf(".ends%s", str_ptr);
            }

            tfree(card->line);
            card->line = new_str;
        }

        if (ciprefix(".subckt", card->line) || ciprefix("x", card->line)) {
            /* remove () */
            str_ptr = skip_non_ws(card->line);
                                        // skip over .subckt, instance name
            str_ptr = skip_ws(str_ptr);
            if (ciprefix(".subckt", card->line)) {
                str_ptr = skip_non_ws(str_ptr); // skip over subckt name
                str_ptr = skip_ws(str_ptr);
            }
            if (*str_ptr == '(') {
                *str_ptr = ' ';
                while (*str_ptr != '\0') {
                    if (*str_ptr == ')') {
                        *str_ptr = ' ';
                        break;
                    }
                    str_ptr++;
                }

                /* Remove the extra white spaces just introduced */
                card->line = inp_remove_ws(card->line);
            }
        }

        if (ciprefix(".para", card->line)) {
            bool is_func = FALSE;
            str_ptr = skip_non_ws(card->line); // skip over .param
            str_ptr = skip_ws(str_ptr);
            while (!isspace_c(*str_ptr) && *str_ptr != '=') {
                if (*str_ptr == '(')
                    is_func = TRUE;
                str_ptr++;
            }

            if (is_func) {
                str_ptr = strchr(card->line, '=');
                if (str_ptr)
                    *str_ptr = ' ';
                str_ptr = card->line + 1;
                str_ptr[0] = 'f';
                str_ptr[1] = 'u';
                str_ptr[2] = 'n';
                str_ptr[3] = 'c';
                str_ptr[4] = ' ';
            }
        }
    }
}


static char *get_instance_subckt(char *line)
{
    char *end_ptr, *inst_name_ptr;
    char *equal_ptr = strchr(line, '=');

    // see if instance has parameters
    if (equal_ptr) {
        end_ptr = skip_back_ws(equal_ptr, line);
        end_ptr = skip_back_non_ws(end_ptr, line);
    }
    else {
        end_ptr = strchr(line, '\0');
    }

    end_ptr = skip_back_ws(end_ptr, line);

    inst_name_ptr = skip_back_non_ws(end_ptr, line);

    return copy_substring(inst_name_ptr, end_ptr);
}


static char *get_subckt_model_name(char *line)
{
    char *name, *end_ptr;

    name = skip_non_ws(line); // eat .subckt|.model
    name = skip_ws(name);

    

    end_ptr = skip_non_ws(name);

    return copy_substring(name, end_ptr);
}


static char *get_model_name(char *line, int num_terminals)
{
    char *beg_ptr, *end_ptr;
    int i = 0;

    beg_ptr = skip_non_ws(line); /* eat device name */
    beg_ptr = skip_ws(beg_ptr);

    for (i = 0; i < num_terminals; i++) { /* skip the terminals */
        beg_ptr = skip_non_ws(beg_ptr);
        beg_ptr = skip_ws(beg_ptr);
    }

    if (*line == 'r') /* special dealing for r models */
        if ((*beg_ptr == '+') || (*beg_ptr == '-') ||
                isdigit_c(*beg_ptr)) { /* looking for a value before model */
            beg_ptr = skip_non_ws(beg_ptr); /* skip the value */
            beg_ptr = skip_ws(beg_ptr);
        }

    end_ptr = skip_non_ws(beg_ptr);

    return copy_substring(beg_ptr, end_ptr);
}


static char *get_model_type(char *line)
{
    char *beg_ptr;

    if (!ciprefix(".model", line))
        return NULL;

    beg_ptr = skip_non_ws(line); /* eat .model */
    beg_ptr = skip_ws(beg_ptr);

    beg_ptr = skip_non_ws(beg_ptr); /* eat model name */
    beg_ptr = skip_ws(beg_ptr);

    return gettok_noparens(&beg_ptr);
}


static char *get_adevice_model_name(char *line)
{
    char *ptr_end, *ptr_beg;

    ptr_end = skip_back_ws(strchr(line, '\0'), line);
    ptr_beg = skip_back_non_ws(ptr_end, line);

    return copy_substring(ptr_beg, ptr_end);
}


/*
 *   To distinguish modelname tokens from other tokens
 *   by checking if token is not a valid ngspice number
 */
static int is_a_modelname(char *s, const char* line)
{
    char *st;
    double testval;
        int error = 0;
        char* evalrc;

    /*token contains a '=' */
    if (strchr(s, '='))
        return FALSE;
    /* first characters not allowed in model name (including '\0')*/
    if (strchr("{*^@\\\'", s[0]))
        return FALSE;

    /* RKM: r100 4k7 are  valid numbers for resistors,
       so not valid model names. */
    if (newcompat.lt && *line == 'r') {
        evalrc = s;
        INPevaluateRKM_R(&evalrc, &error, 0);
        if (*evalrc == '\0' && !error)
            return FALSE;
    }
    if (newcompat.lt && *line == 'c') {
        evalrc = s;
        INPevaluateRKM_C(&evalrc, &error, 0);
        if (*evalrc == '\0' && !error)
            return FALSE;
    }
    if (newcompat.lt && *line == 'l') {
        evalrc = s;
        INPevaluateRKM_L(&evalrc, &error, 0);
        if (*evalrc == '\0' && !error)
            return FALSE;
    }
    /* first character of model name is character from alphabet */
    if (isalpha_c(s[0]))
        return TRUE;

    /* not beeing a valid number */
    testval = strtod(s, &st);
    /* conversion failed, so no number */
    if (eq(s, st)) {
        return TRUE;
    }

    /* test if we have a true number */
    if (*st == '\0' || isspace_c(*st)) {
        return FALSE;
    }

    /* look for the scale factor (alphabetic) and skip it.
     * INPevaluate will not do it because is does not swallow
     * the scale factor from the string.
     */
    switch (*st) {
        case 't':
        case 'T':
        case 'g':
        case 'G':
        case 'k':
        case 'K':
        case 'u':
        case 'U':
        case 'n':
        case 'N':
        case 'p':
        case 'P':
        case 'f':
        case 'F':
        case 'a':
        case 'A':
            st = st + 1;
            break;
        case 'm':
        case 'M':
            if (((st[1] == 'E') || (st[1] == 'e')) &&
                    ((st[2] == 'G') || (st[2] == 'g'))) {
                st = st + 3; /* Meg */
            }
            else if (((st[1] == 'I') || (st[1] == 'i')) &&
                    ((st[2] == 'L') || (st[2] == 'l'))) {
                st = st + 3; /* Mil */
            }
            else {
                st = st + 1; /* m, milli */
            }
            break;
        default:
            break;
    }
    /* test if we have a true scale factor */
    if (*st == '\0' || isspace_c(*st))
        return FALSE;

    /* test if people use Ohms, F, H for RLC, like pF or uOhms */
    if (ciprefix("ohms", st))
        st = st + 4;
    else if (ciprefix("farad", st))
        st = st + 5;
    else if (ciprefix("henry", st))
        st = st + 5;
    else if ((*st == 'f') || (*st == 'h'))
        st = st + 1;
    if (*st == '\0' || isspace_c(*st)) {
        return FALSE;
    }

    /* token starts with non alphanum character */
    return TRUE;
}


struct nlist {
    char **names;
    int num_names;
    int size;
};


static const char *nlist_find(const struct nlist *nlist, const char *name)
{
    int i;
    for (i = 0; i < nlist->num_names; i++)
        if (strcmp(nlist->names[i], name) == 0)
            return nlist->names[i];
    return NULL;
}

#if 0 /* see line 2452 */
static const char *nlist_model_find(
        const struct nlist *nlist, const char *name)
{
    int i;
    for (i = 0; i < nlist->num_names; i++)
        if (model_name_match(nlist->names[i], name))
            return nlist->names[i];
    return NULL;
}
#endif

static void nlist_adjoin(struct nlist *nlist, char *name)
{
    if (nlist_find(nlist, name)) {
        tfree(name);
        return;
    }

    if (nlist->num_names >= nlist->size)
        nlist->names = TREALLOC(char *, nlist->names, nlist->size *= 2);

    nlist->names[nlist->num_names++] = name;
}


static struct nlist *nlist_allocate(int size)
{
    struct nlist *t = TMALLOC(struct nlist, 1);

    t->names = TMALLOC(char *, size);
    t->size = size;

    return t;
}


static void nlist_destroy(struct nlist *nlist)
{
    int i;
    for (i = 0; i < nlist->num_names; i++)
        tfree(nlist->names[i]);

    tfree(nlist->names);
    tfree(nlist);
}


static void get_subckts_for_subckt(struct card *start_card, char *subckt_name,
        struct nlist *used_subckts, struct nlist *used_models,
        bool has_models)
{
    struct card *card;
    int first_new_subckt = used_subckts->num_names;

    bool found_subckt = FALSE;
    int i, fence;

    for (card = start_card; card; card = card->nextcard) {

        char *line = card->line;

        /* no models embedded in these lines */
        if (strchr("*vibefghkt", *line))
            continue;

        if ((ciprefix(".ends", line) || ciprefix(".eom", line)) &&
                found_subckt)
            break;

        if (ciprefix(".subckt", line) || ciprefix(".macro", line)) {
            char *curr_subckt_name = get_subckt_model_name(line);

            if (strcmp(curr_subckt_name, subckt_name) == 0)
                found_subckt = TRUE;

            tfree(curr_subckt_name);
        }

        if (found_subckt) {
            if (*line == 'x') {
                char *inst_subckt_name = get_instance_subckt(line);
                nlist_adjoin(used_subckts, inst_subckt_name);
            }
            else if (*line == 'a') {
                char *model_name = get_adevice_model_name(line);
                nlist_adjoin(used_models, model_name);
            }
            else if (has_models) {
                int num_terminals = get_number_terminals(line);
                if (num_terminals != 0) {
                    char *model_name = get_model_name(line, num_terminals);
                    if (is_a_modelname(model_name, line))
                        nlist_adjoin(used_models, model_name);
                    else
                        tfree(model_name);
                }
            }
        }
    }

    // now make recursive call on instances just found above
    fence = used_subckts->num_names;
    for (i = first_new_subckt; i < fence; i++)
        get_subckts_for_subckt(start_card, used_subckts->names[i],
                used_subckts, used_models, has_models);
}


/*
  iterate through the deck and comment out unused subckts, models
  (don't want to waste time processing everything)
  also comment out .param lines with no parameters defined
*/

void comment_out_unused_subckt_models(struct card *start_card)
{
    struct card *card;
    struct nlist *used_subckts, *used_models;
    int i = 0, fence;
    bool processing_subckt = FALSE, remove_subckt = FALSE, has_models = FALSE;
    int skip_control = 0, nested_subckt = 0;

    used_subckts = nlist_allocate(100);
    used_models = nlist_allocate(100);

    for (card = start_card; card; card = card->nextcard) {
        if (ciprefix(".model", card->line))
            has_models = TRUE;
        if (ciprefix(".cmodel", card->line))
            has_models = TRUE;
        if (ciprefix(".para", card->line) && !strchr(card->line, '='))
            *card->line = '*';
    }

    for (card = start_card; card; card = card->nextcard) {

        char *line = card->line;

        /* no models embedded in these lines */
        if (strchr("*vibefghkt", *line))
            continue;

        /* there is no .subckt, .model or .param inside .control ... .endc */
        if (ciprefix(".control", line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }

        if (ciprefix(".subckt", line) || ciprefix(".macro", line))
            processing_subckt = TRUE;
        if (ciprefix(".ends", line) || ciprefix(".eom", line))
            processing_subckt = FALSE;

        /* no models embedded in these lines */
        if (*line == '.')
            continue;

        if (!processing_subckt) {
            if (*line == 'x') {
                char *subckt_name = get_instance_subckt(line);
                nlist_adjoin(used_subckts, subckt_name);
            }
            else if (*line == 'a') {
                char *model_name = get_adevice_model_name(line);
                nlist_adjoin(used_models, model_name);
            }
            else if (has_models) {
                /* This is a preliminary version, until we have found a
                   reliable method to detect the model name out of the input
                   line (Many options have to be taken into account.). */
                int num_terminals = get_number_terminals(line);
                if (num_terminals != 0) {
                    char *model_name = get_model_name(line, num_terminals);
                    if (is_a_modelname(model_name, line))
                        nlist_adjoin(used_models, model_name);
                    else
                        tfree(model_name);
                }
            } /* if (has_models)  */
        } /* if (!processing_subckt) */
    } /* for loop through all cards */

    fence = used_subckts->num_names;
    for (i = 0; i < fence; i++)
        get_subckts_for_subckt(start_card, used_subckts->names[i],
                used_subckts, used_models, has_models);

    /* comment out any unused subckts, currently only at top level */
    for (card = start_card; card; card = card->nextcard) {

        char *line = card->line;

        if (*line == '*')
            continue;

        if (ciprefix(".subckt", line) || ciprefix(".macro", line)) {
            char *subckt_name = get_subckt_model_name(line);
            /* check if unused, only at top level */
            if (nested_subckt++ == 0)
                remove_subckt = !nlist_find(used_subckts, subckt_name);
            tfree(subckt_name);
        }

        if (ciprefix(".ends", line) || ciprefix(".eom", line)) {
            if (remove_subckt)
                *line = '*';
            if (--nested_subckt == 0)
                remove_subckt = FALSE;
        }

        if (remove_subckt)
            *line = '*'; /* make line a comment */
    }
#if 0
    /* comment out any unused models */
    for (card = start_card; card; card = card->nextcard) {

        char* line = card->line;

        if (*line == '*')
            continue;

        if (has_models &&
            (ciprefix(".model", line) || ciprefix(".cmodel", line))) {
            char* model_type = get_model_type(line);
            char* model_name = get_subckt_model_name(line);

            /* keep R, L, C models because in addition to no. of terminals the
               value may be given, as in RE1 1 2 800 newres dtemp=5, so model
               name may be token no. 4 or 5, and, if 5, will not be detected
               by get_subckt_model_name()*/
            if (!cieq(model_type, "c") && !cieq(model_type, "l") &&
                !cieq(model_type, "r") &&
                !nlist_model_find(used_models, model_name)) {
                *line = '*';
            }

            tfree(model_type);
            tfree(model_name);
        }
    }
#endif
    nlist_destroy(used_subckts);
    nlist_destroy(used_models);
}


#if 0
// find closing paren
static char *
inp_search_closing_paren(char *s)
{
    int count = 0;
    // assert(*s == '(')
    while (*s) {
        if (*s == '(')
            count++;
        if (*s == ')')
            count--;
        if (count == 0)
            return s + 1;
        s++;
    }

    return NULL;
}
#endif


#if 0
/* search backwards for opening paren */
static char *
inp_search_opening_paren(char *s, char *start)
{
    int count = 0;
    // assert(*s == ')')
    while (s >= start) {
        if (*s == '(')
            count--;
        if (*s == ')')
            count++;
        if (count == 0)
            return s;
        s--;
    }

    return NULL;
}
#endif


/* search forward for closing brace */
static char *inp_spawn_brace(char *s)
{
    int count = 0;
    // assert(*s == '{')
    while (*s) {
        if (*s == '{')
            count++;
        if (*s == '}')
            count--;
        if (count == 0)
            return s + 1;
        s++;
    }

    return NULL;
}


/*-------------------------------------------------------------------------*
  removes  " " quotes, returns lower case letters,
  replaces non-printable characters with '_', however if
  non-printable character is the only character in a line,
  replace it by '*'. Leave quotes in .param, .subckt and x
  (subcircuit instance) cards to allow string-valued parameters.
  If there is a XSPICE code model .model line with file input,
  keep quotes and case for the file path.
  *-------------------------------------------------------------------------*/

void inp_casefix(char *string)
{
#ifdef HAVE_CTYPE_H
    /* single non-printable character */
    if (string && !isspace_c(*string) && !isprint_c(*string) &&
            (string[1] == '\0' || isspace_c(string[1]))) {
        *string = '*';
        return;
    }
    if (string) {
        bool keepquotes;

#ifdef XSPICE
        char* tmpstr = NULL;

        /* Special treatment of code model file input. */

        if (ciprefix(".model", string))
            tmpstr = strstr(string, "file=");
#endif
        keepquotes = ciprefix(".param", string); // Allow string params

        while (*string) {
#ifdef XSPICE
            /* exclude file name inside of quotes from getting lower case,
               keep quotes to enable spaces in file path */
            if (string == tmpstr) {
                string = string + 6; // past first quote
                while (*string && *string != '"')
                    string++;
                if (*string)
                    string++; // past second quote
                if (*string == '\0')
                    break;
            }
#endif
            if (*string == '"') {
                if (!keepquotes)
                    *string++ = ' ';
                while (*string && *string != '"')
                    string++;
                if (*string == '\0')
                    continue; /* needed if string is "something ! */
                if (*string == '"' && !keepquotes)
                    *string = ' ';
            }
            if (*string && !isspace_c(*string) && !isprint_c(*string))
                *string = '_';
            if (isupper_c(*string))
                *string = tolower_c(*string);
            string++;
        }
    }
#endif
}


/* Strip all end-of-line comments from a deck
   For cf == TRUE (script files, command files like spinit, .spiceinit)
   and for .control sections only '$ ' is accepted as end-of-line comment,
   to avoid conflict with $variable definition, otherwise we accept '$'. */
static void inp_stripcomments_deck(struct card *c, bool cf)
{
    bool found_control = FALSE;
    for (; c; c = c->nextcard) {
        /* exclude lines between .control and .endc from removing white spaces
         */
        if (ciprefix(".control", c->line))
            found_control = TRUE;
        if (ciprefix(".endc", c->line))
            found_control = FALSE;
        inp_stripcomments_line(c->line, found_control | cf);
    }
}


/*
 * Support for end-of-line comments that begin with any of the following:
 *   ';'
 *   '$' (only outside of a .control section)
 *   '$ '
 *   '//' (like in c++ and as per the numparam code)
 * Any following text to the end of the line is ignored.
 * Note requirement for $ to be followed by a space, if we are inside of a
 * .control section or in a command file. This is to avoid conflict
 * with use of $ in front of a variable.
 * Comments on a continuation line (i.e. line begining with '+') are allowed
 * and are removed before lines are stitched.
 * Lines that contain only an end-of-line comment with or without leading
 * white space are also allowed.

 If there is only white space before the end-of-line comment the
 the whole line is converted to a normal comment line (i.e. one that
 begins with a '*').
 BUG: comment characters in side of string literals are not ignored
 ('$' outside of .control section is o.k. however).

 If the comaptibility mode is PS, LTPS or LTPSA, '$' is treated as a valid
 character, not as end-of-line comment delimiter, except for that it is
 located at the beginning of a line. If inside of a control section,
 still '$ ' is read a an end-of-line comment delimiter.*/
static void inp_stripcomments_line(char *s, bool cs)
{
    char c = ' '; /* anything other than a comment character */
    char *d = s;
    if (*s == '\0')
        return; /* empty line */
    if (*s == '*')
        return; /* line is already a comment */
    /* look for comments */
    while ((c = *d) != '\0') {
        d++;
        if (*d == ';') {
            break;
        }
        /* outside of .control section, and not in PS mode */
        else if (!cs && (c == '$') && !newcompat.ps) {
            /* The character before '&' has to be ',' or ' ' or tab.
               A valid numerical expression directly before '$' is not yet
               supported. */
            if ((d - 2 >= s) &&
                    ((d[-2] == ' ') || (d[-2] == ',') || (d[-2] == '\t'))) {
                d--;
                break;
            }
        }
        else if (cs && (c == '$') &&
                (*d == ' ')) { /* inside of .control section or command file
                                */
            d--; /* move d back to first comment character */
            break;
        }
        else if ((c == '/') && (*d == '/')) {
            d--; /* move d back to first comment character */
            break;
        }
    }
    /* d now points to the first comment character or the null at the string
     * end */

    /* check for special case of comment at start of line */
    if (d == s) {
        *s = '*'; /* turn into normal comment */
        return;
    }

    if (d > s) {
        d--;
        /* d now points to character just before comment */

        /* eat white space at new end of line */
        while (d >= s) {
            if ((*d != ' ') && (*d != '\t'))
                break;
            d--;
        }
        d++;
        /* d now points to the first white space character before the
           end-of-line or end-of-line comment, or it points to the first
           end-of-line comment character, or to the begining of the line */
    }

    /* Check for special case of comment at start of line
       with or without preceeding white space */
    if (d <= s) {
        *s = '*'; /* turn the whole line into normal comment */
        return;
    }

    *d = '\0'; /* terminate line in new location */
}


static void inp_change_quotes(char *s)
{
    bool first_quote = FALSE;

    for (; *s; s++)
        if (*s == '\'') {
            if (first_quote == FALSE) {
                *s = '{';
                first_quote = TRUE;
            }
            else {
                *s = '}';
                first_quote = FALSE;
            }
        }
}


static void add_name(struct names *p, char *name)
{
    if (p->num_names >= N_SUBCKT_W_PARAMS) {
        fprintf(stderr, "ERROR: N_SUBCKT_W_PARMS overflow, more than %d subcircuits\n", N_SUBCKT_W_PARAMS);
        controlled_exit(EXIT_FAILURE);
    }

    p->names[p->num_names++] = name;
}


static char **find_name(struct names *p, char *name)
{
    int i;

    for (i = 0; i < p->num_names; i++)
        if (strcmp(p->names[i], name) == 0)
            return &p->names[i];

    return NULL;
}


static char *inp_fix_subckt(struct names *subckt_w_params, char *s)
{
    struct card *head, *first_param_card, *c;
    char *equal, *beg, *buffer, *ptr1, *ptr2, *new_str;

    equal = strchr(s, '=');
    if (equal && !strstr(s, "params:")) {
        /* get subckt name (ptr1 will point to name) */
        ptr1 = skip_non_ws(s);
        ptr1 = skip_ws(ptr1);
        for (ptr2 = ptr1; *ptr2 && !isspace_c(*ptr2) && !isquote(*ptr2);
                ptr2++)
            ;

        add_name(subckt_w_params, copy_substring(ptr1, ptr2));

        /* go to beginning of first parameter word  */
        /* s    will contain only subckt definition */
        /* beg  will point to start of param list   */
        beg = skip_back_ws(equal, s);
        beg = skip_back_non_ws(beg, s);
        beg[-1] = '\0'; /* fixme can be < s */

        head = insert_new_line(NULL, NULL, 0, 0);
        /* create list of parameters that need to get sorted */
        first_param_card = c = NULL;
        while ((ptr1 = strchr(beg, '=')) != NULL) {
            ptr2 = skip_ws(ptr1 + 1);
            ptr1 = skip_back_ws(ptr1, beg);
            ptr1 = skip_back_non_ws(ptr1, beg);
            /* ptr1 points to beginning of parameter */

            if (*ptr2 == '{')
                ptr2 = inp_spawn_brace(ptr2);
            else
                ptr2 = skip_non_ws(ptr2);

            if (!ptr2) {
                fprintf(stderr, "Error: Missing } in line %s\n", s);
                controlled_exit(EXIT_FAILURE);
            }

            beg = ptr2;

            c = insert_new_line(c, copy_substring(ptr1, ptr2), 0, 0);

            if (!first_param_card)
                first_param_card = c;
        }
        /* now sort parameters in order of dependencies */
        inp_sort_params(first_param_card, head, NULL, NULL);

        /* create new ordered parameter string for subckt call */
        new_str = NULL;
        for (c = head->nextcard; c; c = c->nextcard)
            if (new_str == NULL) {
                new_str = copy(c->line);
            }
            else {
                char *x = tprintf("%s %s", new_str, c->line);
                tfree(new_str);
                new_str = x;
            }

        line_free_x(head, TRUE);

        /* create buffer and insert params: */
        buffer = tprintf("%s params: %s", s, new_str);

        tfree(s);
        tfree(new_str);

        s = buffer;
    }

    return s;
}


/*
 * this function shall:
 *   reduce sequences of whitespace to one space
 *   and to drop even that if it seems to be at a `safe' place to do so
 * safe place means:
 *   before or behind a '='
 *   before or behind an operator within a {} expression
 *     whereby `operator' is classified by `is_arith_char()'
 * fixme:
 *   thats odd and very naive business
 */

char *inp_remove_ws(char *s)
{
    char *x = s;
    char *d = s;

    int brace_level = 0;

    /* preserve at least one whitespace at beginning of line
     * fixme,
     *   is this really necessary ?
     *   or is this an artefact of original inp_remove_ws() implementation ?
     */
    if (isspace_c(*s))
        *d++ = *s++;

    while (*s != '\0') {
        if (*s == '{')
            brace_level++;
        if (*s == '}')
            brace_level--;

        if (isspace_c(*s)) {
            s = skip_ws(s);
            if (!(*s == '\0' || *s == '=' ||
                        ((brace_level > 0) &&
                                (is_arith_char(*s) || *s == ','))))
                *d++ = ' ';
            continue;
        }

        if (*s == '=' ||
                ((brace_level > 0) && (is_arith_char(*s) || *s == ','))) {
            *d++ = *s++;
            s = skip_ws(s);
            continue;
        }

        *d++ = *s++;
    }

    *d = '\0';

    if (d == s)
        return x;

    s = copy(x);
    tfree(x);

    return s;
}


/*
  change quotes from '' to {}
  .subckt name 1 2 3 params: l=1 w=2 --> .subckt name 1 2 3 l=1 w=2
  x1 1 2 3 params: l=1 w=2 --> x1 1 2 3 l=1 w=2
  modify .subckt lines by calling inp_fix_subckt()
  No changes to lines in .control section !
*/

static void inp_fix_for_numparam(
        struct names *subckt_w_params, struct card *c)
{
    bool found_control = FALSE;

    for (; c; c = c->nextcard) {

        if (*(c->line) == '*' || ciprefix(".lib", c->line))
            continue;

        /* exclude lines between .control and .endc from getting quotes
         * changed */
        if (ciprefix(".control", c->line))
            found_control = TRUE;
        if (ciprefix(".endc", c->line))
            found_control = FALSE;

        if (found_control)
            continue;

        inp_change_quotes(c->line);

        if (!newcompat.hs && !newcompat.s3)
            if (ciprefix(".subckt", c->line) || ciprefix("x", c->line)) {
                /* remove params: */
                char *str_ptr = strstr(c->line, "params:");
                if (str_ptr)
                    memcpy(str_ptr, "       ", 7);
            }

        if (ciprefix(".subckt", c->line))
            c->line = inp_fix_subckt(subckt_w_params, c->line);
    }
}


static void inp_remove_excess_ws(struct card *c)
{
    bool found_control = FALSE;

    for (; c; c = c->nextcard) {

        if (*c->line == '*')
            continue;

        /* exclude echo lines between .control and .endc from removing white
         * spaces */
        if (ciprefix(".control", c->line))
            found_control = TRUE;
        if (ciprefix(".endc", c->line))
            found_control = FALSE;

        if (found_control && ciprefix("echo", c->line))
            continue;

        c->line = inp_remove_ws(c->line); /* freed in fcn */
    }
}


static struct card *expand_section_ref(struct card *c, const char *dir_name)
{
    char *line = c->line;

    char *s, *s_e, *y;

    s = skip_non_ws(line);
    while (isspace_c(*s) || isquote(*s))
        s++;
    for (s_e = s; *s_e && !isspace_c(*s_e) && !isquote(*s_e); s_e++)
        ;
    y = s_e;
    while (isspace_c(*y) || isquote(*y))
        y++;

    if (*y) {
        /* library section reference: `.lib <library-file> <section-name>' */

        struct card *section_def;
        char keep_char1, keep_char2;
        char *y_e;
        struct library *lib;

        for (y_e = y; *y_e && !isspace_c(*y_e) && !isquote(*y_e); y_e++)
            ;
        keep_char1 = *s_e;
        keep_char2 = *y_e;
        *s_e = '\0';
        *y_e = '\0';

        lib = read_a_lib(s, dir_name);

        if (!lib) {
            fprintf(stderr, "ERROR, library file %s not found\n", s);
            controlled_exit(EXIT_FAILURE);
        }

        section_def = find_section_definition(lib->deck, y);

        if (!section_def) {
            fprintf(stderr,
                    "ERROR, library file %s, section definition %s not "
                    "found\n",
                    s, y);
            controlled_exit(EXIT_FAILURE);
        }

        /* recursively expand the refered section itself */
        {
            struct card *t = section_def;
            for (; t; t = t->nextcard) {
                if (ciprefix(".endl", t->line))
                    break;
                if (ciprefix(".lib", t->line))
                    t = expand_section_ref(t, lib->habitat);
            }
            if (!t) {
                fprintf(stderr, "ERROR, .endl not found\n");
                controlled_exit(EXIT_FAILURE);
            }
        }

        /* insert the library section definition into `c' */
        {
            struct card *t = section_def;
            for (; t; t = t->nextcard) {
                c = insert_new_line(
                        c, copy(t->line), t->linenum, t->linenum_orig);
                if (t == section_def) {
                    c->line[0] = '*';
                    c->line[1] = '<';
                }
                if (ciprefix(".endl", t->line)) {
                    c->line[0] = '*';
                    c->line[1] = '>';
                    break;
                }
            }
            if (!t) {
                fprintf(stderr, "ERROR, .endl not found\n");
                controlled_exit(EXIT_FAILURE);
            }
        }

        *line = '*'; /* comment out .lib line */
        *s_e = keep_char1;
        *y_e = keep_char2;
    }

    return c;
}


/*
 * recursively expand library section references,
 * either
 *    every library section reference (when the given section_name_ === NULL)
 * or
 *    just those references occuring in the given library section definition
 *
 * Command .libsave saves the loaded and parsed lib, to be read by .include
 */

static void expand_section_references(struct card *c, const char *dir_name)
{
    for (; c; c = c->nextcard) {
        struct card* p = c;
        if (ciprefix(".libsave", c->line)) {
            c = expand_section_ref(c, dir_name);
            char *filename = libprint(p, dir_name);
            fprintf(stdout, "\nLibrary\n%s\nsaved to %s\n", p->line + 9, filename);
            tfree(filename);
        }
        else if (ciprefix(".lib", c->line))
            c = expand_section_ref(c, dir_name);
    }
}


static char *inp_get_subckt_name(char *s)
{
    char *subckt_name, *end_ptr = strchr(s, '=');

    if (end_ptr) {
        end_ptr = skip_back_ws(end_ptr, s);
        end_ptr = skip_back_non_ws(end_ptr, s);
    }
    else {
        end_ptr = strchr(s, '\0');
    }

    end_ptr = skip_back_ws(end_ptr, s);
    subckt_name = skip_back_non_ws(end_ptr, s);

    return copy_substring(subckt_name, end_ptr);
}


static int inp_get_params(
        char *line, char *param_names[], char *param_values[])
{
    char *equal_ptr;
    char *end, *name, *value;
    int num_params = 0;
    char keep;

    while ((equal_ptr = find_assignment(line)) != NULL) {

        /* get parameter name */
        end = skip_back_ws(equal_ptr, line);
        name = skip_back_non_ws(end, line);

        if (num_params == NPARAMS) {
            fprintf(stderr, "Error: to many params in a line, max is %d\n",
                    NPARAMS);
            controlled_exit(EXIT_FAILURE);
        }

        param_names[num_params++] = copy_substring(name, end);

        /* get parameter value */
        value = skip_ws(equal_ptr + 1);

        if (*value == '{')
            end = inp_spawn_brace(value);
        else
            end = skip_non_ws(value);

        if (!end) {
            fprintf(stderr, "Error: Missing } in %s\n", line);
            controlled_exit(EXIT_FAILURE);
        }

        keep = *end;
        *end = '\0';

        if (*value == '{' || isdigit_c(*value) ||
                (*value == '.' && isdigit_c(value[1]))) {
            value = copy(value);
        }
        else {
            value = tprintf("{%s}", value);
        }

        param_values[num_params - 1] = value;
        *end = keep;

        line = end;
    }

    return num_params;
}


static char *inp_fix_inst_line(char *inst_line, int num_subckt_params,
        char *subckt_param_names[], char *subckt_param_values[],
        int num_inst_params, char *inst_param_names[],
        char *inst_param_values[])
{
    char *end, *inst_name, *inst_name_end;
    char *curr_line = inst_line, *new_line = NULL;
    int i, j;

    inst_name_end = skip_non_ws(inst_line);
    inst_name = copy_substring(inst_line, inst_name_end);

    end = strchr(inst_line, '=');
    if (end) {
        end = skip_back_ws(end, inst_line);
        end = skip_back_non_ws(end, inst_line);
        end[-1] = '\0'; /* fixme can be < inst_line */
    }

    for (i = 0; i < num_subckt_params; i++)
        for (j = 0; j < num_inst_params; j++)
            if (strcmp(subckt_param_names[i], inst_param_names[j]) == 0) {
                tfree(subckt_param_values[i]);
                subckt_param_values[i] = copy(inst_param_values[j]);
            }

    for (i = 0; i < num_subckt_params; i++) {
        new_line = tprintf("%s %s", curr_line, subckt_param_values[i]);

        tfree(curr_line);
        tfree(subckt_param_names[i]);
        tfree(subckt_param_values[i]);

        curr_line = new_line;
    }

    for (i = 0; i < num_inst_params; i++) {
        tfree(inst_param_names[i]);
        tfree(inst_param_values[i]);
    }

    tfree(inst_name);

    return curr_line;
}


/* If multiplier parameter 'm' is found on a X line, flag is set
   to TRUE.
   Function is called from inp_fix_inst_calls_for_numparam() */

static bool found_mult_param(int num_params, char *param_names[])
{
    int i;

    for (i = 0; i < num_params; i++)
        if (strcmp(param_names[i], "m") == 0)
            return TRUE;

    return FALSE;
}


/* If a subcircuit invocation (X-line) is found, which contains the
   multiplier parameter 'm', m is added to all lines inside
   the corresponding subcircuit except of some excluded in the code below
   Function is called from inp_fix_inst_calls_for_numparam() */

static int inp_fix_subckt_multiplier(struct names *subckt_w_params,
        struct card *subckt_card, int num_subckt_params,
        char *subckt_param_names[], char *subckt_param_values[])
{
    struct card *card;
    char *new_str;

    subckt_param_names[num_subckt_params] = copy("m");
    subckt_param_values[num_subckt_params] = copy("1");
    num_subckt_params++;

    if (!strstr(subckt_card->line, "params:")) {
        new_str = tprintf("%s params: m=1", subckt_card->line);
        add_name(subckt_w_params, get_subckt_model_name(subckt_card->line));
    }
    else {
        new_str = tprintf("%s m=1", subckt_card->line);
    }

    tfree(subckt_card->line);
    subckt_card->line = new_str;

    for (card = subckt_card->nextcard; card && !ciprefix(".ends", card->line);
            card = card->nextcard) {
        char *curr_line = card->line;
        /* no 'm' for comment line, B, V, E, H and some others that are not
         * using 'm' in their model description */
        if (strchr("*bvehaknopstuwy", curr_line[0]))
            continue;
        /* no 'm' for model cards */
        if (ciprefix(".model", curr_line))
            continue;
        if (newcompat.hs) {
            /* if there is already an m=xx in the instance line, multiply it with the new m */
            char* mult = strstr(curr_line, " m=");
            if (mult) {
                char* beg = copy_substring(curr_line, mult);
                mult = mult + 3;
                char* multval = gettok(&mult);
                /* replace { } or ' ' by ( ) to avoid double braces */
                if (*multval == '{' || *multval == '\'') {
                    *multval = '(';
                }
                char* tmpstr = strchr(multval, '}');
                if (tmpstr) {
                    *tmpstr = ')';
                }
                tmpstr = strchr(multval, '\'');
                if (tmpstr) {
                    *tmpstr = ')';
                }
                new_str = tprintf("%s m={m*%s} %s", beg, multval, mult);
                tfree(beg);
                tfree(multval);
            }
            else {
                new_str = tprintf("%s m={m}", curr_line);
            }
        }
        else {
            new_str = tprintf("%s m={m}", curr_line);
        }

        tfree(card->line);
        card->line = new_str;
    }

    return num_subckt_params;
}


static void inp_fix_inst_calls_for_numparam(
        struct names *subckt_w_params, struct card *deck)
{
    struct card *c;
    char *subckt_param_names[NPARAMS];
    char *subckt_param_values[NPARAMS];
    char *inst_param_names[NPARAMS];
    char *inst_param_values[NPARAMS];
    int i;

    // first iterate through instances and find occurences where 'm'
    // multiplier needs to be added to the subcircuit -- subsequent instances
    // will then need this parameter as well
    for (c = deck; c; c = c->nextcard) {
        char *inst_line = c->line;

        if (*inst_line == '*')
            continue;

        if (ciprefix("x", inst_line)) {
            int num_inst_params = inp_get_params(
                    inst_line, inst_param_names, inst_param_values);
            char *subckt_name = inp_get_subckt_name(inst_line);

            if (found_mult_param(num_inst_params, inst_param_names)) {
                struct card_assoc *a = find_subckt(c->level, subckt_name);
                if (a) {
                    int num_subckt_params = inp_get_params(a->line->line,
                            subckt_param_names, subckt_param_values);

                    if (!found_mult_param(
                                num_subckt_params, subckt_param_names))
                        inp_fix_subckt_multiplier(subckt_w_params, a->line,
                                num_subckt_params, subckt_param_names,
                                subckt_param_values);

                    for (i = 0; i < num_subckt_params; i++) {
                        tfree(subckt_param_names[i]);
                        tfree(subckt_param_values[i]);
                    }
                }
            }

            tfree(subckt_name);
            for (i = 0; i < num_inst_params; i++) {
                tfree(inst_param_names[i]);
                tfree(inst_param_values[i]);
            }
        }
    }

    for (c = deck; c; c = c->nextcard) {
        char *inst_line = c->line;

        if (*inst_line == '*')
            continue;

        if (ciprefix("x", inst_line)) {

            char *subckt_name = inp_get_subckt_name(inst_line);

            if (find_name(subckt_w_params, subckt_name)) {
                struct card *d;
                struct card_assoc* ca = find_subckt(c->level, subckt_name);
                if (ca)
                    d = ca->line;
                else
                    continue;
                if (d) {
                    char *subckt_line = d->line;
                    subckt_line = skip_non_ws(subckt_line);
                    subckt_line = skip_ws(subckt_line);

                    int num_subckt_params = inp_get_params(subckt_line,
                            subckt_param_names, subckt_param_values);
                    int num_inst_params = inp_get_params(
                            inst_line, inst_param_names, inst_param_values);

                    c->line = inp_fix_inst_line(inst_line, num_subckt_params,
                            subckt_param_names, subckt_param_values,
                            num_inst_params, inst_param_names,
                            inst_param_values);
                    for (i = 0; i < num_subckt_params; i++) {
                        tfree(subckt_param_names[i]);
                        tfree(subckt_param_values[i]);
                    }

                    for (i = 0; i < num_inst_params; i++) {
                        tfree(inst_param_names[i]);
                        tfree(inst_param_values[i]);
                    }
                }
            }

            tfree(subckt_name);
        }
    }
}


static struct function *new_function(struct function_env *env, char *name)
{
    struct function *f = TMALLOC(struct function, 1);

    f->name = name;
    f->num_parameters = 0;

    f->next = env->functions;
    env->functions = f;

    return f;
}


static struct function *find_function(struct function_env *env, char *name)
{
    struct function *f;

    for (; env; env = env->up)
        for (f = env->functions; f; f = f->next)
            if (strcmp(f->name, name) == 0)
                return f;

    return NULL;
}


static void free_function(struct function *fcn)
{
    int i;

    tfree(fcn->name);
    tfree(fcn->body);
    tfree(fcn->accept);

    for (i = 0; i < fcn->num_parameters; i++)
        tfree(fcn->params[i]);
}


static void new_function_parameter(struct function *fcn, char *parameter)
{
    if (fcn->num_parameters >= N_PARAMS) {
        fprintf(stderr, "ERROR, N_PARAMS overflow, more than %d parameters\n", N_PARAMS);
        controlled_exit(EXIT_FAILURE);
    }

    fcn->params[fcn->num_parameters++] = parameter;
}


static bool inp_strip_braces(char *s)
{
    int nesting = 0;
    char *d = s;

    for (; *s; s++)
        if (*s == '{') {
            nesting++;
        }
        else if (*s == '}') {
            if (--nesting < 0)
                return FALSE;
        }
        else if (!isspace_c(*s)) {
            *d++ = *s;
        }

    *d++ = '\0';

    return TRUE;
}


static void inp_get_func_from_line(struct function_env *env, char *line)
{
    char *end, *orig_line = line;
    struct function *function;

    /* skip `.func' */
    line = skip_non_ws(line);
    line = skip_ws(line);

    /* get function name */
    end = line;
    while (*end && !isspace_c(*end) && *end != '(')
        end++;

    function = new_function(env, copy_substring(line, end));

    end = skip_ws(end);

    if (*end != '(')
        goto Lerror;

    end = skip_ws(end + 1);

    /* get function parameters */
    for (;;) {
        char *beg = end;
        while (*end && !isspace_c(*end) && *end != ',' && *end != ')')
            end++;
        if (end == beg)
            break;
        new_function_parameter(function, copy_substring(beg, end));
        end = skip_ws(end);
        if (*end != ',')
            break;
        end = skip_ws(end + 1);
        if (*end == ')')
            goto Lerror;
    }

    if (*end != ')')
        goto Lerror;

    end = skip_ws(end + 1);

    // skip an unwanted and non advertised optional '='
    if (*end == '=')
        end = skip_ws(end + 1);

    function->body = copy(end);

    if (inp_strip_braces(function->body)) {
        int i;

        char *accept = TMALLOC(char, function->num_parameters + 1);
        for (i = 0; i < function->num_parameters; i++)
            accept[i] = function->params[i][0];
        accept[i] = '\0';

        function->accept = accept;
        return;
    }

    tfree(function->body);

Lerror:
    // fixme, free()
    fprintf(stderr, "ERROR: failed to parse .func in: %s\n", orig_line);
    controlled_exit(EXIT_FAILURE);
}


/*
 * grab functions at the current .subckt nesting level
 */

static void inp_grab_func(struct function_env *env, struct card *c)
{
    int nesting = 0;

    for (; c; c = c->nextcard) {

        if (*c->line == '*')
            continue;

        if (ciprefix(".subckt", c->line))
            nesting++;
        if (ciprefix(".ends", c->line))
            nesting--;

        if (nesting < 0)
            break;

        if (nesting > 0)
            continue;

        if (ciprefix(".func", c->line)) {
            inp_get_func_from_line(env, c->line);
            *c->line = '*';
        }
    }
}


static char *search_func_arg(
        char *str, struct function *fcn, int *which, char *str_begin)
{
    for (; (str = strpbrk(str, fcn->accept)) != NULL; str++) {
        char before;

        if (str > str_begin)
            before = str[-1];
        else
            before = '\0';

        if (is_arith_char(before) || isspace_c(before) ||
                strchr(",=", before)) {
            int i;
            for (i = 0; i < fcn->num_parameters; i++) {
                size_t len = strlen(fcn->params[i]);
                if (strncmp(str, fcn->params[i], len) == 0) {
                    char after = str[len];
                    if (is_arith_char(after) || isspace_c(after) ||
                            strchr(",=", after)) {
                        *which = i;
                        return str;
                    }
                }
            }
        }
    }

    return NULL;
}


static char *inp_do_macro_param_replace(struct function *fcn, char *params[])
{
    char *str = copy(fcn->body);
    int i;

    char *collect_ptr = NULL;
    char *arg_ptr = str;
    char *rest = str;

    while ((arg_ptr = search_func_arg(arg_ptr, fcn, &i, str)) != NULL) {
        char *p;
        int is_vi = 0;

        /* exclude v(nn, parameter), v(parameter, nn), v(parameter),
           and i(parameter) if here 'parameter' is also a node name */

        /* go backwards from 'parameter' and find '(' */
        for (p = arg_ptr; --p > str;)
            if (*p == '(' || *p == ')') {
                if ((*p == '(') && strchr("vi", p[-1]) &&
                        (p - 2 < str || is_arith_char(p[-2]) ||
                                isspace_c(p[-2]) || strchr(",=", p[-2])))
                    is_vi = 1;
                break;
            }

        /* if we have a true v( or i( */
        if (is_vi) {
            /* go forward and find closing ')' */
            for (p = arg_ptr + 1; *p; p++)
                if (*p == '(' || *p == ')')
                    break;
            /* We have a true v(...) or i(...),
               so skip it, and continue searching for new 'parameter' */
            if (*p == ')') {
                arg_ptr = p;
                continue;
            }
        }

        {
            size_t collect_ptr_len = collect_ptr ? strlen(collect_ptr) : 0;
            size_t len = strlen(rest) + strlen(params[i]) + 1;
            int prefix_len = (int) (arg_ptr - rest);
            if (str_has_arith_char(params[i])) {
                collect_ptr = TREALLOC(
                        char, collect_ptr, collect_ptr_len + len + 2);
                sprintf(collect_ptr + collect_ptr_len, "%.*s(%s)", prefix_len,
                        rest, params[i]);
            }
            else {
                collect_ptr =
                        TREALLOC(char, collect_ptr, collect_ptr_len + len);
                sprintf(collect_ptr + collect_ptr_len, "%.*s%s", prefix_len,
                        rest, params[i]);
            }
        }

        arg_ptr += strlen(fcn->params[i]);
        rest = arg_ptr;
    }

    if (collect_ptr) {
        char *new_str = tprintf("%s%s", collect_ptr, rest);
        tfree(collect_ptr);
        tfree(str);
        str = new_str;
    }

    return str;
}


static char *inp_expand_macro_in_str(struct function_env *env, char *str)
{
    struct function *function;
    char *open_paren_ptr, *close_paren_ptr, *fcn_name, *params[FCN_PARAMS];
    char *curr_ptr, *macro_str, *curr_str = NULL;
    int num_params, i;
    char *orig_ptr = str, *search_ptr = str, *orig_str = copy(str);
    char keep;

    /* If we have '.model mymod mdname(params)', don't treat this as a function,
    but skip '.model mymod mdname' and only then start searching for functions. */
    if (ciprefix(".model", search_ptr)){
        search_ptr = nexttok(search_ptr);
        search_ptr = nexttok(search_ptr);
        char *end;
        findtok_noparen(&search_ptr, &search_ptr, &end);
    }
    // printf("%s: enter(\"%s\")\n", __FUNCTION__, str);
    while ((open_paren_ptr = strchr(search_ptr, '(')) != NULL) {

        fcn_name = open_paren_ptr;
        while (--fcn_name >= search_ptr)
            /* function name consists of numbers, letters and special
             * characters (VALIDCHARS) */
            if (!isalnum_c(*fcn_name) && !strchr(VALIDCHARS, *fcn_name))
                break;
        fcn_name++;

        search_ptr = open_paren_ptr + 1;
        if (open_paren_ptr == fcn_name)
            continue;

        *open_paren_ptr = '\0';

        function = find_function(env, fcn_name);

        *open_paren_ptr = '(';

        if (!function)
            continue;

        /* find the closing paren */
        {
            int num_parens = 1;
            char *c = open_paren_ptr + 1;

            for (; *c; c++) {
                if (*c == '(')
                    num_parens++;
                if (*c == ')' && --num_parens == 0)
                    break;
            }

            if (num_parens) {
                fprintf(stderr,
                        "ERROR: did not find closing parenthesis for "
                        "function call in str: %s\n",
                        orig_str);
                controlled_exit(EXIT_FAILURE);
            }

            close_paren_ptr = c;
        }

        /*
         * if (ciprefix("v(", curr_ptr)) {
         *     // look for any commas and change to ' '
         *     char *str_ptr = curr_ptr;
         *     while (*str_ptr != '\0' && *str_ptr != ')') {
         *         if (*str_ptr == ',' || *str_ptr == '(')
         *             *str_ptr = ' '; str_ptr++; }
         *     if (*str_ptr == ')')
         *         *str_ptr = ' ';
         * }
         */

        /* get the parameters */
        curr_ptr = open_paren_ptr + 1;

        for (num_params = 0; curr_ptr < close_paren_ptr; curr_ptr++) {
            char *beg_parameter;
            int num_parens;
            if (isspace_c(*curr_ptr))
                continue;
            beg_parameter = curr_ptr;
            num_parens = 0;
            for (; curr_ptr < close_paren_ptr; curr_ptr++) {
                if (*curr_ptr == '(')
                    num_parens++;
                if (*curr_ptr == ')')
                    num_parens--;
                if (*curr_ptr == ',' && num_parens == 0)
                    break;
            }
            if (num_params == FCN_PARAMS) {
                fprintf(stderr, "Error: Too many params in fcn, max is %d\n",
                        FCN_PARAMS);
                controlled_exit(EXIT_FAILURE);
            }
            params[num_params++] = inp_expand_macro_in_str(
                    env, copy_substring(beg_parameter, curr_ptr));
        }

        if (function->num_parameters != num_params) {
            fprintf(stderr,
                    "ERROR: parameter mismatch for function call in str: "
                    "%s\n",
                    orig_str);
            controlled_exit(EXIT_FAILURE);
        }

        macro_str = inp_do_macro_param_replace(function, params);
        macro_str = inp_expand_macro_in_str(env, macro_str);
        keep = *fcn_name;
        *fcn_name = '\0';
        {
            size_t curr_str_len = curr_str ? strlen(curr_str) : 0;
            size_t len = strlen(str) + strlen(macro_str) + 3;
            curr_str = TREALLOC(char, curr_str, curr_str_len + len);
            sprintf(curr_str + curr_str_len, "%s(%s)", str, macro_str);
        }
        *fcn_name = keep;
        tfree(macro_str);

        search_ptr = str = close_paren_ptr + 1;

        for (i = 0; i < num_params; i++)
            tfree(params[i]);
    }

    if (curr_str == NULL) {
        curr_str = orig_ptr;
    }
    else {
        if (str != NULL) {
            size_t curr_str_len = strlen(curr_str);
            size_t len = strlen(str) + 1;
            curr_str = TREALLOC(char, curr_str, curr_str_len + len);
            sprintf(curr_str + curr_str_len, "%s", str);
        }
        tfree(orig_ptr);
    }

    tfree(orig_str);
    // printf("%s: --> \"%s\"\n", __FUNCTION__, curr_str);

    return curr_str;
}


static void inp_expand_macros_in_func(struct function_env *env)
{
    struct function *f;

    for (f = env->functions; f; f = f->next)
        f->body = inp_expand_macro_in_str(env, f->body);
}


static struct function_env *new_function_env(struct function_env *up)
{
    struct function_env *env = TMALLOC(struct function_env, 1);

    env->up = up;
    env->functions = NULL;

    return env;
}


static struct function_env *delete_function_env(struct function_env *env)
{
    struct function_env *up = env->up;
    struct function *f;

    for (f = env->functions; f;) {
        struct function *here = f;
        f = f->next;
        free_function(here);
        tfree(here);
    }

    tfree(env);

    return up;
}


static struct card *inp_expand_macros_in_deck(
        struct function_env *env, struct card *c)
{
    env = new_function_env(env);

    inp_grab_func(env, c);

    inp_expand_macros_in_func(env);

    for (; c; c = c->nextcard) {

        if (*c->line == '*')
            continue;

        if (ciprefix(".subckt", c->line)) {
            struct card *subckt = c;
            c = inp_expand_macros_in_deck(env, c->nextcard);
            if (c)
                continue;

            fprintf(stderr, "Error: line %d, missing .ends\n  %s\n",
                    subckt->linenum_orig, subckt->line);
            controlled_exit(EXIT_BAD);
        }

        if (ciprefix(".ends", c->line))
            break;

        c->line = inp_expand_macro_in_str(env, c->line);
    }

    env = delete_function_env(env);

    return c;
}


/* Put {} around tokens for handling in numparam.
   Searches for the next '=' in the line to become active.
   Several exceptions (eg. no 'set' or 'b' lines, no .cmodel lines,
   no lines between .control and .endc, no .option lines).
   Special handling of vectors with [] and complex values with < >

   h_vogt 20 April 2008
   * For xspice and num_pram compatibility .cmodel added
   * .cmodel will be replaced by .model in inp_fix_param_values()
   * and then the entire line is skipped (will not be changed by this
   function).
   * Usage of numparam requires {} around the parameters in the .cmodel line.
   * May be obsolete?
   */

static void inp_fix_param_values(struct card *c)
{
    char *beg_of_str, *end_of_str, *old_str, *equal_ptr, *new_str;
    char *vec_str, *tmp_str, *natok, *buffer, *newvec, *whereisgt;
    bool control_section = FALSE;
    wordlist *nwl;
    int parens;

    for (; c; c = c->nextcard) {
        char *line = c->line;

        if (*line == '*' || (ciprefix(".para", line) && strchr(line, '{')))
            continue;

        if (ciprefix(".control", line)) {
            control_section = TRUE;
            continue;
        }

        if (ciprefix(".endc", line)) {
            control_section = FALSE;
            continue;
        }

        /* no handling of params in "option" lines */
        if (control_section || ciprefix(".option", line))
            continue;

        /* no handling of params in "set" lines */
        if (ciprefix("set", line))
            continue;

        /* no handling of params in B source lines */
        if (*line == 'b')
            continue;

        /* for xspice .cmodel: replace .cmodel with .model and skip entire
         * line) */
        if (ciprefix(".cmodel", line)) {
            *(++line) = 'm';
            *(++line) = 'o';
            *(++line) = 'd';
            *(++line) = 'e';
            *(++line) = 'l';
            *(++line) = ' ';
            continue;
        }

        /* exclude CIDER models */
        if (ciprefix(".model", line) &&
                (strstr(line, "numos") || strstr(line, "numd") ||
                        strstr(line, "nbjt") || strstr(line, "nbjt2") ||
                        strstr(line, "numd2"))) {
            continue;
        }

        /* exclude CIDER devices with ic.file parameter */
        if (strstr(line, "ic.file"))
            continue;

        while ((equal_ptr = find_assignment(line)) != NULL) {

            // special case: .MEASURE {DC|AC|TRAN} result FIND out_variable
            // WHEN out_variable2=out_variable3 no braces around
            // out_variable3. out_variable3 may be v(...) or i(...)
            if (ciprefix(".meas", line))
                if (((equal_ptr[1] == 'v') || (equal_ptr[1] == 'i')) &&
                        (equal_ptr[2] == '(')) {
                    // find closing ')' and skip token v(...) or i(...)
                    while (*equal_ptr != ')' && equal_ptr[1] != '\0')
                        equal_ptr++;
                    line = equal_ptr + 1;
                    continue;
                }

            beg_of_str = skip_ws(equal_ptr + 1);
            /* all cases where no {} have to be put around selected token */
            if (isdigit_c(*beg_of_str) || *beg_of_str == '{' ||
                    *beg_of_str == '.' || *beg_of_str == '"' ||
                    ((*beg_of_str == '-' || *beg_of_str == '+') &&
                            isdigit_c(beg_of_str[1])) ||
                    ((*beg_of_str == '-' || *beg_of_str == '+') &&
                            beg_of_str[1] == '.' &&
                            isdigit_c(beg_of_str[2])) ||
                    ciprefix("true", beg_of_str) ||
                    ciprefix("false", beg_of_str)) {
                line = equal_ptr + 1;
            }
            else if (*beg_of_str == '[') {
                /* A vector following the '=' token: code to put curly
                   brackets around all params
                   inside a pair of square brackets */
                end_of_str = beg_of_str;
                while (*end_of_str != ']' && *end_of_str != '\0')
                    end_of_str++;
                /* string xx yyy from vector [xx yyy] */
                tmp_str = vec_str =
                        copy_substring(beg_of_str + 1, end_of_str);

                /* work on vector elements inside [] */
                nwl = NULL;
                for (;;) {
                    natok = gettok(&vec_str);
                    if (!natok)
                        break;

                    buffer = TMALLOC(char, strlen(natok) + 4);
                    if (isdigit_c(*natok) || *natok == '{' || *natok == '.' ||
                            *natok == '"' ||
                            (*natok == '-' && isdigit_c(natok[1])) ||
                            ciprefix("true", natok) ||
                            ciprefix("false", natok) || eq(natok, "<") ||
                            eq(natok, ">")) {
                        (void) sprintf(buffer, "%s", natok);
                        /* A complex value found inside a vector [< x1 y1> <x2
                         * y2>] */
                        /* < xx and yy > have been dealt with before */
                        /* <xx */
                    }
                    else if (*natok == '<') {
                        if (isdigit_c(natok[1]) ||
                                (natok[1] == '-' && isdigit_c(natok[2]))) {
                            (void) sprintf(buffer, "%s", natok);
                        }
                        else {
                            *natok = '{';
                            (void) sprintf(buffer, "<%s}", natok);
                        }
                        /* yy> */
                    }
                    else if (strchr(natok, '>')) {
                        if (isdigit_c(*natok) ||
                                (*natok == '-' && isdigit_c(natok[1]))) {
                            (void) sprintf(buffer, "%s", natok);
                        }
                        else {
                            whereisgt = strchr(natok, '>');
                            *whereisgt = '}';
                            (void) sprintf(buffer, "{%s>", natok);
                        }
                        /* all other tokens */
                    }
                    else {
                        (void) sprintf(buffer, "{%s}", natok);
                    }
                    tfree(natok);
                    nwl = wl_cons(copy(buffer), nwl);
                    tfree(buffer);
                }
                tfree(tmp_str);
                nwl = wl_reverse(nwl);
                /* new vector elements */
                newvec = wl_flatten(nwl);
                wl_free(nwl);
                /* insert new vector into actual line */
                *equal_ptr = '\0';
                new_str = tprintf(
                        "%s=[%s] %s", c->line, newvec, end_of_str + 1);
                tfree(newvec);

                old_str = c->line;
                c->line = new_str;
                line = new_str + strlen(old_str) + 1;
                tfree(old_str);
            }
            else if (*beg_of_str == '<') {
                /* A complex value following the '=' token: code to put curly
                   brackets around all params inside a pair < > */
                end_of_str = beg_of_str;
                while (*end_of_str != '>' && *end_of_str != '\0')
                    end_of_str++;
                /* string xx yyy from vector [xx yyy] */
                vec_str = copy_substring(beg_of_str + 1, end_of_str);

                /* work on tokens inside <> */
                nwl = NULL;
                for (;;) {
                    natok = gettok(&vec_str);
                    if (!natok)
                        break;

                    buffer = TMALLOC(char, strlen(natok) + 4);
                    if (isdigit_c(*natok) || *natok == '{' || *natok == '.' ||
                            *natok == '"' ||
                            (*natok == '-' && isdigit_c(natok[1])) ||
                            ciprefix("true", natok) ||
                            ciprefix("false", natok)) {
                        (void) sprintf(buffer, "%s", natok);
                    }
                    else {
                        (void) sprintf(buffer, "{%s}", natok);
                    }
                    tfree(natok);
                    nwl = wl_cons(copy(buffer), nwl);
                    tfree(buffer);
                }
                nwl = wl_reverse(nwl);
                /* new elements of complex variable */
                newvec = wl_flatten(nwl);
                wl_free(nwl);
                /* insert new complex value into actual line */
                *equal_ptr = '\0';
                new_str = tprintf(
                        "%s=<%s> %s", c->line, newvec, end_of_str + 1);
                tfree(newvec);

                old_str = c->line;
                c->line = new_str;
                line = new_str + strlen(old_str) + 1;
                tfree(old_str);
            }
            else {
                /* put {} around token to be accepted as numparam */
                end_of_str = beg_of_str;
                parens = 0;
                while (*end_of_str != '\0' &&
                        (!isspace_c(*end_of_str) || (parens > 0))) {
                    if (*end_of_str == '(')
                        parens++;
                    if (*end_of_str == ')')
                        parens--;
                    end_of_str++;
                }

                *equal_ptr = '\0';

                if (*end_of_str == '\0') {
                    new_str = tprintf("%s={%s}", c->line, beg_of_str);
                }
                else {
                    *end_of_str = '\0';
                    new_str = tprintf("%s={%s} %s", c->line, beg_of_str,
                            end_of_str + 1);
                }
                old_str = c->line;
                c->line = new_str;

                line = new_str + strlen(old_str) + 1;
                tfree(old_str);
            }
        }
    }
}


static char *get_param_name(char *line)
{
    char *beg;
    char *equal_ptr = strchr(line, '=');

    if (!equal_ptr) {
        fprintf(stderr, "ERROR: could not find '=' on parameter line '%s'!\n",
                line);
        controlled_exit(EXIT_FAILURE);
    }

    equal_ptr = skip_back_ws(equal_ptr, line);

    beg = skip_back_non_ws(equal_ptr, line);

    return copy_substring(beg, equal_ptr);
}


static char *get_param_str(char *line)
{
    char *equal_ptr = strchr(line, '=');

    if (equal_ptr)
        return skip_ws(equal_ptr + 1);
    else
        return line;
}


struct dependency {
    int level;
    int skip;
    char *param_name;
    char *param_str;
    char *depends_on[DEPENDSON];
    struct card *card;
};


static int inp_get_param_level(
        int param_num, struct dependency *deps, int num_params)
{
    int i, k, l, level = 0;
    static int recounter = 0;
    recounter++;

    if (recounter > 1000) { /* magic number 1000: if larger, stack overflow occurs */
        fprintf(stderr,
            "ERROR: A level depth greater 1000 for dependent parameters is not supported!\n");
        fprintf(stderr,
            "    You probably do have a circular parameter dependency at line\n");
        fprintf(stderr,
            "    %s\n", deps[param_num].card->line);
        recounter = 0;
        controlled_exit(EXIT_FAILURE);
    }

    if (deps[param_num].level != -1) {
        recounter = 0;
        return deps[param_num].level;
    }

    for (i = 0; deps[param_num].depends_on[i]; i++) {

        for (k = 0; k < num_params; k++)
            if (deps[param_num].depends_on[i] == deps[k].param_name)
                break;

        if (k >= num_params) {
            fprintf(stderr,
                    "ERROR: unable to find dependency parameter for %s!\n",
                    deps[param_num].param_name);
            recounter = 0;
            controlled_exit(EXIT_FAILURE);
        }

        l = inp_get_param_level(k, deps, num_params) + 1;

        if (level < l)
            level = l;
    }

    deps[param_num].level = level;
    recounter = 0;

    return level;
}

/* Return the number of terminals for a given device, characterized by
   the first letter of its instance line. Returns 0 upon error. */
int get_number_terminals(char *c)
{
    int i, j, k;
    char *name[12];
    char nam_buf[128];
    bool area_found = FALSE;

    if (!c)
        return 0;

    switch (*c) {
        case 'r':
        case 'c':
        case 'l':
        case 'k':
        case 'f':
        case 'h':
        case 'b':
        case 'v':
        case 'i':
            return 2;
            break;
        case 'd':
            i = 0;
            /* find the first token with "off" or "=" in the line*/
            while ((i < 10) && (*c != '\0')) {
                char *inst = gettok_instance(&c);
                strncpy(nam_buf, inst, sizeof(nam_buf) - 1);
                txfree(inst);
                if ( i > 3 && (search_plain_identifier(nam_buf, "off") || search_plain_identifier(nam_buf, "thermal") || strchr(nam_buf, '=')))
                    break;
                i++;
            }
            return i - 2;
            break;
        case 'x':
            i = 0;
            /* find the first token with "params:" or "=" in the line*/
            while ((i < 100) && (*c != '\0')) {
                char *inst = gettok_instance(&c);
                strncpy(nam_buf, inst, sizeof(nam_buf) - 1);
                txfree(inst);
                if (search_plain_identifier(nam_buf, "params:") || strchr(nam_buf, '='))
                    break;
                i++;
            }
            return i - 2;
            break;
        case 'u':
        case 'j':
        case 'w':
        case 'z':
            return 3;
            break;
        case 't':
        case 'o':
        case 'g':
        case 'e':
        case 's':
        case 'y':
            return 4;
            break;
        case 'm': /* recognition of 4, 5, 6, or 7 nodes for SOI devices needed
                   */
        {
            i = 0;
            char* cc, * ccfree;
            cc = copy(c);
            /* required to make m= 1 a single token m=1 */
            ccfree = cc = inp_remove_ws(cc);
            /* find the first token with "off", "tnodeout", "thermal" or "=" in the line*/
            while ((i < 20) && (*cc != '\0')) {
                char* inst = gettok_instance(&cc);
                strncpy(nam_buf, inst, sizeof(nam_buf) - 1);
                txfree(inst);
                if ( i > 4 && (search_plain_identifier(nam_buf, "off") || strchr(nam_buf, '=') ||
                    search_plain_identifier(nam_buf, "tnodeout") || search_plain_identifier(nam_buf, "thermal")))
                    break;
                i++;
            }
            tfree(ccfree);
            return i - 2;
            break;
        }
        case 'p': /* recognition of up to 100 cpl nodes */
            i = j = 0;
            /* find the last token in the line*/
            while ((i < 100) && (*c != '\0')) {
                char *tmp_inst = gettok_instance(&c);
                strncpy(nam_buf, tmp_inst, 32);
                tfree(tmp_inst);
                if (strchr(nam_buf, '='))
                    j++;
                i++;
            }
            if (i == 100)
                return 0;
            return i - j - 2;
            break;
        case 'q': /* recognition of 3, 4 or 5 terminal bjt's needed */
            /* QXXXXXXX NC NB NE <NS> <NT> MNAME <AREA> <OFF> <IC=VBE, VCE>
             * <TEMP=T> */
            /* 12 tokens maximum */
        {
            char* cc, * ccfree;
            i = j = 0;
            cc = copy(c);
            /* required to make m= 1 a single token m=1 */
            ccfree = cc = inp_remove_ws(cc);
            while ((i < 12) && (*cc != '\0')) {
                char* comma;
                name[i] = gettok_instance(&cc);
                if (search_plain_identifier(name[i], "off") || strchr(name[i], '='))
                    j++;
#ifdef CIDER
                if (search_plain_identifier(name[i], "save") || search_plain_identifier(name[i], "print"))
                    j++;
#endif
                /* If we have IC=VBE, VCE instead of IC=VBE,VCE we need to inc
                 * j */
                if ((comma = strchr(name[i], ',')) != NULL &&
                        (*(++comma) == '\0'))
                    j++;
                /* If we have IC=VBE , VCE ("," is a token) we need to inc j
                 */
                if (eq(name[i], ","))
                    j++;
                i++;
            }
            tfree(ccfree);
            i--;
            area_found = FALSE;
            for (k = i; k > i - j - 1; k--) {
                bool only_digits = TRUE;
                char* nametmp = name[k];
                /* MNAME has to contain at least one alpha character. AREA may
                   be assumed if we have a token with only digits, and where
                   the previous token does not end with a ',' */
                while (*nametmp) {
                    if (isalpha_c(*nametmp) || (*nametmp == ','))
                        only_digits = FALSE;
                    nametmp++;
                }
                if (only_digits && (strchr(name[k - 1], ',') == NULL))
                    area_found = TRUE;
            }
            for (k = i; k >= 0; k--)
                tfree(name[k]);
            if (area_found) {
                return i - j - 2;
            }
            else {
                return i - j - 1;
            }
            break;
        }
#ifdef OSDI
        case 'n': /* Recognize an unknown number of nodes by stopping at tokens with '=' */
        {
            i = 0;
            char* cc, * ccfree;
            cc = copy(c);
            /* required to make m= 1 a single token m=1 */
            ccfree = cc = inp_remove_ws(cc);
            /* find the first token with "off", "tnodeout", "thermal" or "=" in the line*/
            while ((i < 20) && (*cc != '\0')) {
                char* inst = gettok_instance(&cc);
                strncpy(nam_buf, inst, sizeof(nam_buf) - 1);
                txfree(inst);
                if (i > 2 && (strchr(nam_buf, '=')))
                    break;
                i++;
            }
            tfree(ccfree);
            return i - 2;
            break;
        }
#endif
        default:
            return 0;
            break;
    }
}


static char *ya_search_identifier(
        char *str, const char *identifier, char *str_begin);


static void inp_quote_params(struct card *s_c, struct card *e_c,
        struct dependency *deps, int num_params);

/* sort parameters based on parameter dependencies */

static void inp_sort_params(struct card *param_cards,
        struct card *card_bf_start, struct card *s_c, struct card *e_c)
{
    int i, j, num_params, ind = 0, max_level;

    struct card *c;
    int skipped;
    int arr_size;

    struct dependency *deps;

    if (param_cards == NULL)
        return;

    /* determine the number of lines with .param */

    arr_size = 0;
    for (c = param_cards; c; c = c->nextcard)
        if (strchr(c->line, '='))
            arr_size++;

    deps = TMALLOC(struct dependency, arr_size);

    num_params = 0;
    for (c = param_cards; c; c = c->nextcard)
        // ignore .param lines without '='
        if (strchr(c->line, '=')) {
            deps[num_params].depends_on[0] = NULL;
            deps[num_params].level = -1;
            deps[num_params].skip = 0;
            deps[num_params].param_name =
                    get_param_name(c->line); /* copy in fcn */
            deps[num_params].param_str = copy(get_param_str(c->line));
            deps[num_params].card = c;
            num_params++;
        }

    // look for duplicately defined parameters and mark earlier one to skip
    // param list is ordered as defined in netlist

    skipped = 0;
    for (i = 0; i < num_params; i++) {
        for (j = i + 1; j < num_params; j++)
            if (strcmp(deps[i].param_name, deps[j].param_name) == 0)
                break;
        if (j < num_params) {
            deps[i].skip = 1;
            skipped++;
        }
    }

    for (i = 0; i < num_params; i++)
        if (!deps[i].skip) {
            char *param = deps[i].param_name;
            for (j = 0; j < num_params; j++)
                if (j != i &&
                        search_plain_identifier(deps[j].param_str, param)) {
                    for (ind = 0; deps[j].depends_on[ind]; ind++)
                        ;
                    deps[j].depends_on[ind++] = param;
                    if (ind == DEPENDSON) {
                        fprintf(stderr, "Error in netlist: Too many parameter dependencies (> %d)\n", ind);
                        fprintf(stderr, "    Please check your netlist.\n");
                        controlled_exit(EXIT_BAD);
                    }
                    deps[j].depends_on[ind] = NULL;
                }
        }

    max_level = 0;
    for (i = 0; i < num_params; i++) {
        deps[i].level = inp_get_param_level(i, deps, num_params);
        if (max_level < deps[i].level)
            max_level = deps[i].level;
    }

    c = card_bf_start;

    ind = 0;
    for (i = 0; i <= max_level; i++)
        for (j = 0; j < num_params; j++)
            if (!deps[j].skip && deps[j].level == i) {
                c = insert_deck(c, deps[j].card);
                ind++;
            }
            else if (deps[j].skip) {
                line_free_x(deps[j].card, FALSE);
                deps[j].card = NULL;
            }

    num_params -= skipped;
    if (ind != num_params) {
        fprintf(stderr,
                "ERROR: found wrong number of parameters during levelization "
                "( %d instead of %d parameter s)!\n",
                ind, num_params);
        controlled_exit(EXIT_FAILURE);
    }

    inp_quote_params(s_c, e_c, deps, num_params);

    // clean up memory
    for (i = 0; i < arr_size; i++) {
        tfree(deps[i].param_name);
        tfree(deps[i].param_str);
    }

    tfree(deps);
}


static void inp_add_params_to_subckt(
        struct names *subckt_w_params, struct card *subckt_card)
{
    struct card *card = subckt_card->nextcard;
    char *subckt_line = subckt_card->line;
    char *new_line, *param_ptr, *subckt_name, *end_ptr;

    for (; card; card = card->nextcard) {

        char *curr_line = card->line;

        if (!ciprefix(".para", curr_line))
            break;

        param_ptr = strchr(curr_line, ' ');
        param_ptr = skip_ws(param_ptr);

        if (!strstr(subckt_line, "params:")) {
            new_line = tprintf("%s params: %s", subckt_line, param_ptr);

            subckt_name = skip_non_ws(subckt_line);
            subckt_name = skip_ws(subckt_name);
            end_ptr = skip_non_ws(subckt_name);
            add_name(subckt_w_params, copy_substring(subckt_name, end_ptr));
        }
        else {
            new_line = tprintf("%s %s", subckt_line, param_ptr);
        }

        tfree(subckt_line);
        subckt_line = new_line;

        *curr_line = '*';
    }

    subckt_card->line = subckt_line;
}


/*
 * process a sequence of decks
 *   starting from a         `.suckt' deck
 *   upto the corresponding  `.ends'  deck
 * return a pointer to the terminating `.ends' deck
 *
 * recursivly descend
 *   when another `.subckt' is found
 *
 * parameters are removed from the main list
 *   and collected into a local list `first_param_card'
 * then processed and reinserted into the main list
 *
 */

static struct card *inp_reorder_params_subckt(
        struct names *subckt_w_params, struct card *subckt_card)
{
    struct card *first_param_card = NULL;
    struct card *last_param_card = NULL;

    struct card *prev_card = subckt_card;
    struct card *c = subckt_card->nextcard;

    /* move .param lines to beginning of deck */
    while (c != NULL) {

        char *curr_line = c->line;

        if (*curr_line == '*') {
            prev_card = c;
            c = c->nextcard;
            continue;
        }

        if (ciprefix(".subckt", curr_line)) {
            prev_card = inp_reorder_params_subckt(subckt_w_params, c);
            c = prev_card->nextcard;
            continue;
        }

        if (ciprefix(".ends", curr_line)) {
            if (first_param_card) {
                inp_sort_params(first_param_card, subckt_card,
                        subckt_card->nextcard, c);
                inp_add_params_to_subckt(subckt_w_params, subckt_card);
            }
            return c;
        }

        if (ciprefix(".para", curr_line)) {
            prev_card->nextcard = c->nextcard;

            last_param_card = insert_deck(last_param_card, c);

            if (!first_param_card)
                first_param_card = last_param_card;

            c = prev_card->nextcard;
            continue;
        }

        prev_card = c;
        c = c->nextcard;
    }

    /* the terminating `.ends' deck wasn't found */
    fprintf(stderr, "Error: Missing .ends statement\n");
    controlled_exit(EXIT_FAILURE);
}


static void inp_reorder_params(
        struct names *subckt_w_params, struct card *list_head)
{
    struct card *first_param_card = NULL;
    struct card *last_param_card = NULL;

    struct card *prev_card = list_head;
    struct card *c = prev_card->nextcard;

    /* move .param lines to beginning of deck */
    while (c != NULL) {

        char *curr_line = c->line;

        if (*curr_line == '*') {
            prev_card = c;
            c = c->nextcard;
            continue;
        }

        if (ciprefix(".subckt", curr_line)) {
            prev_card = inp_reorder_params_subckt(subckt_w_params, c);
            c = prev_card->nextcard;
            continue;
        }

        /* check for an unexpected extra `.ends' deck */
        if (ciprefix(".ends", curr_line)) {
            fprintf(stderr, "Error: Unexpected extra .ends in line:\n  %s.\n",
                    curr_line);
            controlled_exit(EXIT_FAILURE);
        }

        if (ciprefix(".para", curr_line)) {
            prev_card->nextcard = c->nextcard;

            last_param_card = insert_deck(last_param_card, c);

            if (!first_param_card)
                first_param_card = last_param_card;

            c = prev_card->nextcard;
            continue;
        }

        prev_card = c;
        c = c->nextcard;
    }

    inp_sort_params(first_param_card, list_head, list_head->nextcard, NULL);
}


// iterate through deck and find lines with multiply defined parameters
//
// split line up into multiple lines and place those new lines immediately
// after the current multi-param line in the deck

static int inp_split_multi_param_lines(struct card *card, int line_num)
{
    for (; card; card = card->nextcard) {

        char *curr_line = card->line;

        if (*curr_line == '*')
            continue;

        if (ciprefix(".para", curr_line)) {

            char *equal_ptr, **array;
            int i, counter = 0;

            while ((equal_ptr = find_assignment(curr_line)) != NULL) {
                counter++;
                curr_line = equal_ptr + 1;
            }

            if (counter <= 1)
                continue;

            array = TMALLOC(char *, counter);

            // need to split multi param line
            curr_line = card->line;
            counter = 0;
            while ((equal_ptr = find_assignment(curr_line)) != NULL) {

                char *beg_param, *end_param;

                int expression_depth = 0;
                int paren_depth = 0;

                beg_param = skip_back_ws(equal_ptr, curr_line);
                beg_param = skip_back_non_ws(beg_param, curr_line);
                end_param = skip_ws(equal_ptr + 1);
                while (*end_param && !isspace_c(*end_param)) {
                    /* Advance over numeric or string expression. */

                    if (*end_param == '"') {
                        /* RHS is quoted string. */

                        end_param++;
                        while (*end_param != '\0' && *end_param != '"')
                            end_param++;
                        if (*end_param == '"')
                            end_param++;
                    } else if (*end_param == ',' && paren_depth == 0) {
                        break;
                    } else {
                        while (*end_param != '\0' && *end_param != '"' &&
                               (!isspace_c(*end_param) ||
                                expression_depth || paren_depth)) {
                            if (*end_param == ',' && paren_depth == 0)
                                break;
                            if (*end_param == '{')
                                ++expression_depth;
                            else if (*end_param == '(')
                                ++paren_depth;
                            else if (*end_param == '}' && expression_depth > 0)
                                --expression_depth;
                            else if (*end_param == ')' && paren_depth > 0)
                                --paren_depth;
                            end_param++;
                        }
                    }
                }

                if (end_param[-1] == ',')
                    end_param--;

                array[counter++] = tprintf(".param %.*s",
                        (int) (end_param - beg_param), beg_param);

                curr_line = end_param;
            }

            // comment out current multi-param line
            *(card->line) = '*';
            // insert new param lines immediately after current line
            for (i = 0; i < counter; i++)
                card = insert_new_line(card, array[i], line_num++, card->linenum_orig);

            tfree(array);
        }
    }

    return line_num;
}


static int identifier_char(char c)
{
    return (c == '_') || isalnum_c(c);
}


static bool b_transformation_wanted(const char *p)
{
    const char *start = p;

    for (p = start; (p = strpbrk(p, "vith")) != NULL; p++) {
        if (p > start && identifier_char(p[-1]))
            continue;
        if (strncmp(p, "v(", 2) == 0 || strncmp(p, "i(", 2) == 0)
            return TRUE;
        if (strncmp(p, "temper", 6) == 0 && !identifier_char(p[6]))
            return TRUE;
        if (strncmp(p, "hertz", 5) == 0 && !identifier_char(p[5]))
            return TRUE;
        if (strncmp(p, "time", 4) == 0 && !identifier_char(p[4]))
            return TRUE;
    }

    return FALSE;
}


char *search_identifier(char *str, const char *identifier, char *str_begin)
{
    if (str && identifier) {
        while ((str = strstr(str, identifier)) != NULL) {
            char before;

            if (str > str_begin)
                before = str[-1];
            else
                before = '\0';

            if (is_arith_char(before) || isspace_c(before) ||
                    strchr("=,{", before)) {
                char after = str[strlen(identifier)];
                if (is_arith_char(after) || isspace_c(after) ||
                        strchr(",}", after))
                    return str;
            }

            str++;
        }
    }
    return NULL;
}


char *ya_search_identifier(char *str, const char *identifier, char *str_begin)
{
    if (str && identifier) {
        while ((str = strstr(str, identifier)) != NULL) {
            char before;

            if (str > str_begin)
                before = str[-1];
            else
                before = '\0';

            if (is_arith_char(before) || isspace_c(before) ||
                    before == ',' || (str <= str_begin)) {
                char after = str[strlen(identifier)];
                if (is_arith_char(after) || isspace_c(after) ||
                            after == '\0' || after == ',')
                    break;
            }

            str++;
        }
    }
    return str;
}

/* Check for 'identifier' being in string str, surrounded by chars
   not being a member of alphanumeric or '_' characters. */
char *search_plain_identifier(char *str, const char *identifier)
{
    if (str && identifier && *identifier != '\0') {
        char *str_begin = str;
        while ((str = strstr(str, identifier)) != NULL) {
            char before;

            if (str > str_begin)
                before = str[-1];
            else
                before = '\0';

            if (!before || !identifier_char(before)) {
                char after = str[strlen(identifier)];
                if (!after || !identifier_char(after))
                    return str;
            }

            str += strlen(identifier);
        }
    }
    return NULL;
}

/* return a string that consists of tc1 and tc2 evaluated
   or having a rhs for numparam expansion {...}.
   The retun string has to be freed by the caller after its usage. */
static char* eval_tc(char* line, char *tline) {
    double tc1, tc2;
    char *str_ptr, *tc1_ptr, *tc2_ptr, *tc1_str = NULL, *tc2_str = NULL;
    char* cut_line = line;
    str_ptr = strstr(cut_line, "tc1=");
    if (str_ptr) {
        /* We need to have 'tc1=something */
        if (str_ptr[4]) {
            tc1_ptr = str_ptr + 4;
            int error = 0;
            tc1 = INPevaluate(&tc1_ptr, &error, 1);
            /*We have a value and create the tc1 string */
            if (error == 0) {
                tc1_str = tprintf("tc1=%15.8e", tc1);
            }
            else if (error == 1 && *tc1_ptr == '{' && *(tc1_ptr + 1) != '}') {
                char* bra = gettok_char(&tc1_ptr, '}', TRUE, TRUE);
                if (bra) {
                    tc1_str = tprintf("tc1=%s", bra);
                    tfree(bra);
                }
                else {
                    fprintf(stderr, "Warning: Cannot copy tc1 in line\n   %s\n   ignored\n", tline);
                    tc1_str = copy(" ");
                }
            }
            else {
                fprintf(stderr, "Warning: Cannot copy tc1 in line\n   %s\n   ignored\n", tline);
                tc1_str = copy(" ");
            }
        }
    }
    else {
        tc1_str = copy(" ");
    }
    cut_line = line;
    str_ptr = strstr(cut_line, "tc2=");
    if (str_ptr) {
        /* We need to have 'tc2=something */
        if (str_ptr[4]) {
            tc2_ptr = str_ptr + 4;
            int error = 0;
            tc2 = INPevaluate(&tc2_ptr, &error, 1);
            /*We have a value and create the tc2 string */
            if (error == 0) {
                tc2_str = tprintf("tc2=%15.8e", tc2);
            }
            else if (error == 1 && *tc2_ptr == '{' && *(tc2_ptr + 1) != '}') {
                char* bra = gettok_char(&tc2_ptr, '}', TRUE, TRUE);
                if (bra) {
                    tc2_str = tprintf("tc2=%s", bra);
                    tfree(bra);
                }
                else {
                    fprintf(stderr, "Warning: Cannot copy tc2 in line\n   %s\n   ignored\n", tline);
                    tc2_str = copy(" ");
                }
            }
            else {
                fprintf(stderr, "Warning: Cannot copy tc2 in line\n   %s\n   ignored\n", tline);
                tc2_str = copy(" ");
            }
        }
    }
    else {
        tc2_str = copy(" ");
    }
    char* ret_str = tprintf("%s %s", tc1_str, tc2_str);
    tfree(tc1_str);
    tfree(tc2_str);

    return ret_str;
}

/* return a string that consists of m evaluated (like m=5)
   or having a rhs for numparam expansion m={...}.
   The return string has to be freed by the caller after its usage. */
static char* eval_m(char* line, char* tline) {
    double m;
    char* str_ptr, * m_ptr, * m_str = NULL;
    char* cut_line = line;
    str_ptr = strstr(cut_line, " m=");
    if (str_ptr) {
        /* We need to have 'm=something */
        if (str_ptr[3]) {
            m_ptr = str_ptr + 3;
            int error = 0;
            m = INPevaluate(&m_ptr, &error, 1);
            /*We have a value and create the m string */
            if (error == 0) {
                m_str = tprintf("m=%15.8e", m);
            }
            else if (error == 1 && *m_ptr == '{' && *(m_ptr + 1) != '\0' && *(m_ptr + 1) != '}') {
                char* bra = gettok_char(&m_ptr, '}', TRUE, TRUE);
                if (bra) {
                    m_str = tprintf("m=%s", bra);
                    tfree(bra);
                }
                else {
                    fprintf(stderr, "Warning: Cannot copy m in line\n   %s\n   ignored\n", tline);
                    m_str = copy(" ");
                }
            }
            else {
                fprintf(stderr, "Warning: Cannot copy m in line\n   %s\n   ignored\n", tline);
                m_str = copy(" ");
            }
        }
    }
    else {
        m_str = copy(" ");
    }

    return m_str;
}

/* return a string that consists of the m value only.
   If m is not given, return string "1".
   The return string has to be freed by the caller after its usage. */
static char* eval_mvalue(char* line, char* tline) {
    double m;
    char* str_ptr, * m_ptr, * m_str = NULL;
    char* cut_line = line;
    str_ptr = strstr(cut_line, " m=");
    if (str_ptr) {
        /* We need to have 'm=something */
        if (str_ptr[3]) {
            m_ptr = str_ptr + 3;
            int error = 0;
            m = INPevaluate(&m_ptr, &error, 1);
            /*We have a value and create the m string */
            if (error == 0) {
                m_str = tprintf("%15.8e", m);
            }
            else if (error == 1 && *m_ptr == '{' && *(m_ptr + 1) != '\0' && *(m_ptr + 1) != '}') {
                char* bra = gettok_char(&m_ptr, '}', TRUE, TRUE);
                if (bra) {
                    m_str = tprintf("%s", bra);
                    tfree(bra);
                }
                else {
                    fprintf(stderr, "Warning: Cannot copy m in line\n   %s\n   ignored\n", tline);
                    m_str = copy(" ");
                }
            }
            else {
                fprintf(stderr, "Warning: Cannot copy m in line\n   %s\n   ignored\n", tline);
                m_str = copy(" ");
            }
        }
    }
    else {
        m_str = copy("1");
    }

    return m_str;
}


/* ps compatibility:
   Exxx n1 n2 TABLE {0.45*v(1)} = (-1, -0.5) (-0.5, 0) (0, 2) (0.5, 2) (1, 1)
   -->
   exxx n1 n2 exxx_int1 0 1
   bexxx exxx_int2 0 v=   4.5000000000e-01 * v(1)
   aexxx %v(exxx_int2) %v(exxx_int1) xfer_exxx
   .model xfer_exxx pwl(x_array=[-1 -0.5 0 0.5 1 ] y_array=[-0.5 0 2 2 1 ]
         input_domain=0.1 fraction=TRUE)

   gd16 16 1 table {v(16,1)} ((-100,-100e-15)(0,0)(1m,1u)(2m,1m))
    -->
   gd16 16 1 gd16_int1 0 1
   bgd16 gd16_int2 0 v= v(16,1)
   agd16 %v(gd16_int2) %v(gd16_int1) xfer_gd16
   .model xfer_gd16 pwl(x_array=[-100 0 1m 2m ] y_array=[-100e-15 0 1u 1m ]
       input_domain=0.1 fraction=TRUE)
*/

/* hs compatibility:
   Exxx n1 n2 VCVS n3 n4 gain --> Exxx n1 n2 n3 n4 gain
   Gxxx n1 n2 VCCS n3 n4 tr --> Gxxx n1 n2 n3 n4 tr

   Two step approach to keep the original names for reuse,
   i.e. for current measurements like i(Exxx):
   Exxx n1 n2 VOL = {equation}
   -->
   Exxx n1 n2 int1 0 1
   BExxx int1 0 V = {equation}

   Gxxx n1 n2 CUR = {equation}
   -->
   Gxxx n1 n2 int1 0 1
   BGxxx int1 0 V = {equation}

   Do the following transformations only if {equation} contains
   simulation output like v(node), v(node1, node2), i(branch).
   Otherwise let do numparam the substitutions (R=const is handled
   in inp2r.c).

   Rxxx n1 n2 R = {equation} or Rxxx n1 n2 {equation}
   -->
   BRxxx n1 n2 I = V(n1,n2)/{equation}

   Unfortunately the capability for ac noise calculation of
   resistance may be lost.

   Cxxx n1 n2 C = {equation} or Cxxx n1 n2 {equation}
   -->
   Exxx  n-aux 0  n2 n1  1
   Cxxx  n-aux 0         1
   Bxxx  n1 n2  I = i(Exxx) * equation

   Lxxx n1 n2 L = {equation} or Lxxx n1 n2 {equation}
   -->
   Fxxx n-aux 0  Bxxx -1
   Lxxx n-aux 0      1
   Bxxx n1 n2 V = v(n-aux) * 1e-16

*/

static void inp_compat(struct card *card)
{
    char *str_ptr, *cut_line, *title_tok, *node1, *node2;
    char *out_ptr, *exp_ptr, *beg_ptr, *end_ptr, *copy_ptr, *del_ptr;
    char *xline, *x2line = NULL, *x3line = NULL, *x4line = NULL;
    size_t xlen, i, pai = 0, paui = 0, ii;
    char *ckt_array[100];

    int skip_control = 0;

    char *equation;

    for (; card; card = card->nextcard) {

        char *curr_line = card->line;
        int currlinenumber = card->linenum_orig;

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

        if (*curr_line == '*')
            continue;

        if (*curr_line == 'e') {
            /*    Exxx n1 n2 VCVS n3 n4 gain --> Exxx n1 n2 n3 n4 gain
                  remove vcvs */
            replace_token(curr_line, "vcvs", 4, 7);

            /* Exxx n1 n2 value={equation}
               -->
               Exxx n1 n2   vol={equation} */
            if ((str_ptr = search_plain_identifier(curr_line, "value")) !=
                    NULL) {
                if (str_ptr[5] == '=')
                    *str_ptr++ = ' ';
                memcpy(str_ptr, " vol=", 5);
            }
            /* Exxx n1 n2 TABLE {expression} = (x0, y0) (x1, y1) (x2, y2)
               -->
             Exxx n1 n2 Exxx_int1 0 1
             BExxx Exxx_int2 0 v = expression
             aExxx %v(Exxx_int2) %v(Exxx_int1) xfer_Exxx
             .model xfer_Exxx pwl(x_array=[x0 x1 x2]
                   y_array=[y0 y1 y2]
                   input_domain=0.1 fraction=TRUE)
            */
            if ((str_ptr = search_plain_identifier(curr_line, "table")) != NULL) {
                char *expression, *firstno, *secondno;
                DS_CREATE(dxar, 200);
                DS_CREATE(dyar, 200);
                cut_line = curr_line;
                /* title and nodes */
                title_tok = gettok(&cut_line);
                node1 = gettok(&cut_line);
                node2 = gettok(&cut_line);
                // Exxx  n1 n2 int1 0 1
                ckt_array[0] = tprintf("%s %s %s %s_int1 0 1", title_tok,
                        node1, node2, title_tok);
                // skip "table"
                cut_line = skip_ws(cut_line);
                if (ciprefix("table", cut_line)) {
                    /* a regular TABLE line */
                    cut_line += 5;
                    // compatibility, allow table = {expr} {pairs}
                    if (*cut_line == '=')
                        *cut_line++ = ' ';
                    // get the expression
                    str_ptr = gettok_char(&cut_line, '{', FALSE, FALSE);
                    expression = gettok_char(
                            &cut_line, '}', TRUE, TRUE); /* expression */
                    if (!expression || !str_ptr) {
                        fprintf(stderr,
                                "Error: bad syntax in line %d\n  %s\n",
                                card->linenum_orig, card->line);
                        controlled_exit(EXIT_BAD);
                    }
                    tfree(str_ptr);
                    /* remove '{' and '}' from expression */
                    if ((str_ptr = strchr(expression, '{')) != NULL)
                        *str_ptr = ' ';
                    if ((str_ptr = strchr(expression, '}')) != NULL)
                        *str_ptr = ' ';
                    /* cut_line may now have a '=', if yes, it will have '{'
                       and '}' (braces around token after '=') */
                    if ((str_ptr = strchr(cut_line, '=')) != NULL)
                        *str_ptr = ' ';
                    if ((str_ptr = strchr(cut_line, '{')) != NULL)
                        *str_ptr = ' ';
                    if ((str_ptr = strchr(cut_line, '}')) != NULL)
                        *str_ptr = ' ';

                    /* E51 50 51 E51_int1 0 1
                       BE51 e51_int2 0 v = V(40,41)
                       ae51 %v(e51_int2) %v(e51_int1) xfer_e51
                       .model xfer_e51 pwl(x_array=[-10 0 1m 2m 3m]
                       + y_array=[-1n 0 1m 1 100]
                       + input_domain=0.1 fraction=TRUE)
                     */
                    ckt_array[1] = tprintf("b%s %s_int2 0 v = %s", title_tok,
                            title_tok, expression);
                    ckt_array[2] = tprintf(
                            "a%s %%v(%s_int2) %%v(%s_int1) xfer_%s",
                            title_tok, title_tok, title_tok, title_tok);
                    /* (x0, y0) (x1, y1) (x2, y2) to x0 x1 x2, y0 y1 y2 */
                    int ipairs = 0;
                    char* pair_line = cut_line;
                    while (*cut_line != '\0') {
                        firstno = gettok_node(&cut_line);
                        secondno = gettok_node(&cut_line);
                        if ((!firstno && secondno) ||
                                (firstno && !secondno)) {
                            fprintf(stderr, "Error: Missing token in %s\n",
                                    curr_line);
                            break;
                        }
                        else if (!firstno && !secondno)
                            continue;
                        sadd(&dxar, firstno);
                        cadd(&dxar, ' ');
                        sadd(&dyar, secondno);
                        cadd(&dyar, ' ');
                        tfree(firstno);
                        tfree(secondno);
                        ipairs++;
                    }

                    /* There is a strange usage of the TABLE function:
                       A single pair (x0, y0) will return a constant voltage y0 */
                    if (ipairs == 1) {
                        tfree(ckt_array[1]);
                        tfree(ckt_array[2]);
                        firstno = gettok_node(&pair_line);
                        tfree(firstno);
                        secondno = gettok_node(&pair_line);
                        ckt_array[1] = tprintf("v%s %s_int1 0 %s", title_tok,
                            title_tok, secondno);
                        tfree(secondno);
                        // comment out current variable e line
                        *(card->line) = '*';
                        // insert new lines immediately after current line
                        for (i = 0; i < 2; i++)
                            card = insert_new_line(card, ckt_array[i], (int)i + 1, currlinenumber);
                    }
                    else {
                        ckt_array[3] = tprintf(
                            ".model xfer_%s pwl(x_array=[%s] y_array=[%s] "
                            "input_domain=0.1 fraction=TRUE limit=TRUE)",
                            title_tok, ds_get_buf(&dxar), ds_get_buf(&dyar));
                        // comment out current variable e line
                        *(card->line) = '*';
                        // insert new lines immediately after current line
                        for (i = 0; i < 4; i++)
                            card = insert_new_line(card, ckt_array[i], (int)i + 1, currlinenumber);
                    }
                    tfree(expression);
                    tfree(title_tok);
                    tfree(node1);
                    tfree(node2);
                    ds_free(&dxar);
                    ds_free(&dyar);
                }
             }

            /* Exxx n1 n2 VOL = {equation}
               -->
               Exxx n1 n2 int1 0 1
               BExxx int1 0 V = {equation}
            */
            /* search for ' vol=' or ' vol =' */
            if (((str_ptr = strchr(curr_line, '=')) != NULL) &&
                    prefix("vol",
                            skip_back_non_ws(skip_back_ws(str_ptr, curr_line),
                                    curr_line))) {
                cut_line = curr_line;
                /* title and nodes */
                title_tok = gettok(&cut_line);
                node1 = gettok(&cut_line);
                node2 = gettok(&cut_line);
                /* Find equation, starts with '{', till end of line */
                str_ptr = strchr(cut_line, '{');
                if (str_ptr == NULL) {
                    fprintf(stderr, "ERROR: mal formed E line: %s\n",
                            curr_line);
                    controlled_exit(EXIT_FAILURE);
                }

                // Exxx  n1 n2 int1 0 1
                ckt_array[0] = tprintf("%s %s %s %s_int1 0 1",
                        title_tok, node1, node2, title_tok);
                // BExxx int1 0 V = {equation}
                ckt_array[1] = tprintf("b%s %s_int1 0 v = %s",
                        title_tok, title_tok, str_ptr);

                // comment out current variable e line
                *(card->line) = '*';
                // insert new B source line immediately after current line
                for (i = 0; i < 2; i++)
                    card = insert_new_line(card, ckt_array[i], (int)i + 1, currlinenumber);

                tfree(title_tok);
                tfree(node1);
                tfree(node2);
            }
        }
        else if (*curr_line == 'g') {
            /* Gxxx n1 n2 VCCS n3 n4 tr --> Gxxx n1 n2 n3 n4 tr
               remove vccs */
            replace_token(curr_line, "vccs", 4, 7);

            /* Gxxx n1 n2 value={equation}
               -->
               Gxxx n1 n2   cur={equation} */
            if ((str_ptr = search_plain_identifier(curr_line, "value")) !=
                    NULL) {
                if (str_ptr[5] == '=')
                    *str_ptr++ = ' ';
                memcpy(str_ptr, " cur=", 5);
            }

            /* Gxxx n1 n2 TABLE {expression} = (x0, y0) (x1, y1) (x2, y2)
               -->
             Gxxx n1 n2 Gxxx_int1 0 1
             BGxxx Gxxx_int2 0 v = expression
             aGxxx %v(Gxxx_int2) %v(Gxxx_int1) xfer_Gxxx
             .model xfer_Gxxx pwl(x_array=[x0 x1 x2]
                   y_array=[y0 y1 y2]
                   input_domain=0.1 fraction=TRUE)
            */
            if ((str_ptr = search_plain_identifier(curr_line, "table")) != NULL) {
                char *expression, *firstno, *secondno;
                char *m_ptr, *m_token;
                DS_CREATE(dxar, 200);
                DS_CREATE(dyar, 200);
                cut_line = curr_line;
                /* title and nodes */
                title_tok = gettok(&cut_line);
                /* the title in the a instance should not contain %, [, nor ]
                   replace it by '_' */
                char* stok = copy(title_tok);
                char* ntok = stok;
                while (*ntok != '\0') {
                    if (*ntok == '[' || *ntok == ']' || *ntok == '%')
                        *ntok = '_';
                    ntok++;
                }
                node1 = gettok(&cut_line);
                node2 = gettok(&cut_line);
                // Gxxx  n1 n2 int1 0 1
                // or
                // Gxxx  n1 n2 int1 0 m='expr'
                /* find multiplier m at end of line */
                m_ptr = strstr(cut_line, "m=");
                if (m_ptr) {
                    m_token = copy(m_ptr + 2); // get only the expression
                    *m_ptr = '\0';
                }
                else
                    m_token = copy("1");
                ckt_array[0] = tprintf("%s %s %s %s_int1 0 %s",
                        title_tok, node1, node2, stok, m_token);
                // skip "table"
                cut_line = skip_ws(cut_line);
                if (!ciprefix("table", cut_line)) {
                    fprintf(stderr, "Error: bad syntax in line %d\n  %s\n",
                            card->linenum_orig, card->line);
                    controlled_exit(EXIT_BAD);
                }
                cut_line += 5;
                // compatibility, allow table = {expr} {pairs}
                if (*cut_line == '=')
                    *cut_line++ = ' ';
                // get the expression
                str_ptr =  gettok_char(&cut_line, '{', FALSE, FALSE);
                expression = gettok_char(&cut_line, '}', TRUE, TRUE);
                if (!expression || !str_ptr) {
                    fprintf(stderr, "Error: bad syntax in line %d\n  %s\n",
                            card->linenum_orig, card->line);
                    controlled_exit(EXIT_BAD);
                }
                tfree(str_ptr);
                /* remove '{' and '}' from expression */
                if ((str_ptr = strchr(expression, '{')) != NULL)
                    *str_ptr = ' ';
                if ((str_ptr = strchr(expression, '}')) != NULL)
                    *str_ptr = ' ';
                /* cut_line may now have a '=', if yes, it will have '{' and
                   '}' (braces around token after '=') */
                if ((str_ptr = strchr(cut_line, '=')) != NULL)
                    *str_ptr = ' ';
                /* FIXME: To enable adding expressions in {} as pwl parameters, we need an intelligent
                   removal of {}, not just brute force as following now. */
                if ((str_ptr = strchr(cut_line, '{')) != NULL)
                    *str_ptr = ' ';
                if ((str_ptr = strchr(cut_line, '}')) != NULL)
                    *str_ptr = ' ';

                /* GD51 50 51 gd51_int1 0 1
                   BGD51 gd51_int2 0 v = V(50,51)
                   agd51 %v(gd51_int2) %v(gd51_int1) xfer_gd51
                   .model xfer_gd51 pwl(x_array=[-10 0 1m 2m 3m]
                   + y_array=[-1n 0 1m 1 100]
                   + input_domain=0.1 fraction=TRUE)
                 */
                ckt_array[1] = tprintf("b%s %s_int2 0 v = %s", title_tok,
                        stok, expression);
                ckt_array[2] = tprintf("a%s %%v(%s_int2) %%v(%s_int1) xfer_%s",
                        stok, stok, stok, stok);
                /* (x0, y0) (x1, y1) (x2, y2) to x0 x1 x2, y0 y1 y2 */
                int ipairs = 0;
                char* pair_line = cut_line;
                while (*cut_line != '\0') {
                    /* If we have expressions in {}, we copy the complete expression,
                       otherwise only the next token. */
                    if (*cut_line == '{')
                        firstno = gettok_char(&cut_line, '}', TRUE, TRUE);
                    else
                        firstno = gettok_node(&cut_line);
                    if (*cut_line == '{')
                        secondno = gettok_char(&cut_line, '}', TRUE, TRUE);
                    else
                        secondno = gettok_node(&cut_line);
                    if ((!firstno && secondno) || (firstno && !secondno)) {
                        fprintf(stderr, "Error: Missing token in %s\n",
                                curr_line);
                        break;
                    }
                    else if (!firstno && !secondno)
                        continue;
                    sadd(&dxar, firstno);
                    cadd(&dxar, ' ');
                    sadd(&dyar, secondno);
                    cadd(&dyar, ' ');
                    tfree(firstno);
                    tfree(secondno);
                    ipairs++;
                }

                /* There is a strange usage of the TABLE function:
                   A single pair (x0, y0) will return a constant current y0 */
                if (ipairs == 1) {
                    tfree(ckt_array[1]);
                    tfree(ckt_array[2]);
                    firstno = gettok_node(&pair_line);
                    tfree(firstno);
                    secondno = gettok_node(&pair_line);
                    ckt_array[1] = tprintf("v%s %s_int1 0 %s", title_tok,
                        stok, secondno);
                    tfree(secondno);
                    // comment out current variable e line
                    *(card->line) = '*';
                    // insert new lines immediately after current line
                    for (i = 0; i < 2; i++)
                        card = insert_new_line(card, ckt_array[i], (int)i + 1, currlinenumber);
                }
                else {
                    ckt_array[3] = tprintf(".model xfer_%s pwl(x_array=[%s] y_array=[%s] "
                        "input_domain=0.1 fraction=TRUE limit=TRUE)", stok, ds_get_buf(&dxar), ds_get_buf(&dyar));
                    // comment out current variable g line
                    *(card->line) = '*';
                    // insert new lines immediately after current line
                    for (i = 0; i < 4; i++)
                        card = insert_new_line(card, ckt_array[i], (int)i + 1, currlinenumber);
                }

                tfree(expression);
                tfree(title_tok);
                tfree(stok);
                tfree(node1);
                tfree(node2);
                tfree(m_token);
                ds_free(&dxar);
                ds_free(&dyar);
            }
            /*
              Gxxx n1 n2 CUR = {equation}
              -->
              Gxxx n1 n2 int1 0 1
              BGxxx int1 0 V = {equation}
            */
            /* search for ' cur=' or ' cur =' */
            if (((str_ptr = strchr(curr_line, '=')) != NULL) &&
                    prefix("cur",
                            skip_back_non_ws(skip_back_ws(str_ptr, curr_line),
                                    curr_line))) {
                char *m_ptr, *m_token;
                cut_line = curr_line;
                /* title and nodes */
                title_tok = gettok(&cut_line);
                node1 = gettok(&cut_line);
                node2 = gettok(&cut_line);
                /* Find equation, starts with '{', till end of line */
                str_ptr = strchr(cut_line, '{');
                if (str_ptr == NULL) {
                    fprintf(stderr, "ERROR: mal formed G line: %s\n",
                            curr_line);
                    controlled_exit(EXIT_FAILURE);
                }
                /* find multiplier m at end of line */
                m_ptr = strstr(cut_line, "m=");
                if (m_ptr) {
                    m_token = copy(m_ptr + 2); // get only the expression
                    *m_ptr = '\0';
                }
                else
                    m_token = copy("1");
                // Gxxx  n1 n2 int1 0 1
                // or
                // Gxxx  n1 n2 int1 0 m='expr'
                ckt_array[0] = tprintf("%s %s %s %s_int1 0 %s",
                        title_tok, node1, node2, title_tok, m_token);
                // BGxxx int1 0 V = {equation}
                ckt_array[1] = tprintf("b%s %s_int1 0 v = %s",
                        title_tok, title_tok, str_ptr);

                // comment out current variable g line
                *(card->line) = '*';
                // insert new B source line immediately after current line
                for (i = 0; i < 2; i++)
                    card = insert_new_line(card, ckt_array[i], (int)i + 1, currlinenumber);

                tfree(title_tok);
                tfree(m_token);
                tfree(node1);
                tfree(node2);
            }
        }

        /* F element compatibility */
        else if (*curr_line == 'f') {
            char *equastr, *vnamstr;
            /* Fxxx n1 n2 CCCS vnam gain --> Fxxx n1 n2 vnam gain
               remove cccs */
            replace_token(curr_line, "cccs", 4, 6);

            /* Deal with
               Fxxx n1 n2 vnam {equation}
               if equation contains the 'temper' token */
            if (search_identifier(curr_line, "temper", curr_line)) {
                cut_line = curr_line;
                title_tok = gettok(&cut_line);
                node1 = gettok(&cut_line);
                node2 = gettok(&cut_line);
                vnamstr = gettok(&cut_line);
                equastr = gettok(&cut_line);
                /*
                Fxxx n1 n2 vnam {equation}
                -->
                Fxxx n1 n2 vbFxxx -1
                bFxxx int1 0 i = i(vnam)*{equation}
                vbFxxx int1 0 0
                */
                // Fxxx n1 n2 VBFxxx -1
                ckt_array[0] = tprintf("%s %s %s vb%s -1",
                        title_tok, node1, node2, title_tok);
                // BFxxx BFxxx_int1 0 I = I(vnam)*{equation}
                ckt_array[1] = tprintf("b%s %s_int1 0 i = i(%s) * (%s)",
                        title_tok, title_tok, vnamstr, equastr);
                // VBFxxx int1 0 0
                ckt_array[2] = tprintf("vb%s %s_int1 0 dc 0",
                        title_tok, title_tok);
                // comment out current variable f line
                *(card->line) = '*';
                // insert new three lines immediately after current line
                for (i = 0; i < 3; i++)
                    card = insert_new_line(card, ckt_array[i], (int)i + 1, currlinenumber);

                tfree(title_tok);
                tfree(vnamstr);
                tfree(equastr);
                tfree(node1);
                tfree(node2);
            }
        }
        /* H element compatibility */
        else if (*curr_line == 'h') {
            char *equastr, *vnamstr;
            /* Hxxx n1 n2 CCVS vnam transres --> Hxxx n1 n2 vnam transres
               remove cccs */
            replace_token(curr_line, "ccvs", 4, 6);

            /* Deal with
               Hxxx n1 n2 vnam {equation}
               if equation contains the 'temper' token */
            if (search_identifier(curr_line, "temper", curr_line)) {
                cut_line = curr_line;
                title_tok = gettok(&cut_line);
                node1 = gettok(&cut_line);
                node2 = gettok(&cut_line);
                vnamstr = gettok(&cut_line);
                equastr = gettok(&cut_line);
                /*
                Hxxx n1 n2 vnam {equation}
                -->
                Hxxx n1 n2 vbHxxx -1
                bHxxx int1 0 i = i(vnam)*{equation}
                vbHxxx int1 0 0
                */
                // Hxxx n1 n2 VBHxxx -1
                ckt_array[0] = tprintf("%s %s %s vb%s -1",
                        title_tok, node1, node2, title_tok);
                // BHxxx BHxxx_int1 0 I = I(vnam)*{equation}
                ckt_array[1] = tprintf("b%s %s_int1 0 i = i(%s) * (%s)",
                        title_tok, title_tok, vnamstr, equastr);
                // VBHxxx int1 0 0
                ckt_array[2] =
                        tprintf("vb%s %s_int1 0 dc 0", title_tok, title_tok);
                // comment out current variable h line
                *(card->line) = '*';
                // insert new three lines immediately after current line
                for (i = 0; i < 3; i++)
                    card = insert_new_line(card, ckt_array[i], (int)i + 1, currlinenumber);

                tfree(title_tok);
                tfree(vnamstr);
                tfree(equastr);
                tfree(node1);
                tfree(node2);
            }
        }

        /* Rxxx n1 n2 R = {equation} or Rxxx n1 n2 {equation}
           -->
           BRxxx pos neg I = V(pos, neg)/{equation}
        */
        else if (*curr_line == 'r') {
            cut_line = curr_line;
            /* make BRxxx pos neg I = V(pos, neg)/{equation}*/
            title_tok = gettok(&cut_line);
            node1 = gettok(&cut_line);
            node2 = gettok(&cut_line);
            /* check only after skipping Rname and nodes, either may contain
             * time (e.g. Rtime)*/
            if (!b_transformation_wanted(cut_line)) {
                tfree(title_tok);
                tfree(node1);
                tfree(node2);
                continue;
            }

            /* Find equation, starts with '{', till end of line */
            str_ptr = strchr(cut_line, '{');
            if (str_ptr == NULL) {
                fprintf(stderr, "ERROR: mal formed R line: %s\n", curr_line);
                fprintf(stderr, "    {...} or '...' around equation's right hand side are missing!\n");
                controlled_exit(EXIT_FAILURE);
            }
            else
                equation = gettok_char(&str_ptr, '}', TRUE, TRUE);

            /* evauate tc1 and tc2 */
            char* tcrstr = eval_tc(cut_line, card->line);

            /* evaluate m */
            char* mstr = eval_m(cut_line, card->line);

            /* white noise model by x2line, x3line, x4line */
            /* if variable enable_noisy_r is set */
            bool rnoise = cp_getvar("enable_noisy_r", CP_BOOL, NULL, 0);
            /* if instance parameter noisy=1 (or noise=1) is set */
            if (strstr(cut_line, "noisy=1") || strstr(cut_line, "noise=1"))
                rnoise = TRUE;
            else if (strstr(cut_line, "noisy=0") || strstr(cut_line, "noise=0"))
                rnoise = FALSE;

            /* tc1, tc2, and m are enabled */
            xline = tprintf("b%s %s %s i = v(%s, %s)/(%s) %s %s reciproctc=1 reciprocm=0",
                    title_tok, node1, node2, node1, node2, equation, tcrstr, mstr);
            if (rnoise) {
                x2line = tprintf("b%s_1 %s %s i = i(v%s_3)/sqrt(%s)",
                        title_tok, node1, node2, title_tok, equation);
                x3line = tprintf("r%s_2 %s_3 0 1.0 %s",
                        title_tok, title_tok, tcrstr);
                x4line = tprintf("v%s_3 %s_3 0 0", title_tok, title_tok);
            }

            tfree(tcrstr);
            tfree(mstr);

            // comment out current old R line
            *(card->line) = '*';
            // insert new B source line immediately after current line
            card = insert_new_line(card, xline, 1, currlinenumber);
            if (rnoise) {
                card = insert_new_line(card, x2line, 2, currlinenumber);
                card = insert_new_line(card, x3line, 3, currlinenumber);
                card = insert_new_line(card, x4line, 4, currlinenumber);
            }

            tfree(title_tok);
            tfree(node1);
            tfree(node2);
            tfree(equation);
        }
        /* Cxxx n1 n2 C = {equation} or Cxxx n1 n2 {equation}
           -->
           Exxx  n-aux 0  n2 n1  1
           Cxxx  n-aux 0         1
           Bxxx  n1 n2  I = i(Exxx) * equation
           or
           Cxxx n1 n2 Q = {equation}
           -->
           Gxxx  n1 n2 n-aux 0  1
           Lxxx  n-aux 0        1
           Bxxx  0 n-aux I = equation
        */
        else if (*curr_line == 'c') {
            cut_line = curr_line;
            title_tok = gettok(&cut_line);
            node1 = gettok(&cut_line);
            node2 = gettok(&cut_line);
            /* check only after skipping Cname and nodes, either may contain
             * time (e.g. Ctime) - for charge formula transformation in any case */
            if ((!strstr(curr_line, "q=")) && (!b_transformation_wanted(cut_line))) {
                tfree(title_tok);
                tfree(node1);
                tfree(node2);
                continue;
            }
            /* Find equation, starts with '{', till end of line */
            str_ptr = strchr(cut_line, '{');
            if (str_ptr == NULL) {
                fprintf(stderr, "ERROR: mal formed C line: %s\n", curr_line);
                fprintf(stderr, "    {...} or '...' around equation's right hand side are missing!\n");
                controlled_exit(EXIT_FAILURE);
            }
            else
                equation = gettok_char(&str_ptr, '}', TRUE, TRUE);

            /* evauate tc1 and tc2 */
            char* tcrstr = eval_tc(cut_line, card->line);

            /* evaluate m */
            char* mstr = eval_mvalue(cut_line, card->line);

            if (strstr(curr_line, "c=")) { /* capacitance formulation */
                // Exxx  n-aux 0  n2 n1  1
                ckt_array[0] = tprintf("e%s %s_int1 0 %s %s %s", title_tok,
                        title_tok, node2, node1, mstr);
                // Cxxx  n-aux 0  1
                ckt_array[1] = tprintf("c%s %s_int1 0 1", title_tok, title_tok);
                // Bxxx  n1 n2  I = i(Exxx) * equation
                ckt_array[2] = tprintf("b%s %s %s i = i(e%s) * (%s) "
                                        "%s reciproctc=1",
                        title_tok, node1, node2, title_tok, equation, tcrstr);
            } else {                       /* charge formulation */
                // Gxxx  n1 n2 n-aux 0  1
                ckt_array[0] = tprintf("g%s %s %s %s_int1 0 %s",
                            title_tok, node1, node2, title_tok, mstr);
                // Lxxx  n-aux 0 1
                ckt_array[1] = tprintf("l%s %s_int1 0 1", title_tok, title_tok);
                // Bxxx  0 n-aux I = equation
                ckt_array[2] = tprintf("b%s 0 %s_int1 i = (%s) "
                                        "%s reciproctc=1",
                        title_tok, title_tok, equation, tcrstr);
            }
            tfree(tcrstr);
            tfree(mstr);
            // comment out current variable capacitor line
            *(card->line) = '*';
            // insert new B source line immediately after current line
            for (i = 0; i < 3; i++)
                card = insert_new_line(card, ckt_array[i], (int)i + 1, currlinenumber);

            tfree(title_tok);
            tfree(node1);
            tfree(node2);
            tfree(equation);
        }

        /* Lxxx n1 n2 L = {equation} or Lxxx n1 n2 {equation}
           -->
           Fxxx n-aux 0  Bxxx -1
           Lxxx n-aux 0      1
           Bxxx n1 n2 V = v(n-aux) * equation
        */
        else if (*curr_line == 'l') {
            cut_line = curr_line;
            /* title and nodes */
            title_tok = gettok(&cut_line);
            node1 = gettok(&cut_line);
            node2 = gettok(&cut_line);
            if (!b_transformation_wanted(cut_line)) {
                tfree(title_tok);
                tfree(node1);
                tfree(node2);
                continue;
            }

            /* Find equation, starts with '{', till end of line */
            str_ptr = strchr(cut_line, '{');
            if (str_ptr == NULL) {
                fprintf(stderr, "ERROR: mal formed L line: %s\n", curr_line);
                fprintf(stderr, "    {...} or '...' around equation's right hand side are missing!\n");
                controlled_exit(EXIT_FAILURE);
            }
            else
                equation = gettok_char(&str_ptr, '}', TRUE, TRUE);

            /* evauate tc1 and tc2 */
            char* tcrstr = eval_tc(cut_line, card->line);

            /* evaluate m */
            char* mstr = eval_mvalue(cut_line, card->line);

            // Fxxx  n-aux 0  Bxxx  1
            ckt_array[0] = tprintf("f%s %s_int2 0 b%s -1",
                    title_tok, title_tok, title_tok);
            // Lxxx  n-aux 0  1
            ckt_array[1] = tprintf("l%s %s_int2 0 1", title_tok, title_tok);
            // Bxxx  n1 n2  V = v(n-aux) * equation
            ckt_array[2] = tprintf("b%s %s %s v = v(%s_int2) * (%s) / %s "
                                    "%s reciproctc=0",
                    title_tok, node2, node1, title_tok, equation, mstr, tcrstr);

            tfree(tcrstr);
            tfree(mstr);
            // comment out current variable inductor line
            *(card->line) = '*';
            // insert new B source line immediately after current line
            for (i = 0; i < 3; i++)
                card = insert_new_line(card, ckt_array[i], (int)i + 1, currlinenumber);

            tfree(title_tok);
            tfree(node1);
            tfree(node2);
            tfree(equation);
        }
        /* .probe -> .save
           .print, .plot, .save, .four,
           An ouput vector may be replaced by the following:
           myoutput=par('expression')
           .meas
           A vector out_variable may be replaced by
           par('expression')
        */
        else if (*curr_line == '.') {
            // replace .probe by .save
            if ((str_ptr = strstr(curr_line, ".probe")) != NULL)
                memcpy(str_ptr, ".save ", 6);

            /* Various formats for measure statement:
             * .MEASURE {DC|AC|TRAN} result WHEN out_variable=val
             * + <TD=td> <FROM=val> <TO=val>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
             *
             * .MEASURE {DC|AC|TRAN} result WHEN out_variable=out_variable2
             * + <TD=td> <FROM=val> <TO=val>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
             *
             * .MEASURE {DC|AC|TRAN} result FIND out_variable
             * + WHEN out_variable2=val
             * + <TD=td> <FROM=val> <TO=val>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
             *
             * .MEASURE {DC|AC|TRAN} result FIND out_variable
             * + WHEN out_variable2=out_variable3
             * + <TD=td>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
             *
             * .MEASURE {DC|AC|TRAN} result FIND out_variable AT=val
             * + <FROM=val> <TO=val>
             *
             * .MEASURE {DC|AC|TRAN} result {AVG|MIN|MAX|MIN_AT|MAX_AT|PP|RMS}
             * + out_variable
             * + <TD=td> <FROM=val> <TO=val>
             *
             * .MEASURE {DC|AC|TRAN} result INTEG<RAL> out_variable
             * + <TD=td> <FROM=val> <TO=val>
             *
             * .MEASURE {DC|AC|TRAN} result DERIV<ATIVE> out_variable AT=val
             *
             * .MEASURE {DC|AC|TRAN} result DERIV<ATIVE> out_variable
             * + WHEN out_variable2=val
             * + <TD=td>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
             *
             * .MEASURE {DC|AC|TRAN} result DERIV<ATIVE> out_variable
             * + WHEN out_variable2=out_variable3
             * + <TD=td>
             * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>

             The user may set any out_variable to par(' expr ').
             We have to replace this by v(pa_xx) and generate a B source line.

             * ------------------------------------------------------------ */
            if (ciprefix(".meas", curr_line)) {
                if (strstr(curr_line, "par(") == NULL)
                    continue;
                cut_line = curr_line;
                // search for 'par('
                while ((str_ptr = strstr(cut_line, "par(")) != NULL) {
                    if (pai > 99) {
                        fprintf(stderr,
                                "ERROR: More than 99 function calls to "
                                "par()\n");
                        fprintf(stderr, "  Limited to 99 per input file\n");
                        controlled_exit(EXIT_FAILURE);
                    }

                    // we have ' par({ ... })', the right delimeter is a ' '
                    // or '='
                    if (ciprefix(" par({", (str_ptr - 1))) {
                        // find expression
                        beg_ptr = end_ptr = str_ptr + 5;
                        while ((*end_ptr != ' ') && (*end_ptr != '=') &&
                                (*end_ptr != '\0')) {
                            end_ptr++;
                        }
                        exp_ptr = copy_substring(beg_ptr, end_ptr - 2);
                        cut_line = str_ptr;
                        // generate node
                        out_ptr = tprintf("pa_%02d", (int) pai);
                        // Bout_ptr  out_ptr 0  V = v(expr_ptr)
                        ckt_array[pai] = tprintf("b%s %s 0 v = %s",
                                out_ptr, out_ptr, exp_ptr);
                        ckt_array[++pai] = NULL;
                        // length of the replacement V(out_ptr)
                        del_ptr = copy_ptr = tprintf("v(%s)", out_ptr);
                        // length of the replacement part in original line
                        xlen = strlen(exp_ptr) + 7;
                        // copy the replacement without trailing '\0'
                        for (ii = 0; ii < xlen; ii++)
                            if (*copy_ptr)
                                *cut_line++ = *copy_ptr++;
                            else
                                *cut_line++ = ' ';

                        tfree(del_ptr);
                        tfree(exp_ptr);
                        tfree(out_ptr);
                    }
                    // or we have '={par({ ... })}', the right delimeter is a
                    // ' '
                    else if (ciprefix("={par({", (str_ptr - 2))) {
                        // find expression
                        beg_ptr = end_ptr = str_ptr + 5;
                        while ((*end_ptr != ' ') && (*end_ptr != '\0'))
                            end_ptr++;
                        exp_ptr = copy_substring(beg_ptr, end_ptr - 3);
                        // generate node
                        out_ptr = tprintf("pa_%02d", (int) pai);
                        // Bout_ptr  out_ptr 0  V = v(expr_ptr)
                        ckt_array[pai] = tprintf("b%s %s 0 v = %s",
                                out_ptr, out_ptr, exp_ptr);
                        ckt_array[++pai] = NULL;
                        // length of the replacement V(out_ptr)
                        del_ptr = copy_ptr = tprintf("v(%s)", out_ptr);
                        // length of the replacement part in original line
                        xlen = strlen(exp_ptr) + 9;
                        // skip '='
                        cut_line++;
                        // copy the replacement without trailing '\0'
                        for (ii = 0; ii < xlen; ii++)
                            if (*copy_ptr)
                                *cut_line++ = *copy_ptr++;
                            else
                                *cut_line++ = ' ';

                        tfree(del_ptr);
                        tfree(exp_ptr);
                        tfree(out_ptr);
                    }
                    else {
                        // nothing to replace
                        cut_line = str_ptr + 1;
                        continue;
                    }

                } // while 'par'
                // no replacement done, go to next line
                if (pai == paui)
                    continue;
                // remove white spaces
                card->line = inp_remove_ws(curr_line);
                // insert new B source line immediately after current line
                for (ii = paui; ii < pai; ii++)
                    card = insert_new_line(card, ckt_array[ii], (int)ii + 1, currlinenumber);

                paui = pai;
            }
            else if ((ciprefix(".save", curr_line)) ||
                    (ciprefix(".four", curr_line)) ||
                    (ciprefix(".print", curr_line)) ||
                    (ciprefix(".plot", curr_line))) {
                if (strstr(curr_line, "par(") == NULL)
                    continue;
                cut_line = curr_line;
                // search for 'par('
                while ((str_ptr = strstr(cut_line, "par(")) != NULL) {
                    if (pai > 99) {
                        fprintf(stderr,
                                "ERROR: More than 99 function calls to "
                                "par()\n");
                        fprintf(stderr, "  Limited to 99 per input file\n");
                        controlled_exit(EXIT_FAILURE);
                    }

                    // we have ' par({ ... })'
                    if (ciprefix(" par({", (str_ptr - 1))) {

                        // find expression
                        beg_ptr = end_ptr = str_ptr + 5;
                        while ((*end_ptr != ' ') && (*end_ptr != '\0'))
                            end_ptr++;
                        exp_ptr = copy_substring(beg_ptr, end_ptr - 2);
                        cut_line = str_ptr;
                        // generate node
                        out_ptr = tprintf("pa_%02d", (int) pai);
                        // Bout_ptr  out_ptr 0  V = v(expr_ptr)
                        ckt_array[pai] = tprintf("b%s %s 0 v = %s",
                                out_ptr, out_ptr, exp_ptr);
                        ckt_array[++pai] = NULL;
                        // length of the replacement V(out_ptr)
                        del_ptr = copy_ptr = tprintf("%s", out_ptr);
                        // length of the replacement part in original line
                        xlen = strlen(exp_ptr) + 7;
                        // copy the replacement without trailing '\0'
                        for (ii = 0; ii < xlen; ii++)
                            if (*copy_ptr)
                                *cut_line++ = *copy_ptr++;
                            else
                                *cut_line++ = ' ';

                        tfree(del_ptr);
                        tfree(exp_ptr);
                        tfree(out_ptr);
                    }
                    // or we have '={par({ ... })}'
                    else if (ciprefix("={par({", str_ptr - 2)) {

                        // find myoutput
                        beg_ptr = end_ptr = str_ptr - 2;
                        while (*beg_ptr != ' ')
                            beg_ptr--;
                        out_ptr = copy_substring(beg_ptr + 1, end_ptr);
                        cut_line = beg_ptr + 1;
                        // find expression
                        beg_ptr = end_ptr = str_ptr + 5;
                        while ((*end_ptr != ' ') && (*end_ptr != '\0'))
                            end_ptr++;
                        exp_ptr = copy_substring(beg_ptr, end_ptr - 3);
                        // Bout_ptr  out_ptr 0  V = v(expr_ptr)
                        ckt_array[pai] = tprintf("b%s %s 0 v = %s",
                                out_ptr, out_ptr, exp_ptr);
                        ckt_array[++pai] = NULL;
                        // length of the replacement V(out_ptr)
                        del_ptr = copy_ptr = tprintf("%s", out_ptr);
                        // length of the replacement part in original line
                        xlen = strlen(out_ptr) + strlen(exp_ptr) + 10;
                        // copy the replacement without trailing '\0'
                        for (ii = 0; ii < xlen; ii++)
                            if (*copy_ptr)
                                *cut_line++ = *copy_ptr++;
                            else
                                *cut_line++ = ' ';

                        tfree(del_ptr);
                        tfree(exp_ptr);
                        tfree(out_ptr);
                    }
                    // nothing to replace
                    else
                        cut_line = str_ptr + 1;
                } // while 'par('
                // no replacement done, go to next line
                if (pai == paui)
                    continue;
                // remove white spaces
                card->line = inp_remove_ws(curr_line);
                // comment out current variable capacitor line
                // *(ckt_array[0]) = '*';
                // insert new B source line immediately after current line
                for (ii = paui; ii < pai; ii++)
                    card = insert_new_line(card, ckt_array[ii], (int)ii + 1, currlinenumber);

                paui = pai;
                // continue;
            } // if .print etc.
        } // if ('.')
    }
}


/* replace a token (length 4 char) in string by spaces, if it is found
   at the correct position and the total number of tokens is o.k. */

static void replace_token(
        char *string, char *token, int wherereplace, int total)
{
    int count = 0, i;
    char *actstring = string;

    /* token to be replaced not in string */
    if (strstr(string, token) == NULL)
        return;

    /* get total number of tokens */
    while (*actstring) {
        actstring = nexttok(actstring);
        count++;
    }
    /* If total number of tokens correct */
    if (count == total) {
        actstring = string;
        for (i = 1; i < wherereplace; i++)
            actstring = nexttok(actstring);
        /* If token to be replaced at right position */
        if (ciprefix(token, actstring)) {
            actstring[0] = ' ';
            actstring[1] = ' ';
            actstring[2] = ' ';
            actstring[3] = ' ';
        }
    }
}


/* lines for B sources (except for pwl lines): no parsing in numparam code,
   just replacement of parameters. pwl lines are still handled in numparam.
   Parsing for all other B source lines are done in the B source parser.
   To achive this, do the following:
   Remove all '{' and '}' --> no parsing of equations in numparam
   Place '{' and '}' directly around all potential parameters,
   but skip function names like exp (search for 'exp(' to detect fcn name),
   functions containing nodes like v(node), v(node1, node2), i(branch)
   and other keywords like TEMPER. --> Only parameter replacement in numparam
*/
static void inp_bsource_compat(struct card *card)
{
    char *equal_ptr, *new_str, *final_str;
    int skip_control = 0;

    for (; card; card = card->nextcard) {

        char *curr_line = card->line;

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

        if (*curr_line == 'b') {
            /* remove white spaces of everything inside {}*/
            card->line = inp_remove_ws(card->line);
            curr_line = card->line;
            /* exclude special pwl lines */
            if (strstr(curr_line, "=pwl("))
                continue;
            /* store starting point for later parsing, beginning of
             * {expression} */
            equal_ptr = strchr(curr_line, '=');
            /* check for errors */
            if (equal_ptr == NULL) {
                fprintf(stderr, "ERROR: mal formed B line: %s\n", curr_line);
                controlled_exit(EXIT_FAILURE);
            }
            /* prepare to skip parsing in numparam with expressions */
            new_str = inp_modify_exp(equal_ptr + 1);
            final_str = tprintf("%.*s %s", (int) (equal_ptr + 1 - curr_line),
                    curr_line, new_str);

            // comment out current line (old B source line)
            *(card->line) = '*';
            // insert new B source line immediately after current line
            /* Copy old line numbers into new B source line */
            card = insert_new_line(
                    card, final_str, card->linenum, card->linenum_orig);

            tfree(new_str);
        } /* end of if 'b' */
    } /* end of for loop */
}


/* Find all expressions containing the keyword 'temper',
 * except for B lines and some other exclusions. Prepare
 * these expressions by calling inp_modify_exp() and return
 * a modified card->line
 */

static bool inp_temper_compat(struct card *card)
{
    int skip_control = 0;
    char *beg_str, *end_str, *beg_tstr, *end_tstr, *exp_str;

    bool with_temper = FALSE;
    for (; card; card = card->nextcard) {

        char *new_str = NULL;
        char *curr_line = card->line;

        if (curr_line == NULL)
            continue;
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
        /* exclude some elements */
        if (strchr("*vbiegfh", curr_line[0]))
            continue;
        /* exclude all dot commands except .model */
        if (curr_line[0] == '.' && !prefix(".model", curr_line))
            continue;
        /* exclude lines not containing 'temper' */
        if (!strstr(curr_line, "temper"))
            continue;
        /* now start processing of the remaining lines containing 'temper' */
        /* remove white spaces of everything inside {}*/
        card->line = inp_remove_ws(card->line);
        curr_line = card->line;

        beg_str = beg_tstr = curr_line;
        while ((beg_tstr = search_identifier(
                        beg_tstr, "temper", curr_line)) != NULL) {
            char *modified_exp;
            /* set the global variable */
            with_temper = TRUE;
            /* find the expression: first go back to the opening '{',
               then find the closing '}' */
            while ((*beg_tstr) != '{')
                beg_tstr--;
            end_str = end_tstr = beg_tstr;
            exp_str = gettok_char(&end_tstr, '}', TRUE, TRUE);
            /* modify the expression string */
            modified_exp = inp_modify_exp(exp_str);
            tfree(exp_str);
            /* add the intermediate string between previous and next
             * expression to the new line */
            new_str =
                    INPstrCat(new_str, ' ', copy_substring(beg_str, end_str));
            /* add the modified expression string to the new line */
            new_str = INPstrCat(new_str, ' ', modified_exp);
            new_str = INPstrCat(new_str, ' ', copy(" "));
            /* move on to the next intermediate string */
            beg_str = beg_tstr = end_tstr;
        }
        if (*beg_str)
            new_str = INPstrCat(new_str, ' ', copy(beg_str));
        tfree(card->line);
        card->line = inp_remove_ws(new_str);
    }
    return with_temper;
}


/* lines containing expressions with keyword 'temper':
 * no parsing in numparam code, just replacement of parameters.
 * Parsing done with B source parser in function inp_parse_temper
 * in inp.c. Evaluation is the done with fcn inp_evaluate_temper
 * from inp.c, taking the actual temperature into account.
 * To achive this, do the following here:
 * Remove all '{' and '}' --> no parsing of equations in numparam
 * Place '{' and '}' directly around all potential parameters,
 * but skip function names like exp (search for 'exp(' to detect fcn name),
 * functions containing nodes like v(node), v(node1, node2), i(branch)
 * and other keywords like TEMPER. --> Only parameter replacement in numparam
 */

static char *inp_modify_exp(/* NOT CONST */ char *expr)
{
    char *s;
    wordlist *wl = NULL, *wlist = NULL;

    /* Scan the expression and replace all '{' and '}' with ' '.
       As soon as we encounter a tc1=, tc2=, or m=, stop it. */
    for (s = expr; *s && !(ciprefix("tc1=", s) || ciprefix("tc2=", s) || ciprefix("m=", s)) ; s++) {
        if ((*s == '{') || (*s == '}')) {
            *s = ' ';
        }
    }

    /* scan the expression */
    s = expr;
    while (*(s = skip_ws(s))) {

        static bool c_arith_prev = FALSE;
        bool c_arith = FALSE;
        char c_prev = '\0';
        char c = *s;

        wl_append_word(&wlist, &wl, NULL);

        if ((c == ',') || (c == '(') || (c == ')') || (c == '*') ||
                (c == '/') || (c == '^') || (c == '+') || (c == '?') ||
                (c == ':') || (c == '-')) {
            if ((c == '*') && (s[1] == '*')) {
                wl->wl_word = tprintf("**");
                s += 2;
            }
            else if (c == '-' && c_arith_prev && c_prev != ')') {
                /* enter whole number string if '-' is a sign */
                int error1;
                /* allow 100p, 5MEG etc. */
                double dvalue = INPevaluate(&s, &error1, 0);
                if (error1) {
                    wl->wl_word = tprintf("%c", c);
                    s++;
                }
                else {
                    wl->wl_word = tprintf("%18.10e", dvalue);
                    /* skip the `unit', FIXME INPevaluate() should do this */
                    while (isalpha_c(*s))
                        s++;
                }
            }
            else {
                wl->wl_word = tprintf("%c", c);
                s++;
            }
            c_arith = TRUE;
        }
        else if ((c == '>') || (c == '<') || (c == '!') || (c == '=')) {
            /* >=, <=, !=, ==, <>, ... */
            char *beg = s++;
            if ((*s == '=') || (*s == '<') || (*s == '>')) {
                s++;
            }
            wl->wl_word = copy_substring(beg, s);
        }
        else if ((c == '|') || (c == '&')) {
            char *beg = s++;
            if ((*s == '|') || (*s == '&'))
                s++;
            wl->wl_word = copy_substring(beg, s);
        }
        else if (isalpha_c(c) || c == '_') {

            char buf[512];
            int i = 0;

            if (((c == 'v') || (c == 'i')) && (s[1] == '(')) {
                while (*s != ')') {
                    buf[i++] = *s++;
                }
                buf[i++] = *s++;
                buf[i] = '\0';
                wl->wl_word = copy(buf);
            }
            else {
                while (isalnum_c(*s) || (*s == '!') || (*s == '#') ||
                        (*s == '$') || (*s == '%') || (*s == '_') ||
                        (*s == '[') || (*s == ']')) {
                    buf[i++] = *s++;
                }
                buf[i] = '\0';
                /* no parens {} around time, hertz, temper, the constants
                   pi and e which are defined in inpptree.c, around pwl and
                   temp. coeffs */
                if ((*s == '(') || cieq(buf, "hertz") ||
                        cieq(buf, "temper") || cieq(buf, "time") ||
                        cieq(buf, "pi") || cieq(buf, "e") ||
                        cieq(buf, "pwl")) {
                    wl->wl_word = copy(buf);
                }
                /* no parens {} around instance parameters temp and dtemp (on left hand side) */
                else if ((*s == '=') &&
                    (cieq(buf, "dtemp") || cieq(buf, "temp"))) {
                    wl->wl_word = copy(buf);
                }
                /* as soon as we encounter tc1= or tc2= (temp coeffs.) or
                   m= (multiplier), the expression is done */
                else if ((*s == '=') && (cieq(buf, "tc1") || cieq(buf, "tc2") ||
                        cieq(buf, "reciproctc") || cieq(buf, "m") || cieq(buf, "reciprocm"))) {
                    wl->wl_word = tprintf("%s%s", buf, s);
                    break;
                }
                else {
                    /* {} around all other tokens */
                    wl->wl_word = tprintf("({%s})", buf);
                }
            }
        }
        else if (isdigit_c(c) || (c == '.')) { /* allow .5 format too */
            int error1;
            /* allow 100p, 5MEG etc. */
            double dvalue = INPevaluate(&s, &error1, 0);
            wl->wl_word = tprintf("%18.10e", dvalue);
            /* skip the `unit', FIXME INPevaluate() should do this */
            while (isalpha_c(*s)) {
                s++;
            }
        }
        else { /* strange char */
            printf("Preparing expression for numparam\nWhat is this?\n%s\n",
                    s);
            wl->wl_word = tprintf("%c", *s++);
        }
        c_prev = c;
        c_arith_prev = c_arith;
    }

    expr = wl_flatten(wlist);
    wl_free(wlist);

    return expr;
}


/*
 * destructively fetch a token from the input string
 *   token is either quoted, or a plain nonwhitespace sequence
 * function will return the place from where to continue
 */

static char *get_quoted_token(char *string, char **token)
{
    char *s = skip_ws(string);

    if (!*s) /* nothing found */
        return string;

    if (isquote(*s)) {
        /* we may find single ' or double " quotes */
        char thisquote = *s;
        char *t = ++s;

        while (*t && !(*t == thisquote))
            t++;

        if (!*t) { /* teriminator quote not found */
            *token = NULL;
            return string;
        }

        *t++ = '\0';

        *token = s;
        return t;
    }
    else {

        char *t = skip_non_ws(s);

        if (t == s) { /* nothing found */
            *token = NULL;
            return string;
        }

        if (*t)
            *t++ = '\0';

        *token = s;
        return t;
    }
}


/* Option RSERIES=rval
 * Lxxx n1 n2 Lval
 * -->
 * Lxxx n1 n2_intern__ Lval
 * RLxxx_n2_intern__ n2_intern__ n2 rval
 */

static void inp_add_series_resistor(struct card *deck)
{
    int skip_control = 0;
    struct card *card;
    char *rval = NULL;

    for (card = deck; card; card = card->nextcard) {
        char *curr_line = card->line;
        if (*curr_line != '*' && strstr(curr_line, "option")) {
            char *t = strstr(curr_line, "rseries");
            if (t) {
                tfree(rval);

                t += 7;
                if (*t++ == '=')
                    rval = gettok(&t);

                /* default to "1e-3" if no value given */
                if (!rval)
                    rval = copy("1e-3");
            }
        }
    }

    if (!rval)
        return;

    fprintf(stdout,
            "\nOption rseries given: \n"
            "resistor %s Ohms added in series to each inductor L\n\n",
            rval);

    for (card = deck; card; card = card->nextcard) {
        char *cut_line = card->line;

        /* exclude any command inside .control ... .endc */
        if (ciprefix(".control", cut_line)) {
            skip_control++;
            continue;
        }
        else if (ciprefix(".endc", cut_line)) {
            skip_control--;
            continue;
        }
        else if (skip_control > 0) {
            continue;
        }

        if (ciprefix("l", cut_line)) {

            int currlinenumber = card->linenum_orig;
            char *title_tok = gettok(&cut_line);
            char *node1 = gettok(&cut_line);
            char *node2 = gettok(&cut_line);

            /* new L line and new R line */
            char *newL = tprintf("%s %s %s_intern__ %s", title_tok, node1,
                    title_tok, cut_line);
            char *newR = tprintf("R%s_intern__ %s_intern__ %s %s", title_tok,
                    title_tok, node2, rval);

            // comment out current L line
            *(card->line) = '*';

            // insert new new L and R lines immediately after current line
            card = insert_new_line(card, newL, 1, currlinenumber);
            card = insert_new_line(card, newR, 2, currlinenumber);

            tfree(title_tok);
            tfree(node1);
            tfree(node2);
        }
    }

    tfree(rval);
}


/*
 * rewrite
 *   .subckt node1 node2 node3 name params: l={x} w={y}
 * to
 *   .subckt node1 node2 node3 name
 *   .param l={x} w={y}
 */

static void subckt_params_to_param(struct card *card)
{
    for (; card; card = card->nextcard) {
        char *curr_line = card->line;
        if (ciprefix(".subckt", curr_line)) {
            char *cut_line, *new_line;
            cut_line = strstr(curr_line, "params:");
            if (!cut_line)
                continue;
            /* new_line starts with "params: " */
            new_line = copy(cut_line);
            /* replace "params:" by ".param " */
            memcpy(new_line, ".param ", 7);
            /* card->line ends with subcircuit name */
            cut_line[-1] = '\0';
            /* insert new_line after card->line */
            insert_new_line(card, new_line, card->linenum + 1, card->linenum_orig);
        }
    }
}


/* If XSPICE option is not selected, run this function to alert and exit
   if the 'poly' option is found in e, g, f, or h controlled sources. */

#ifndef XSPICE

static void inp_poly_err(struct card *card)
{
    size_t skip_control = 0;

    for (; card; card = card->nextcard) {

        char *curr_line = card->line;

        if (*curr_line == '*')
            continue;

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

        /* get the fourth token in a controlled source line and exit,
           if it is 'poly' */
        if ((ciprefix("e", curr_line)) || (ciprefix("g", curr_line)) ||
                (ciprefix("f", curr_line)) || (ciprefix("h", curr_line))) {
            curr_line = nexttok(curr_line);
            curr_line = nexttok(curr_line);
            curr_line = nexttok(curr_line);
            if (ciprefix("poly", curr_line)) {
                fprintf(stderr,
                        "\nError: XSPICE is required to run the 'poly' "
                        "option in line %d\n",
                        card->linenum_orig);
                fprintf(stderr, "  %s\n", card->line);
                fprintf(stderr,
                        "\nSee manual chapt. 31 for installation "
                        "instructions\n");
                controlled_exit(EXIT_BAD);
            }
        }
    }
}

#endif

/* Print the parsed library to lib_out?.lib, with ? a growing number
   if multiple libs are saved in a single run. Don't save the .libsave line.*/
static char* libprint(struct card* t, const char *dir_name)
{
    struct card* tmp;
    static int npr = 1;
    char *outfile = tprintf("%s/lib_out%d.lib", dir_name, npr);
    npr++;
    FILE* fd = fopen(outfile, "w");
    if (fd) {
        for (tmp = t; tmp; tmp = tmp->nextcard)
            if (*(tmp->line) != '*' && !ciprefix(".libsave", tmp->line))
                fprintf(fd, "%s\n", tmp->line);
        fclose(fd);
    }
    else {
        fprintf(stderr, "Warning: Can't open file %s \n    command .libsave ignored!\n", outfile);
    }
    return outfile;
}


/* Used for debugging. You may add
 *   tprint(working);
 * somewhere in function inp_readall() of this file to have
 *   a printout of the actual deck written to file "tprint-out.txt" */
void tprint(struct card *t)
{
    struct card *tmp;
    static int npr;
    char outfile[100];
    sprintf(outfile, "tprint-out%d.txt", npr);
    npr++;
    /*debug: print into file*/
    FILE *fd = fopen(outfile, "w");
    for (tmp = t; tmp; tmp = tmp->nextcard)
        if (*(tmp->line) != '*')
            fprintf(fd, "%6d  %6d  %s\n", tmp->linenum_orig, tmp->linenum,
                    tmp->line);
    fprintf(fd,
            "\n**************************************************************"
            "*******************\n");
    fprintf(fd,
            "****************************************************************"
            "*****************\n");
    fprintf(fd,
            "****************************************************************"
            "*****************\n\n");
    for (tmp = t; tmp; tmp = tmp->nextcard)
        fprintf(fd, "%6d  %6d  %s\n", tmp->linenum_orig, tmp->linenum,
                tmp->line);
    fprintf(fd,
            "\n**************************************************************"
            "*******************\n");
    fprintf(fd,
            "****************************************************************"
            "*****************\n");
    fprintf(fd,
            "****************************************************************"
            "*****************\n\n");
    for (tmp = t; tmp; tmp = tmp->nextcard)
        if (*(tmp->line) != '*')
            fprintf(fd, "%s\n", tmp->line);
    fclose(fd);
}


/* prepare .if and .elseif for numparam
   .if(expression) --> .if{expression} */

static void inp_dot_if(struct card *card)
{
    for (; card; card = card->nextcard) {

        char *curr_line = card->line;

        if (*curr_line == '*')
            continue;

        if (ciprefix(".if", curr_line) || ciprefix(".elseif", curr_line)) {
            char *firstbr = strchr(curr_line, '(');
            char *lastbr = strrchr(curr_line, ')');
            if ((!firstbr) || (!lastbr)) {
                fprintf(cp_err, "Error in netlist line no. %d\n",
                        card->linenum_orig);
                fprintf(cp_err, "   Bad syntax: %s\n\n", curr_line);
                controlled_exit(EXIT_BAD);
            }
            *firstbr = '{';
            *lastbr = '}';
        }
    }
}


/* Convert .param lines containing keyword 'temper' into .func lines:
 * .param xxx1 = 'temper + 25'  --->  .func xxx1() 'temper + 25'
 * Add info about the functions (name, subcircuit depth, number of
 * subckt) to linked list new_func.
 * Then scan new_func, for each xxx1 scan all lines of deck,
 * find all xxx1 and convert them to a function:
 * xxx1   --->  xxx1()
 * If this happens to be in another .param line, convert it to .func,
 * add info to end of new_func and continue scanning.
 */

static char *inp_functionalise_identifier(char *curr_line, char *identifier);

static void inp_fix_temper_in_param(struct card *deck)
{
    int skip_control = 0, subckt_depth = 0, j, *sub_count;
    char *funcbody, *funcname;
    struct func_temper *f, *funcs = NULL, **funcs_tail_ptr = &funcs;
    struct card *card;

    sub_count = TMALLOC(int, 16);
    for (j = 0; j < 16; j++)
        sub_count[j] = 0;

    /* first pass: determine all .param with temper inside and replace by
     * .func
     * .param xxx1 = 'temper + 25'
     * will become
     * .func xxx1() 'temper + 25'
     */
    card = deck;
    for (; card; card = card->nextcard) {

        char *curr_line = card->line;

        if (*curr_line == '*')
            continue;

        /* determine nested depths of subcircuits */
        if (ciprefix(".subckt", curr_line)) {
            subckt_depth++;
            sub_count[subckt_depth]++;
            continue;
        }
        else if (ciprefix(".ends", curr_line)) {
            subckt_depth--;
            continue;
        }

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

        if (ciprefix(".para", curr_line)) {

            char *p, *temper, *equal_ptr, *lhs_b, *lhs_e;

            temper = search_identifier(curr_line, "temper", curr_line);

            if (!temper)
                continue;

            equal_ptr = find_assignment(curr_line);

            if (!equal_ptr) {
                fprintf(stderr,
                        "ERROR: could not find '=' on parameter line '%s'!\n",
                        curr_line);
                controlled_exit(EXIT_FAILURE);
            }

            /* .param lines with `,' separated multiple parameters
             *    must have been split in inp_split_multi_param_lines()
             */

            if (find_assignment(equal_ptr + 1)) {
                fprintf(stderr, "ERROR: internal error on line '%s'!\n",
                        curr_line);
                controlled_exit(EXIT_FAILURE);
            }

            lhs_b = skip_non_ws(curr_line); // eat .param
            lhs_b = skip_ws(lhs_b);

            lhs_e = skip_back_ws(equal_ptr, curr_line);

            /* skip if this is a function already */
            p = strpbrk(lhs_b, "(,)");
            if (p && p < lhs_e)
                continue;

            if (temper < equal_ptr) {
                fprintf(stderr,
                        "Error: you cannot assign a value to TEMPER\n"
                        "  Line no. %d, %s\n",
                        card->linenum, curr_line);
                controlled_exit(EXIT_BAD);
            }

            funcname = copy_substring(lhs_b, lhs_e);
            funcbody = copy(equal_ptr + 1);

            *funcs_tail_ptr = inp_new_func(
                    funcname, funcbody, card, sub_count, subckt_depth);
            funcs_tail_ptr = &(*funcs_tail_ptr)->next;

            tfree(funcbody);
        }
    }

    /* second pass */
    /* for each .func entry in `funcs' start the insertion operation:
       search each line from the deck which has the suitable subcircuit
       nesting data. for tokens xxx equalling the funcname, replace xxx by
       xxx(). if the replacement is done in a .param line then convert it to a
       .func line and append an entry to `funcs'. Continue up to the very end
       of `funcs'.
     */

    for (f = funcs; f; f = f->next) {

        for (j = 0; j < 16; j++)
            sub_count[j] = 0;

        card = deck;
        for (; card; card = card->nextcard) {

            char *new_str = NULL; /* string we assemble here */
            char *curr_line = card->line;
            char *firsttok_str;

            if (*curr_line == '*')
                continue;

            /* determine nested depths of subcircuits */
            if (ciprefix(".subckt", curr_line)) {
                subckt_depth++;
                sub_count[subckt_depth]++;
                continue;
            }
            else if (ciprefix(".ends", curr_line)) {
                subckt_depth--;
                continue;
            }

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

            /* exclude lines which do not have the same subcircuit
               nesting depth and number as found in f */
            if (subckt_depth != f->subckt_depth)
                continue;
            if (sub_count[subckt_depth] != f->subckt_count)
                continue;

            /* remove first token, ignore it here, restore it later */
            firsttok_str = gettok(&curr_line);
            if (*curr_line == '\0') {
                tfree(firsttok_str);
                continue;
            }

            new_str = inp_functionalise_identifier(curr_line, f->funcname);

            if (new_str == curr_line) {
                tfree(firsttok_str);
                continue;
            }

            /* restore first part of the line */
            new_str = INPstrCat(firsttok_str, ' ', new_str);
            new_str = inp_remove_ws(new_str);

            /* if we have inserted into a .param line, convert to .func */
            if (prefix(".para", new_str)) {
                char *new_tmp_str = new_str;
                new_tmp_str = nexttok(new_tmp_str);
                funcname = gettok_char(&new_tmp_str, '=', FALSE, FALSE);
                funcbody = copy(new_tmp_str + 1);
                *funcs_tail_ptr = inp_new_func(
                        funcname, funcbody, card, sub_count, subckt_depth);
                funcs_tail_ptr = &(*funcs_tail_ptr)->next;
                tfree(new_str);
                tfree(funcbody);
            }
            else {
                /* Or just enter new line into deck */
                insert_new_line(card, new_str, 0, card->linenum);
                *card->line = '*';
            }
        }
    }

    /* final memory clearance */
    tfree(sub_count);
    inp_delete_funcs(funcs);
}


/* Convert .param lines containing function 'agauss' and others
 *  (function name handed over by *fcn),  into .func lines:
 * .param xxx1 = 'aunif()'  --->  .func xxx1() 'aunif()'
 * Add info about the functions (name, subcircuit depth, number of
 * subckt) to linked list new_func.
 * Then scan new_func, for each xxx1 scan all lines of deck,
 * find all xxx1 and convert them to a function:
 * xxx1   --->  xxx1()
 *
 * In a second step, after subcircuits have been expanded, all occurencies
 * of agauss in a b-line are replaced by their suitable value (function
 * eval_agauss() in inp.c).
 */

static void inp_fix_agauss_in_param(struct card *deck, char *fcn)
{
    int skip_control = 0, subckt_depth = 0, j, *sub_count;
    char *funcbody, *funcname;
    struct func_temper *f, *funcs = NULL, **funcs_tail_ptr = &funcs;
    struct card *card;

    sub_count = TMALLOC(int, 16);
    for (j = 0; j < 16; j++)
        sub_count[j] = 0;

    /* first pass:
     *   determine all .param with agauss inside and replace by .func
     *   convert
     *     .param xxx1 = 'agauss(x,y,z) * 25'
     *   to
     *     .func xxx1() 'agauss(x,y,z) * 25'
     */
    card = deck;
    for (; card; card = card->nextcard) {

        char *curr_line = card->line;

        if (*curr_line == '*')
            continue;

        /* determine nested depths of subcircuits */
        if (ciprefix(".subckt", curr_line)) {
            subckt_depth++;
            sub_count[subckt_depth]++;
            continue;
        }
        else if (ciprefix(".ends", curr_line)) {
            subckt_depth--;
            continue;
        }

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

        if (ciprefix(".para", curr_line)) {

            char *p, *temper, *equal_ptr, *lhs_b, *lhs_e;

            temper = search_identifier(curr_line, fcn, curr_line);

            if (!temper)
                continue;

            equal_ptr = find_assignment(curr_line);

            if (!equal_ptr) {
                fprintf(stderr,
                        "ERROR: could not find '=' on parameter line '%s'!\n",
                        curr_line);
                controlled_exit(EXIT_FAILURE);
            }

            /* .param lines with `,' separated multiple parameters
             *   must have been split in inp_split_multi_param_lines()
             */

            if (find_assignment(equal_ptr + 1)) {
                fprintf(stderr, "ERROR: internal error on line '%s'!\n",
                        curr_line);
                controlled_exit(EXIT_FAILURE);
            }

            lhs_b = skip_non_ws(curr_line); // eat .param
            lhs_b = skip_ws(lhs_b);

            lhs_e = skip_back_ws(equal_ptr, curr_line);

            /* skip if this is a function already */
            p = strpbrk(lhs_b, "(,)");
            if (p && p < lhs_e)
                continue;

            if (temper < equal_ptr) {
                fprintf(stderr,
                        "Error: you cannot assign a value to %s\n"
                        "  Line no. %d, %s\n",
                        fcn, card->linenum, curr_line);
                controlled_exit(EXIT_BAD);
            }

            funcname = copy_substring(lhs_b, lhs_e);
            funcbody = copy(equal_ptr + 1);

            *funcs_tail_ptr = inp_new_func(
                    funcname, funcbody, card, sub_count, subckt_depth);
            funcs_tail_ptr = &(*funcs_tail_ptr)->next;

            tfree(funcbody);
        }
    }

    /* second pass:
     *   for each .func entry in `funcs' start the insertion operation:
     *      search each line from the deck which has the suitable
     *      subcircuit nesting data.
     *   for tokens xxx equalling the funcname, replace xxx by xxx().
     */

    for (f = funcs; f; f = f->next) {

        for (j = 0; j < 16; j++)
            sub_count[j] = 0;

        card = deck;
        for (; card; card = card->nextcard) {

            char *new_str = NULL; /* string we assemble here */
            char *curr_line = card->line;
            char *firsttok_str;

            if (*curr_line == '*')
                continue;

            /* determine nested depths of subcircuits */
            if (ciprefix(".subckt", curr_line)) {
                subckt_depth++;
                sub_count[subckt_depth]++;
                continue;
            }
            else if (ciprefix(".ends", curr_line)) {
                subckt_depth--;
                continue;
            }

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

            /* if function is not at top level,
               exclude lines which do not have the same subcircuit
               nesting depth and number as found in f */
            if (f->subckt_depth > 0) {
                if (subckt_depth != f->subckt_depth)
                    continue;
                if (sub_count[subckt_depth] != f->subckt_count)
                    continue;
            }

            /* remove first token, ignore it here, restore it later */
            firsttok_str = gettok(&curr_line);
            if (*curr_line == '\0') {
                tfree(firsttok_str);
                continue;
            }

            new_str = inp_functionalise_identifier(curr_line, f->funcname);

            if (new_str == curr_line) {
                tfree(firsttok_str);
                continue;
            }

            /* restore first part of the line */
            new_str = INPstrCat(firsttok_str, ' ', new_str);
            new_str = inp_remove_ws(new_str);

            *card->line = '*';
            /* Enter new line into deck */
            insert_new_line(card, new_str, 0, card->linenum);
        }
    }
    /* final memory clearance */
    tfree(sub_count);
    inp_delete_funcs(funcs);
}


/* append "()" to each 'identifier' in 'curr_line',
 *   unless already there */
static char *inp_functionalise_identifier(char *curr_line, char *identifier)
{
    size_t len = strlen(identifier);
    char *p, *str = curr_line;

    /* Start replacing identifier by func only after the first '=' or '{' */
    char* estr1 = strchr(curr_line, '=');
    char* estr2 = strchr(curr_line, '{');
    char* estr;

    if (!estr1 && !estr2)
        return str;

    if (estr1 && estr2)
        estr = (estr1 < estr2) ? estr1 : estr2;
    else if (estr1)
        estr = estr1;
    else
        estr = estr2;

    for (p = estr; (p = search_identifier(p, identifier, str)) != NULL;)
        if (p[len] != '(') {
            int prefix_len = (int) (p + len - str);
            char *x = str;
            str = tprintf("%.*s()%s", prefix_len, str, str + prefix_len);
            if (x != curr_line)
                tfree(x);
            p = str + prefix_len + 2;
        }
        else {
            p++;
        }

    return str;
}


/* enter function name, nested .subckt depths, and
 * number of .subckt at given level into struct new_func
 * and add line to deck
 */

static struct func_temper *inp_new_func(char *funcname, char *funcbody,
        struct card *card, int *sub_count, int subckt_depth)
{
    struct func_temper *f;
    char *new_str;

    f = TMALLOC(struct func_temper, 1);
    f->funcname = funcname;
    f->next = NULL;
    f->subckt_depth = subckt_depth;
    f->subckt_count = sub_count[subckt_depth];

    /* replace line in deck */
    new_str = tprintf(".func %s() %s", funcname, funcbody);

    *card->line = '*';
    insert_new_line(card, new_str, 0, card->linenum);

    return f;
}


static void inp_delete_funcs(struct func_temper *f)
{
    while (f) {
        struct func_temper *f_next = f->next;
        tfree(f->funcname);
        tfree(f);
        f = f_next;
    }
}


/* look for unquoted parameters and quote them */
/* FIXME, this function seems to be useless and/or buggy and/or naive */
static void inp_quote_params(struct card *c, struct card *end_c,
        struct dependency *deps, int num_params)
{
    bool in_control = FALSE;

    for (; c && c != end_c; c = c->nextcard) {

        int i, j, num_terminals;

        char *curr_line = c->line;

        if (ciprefix(".control", curr_line)) {
            in_control = TRUE;
            continue;
        }

        if (ciprefix(".endc", curr_line)) {
            in_control = FALSE;
            continue;
        }

        if (in_control || curr_line[0] == '.' || curr_line[0] == '*')
            continue;

        num_terminals = get_number_terminals(curr_line);

        if (num_terminals <= 0)
            continue;

        /* There are devices that should not get quotes around tokens
           following after the terminals. These may be model names or control
           voltages. See bug 384  or Skywater issue 327 */
        if (strchr("fhmouydqjzsw", *curr_line))
            num_terminals++;

        for (i = 0; i < num_params; i++) {

            char *s = curr_line;

            for (j = 0; j < num_terminals + 1; j++) {
                s = skip_non_ws(s);
                s = skip_ws(s);
            }

            while ((s = ya_search_identifier(
                            s, deps[i].param_name, curr_line)) != NULL) {

                char *rest = s + strlen(deps[i].param_name);

                if (s > curr_line && (isspace_c(s[-1]) || s[-1] == '=' || s[-1] == ',') &&
                        (isspace_c(*rest) || *rest == '\0'  || *rest == ',' || *rest == ')')) {
                    int prefix_len;

                    if (isspace_c(s[-1])) {
                        s = skip_back_ws(s, curr_line);
                        if (s > curr_line && s[-1] == '{')
                            s--;
                    }

                    if (isspace_c(*rest)) {
                        /* possible case: "{  length }" -> {length} */
                        rest = skip_ws(rest);
                        if (*rest == '}')
                            rest++;
                        else
                            rest--;
                    }

                    prefix_len = (int) (s - curr_line);

                    curr_line = tprintf("%.*s {%s}%s", prefix_len, curr_line,
                            deps[i].param_name, rest);
                    s = curr_line + prefix_len + strlen(deps[i].param_name) +
                            3;

                    tfree(c->line);
                    c->line = curr_line;
                }
                else {
                    s += strlen(deps[i].param_name);
                }
            }
        }
        /* Now check if we have nested {..{  }...}, which is not accepted by numparam code.
           Replace the inner { } by ( ). Do this only when this is not a behavioral device
           which will become a B source. B source handling is special in inp.c. */
        char* cut_line = c->line;
        if (!b_transformation_wanted(cut_line)) {
            cut_line = strchr(cut_line, '{');
            if (cut_line) {
                int level = 1;
                cut_line++;
                while (*cut_line != '\0') {
                    if (*cut_line == '{') {
                        level++;
                        if (level > 1)
                            *cut_line = '(';
                    }
                    else if (*cut_line == '}') {
                        if (level > 1)
                            *cut_line = ')';
                        level--;
                    }
                    cut_line++;
                }
            }
        }
    }
}


/* VDMOS special:
   Check for 'vdmos' in .model line.
   check if 'pchan', then add p to vdmos and ignore 'pchan'.
   If no 'pchan' is found, add n to vdmos.
   Ignore annotations on Vds, Ron, Qg, and mfg.
   Assemble all other tokens in a wordlist, and flatten it
   to become the new .model line.
*/
static int inp_vdmos_model(struct card *deck)
{
#define MODNUMBERS 2048

    struct card *card;
    struct card *vmodels[MODNUMBERS]; /* list of pointers to vdmos model cards */
    int j = 0;
    vmodels[0] = NULL;

    for (card = deck; card; card = card->nextcard) {

        char* curr_line, * cut_line, * token, * new_line;
        wordlist* wl = NULL, * wlb;

        curr_line = cut_line = card->line;

        if (ciprefix(".model", curr_line) && strstr(curr_line, "vdmos")) {
            cut_line = strstr(curr_line, "vdmos");
            wl_append_word(&wl, &wl, copy_substring(curr_line, cut_line));
            wlb = wl;
            if (strstr(cut_line, "pchan")) {
                wl_append_word(NULL, &wl, copy("vdmosp ("));
            }
            else {
                wl_append_word(NULL, &wl, copy("vdmosn ("));
            }
            cut_line = cut_line + 5;

            cut_line = skip_ws(cut_line);
            if (*cut_line == '(')
                cut_line = cut_line + 1;
            new_line = NULL;
            while (cut_line && *cut_line) {
                token = gettok_model(&cut_line);
                if (!ciprefix("pchan", token) && !ciprefix("ron=", token) &&
                    !ciprefix("vds=", token) && !ciprefix("qg=", token) &&
                    !ciprefix("mfg=", token) && !ciprefix("nchan", token))
                    wl_append_word(NULL, &wl, token);
                else
                    tfree(token);
                if (*cut_line == ')') {
                    wl_append_word(NULL, &wl, copy(")"));
                    break;
                }
            }
            new_line = wl_flatten(wlb);
            tfree(card->line);
            card->line = new_line;
            wl_free(wlb);

            /* add model card pointer to list */
            vmodels[j] = card;
            j++;
            if (j == MODNUMBERS) {
                vmodels[j - 1] = NULL;
                break;
            }
            vmodels[j] = NULL;
        }
    }

    /* we don't have vdmos models, so return */
    if (vmodels[0] == NULL)
        return 0;
    if (j == MODNUMBERS)
        fprintf(cp_err, "Warning: Syntax check for VDMOS instances is limited to %d .model cards\n", MODNUMBERS);

    for (card = deck; card; card = card->nextcard) {
        /* we have a VDMOS instance line with 'thermal' flag and thus need exactly 5 nodes
         */
        int i;
        char *curr_line = card->line;
        if (curr_line[0] == 'm' && strstr(curr_line, "thermal")) {
            /* move to model name */
            for (i = 0; i < 6; i++)
                curr_line = nexttok(curr_line);
            if (!curr_line || !*curr_line) {
                fprintf(cp_err,
                    "Error: We need exactly 5 nodes\n"
                    "    drain, gate, source, tjunction, tcase\n"
                    "    in VDMOS instance line with thermal model\n"
                    "    %s\n", card->line);
                fprintf(stderr, "No circuit loaded!\n");
                return 1;
            }
            /* next token is the model name of instance */
            char* instmodname = gettok(&curr_line);
            i = 0;
            while (vmodels[i]) {
                char* mod = vmodels[i]->line;
                mod = nexttok(mod); /* skip .model */
                if (ciprefix(instmodname, mod)) {
                    tfree(instmodname);
                    return 0;
                }
                i++;
            }
            fprintf(cp_err,
                "Error: We need exactly 5 nodes\n"
                "    drain, gate, source, tjunction, tcase\n"
                "    in VDMOS instance line with thermal model\n"
                "    %s\n", card->line);
            fprintf(stderr, "No circuit loaded!\n");
            tfree(instmodname);
            return 1;
        }
    }
    return 0;
}


/* storage for devices which get voltage source added */
struct replace_currm
{
    struct card *s_start;
    struct card *cline;
    char *rtoken;
    struct replace_currm *next;
};

/* check if fourth token of sname starts with POLY */
static bool is_poly_source(char *sname)
{
    char *nstr = nexttok(sname);
    nstr = nexttok(nstr);
    nstr = nexttok(nstr);
    if (ciprefix("POLY", nstr))
        return TRUE;
    else
        return FALSE;
}

/* Measure current in node 1 of all devices, e.g. I, B, F, G.
   I(V...) will be ignored, I(E...) and I(H...) will be undone if
   they are simple linear sources, however E nonlinear voltage
   source will be converted later to B source,
   therefore we need to add current measurement here.
   First find all ocurrencies of i(XYZ), store their cards, then
   search for XYZ, but only within respective subcircuit, or if
   all happens at top level. Other hierarchy is ignored for now.
   Replace I(XYZ) bx I(V_XYZ), add voltage source V_XYZ with
   suitable extra nodes.
*/
static void inp_meas_current(struct card *deck)
{
    struct card *card, *subc_start = NULL, *subc_prev = NULL;
    struct replace_currm *new_rep, *act_rep = NULL, *rep = NULL;
    char *s, *t, *u, *v, *w;
    int skip_control = 0, subs = 0, sn = 0;

    /* scan through deck and find i(xyz), replace by i(v_xyz) */
    for (card = deck; card; card = card->nextcard) {

        char *curr_line = card->line;

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

        if (*curr_line == '*')
            continue;

        if (*curr_line == '.') {
            if (ciprefix(".subckt", curr_line)) {
                subs++;
                subc_prev = subc_start;
                subc_start = card;
            }
            else if (ciprefix(".ends", curr_line)) {
                subs--;
                subc_start = subc_prev;
            }
            else
                continue;
        }

        if (!strstr(curr_line, "i("))
            continue;

        s = v = w = stripWhiteSpacesInsideParens(curr_line);
        while (s) {
            /* i( may occur more than once in a line */
            s = u = strstr(s, "i(");
            /* we have found it, but not (in error) at the beginning of the
             * line */
            if (s && s > v) {
                /* %i( may be part of the node definition in a XSPICE instance, so skip it here */
                if (*v == 'a' && s[-1] == '%') {
                    s++;
                    continue;
                }
                 /* '{' if at beginning of expression, '=' possible in B-line */
                else if (is_arith_char(s[-1]) || s[-1] == '{' || s[-1] == '=' ||
                        isspace_c(s[-1])) {
                    s += 2;
                    if (*s == 'v') {
                        // printf("i(v...) found in\n%s\n not converted!\n\n",
                        // curr_line);
                        continue;
                    }
                    else {
                        char *beg_str, *new_str;
                        get_r_paren(&u);
                        /* token containing name of devices to be measured */
                        t = copy_substring(s, --u);
                        if (ft_ngdebug)
                            printf("i(%s) found in\n%s\n\n", t, v);

                        /* new entry to the end of struct rep */
                        new_rep = TMALLOC(struct replace_currm, 1);
                        new_rep->s_start = subc_start;
                        new_rep->next = NULL;
                        new_rep->cline = card;
                        new_rep->rtoken = t;
                        if (act_rep) {
                            act_rep->next = new_rep;
                            act_rep = act_rep->next;
                        }
                        else
                            rep = act_rep = new_rep;
                        /* change line, convert i(XXX) to i(v_XXX) */
                        beg_str = copy_substring(v, s);
                        new_str = tprintf("%s%s%s", beg_str, "v_", s);
                        if (ft_ngdebug)
                            printf("converted to\n%s\n\n", new_str);
                        tfree(card->line);
                        card->line = s = v = new_str;
                        s++;
                        tfree(beg_str);
                    }
                }
                else
                    s++;
            }
        }
        tfree(w);
    }

    /* return if we did not find any i( */
    if (rep == NULL) {
        return;
    }

    /* scan through all the devices, search for xyz, modify node 1 by adding
       _vmeas, add a line with zero voltage v_xyz, having original node 1 and
       modified node 1. Do this within the top level or the same level of
       subcircuit only. */
    new_rep = rep;
    for (; rep; rep = rep->next) {
        card = rep->s_start;
        subs = 0;
        if (card)
            card = card->nextcard;
        else
            card = deck;
        for (; card; card = card->nextcard) {
            char *tok, *new_tok, *node1, *new_line;
            char *curr_line = card->line;
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

            if (*curr_line == '*')
                continue;

            if (*curr_line == '\0')
                continue;

            if (*curr_line == '.') {
                if (ciprefix(".subckt", curr_line))
                    subs++;
                else if (ciprefix(".ends", curr_line))
                    subs--;
                else
                    continue;
            }
            if (subs > 0)
                continue;
            /* We are at now top level or in top level of subcircuit
               where i(xyz) has been found */
            tok = gettok(&curr_line);
            /* done when end of subcircuit is reached */
            if (eq(".ends", tok) && rep->s_start) {
                tfree(tok);
                break;
            }
            if (eq(rep->rtoken, tok)) {
                /* special treatment if we have an e (VCVS) or h (CCVS)
                source: check if it is a simple linear source, if yes, don't
                do a replacement, instead undo the already done name
                conversion */
                if (((tok[0] == 'e') || (tok[0] == 'h')) &&
                        !strchr(curr_line, '=') &&
                        !is_poly_source(card->line)) {
                    /* simple linear e source */
                    char *searchstr = tprintf("i(v_%s)", tok);
                    char *thisline = rep->cline->line;
                    char *findstr = strstr(thisline, searchstr);
                    while (findstr) {
                        if (prefix(searchstr, findstr))
                            memcpy(findstr, "  i(", 4);
                        findstr = strstr(thisline, searchstr);
                        if (ft_ngdebug)
                            printf("i(%s) moved back to i(%s) in\n%s\n\n",
                                    searchstr, tok, rep->cline->line);
                    }
                    tfree(searchstr);
                    tfree(tok);
                    continue;
                }
                node1 = gettok(&curr_line);
                /* Add _vmeas only once to first device node.
                   Continue if we already have modified device "tok" */
                if (!strstr(node1, "_vmeas")) {
                    new_line = tprintf("%s %s_vmeas_%d %s",
                            tok, node1, sn, curr_line);
                    tfree(card->line);
                    card->line = new_line;
                }

                new_tok = tprintf("v_%s", tok);
                /* We have already added a line v_xyz to the deck */
                if (!ciprefix(new_tok, card->nextcard->line)) {
                    /* add new line */
                    new_line = tprintf("%s %s %s_vmeas_%d 0",
                            new_tok, node1, node1, sn);
                    /* insert new_line after card->line */
                    insert_new_line(card, new_line, card->linenum + 1, card->linenum_orig);
                }
                sn++;
                tfree(new_tok);
                tfree(node1);
            }
            tfree(tok);
        }
    }

    /* free rep */
    while (new_rep) {
        struct replace_currm *repn = new_rep->next;
        tfree(new_rep->rtoken);
        tfree(new_rep);
        new_rep = repn;
    }
}


/* syntax check:
   Check if we have a .control ... .endc pair,
   a .if ... .endif pair, a .suckt ... .ends pair */
static void inp_check_syntax(struct card *deck)
{
    struct card *card;
    int check_control = 0, check_subs = 0, check_if = 0, check_ch = 0, ii;
    bool mwarn = FALSE;
    char* subs[10];  /* store subckt lines */
    int ends = 0;  /* store .ends line numbers */

    /* prevent crash in inp.c, fcn inp_spsource: */
    if (ciprefix(".param", deck->line) || ciprefix(".meas", deck->line)) {
        fprintf(cp_err, "\nError: title line is missing!\n\n");
        controlled_exit(EXIT_BAD);
    }


    /* When '.probe alli' is set, disable auto bridging and set a flag */
    for (card = deck; card; card = card->nextcard) {
        char* cut_line = card->line;
        if (ciprefix(".probe", cut_line) && search_plain_identifier(cut_line, "alli")) {
            int i = 0;
            bool bi = TRUE;
            cp_vset("auto_bridge", CP_NUM, &i);
            cp_vset("probe_alli_given", CP_BOOL, &bi);
            break;
        }
    }

    for (ii = 0; ii < 10; ii++)
        subs[ii] = NULL;

    for (card = deck; card; card = card->nextcard) {
        char *cut_line = card->line;
        if (*cut_line == '*' || *cut_line == '\0')
            continue;
        // check for unusable leading characters and change them to '*'
        if (strchr("=[]?()&%$\"!:,\f", *cut_line)) {
            if (ft_stricterror) {
                fprintf(stderr, "Error: '%c' is not allowed as first character in line %s.\n", *cut_line, cut_line);
                controlled_exit(EXIT_BAD);
            }
            else {
                if (!check_ch) {
                    fprintf(stderr, "Warning: Unusual leading characters like '%c' or others out of '= [] ? () & %% $\"!:,\\f'\n", *cut_line);
                    fprintf(stderr, "    in netlist or included files, will be replaced with '*'.\n");
                    fprintf(stderr, "    Check line no %d:  %s\n\n", card->linenum_orig, cut_line);
                    check_ch = 1; /* just one warning */
                }
                *cut_line = '*';
            }
        }
        /* leading end-of-line delimiter ';' silently change to '*' */
        else if (*cut_line == ';') {
            *cut_line = '*';
        }
        // check for .control ... .endc
        if (ciprefix(".control", cut_line)) {
            if (check_control > 0) {
                fprintf(cp_err,
                        "\nError: Nesting of .control statements is not "
                        "allowed!\n\n");
                controlled_exit(EXIT_BAD);
            }
            check_control++;
            continue;
        }
        else if (ciprefix(".endc", cut_line)) {
            check_control--;
            continue;
        }
        // check for .subckt ... .ends
        else if (ciprefix(".subckt", cut_line)) {
            // warn if m=xx on .subckt line
            if (newcompat.hs && !mwarn) {
                if (strstr(cut_line, " m=") || strstr(cut_line, " m =")) {
                    fprintf(stderr, "Warning: m=xx on .subckt line will override multiplier m hierarchy!\n\n");
                    mwarn = TRUE;
                }
            }
            // nesting may be critical if params are involved
            if (check_subs > 0 && strchr(cut_line, '='))
                fprintf(cp_err,
                        "\nWarning: Nesting of subcircuits with parameters "
                        "is only marginally supported!\n\n");
            if (check_subs < 10)
                subs[check_subs] = cut_line;
            else
                fprintf(stderr, "Warning: .subckt nesting larger than 10, check may not catch all errors\n");
            check_subs++;
            continue;
        }
        else if (ciprefix(".ends", cut_line)) {
            check_subs--;
            if (check_subs >= 0 && check_subs < 10)
                subs[check_subs] = NULL;
            else if (ends == 0) /* store first occurence */
                ends = card->linenum_orig;
            continue;
        }
        // check for .if ... .endif
        if (ciprefix(".if", cut_line)) {
            check_if++;
            has_if = TRUE;
            continue;
        }
        else if (ciprefix(".endif", cut_line)) {
            check_if--;
            continue;
        }
        /* check for missing ac <val> in voltage or current source */
        if (check_control == 0 && strchr("VvIi", *cut_line)) {
            int err = 0;
            char* acline;
            /* skip instance name and nodes */
            acline = nexttok(cut_line);
            acline = nexttok(acline);
            acline = nexttok(acline);
            if (!acline) {
                fprintf(stderr, "Error in line   %s\n", cut_line);
                fprintf(stderr, "    Not enough parameters\n");
                controlled_exit(EXIT_BAD);
            }
            acline = search_plain_identifier(acline, "ac");
            if (acline == NULL)
                continue;
            /* skip ac */
            char* nacline = acline + 2;
             /* skip white spaces */
            nacline = skip_ws(nacline);
            /* if no numberr token, go to */
            if (*nacline == '\0')
                err = 1;
            else {
                /* skip potential = , found by make check */
                if (*nacline == '=')
                    nacline++;
                char* nnacline = nacline;
                /* get first token after ac */
                char* numtok = gettok_node(&nnacline);
                if (numtok) {
                    char* numtokfree = numtok;
                    /* Check if token is a parameter, to be filled in later */
                    if (*numtok == '\'' || *numtok == '{') {
                        err = 0;
                    }
                    else {
                        /* check if token is a valid number */
                        INPevaluate(&numtok, &err, 0);
                    }
                    tfree(numtokfree);
                }
                else
                    err = 1;
            }
            /* if no number, replace 'ac' by 'ac 1 0' */
            if (err){
                char *begstr = copy_substring(cut_line, acline);
                char* newline = tprintf("%s  ac ( 1 0 ) %s", begstr, nacline);
                tfree(begstr);
                tfree(card->line);
                card->line = newline;
            }
            continue;
        }
    }

    if (check_control > 0) {
        fprintf(cp_err, "\nWarning: Missing .endc statement!\n");
        fprintf(cp_err, "    This may cause subsequent errors.\n\n");
    }
    if (check_control < 0) {
        fprintf(cp_err, "\nWarning: Missing .control statement!\n");
        fprintf(cp_err, "    This may cause subsequent errors.\n\n");
    }
    if (check_subs != 0) {
        fprintf(cp_err,
                "\nError: Mismatch of .subckt ... .ends statements!\n");
        fprintf(cp_err, "    This will cause subsequent errors.\n\n");
        if (ends > 0)
            fprintf(cp_err, "Check .ends in line number %d\n", ends);
        else
            fprintf(cp_err, "Check line %s\n", subs[0]);
        controlled_exit(EXIT_BAD);
    }
    if (check_if != 0) {
        fprintf(cp_err, "\nError: Mismatch of .if ... .endif statements!\n");
        fprintf(cp_err, "    This may cause subsequent errors.\n\n");
    }
}

/* remove the mfg=mfgname entry from the .model cards */
static void rem_mfg_from_models(struct card *deck)
{
    struct card *card;
    for (card = deck; card; card = card->nextcard) {

        char *curr_line, *end, *start;

        curr_line = start = card->line;
        if (*curr_line == '*' || *curr_line == '\0')
            continue;
        /* remove mfg=name */
        if (ciprefix(".model", curr_line)) {
            start = search_plain_identifier(curr_line, "mfg");
            if (start && start[3] == '=') {
                end = nexttok(start);
                if (*end == '\0')
                    *start = '\0';
                else
                    while (start < end) {
                        *start = ' ';
                        start++;
                    }
            }
            start = search_plain_identifier(curr_line, "icrating");
            if (start && start[8] == '=') {
                end = nexttok(start);
                if (*end == '\0')
                    *start = '\0';
                else
                    while (start < end) {
                        *start = ' ';
                        start++;
                    }
            }
            start = search_plain_identifier(curr_line, "vceo");
            if (start && start[4] == '=') {
                end = nexttok(start);
                if (*end == '\0')
                    *start = '\0';
                else
                    while (start < end) {
                        *start = ' ';
                        start++;
                    }
            }
            start = search_plain_identifier(curr_line, "type");
            if (start && start[4] == '=') {
                end = nexttok(start);
                if (*end == '\0')
                    *start = '\0';
                else
                    while (start < end) {
                        *start = ' ';
                        start++;
                    }
            }
        }
    }
}

/* model type as input, element identifier as output */
static char inp_get_elem_ident(char *type)
{
    if (cieq(type, "r"))
        return 'r';
    else if (cieq(type, "c"))
        return 'c';
    else if (cieq(type, "l"))
        return 'l';
    else if (cieq(type, "nmos"))
        return 'm';
    else if (cieq(type, "pmos"))
        return 'm';
    else if (cieq(type, "numos"))
        return 'm';
    else if (cieq(type, "d"))
        return 'd';
    else if (cieq(type, "numd"))
        return 'd';
    else if (cieq(type, "numd2"))
        return 'd';
    else if (cieq(type, "npn"))
        return 'q';
    else if (cieq(type, "pnp"))
        return 'q';
    else if (cieq(type, "nbjt"))
        return 'q';
    else if (cieq(type, "nbjt2"))
        return 'q';
    else if (cieq(type, "njf"))
        return 'j';
    else if (cieq(type, "pjf"))
        return 'j';
    else if (cieq(type, "nmf"))
        return 'z';
    else if (cieq(type, "pmf"))
        return 'z';
    else if (cieq(type, "nhfet"))
        return 'z';
    else if (cieq(type, "phfet"))
        return 'z';
    else if (cieq(type, "sw"))
        return 's';
    else if (cieq(type, "csw"))
        return 'w';
    else if (cieq(type, "txl"))
        return 'y';
    else if (cieq(type, "cpl"))
        return 'p';
    else if (cieq(type, "ltra"))
        return 'o';
    else if (cieq(type, "urc"))
        return 'u';
    else if (ciprefix("vdmos", type))
        return 'm';
    if (cieq(type, "res"))
        return 'r';
    /* xspice code models do not have unique type names,
       but could also be an OSDI/OpenVAF model. */
    else
        return 'a';
}


static struct card_assoc *find_subckt_1(
        struct nscope *scope, const char *name)
{
    struct card_assoc *p = scope->subckts;
    for (; p; p = p->next)
        if (eq(name, p->name))
            break;
    return p;
}


static struct card_assoc *find_subckt(struct nscope *scope, const char *name)
{
    for (; scope; scope = scope->next) {
        struct card_assoc *p = find_subckt_1(scope, name);
        if (p)
            return p;
    }
    return NULL;
}


static void add_subckt(struct nscope *scope, struct card *subckt_line)
{
    char *n = skip_ws(skip_non_ws(subckt_line->line));
    char *name = copy_substring(n, skip_non_ws(n));
    if (find_subckt_1(scope, name)) {
        fprintf(stderr, "Warning: redefinition of .subckt %s, ignored\n",
                name);
        /* rename the redefined subcircuit */
        *n = '_';
    }
    struct card_assoc *entry = TMALLOC(struct card_assoc, 1);
    entry->name = name;
    entry->line = subckt_line;
    entry->next = scope->subckts;
    scope->subckts = entry;
}


/* linked list of models, includes use info */
struct modellist {
    struct card *model;
    char *modelname;
    bool used;
    char elemb;
    struct modellist *next;
};


static struct modellist *inp_find_model_1(
        struct nscope *scope, const char *name)
{
    struct modellist *p = scope->models;
    for (; p; p = p->next)
        if (model_name_match(name, p->modelname))
            break;
    return p;
}


static struct modellist *inp_find_model(
        struct nscope *scope, const char *name)
{
    for (; scope; scope = scope->next) {
        struct modellist *p = inp_find_model_1(scope, name);
        if (p)
            return p;
    }
    return NULL;
}

/* scan through deck and add level information to all struct card
 * depending on nested subcircuits */
struct nscope *inp_add_levels(struct card *deck)
{
    struct card *card;
    int skip_control = 0;

    struct nscope *root = TMALLOC(struct nscope, 1);
    root->next = NULL;
    root->subckts = NULL;
    root->models = NULL;

    struct nscope *lvl = root;

    for (card = deck; card; card = card->nextcard) {

        char *curr_line = card->line;

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

        if (*curr_line == '.') {
            if (ciprefix(".subckt", curr_line)) {
                add_subckt(lvl, card);
                struct nscope *scope = TMALLOC(struct nscope, 1);
                // lvl->name = ..., or just point to the deck
                scope->next = lvl;
                scope->subckts = NULL;
                scope->models = NULL;
                lvl = card->level = scope;
            }
            else if (ciprefix(".ends", curr_line)) {
                if (lvl == root) {
                    fprintf(stderr, "Error: .subckt/.ends not balanced\n");
                    controlled_exit(1);
                }
                card->level = lvl;
                lvl = lvl->next;
            }
            else {
                card->level = lvl;
            }
        }
        else {
            card->level = lvl;
        }
    }

    if (lvl != root)
        fprintf(stderr, "nesting error\n");

    return root;
}

/* remove the level and subckts entries */
void inp_rem_levels(struct nscope *root)
{
    struct card_assoc *p = root->subckts;
    while (p) {
        inp_rem_levels(p->line->level);
        tfree(p->name);
        struct card_assoc *pn = p->next;
        tfree(p);
        p = pn;
    }
    tfree(root);
}

static void rem_unused_xxx(struct nscope *level)
{
    struct modellist *m = level->models;
    while (m) {
        struct modellist *next_m = m->next;
        if (!m->used)
            m->model->line[0] = '*';
        tfree(m->modelname);
        tfree(m);
        m = next_m;
    }
    level->models = NULL;

    struct card_assoc *p = level->subckts;
    for (; p; p = p->next)
        rem_unused_xxx(p->line->level);
}


static void mark_all_binned(struct nscope *scope, char *name)
{
    struct modellist *p = scope->models;

    for (; p; p = p->next)
        if (model_name_match(name, p->modelname))
            p->used = TRUE;
}


void inp_rem_unused_models(struct nscope *root, struct card *deck)
{
    struct card *card;
    int skip_control = 0;

    /* create a list of .model */
    for (card = deck; card; card = card->nextcard) {

        char *curr_line = card->line;

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

        if (*curr_line == '*')
            continue;

        if (ciprefix(".model", curr_line)) {
            struct modellist *modl_new;
            modl_new = TMALLOC(struct modellist, 1);
            char *model_type = get_model_type(curr_line);
            if (!model_type) {
                fprintf(stderr, "Error: no model type given in line %s!\n", curr_line);
                tfree(modl_new);
                controlled_exit(EXIT_BAD);
            }
            modl_new->elemb = inp_get_elem_ident(model_type);
            modl_new->modelname = get_subckt_model_name(curr_line);
            modl_new->model = card;
            modl_new->used = FALSE;
            modl_new->next = card->level->models;
            card->level->models = modl_new;
            tfree(model_type);
        }
    }

    /* scan through all element lines  that require or may need a model */
    for (card = deck; card; card = card->nextcard) {

        char *curr_line = card->line;

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

        switch (*curr_line) {
            case '*':
            case '.':
            case 'v':
            case 'i':
            case 'b':
            case 'x':
            case 'e':
            case 'h':
            case 'g':
            case 'f':
            case 'k':
            case 't':
                continue;
                break;
            default:
                break;
        }

        /* check if correct model name */
        int num_terminals = get_number_terminals(curr_line);
        /* num_terminals may be 0 for a elements */
        if ((num_terminals != 0) || (*curr_line == 'a')) {
            char *elem_model_name;
            if (*curr_line == 'a')
                elem_model_name = get_adevice_model_name(curr_line);
            else
                elem_model_name = get_model_name(curr_line, num_terminals);

            /* ignore certain cases, for example
             *    'C5 node1 node2 42.0' or 'R2 node1 node2 4k7'
             */
            if (is_a_modelname(elem_model_name, curr_line)) {

                struct modellist *m =
                        inp_find_model(card->level, elem_model_name);
                if (m) {
                    if (*curr_line != m->elemb && !(*curr_line == 'n' && m->elemb == 'a'))
                        fprintf(stderr,
                                "warning, model type mismatch in line\n    "
                                "%s\n",
                                curr_line);
                    mark_all_binned(m->model->level, elem_model_name);
                }
                else {
                    fprintf(stderr, "warning, can't find model '%s' from line\n    "
                           "%s\n",
                            elem_model_name, curr_line);
                }
            }

            tfree(elem_model_name);
        }
    }

    // disable unused .model lines, and free the models assoc lists
    rem_unused_xxx(root);
}

/* Markus Kuhn <http://www.cl.cam.ac.uk/~mgk25/> -- 2005-03-30
 * License: Modified BSD (see http://www.cl.cam.ac.uk/~mgk25/short-license.html)
 * The utf8_check() function scans the '\0'-terminated string starting
 * at s. It returns a pointer to the first byte of the first malformed
 * or overlong UTF-8 sequence found, or NULL if the string contains
 * only correct UTF-8. It also spots UTF-8 sequences that could cause
 * trouble if converted to UTF-16, namely surrogate characters
 * (U+D800..U+DFFF) and non-Unicode positions (U+FFFE..U+FFFF).
 * In addition we check for some ngspice-specific characters like � etc.*/
#ifndef EXT_ASC
static unsigned char*
utf8_check(unsigned char *s)
{
    while (*s) {
        if (*s < 0x80)
            /* 0xxxxxxx */
            s++;
        else if (*s == 0xb5) {
            /* translate ansi micro � to u */
            *s = 'u';
            s++;
        }
        else if (s[0] == 0xc2 && s[1] == 0xb5) {
            /* translate utf-8 micro � to u */
            s[0] = 'u';
            /* remove second byte */
            unsigned char *y = s + 1;
            unsigned char *z = s + 2;
            while (*z) {
                *y++ = *z++;
            }
            *y = '\0';
            s++;
        }
        else if ((s[0] & 0xe0) == 0xc0) {
            /* 110XXXXx 10xxxxxx */
            if ((s[1] & 0xc0) != 0x80 ||
                (s[0] & 0xfe) == 0xc0)                        /* overlong? */
                return s;
            else
                s += 2;
        }
        else if ((s[0] & 0xf0) == 0xe0) {
            /* 1110XXXX 10Xxxxxx 10xxxxxx */
            if ((s[1] & 0xc0) != 0x80 ||
                (s[2] & 0xc0) != 0x80 ||
                (s[0] == 0xe0 && (s[1] & 0xe0) == 0x80) ||    /* overlong? */
                (s[0] == 0xed && (s[1] & 0xe0) == 0xa0) ||    /* surrogate? */
                (s[0] == 0xef && s[1] == 0xbf &&
                (s[2] & 0xfe) == 0xbe))                      /* U+FFFE or U+FFFF? */
                return s;
            else
                s += 3;
        }
        else if ((s[0] & 0xf8) == 0xf0) {
            /* 11110XXX 10XXxxxx 10xxxxxx 10xxxxxx */
            if ((s[1] & 0xc0) != 0x80 ||
                (s[2] & 0xc0) != 0x80 ||
                (s[3] & 0xc0) != 0x80 ||
                (s[0] == 0xf0 && (s[1] & 0xf0) == 0x80) ||    /* overlong? */
                (s[0] == 0xf4 && s[1] > 0x8f) || s[0] > 0xf4) /* > U+10FFFF? */
                return s;
            else
                s += 4;
        }
        else
            return s;
    }

    return NULL;
}

/* Scan through input deck and check for utf-8 syntax errors */
static void
utf8_syntax_check(struct card *deck)
{
    struct card *card;
    unsigned char *s;

    for (card = deck; card; card = card->nextcard) {

        char *curr_line = card->line;

        if (*curr_line == '*')
            continue;

        s = utf8_check((unsigned char*)curr_line);

        if (s) {
            fprintf(stderr, "Error: UTF-8 syntax error in input deck,\n    line %d at token/word %s\n", card->linenum_orig, s);
            controlled_exit(1);
        }
    }
}
#endif

/* if .dc (TEMPER) -15 75 5 if found, replace it by .dc TEMP -15 75 5. */
static void inp_repair_dc_ps(struct card* deck) {
    struct card* card;

    for (card = deck; card; card = card->nextcard) {
        char* curr_line = card->line;
        if (ciprefix(".dc", curr_line)) {
            char* tempstr = strstr(curr_line, "(temper)");
            if (tempstr) {
                memcpy(tempstr, "temp    ", 8);
            }
        }
    }
}

#ifdef XSPICE
/* spice2g6 allows to omit the poly(n) statement, if the
   polynomial is one-dimensional (n==1).
   For compatibility with the XSPIXE code, we have to add poly(1) appropriately. */
static int inp_poly_2g6_compat(struct card* deck) {
    struct card* card;
    int skip_control = 0;

    for (card = deck; card; card = card->nextcard) {
        char* curr_line = card->line;
        char* thisline = curr_line;

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


        switch (*thisline) {
        case 'h':
        case 'g':
        case 'e':
        case 'f':
            curr_line = nexttok_noparens(curr_line);
            curr_line = nexttok_noparens(curr_line);
            curr_line = nexttok_noparens(curr_line);
            if (!curr_line) {
                fprintf(stderr, "Error: bad syntax of line\n   %s\n", thisline);
                fprintf(stderr, "No circuit loaded!\n");
                return 1;
            }
            /* exclude all of the following fourth tokens */
            if (ciprefix("poly", curr_line))
                continue;
            if (ciprefix("value", curr_line))
                continue;
            if (ciprefix("vol", curr_line))
                continue;
            if (ciprefix("table", curr_line))
                continue;
            if (ciprefix("laplace", curr_line))
                continue;
            if (ciprefix("cur", curr_line))
                continue;
            /* the next four are HSPICE specific */
            if (ciprefix("vccs", curr_line))
                continue;
            if (ciprefix("vcvs", curr_line))
                continue;
            if (ciprefix("ccvs", curr_line))
                continue;
            if (ciprefix("cccs", curr_line))
                continue;
            break;
        default:
            continue;
        }
        /* go beyond the usual nodes and sources */
        switch (*thisline) {
        case 'g':
        case 'e':
            curr_line = nexttok_noparens(curr_line);
            curr_line = nexttok_noparens(curr_line);
            if (!curr_line) {
                fprintf(stderr, "Error: not enough parameters in line\n   %s\n", thisline);
                fprintf(stderr, "No circuit loaded!\n");
                return 1;
            }
            /* The next token may be a simple text token or an expression
               enclosed in brackets */
            if (*curr_line == '{') {
                char* tmptok = gettok_char(&curr_line, '}', TRUE, TRUE);
                tfree(tmptok);
            }
            else
                curr_line = nexttok(curr_line);
            if (!curr_line) {
                fprintf(stderr, "Error: not enough parameters in line\n   %s\n", thisline);
                fprintf(stderr, "No circuit loaded!\n");
                return 1;
            }
            if (*curr_line == '\0')
                continue;
            break;
        case 'f':
        case 'h':
            curr_line = nexttok(curr_line);
            if (!curr_line) {
                fprintf(stderr, "Error: not enough parameters in line\n   %s\n", thisline);
                fprintf(stderr, "No circuit loaded!\n");
                return 1;
            }
            /* The next token may be a simple text token or an expression
               enclosed in brackets */
            if (*curr_line == '{') {
                char* tmptok = gettok_char(&curr_line, '}', TRUE, TRUE);
                tfree(tmptok);
            }
            else
                curr_line = nexttok(curr_line);
            if (!curr_line) {
                fprintf(stderr, "Error: not enough parameters in line\n   %s\n", thisline);
                fprintf(stderr, "No circuit loaded!\n");
                return 1;
            }
            if (*curr_line == '\0')
                continue;
            break;
        }
        /* finish if these end tokens are found */
        if (ciprefix("ic=", curr_line)) {
            continue;
        }
        if (ciprefix("m=", curr_line)) {
            continue;
        }
        /* this now seems to be a spice2g6 poly one-dimensional source */
        /* insert poly(1) as the fourth token */
        curr_line = nexttok(thisline);
        curr_line = nexttok(curr_line);
        curr_line = nexttok(curr_line);
        char *endofline = copy(curr_line);
        *curr_line = '\0';
        char* newline = tprintf("%s poly(1) %s", thisline, endofline);
        tfree(card->line);
        card->line = newline;
        tfree(endofline);
    }
    return 0;
}
#endif
