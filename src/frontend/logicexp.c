/*
    logicexp.c

    Convert PSpice LOGICEXP logic expressions into XSPICE gates.
    Extract typical timing delay estimates from PINDLY statements and
    insert buffers and tristates with these delays.

    Reference: PSpice A/D Reference Guide version 16.6
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "ngspice/memory.h"
#include "ngspice/macros.h"
#include "ngspice/bool.h"
#include "ngspice/ngspice.h"
#include "ngspice/stringskip.h"
#include "ngspice/stringutil.h"
#include "ngspice/dstring.h"
#include "ngspice/logicexp.h"
#include "ngspice/udevices.h"

/* Turn off/on debug tracing */
#define PRINT_ALL    FALSE
//#define PRINT_ALL    TRUE

/* Start of btree symbol table */
#define SYM_INPUT       1
#define SYM_OUTPUT      2
#define SYM_TMODEL      4
#define SYM_KEY_WORD    8
#define SYM_ID          16
#define SYM_GATE_OP     32
#define SYM_INVERTER    64
#define SYM_OTHER       128

typedef struct sym_entry *SYM_TAB;
struct sym_entry {
    char *name;
    char *alias;
    int attribute;
    int ref_count; // for inverters
    SYM_TAB left;
    SYM_TAB right;
};

static SYM_TAB new_sym_entry(char *name, int attr)
{
    SYM_TAB newp;
    newp = TMALLOC(struct sym_entry, 1);
    newp->left = NULL;
    newp->right = NULL;
    newp->name = TMALLOC(char, strlen(name) + 1);
    strcpy(newp->name, name);
    newp->alias = NULL;
    newp->attribute = attr;
    newp->ref_count = 0;
    return newp;
}

static SYM_TAB insert_sym_tab(char *name, SYM_TAB t, int attr)
{
    int cmp;
    if (t == NULL) {
        t = new_sym_entry(name, attr);
        return t;
    }
    cmp = strcmp(name, t->name);
    if (cmp < 0) {
        t->left = insert_sym_tab(name, t->left, attr);
    } else if (cmp > 0) {
        t->right = insert_sym_tab(name, t->right, attr);
    } else {
        printf("NOTE insert_sym_tab %s already there\n", name);
    }
    return t;
}

static SYM_TAB member_sym_tab(char *name, SYM_TAB t)
{
    int cmp;
    while (t != NULL) {
        cmp = strcmp(name, t->name);
        if (cmp == 0) {
            return t;
        } else if (cmp < 0) {
            t = t->left;
        } else {
            t = t->right;
        }
    }
    return NULL;
}

static SYM_TAB add_sym_tab_entry(char *name, int attr, SYM_TAB *stab)
{
    SYM_TAB entry = NULL;
    entry = member_sym_tab(name, *stab);
    if (!entry) {
        *stab = insert_sym_tab(name, *stab, attr);
        entry = member_sym_tab(name, *stab);
    }
    return entry;
}

static void alias_sym_tab(char *alias, SYM_TAB t)
{
    if (t == NULL) { return; }
    if (t->alias)
        tfree(t->alias);
    t->alias = TMALLOC(char, strlen(alias) + 1);
    strcpy(t->alias, alias);
}

static void delete_sym_tab(SYM_TAB t)
{
    if (t == NULL) { return; }
    delete_sym_tab(t->left);
    delete_sym_tab(t->right);
    if (t->name)
        tfree(t->name);
    if (t->alias)
        tfree(t->alias);
    tfree(t);
}

static void print_sym_tab(SYM_TAB t, BOOL with_addr)
{
    if (t == NULL) { return; }
    print_sym_tab(t->left, with_addr);
    if (with_addr)
        printf("%p --> \n", (void *)t);
    printf("\"%s\"    %d  ref_count=%d", t->name, t->attribute, t->ref_count);
    if (t->alias)
        printf("  alias = \"%s\"", t->alias);
    printf("\n");
    print_sym_tab(t->right, with_addr);
}
/* End of btree symbol table */

/* Start of lexical scanner */
#define LEX_ID       256
#define LEX_OTHER    257
#define LEX_BUF_SZ   512
#define LEX_INIT_SZ  128

typedef struct lexer *LEXER;
struct lexer {
    char *lexer_buf;
    char *lexer_line;
    int lexer_pos;
    int lexer_back;
    SYM_TAB lexer_sym_tab;
    size_t lexer_blen;
};

static LEXER parse_lexer = NULL;

static LEXER new_lexer(char *line)
{
    LEXER lx;
    lx = TMALLOC(struct lexer, 1);
    lx->lexer_line = TMALLOC(char, (strlen(line) + 1));
    strcpy(lx->lexer_line, line);
    lx->lexer_pos = lx->lexer_back = 0;
    lx->lexer_blen = LEX_INIT_SZ;
    lx->lexer_buf = TMALLOC(char, lx->lexer_blen);
    (void) memset(lx->lexer_buf, 0, lx->lexer_blen);
    lx->lexer_sym_tab = NULL;
    return lx;
}

static void delete_lexer(LEXER lx)
{
    if (!lx)
        return;
    if (lx->lexer_buf)
        tfree(lx->lexer_buf);
    if (lx->lexer_line)
        tfree(lx->lexer_line);
    if (lx->lexer_sym_tab)
        delete_sym_tab(lx->lexer_sym_tab);
    tfree(lx);
}

static void lex_init(char *line)
{
    parse_lexer = new_lexer(line);
    return;
}

static int lexer_set_start(char *s, LEXER lx)
{
    char *pos;
    if (!lx)
        return -1;
    pos = strstr(lx->lexer_line, s);
    if (!pos)
        return -1;
    lx->lexer_pos = (int) (pos - &lx->lexer_line[0]);
    lx->lexer_back = lx->lexer_pos;
    return lx->lexer_pos;
}

static int lex_set_start(char *s)
{
    return lexer_set_start(s, parse_lexer);
}

static int lexer_getchar(LEXER lx)
{
    int item = 0;
    item = lx->lexer_line[lx->lexer_pos];
    lx->lexer_back = lx->lexer_pos;
    if (item != 0)
        lx->lexer_pos++;
    return item;
}

static void lexer_putback(LEXER lx)
{
    if (lx->lexer_back >= 0)
        lx->lexer_pos = lx->lexer_back;
}

static int lex_punct(int c)
{
    switch (c) {
    case ',':
    case '{':
    case '}':
    case '(':
    case ')':
    case ':':
    case '.':
        return c;
    default:
        break;
    }
    return 0;
}

static int lex_oper(int c)
{
    switch (c) {
    case '~':
    case '&':
    case '^':
    case '|':
    case '=':
        return c;
    default:
        break;
    }
    return 0;
}

static char *lex_gate_name(int c, BOOL not)
{
    /* returns an XSPICE gate model name */
    static char buf[32];
    switch (c) {
    case '~':
        if (not)
            sprintf(buf, "d__inverter__1");
        else
            sprintf(buf, "d__buffer__1");
        break;
    case '&':
        if (not)
            sprintf(buf, "d__nand__1");
        else
            sprintf(buf, "d__and__1");
        break;
    case '^':
        if (not)
            sprintf(buf, "d__xnor__1");
        else
            sprintf(buf, "d__xor__1");
        break;
    case '|':
        if (not)
            sprintf(buf, "d__nor__1");
        else
            sprintf(buf, "d__or__1");
        break;
    default:
        sprintf(buf, "UNKNOWN");
        break;
    }
    return buf;
}

static char *tmodel_gate_name(int c, BOOL not)
{
    /* Returns an XSPICE model name for the case where
       logicexp does not have a corresponding pindly
       but does have a UGATE timing model (not d0_gate).
    */
    static char buf[32];
    switch (c) {
    case '&':
        if (not)
            sprintf(buf, "dxspice_dly_nand");
        else
            sprintf(buf, "dxspice_dly_and");
        break;
    case '|':
        if (not)
            sprintf(buf, "dxspice_dly_nor");
        else
            sprintf(buf, "dxspice_dly_or");
        break;
    case '^':
        if (not)
            sprintf(buf, "dxspice_dly_xnor");
        else
            sprintf(buf, "dxspice_dly_xor");
        break;
    case '~':
        if (not)
            sprintf(buf, "dxspice_dly_inverter");
        else
            sprintf(buf, "dxspice_dly_buffer");
        break;
    default:
        return NULL;
    }
    return buf;
}

static int lex_gate_op(int c)
{
    switch (c) {
    case '&':
    case '^':
    case '|':
        return c;
    default:
        break;
    }
    return 0;
}

static int lex_ident(int c)
{
    /* Pspice and MicroCap are vague about what defines an identifier */
    if (isalnum(c) || c == '_' || c == '/' || c == '-' || c == '+')
        return c;
    else
        return 0;
}

static int lexer_scan(LEXER lx)
{
    int c;
    while (1) {
        lx->lexer_buf[0] = '\0';
        c = lexer_getchar(lx);
        if (c == '\0')
            return 0;
        else if (isspace(c))
            continue;
        else if (lex_punct(c))
            return c;
        else if (lex_oper(c))
            return c;
        else if (lex_ident(c)) {
            size_t i = 0;
            if (c == '+') { // an identifier does not begin with '+'
                lx->lexer_buf[0] = (char) c;
                lx->lexer_buf[1] = '\0';
                return LEX_OTHER;
            }
            while (lex_ident(c)) {
                if (i >= lx->lexer_blen) {
                    lx->lexer_blen *= 2;
                    lx->lexer_buf =
                        TREALLOC(char, lx->lexer_buf, lx->lexer_blen);
                }
                lx->lexer_buf[i] = (char) c;
                i++;
                c = lexer_getchar(lx);
            }
            if (i >= lx->lexer_blen) {
                lx->lexer_blen *= 2;
                lx->lexer_buf =
                    TREALLOC(char, lx->lexer_buf, lx->lexer_blen);
            }
            lx->lexer_buf[i] = '\0';
            if (c != '\0')
                lexer_putback(lx);
            return LEX_ID;
        } else {
            lx->lexer_buf[0] = (char) c;
            lx->lexer_buf[1] = '\0';
            return LEX_OTHER;
        }
    }
}

static int lex_scan(void)
{
    return lexer_scan(parse_lexer);
}

static BOOL lex_all_digits(char *str)
{
    size_t i, slen;
    if (!str) { return FALSE; }
    slen = strlen(str);
    if (slen < 1) { return FALSE; }
    for (i = 0; i < slen; i++) {
        if (!isdigit(str[i])) { return FALSE; }
    }
    return TRUE;
}
/* End of lexical scanner */

/* Start parse table */
typedef struct table_line *TLINE;
struct table_line {
    char *line;
    int depth;  /* expression nesting depth, outermost depth == 1 */
    TLINE next;
};

typedef struct parse_table *PTABLE;
struct parse_table {
    TLINE first;
    TLINE last;
    unsigned int entry_count;
};

static PTABLE parse_tab = NULL;
static PTABLE gen_tab = NULL;

static PTABLE new_parse_table(void)
{
    PTABLE pt;
    pt = TMALLOC(struct parse_table, 1);
    pt->first = pt->last = NULL;
    pt->entry_count = 0;
    return pt;
}

static void delete_parse_table(PTABLE pt)
{
    TLINE t, next;
    if (!pt)
        return;
    next = pt->first;
    while (next) {
        t = next;
        tfree(t->line);
        next = t->next;
        tfree(t);
    }
    tfree(pt);
}

static void delete_parse_gen_tables(void)
{
    delete_parse_table(parse_tab);
    delete_parse_table(gen_tab);
    parse_tab = gen_tab = NULL;
}

static void init_parse_tables(void)
{
    parse_tab = new_parse_table();
    gen_tab = new_parse_table();
}

static TLINE ptab_new_line(char *line)
{
    TLINE t = NULL;
    t = TMALLOC(struct table_line, 1);
    t->next = NULL;
    t->line = TMALLOC(char, (strlen(line) + 1));
    strcpy(t->line, line);
    t->depth = 0;
    return t;
}

static TLINE add_common(char *line, BOOL ignore_blank)
{
    if (!line)
        return NULL;
    if (ignore_blank) {
        if (line[0] == '\0') {
            return NULL;
        } else if (line[0] == '\n' && strlen(line) < 2) {
            return NULL;
        }
    }
    return ptab_new_line(line);
}

static TLINE add_to_parse_table(PTABLE pt, char *line, BOOL ignore_blank)
{
    TLINE t;
    if (!pt)
        return NULL;
    t = add_common(line, ignore_blank);
    if (!t)
        return NULL;
    t->next = NULL;
    if (!pt->first) {
        pt->first = pt->last = t;
    } else {
        pt->last->next = t;
        pt->last = t;
    }
    pt->entry_count++;
    return t;
}

static TLINE ptab_add_line(char *line, BOOL ignore_blank, int depth)
{
    TLINE t;
    t = add_to_parse_table(parse_tab, line, ignore_blank);
    if (t)
        t->depth = depth;
    return t;
}

static TLINE gen_tab_add_line(char *line, BOOL ignore_blank)
{
    TLINE t;
    t = add_to_parse_table(gen_tab, line, ignore_blank);
    return t;
}

static char *get_temp_from_line(char *line, BOOL begin, DSTRING *pds)
{
    /* First occurrence of "tmpx.." on the line, x is a digit */
    /* If begin is TRUE then "tmpx.." must be at the start of line */
    char *p, *q;
    int j = 0;
    p = strstr(line, "tmp");
    if (!p)
        return NULL;
    if (begin && p != line)
        return NULL;
    ds_clear(pds);
    p += 3;
    if (!isdigit(p[0]))
        return NULL;
    ds_cat_str(pds, "tmp");
    for (q = p, j = 0; isdigit(q[j]) || q[j] == '_'; j++) {
        ds_cat_char(pds, q[j]);
    }
    ds_cat_char(pds, '\0');
    return ds_get_buf(pds);
}

static char *find_temp_begin(char *line, DSTRING *pds)
{
    return get_temp_from_line(line, TRUE, pds);
}

static char *find_temp_anywhere(char *line, DSTRING *pds)
{
    return get_temp_from_line(line, FALSE, pds);
}

static TLINE tab_find(PTABLE pt, char *str, BOOL start_of_line)
{
    TLINE t;
    size_t len;

    if (!pt)
        return NULL;
    t = pt->first;
    len = strlen(str);
    while (t) {
        if (start_of_line) {
            if (strncmp(t->line, str, len) == 0)
                return t;
        } else {
            if (strstr(t->line, str))
                return t;
        }
        t = t->next;
    }
    return NULL;
}

static void ptable_print(PTABLE pt)
{
    TLINE t;
    if (!pt)
        return;
    t = pt->first;
    printf("entry_count %u\n", pt->entry_count);
    while (t) {
        if (t->depth > 1) {
            int i;
            for (i = 1; i < t->depth; i++) {
                printf("  ");
            }
        }
        printf("%s", t->line);
        if (t->depth > 0)
            printf(" ...[%d]", t->depth);
        printf("\n");
        t = t->next;
    }
}
/* End parse table */

/* Start of logicexp parser */
static char *get_inst_name(void);
static void aerror(char *s);
static BOOL amatch(int t);
static BOOL bexpr(void);
static BOOL bfactor(void);
static BOOL bparse(char *line, BOOL new_lexer);

static int lookahead = 0;
static int adepth = 0;
static int max_adepth = 0;
static DSTRING d_curr_line;
static int number_of_instances = 0;
static BOOL use_tmodel_delays = FALSE;

static void cleanup_parser(void)
{
    delete_lexer(parse_lexer);
    parse_lexer = NULL;
    delete_parse_gen_tables();
}

static char *get_inst_name(void)
{
    static char name[64];
    static int number = 0;
    number++;
    (void) sprintf(name, "a_%d", number);
    number_of_instances++;
    return name;
}

static char *get_inverter_output_name(char *input, DSTRING *pds)
{
    LEXER lx = parse_lexer;
    // FIX ME keep this name in the symbol table to ensure uniqueness
    ds_clear(pds);
    ds_cat_printf(pds, "inv_out__%s", input);
    if (member_sym_tab(ds_get_buf(pds), lx->lexer_sym_tab))
        fprintf(stderr, "ERROR %s is already in use\n", ds_get_buf(pds));
    return ds_get_buf(pds);
}

static char *get_inv_tail(char *str, DSTRING *pds)
{
    char *p = NULL, *q = NULL;
    int j = 0;
    size_t slen = strlen("inv_out__");
    p = strstr(str, "inv_out__");
    if (!p)
        return NULL;
    ds_clear(pds);
    for (q = p + slen, j = 0; q[j] != '\0' && !isspace(q[j]); j++) {
        ds_cat_char(pds, q[j]);
    }
    ds_cat_char(pds, '\0');
    return ds_get_buf(pds);
}

static void gen_models(void)
{
    DS_CREATE(model, 64);

    ds_clear(&model);
    ds_cat_printf(&model,
    ".model d_inv_zero_delay d_inverter(inertial_delay=true rise_delay=1.0e-12 fall_delay=1.0e-12)");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model,
    ".model d__inverter__1 d_inverter(inertial_delay=true rise_delay=1.0e-12 fall_delay=1.0e-12)");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model,
    ".model d__buffer__1 d_buffer(inertial_delay=true rise_delay=1.0e-12 fall_delay=1.0e-12)");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model,
    ".model d__nand__1 d_nand(inertial_delay=true rise_delay=1.0e-12 fall_delay=1.0e-12)");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model,
    ".model d__and__1 d_and(inertial_delay=true rise_delay=1.0e-12 fall_delay=1.0e-12)");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model,
    ".model d__xnor__1 d_xnor(inertial_delay=true rise_delay=1.0e-12 fall_delay=1.0e-12)");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model,
    ".model d__xor__1 d_xor(inertial_delay=true rise_delay=1.0e-12 fall_delay=1.0e-12)");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model,
    ".model d__nor__1 d_nor(inertial_delay=true rise_delay=1.0e-12 fall_delay=1.0e-12)");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model,
    ".model d__or__1 d_or(inertial_delay=true rise_delay=1.0e-12 fall_delay=1.0e-12)");
    u_add_instance(ds_get_buf(&model));

    ds_free(&model);
}

static void aerror(char *s)
{
    LEXER lx = parse_lexer;
    printf("%s [%s]\n", s, lx->lexer_line + lx->lexer_pos);
    fflush(stdout);
    cleanup_parser();
}

char *get_temp_name(void)
{
    static char name[64];
    static int number = 0;
    number++;
    (void) sprintf(name, "tmp%d", number);
    return name;
}

static BOOL amatch(int t)
{
    if (lookahead == t) {
        lookahead = lex_scan();
    } else {
        printf("expect = %d lookahead = %d lexer_buf \"%s\"\n",
            t, lookahead, parse_lexer->lexer_buf);
        aerror("amatch: syntax error");
        return FALSE;
    }
    return TRUE;
}

#define AMATCH_BFACTOR(n) \
{ \
    if (!amatch((n))) { \
        return FALSE; \
    } \
}

static BOOL bfactor(void)
{
    /* factor is : ['~'] rest
       where rest is: input_name_id | '(' expr ')' | error
       [] means optional
    */
    BOOL is_not = FALSE;
    SYM_TAB entry = NULL;
    LEXER lx = parse_lexer;

    adepth++;

    if (lookahead == '~') {
        is_not = TRUE;
        lookahead = lex_scan();
    }

    if (lookahead == LEX_ID) {
        entry = add_sym_tab_entry(lx->lexer_buf, SYM_ID, &lx->lexer_sym_tab);
        if (is_not) {
            DS_CREATE(dstr, 128);
            ds_clear(&dstr);
            ds_cat_printf(&d_curr_line, "%s ",
                get_inverter_output_name(lx->lexer_buf, &dstr));
            ds_free(&dstr);
            entry->attribute |= SYM_INVERTER;
            entry->ref_count++;
        } else {
            ds_cat_printf(&d_curr_line, "%s ", lx->lexer_buf);
        }

        lookahead = lex_scan();

    } else if (lookahead == '(') {
        DS_CREATE(tmpnam, 64);

        ds_clear(&tmpnam);
        if (adepth > max_adepth)
            max_adepth = adepth;

        ds_cat_str(&tmpnam, get_temp_name());
        (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE, adepth);
        ds_clear(&d_curr_line);
        ds_cat_printf(&d_curr_line, "%s__%d <- ", ds_get_buf(&tmpnam), adepth);

        if (is_not) {
            ds_cat_printf(&d_curr_line, "~ %c", lookahead);
        } else {
            ds_cat_printf(&d_curr_line, "%c", lookahead);
        }
        (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE, adepth);
        ds_clear(&d_curr_line);

        lookahead = lex_scan();
        if (!bexpr()) {
            cleanup_parser();
            return FALSE;
        }

        (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE, adepth);
        ds_clear(&d_curr_line);

        ds_cat_printf(&d_curr_line, "%c -> %s__%d", lookahead,
            ds_get_buf(&tmpnam), adepth);
        (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE, adepth);
        ds_clear(&d_curr_line);

        ds_free(&tmpnam);
        AMATCH_BFACTOR(')');

    } else {
        aerror("bfactor: syntax error");
        return FALSE;
    }
    adepth--;
    return TRUE;
}

static BOOL bexpr(void)
{
    /* expr is: factor { gate_op factor }+
       where {}+ means 0 or more times.
    */
    if (!bfactor()) {
        cleanup_parser();
        return FALSE;
    }

    while (lex_gate_op(lookahead)) {
        ds_cat_printf(&d_curr_line, "%c ", lookahead);

        lookahead = lex_scan();
        if (!bfactor()) {
            cleanup_parser();
            return FALSE;
        }
    }
    return TRUE;
}

#define AMATCH_BSTMT(n) \
{ \
    if (!amatch((n))) { \
        ds_free(&tname); ds_free(&assign); \
        return FALSE; \
    } \
}

static BOOL bstmt(void)
{
    /* A stmt is: output_name_id = '{' expr '}' */
    BOOL verbose = PRINT_ALL;
    int end_pos = 0, start_pos = 0;
    SYM_TAB entry = NULL;
    DS_CREATE(tname, 64);
    DS_CREATE(assign, LEX_BUF_SZ);

    if (lookahead == LEX_ID) {
        entry = add_sym_tab_entry(parse_lexer->lexer_buf, SYM_ID,
            &parse_lexer->lexer_sym_tab);
    } else {
        aerror("bstmt: syntax error");
        return FALSE;
    }

    adepth++;
    if (adepth > max_adepth)
        max_adepth = adepth;

    if (verbose) {
        start_pos = parse_lexer->lexer_pos;
        printf("* %s", parse_lexer->lexer_buf);
    }

    AMATCH_BSTMT(LEX_ID);
    AMATCH_BSTMT('=');

    ds_clear(&assign);
    ds_cat_printf(&assign, "%s =", entry->name);
    (void) ptab_add_line(ds_get_buf(&assign), TRUE, adepth);

    AMATCH_BSTMT('{');

    ds_clear(&tname);
    ds_cat_str(&tname, get_temp_name());
    ds_cat_printf(&d_curr_line, "%s__%d <- (", ds_get_buf(&tname), adepth);
    (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE, adepth);
    ds_clear(&d_curr_line);

    if (!bexpr()) {
        cleanup_parser();
        ds_free(&assign);
        ds_free(&tname);
        return FALSE;
    }

    if (ds_get_length(&d_curr_line) > 0) {
        (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE, adepth);
    }
    ds_clear(&d_curr_line);
    ds_cat_printf(&d_curr_line, ") -> %s__%d", ds_get_buf(&tname), adepth);
    (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE, adepth);
    ds_clear(&d_curr_line);

    if (verbose) {
        DS_CREATE(stmt_str, 128);
        end_pos = parse_lexer->lexer_pos;
        ds_cat_mem(&stmt_str, &parse_lexer->lexer_line[start_pos],
            (size_t) (end_pos - start_pos));
        printf("%s\n", ds_get_buf(&stmt_str));
        ds_free(&stmt_str);
    }

    AMATCH_BSTMT('}');

    ds_free(&assign);
    ds_free(&tname);
    adepth--;
    return TRUE;
}

static PTABLE optimize_gen_tab(PTABLE pt)
{
    /* This function compacts the gen_tab, returning a new PTABLE.
       Aliases are transformed and removed as described below.
       Usually, optimize_gen_tab is called a second time on the
       PTABLE created by the first call. The algorithm here will
       only transform one level of aliases.
    */
    TLINE t = NULL;
    LEXER lxr = NULL;
    int val, idnum = 0, tok_count = 0;
    SYM_TAB entry = NULL, alias_tab = NULL;
    BOOL found_tilde = FALSE, starts_with_temp = FALSE;
    BOOL prit = PRINT_ALL;
    PTABLE new_gen = NULL;
    DS_CREATE(scratch, LEX_BUF_SZ);
    DS_CREATE(alias, 64);
    DS_CREATE(non_tmp_name, 64);
    DS_CREATE(tmp_name, 64);
    DS_CREATE(find_str, 128);

    if (!pt || !pt->first) {
        ds_free(&scratch);
        ds_free(&alias);
        ds_free(&non_tmp_name);
        ds_free(&tmp_name);
        ds_free(&find_str);
        return NULL;
    }
    t = pt->first;
    lxr = new_lexer(t->line);
    /* Look for tmp... = another_name
         t1 = name1 (alias for t1)
         t2 = name2 (alias for t2)
         t3 = t1 op t2
       during second pass transform
         ignore t1, t2
         t3 = name1 op name2
    */
    while (t) {
        idnum = 0;
        val = lexer_scan(lxr);
        ds_clear(&alias);
        entry = NULL;
        found_tilde = FALSE;
        if (find_temp_begin(t->line, &find_str))
            starts_with_temp = TRUE;
        else
            starts_with_temp = FALSE;
        tok_count = 0;
        while (val != '\0') {
            tok_count++;
            if (val == LEX_ID) {
                idnum++;
                if (idnum == 1) {
                    entry = add_sym_tab_entry(lxr->lexer_buf, SYM_ID,
                        &alias_tab);
                } else if (idnum == 2) {
                    ds_cat_str(&alias, lxr->lexer_buf);
                }
            } else if (val == '~') {
                found_tilde = TRUE;
                if (tok_count != 3) {
                    goto quick_return;
                }
            } else if (val == '=') {
                if (tok_count != 2) {
                    goto quick_return;
                }
            }
            val = lexer_scan(lxr);
        }
        if (starts_with_temp && !found_tilde && idnum == 2)
            alias_sym_tab(ds_get_buf(&alias), entry);
        t = t->next;
        if (t) {
            delete_lexer(lxr);
            lxr = new_lexer(t->line);
        }
    }
    if (prit) {
        printf("alias_tab:\n");
        print_sym_tab(alias_tab, FALSE);
    }
    delete_lexer(lxr);


    /* Second pass, replace names by their aliases.
       Perform transformation as mentioned above.
       Transform:
         t1 = t2 op t3 {op t4 ...} (t* can also be name*, not just tmps)
         lhs = t1 (lhs of original x = { expr } statement)
       into:
         ignore lhs = t1
         lhs = t2 op t3 {op t4...}
       NOTE that lhs_= t1 should be the last entry in gen_tab.
       lhs = t1 (from stmt lhs = { expr }) is the top-most level
       in the parse tree, and is encountered last in the evaluation order.
    */
    new_gen = new_parse_table();
    ds_clear(&scratch);
    t = pt->first;
    lxr = new_lexer(t->line);
    while (t) { // while (t) second pass
        BOOL skip = FALSE;

        val = lexer_scan(lxr);
        idnum = 0;
        entry = NULL;
        if (find_temp_begin(t->line, &find_str))
            starts_with_temp = TRUE;
        else
            starts_with_temp = FALSE;
        tok_count = 0;
        ds_clear(&scratch);
        ds_clear(&non_tmp_name);
        ds_clear(&tmp_name);
        while (val != '\0' && !skip) {
            tok_count++;
            if (val == LEX_ID) {
                idnum++;
                entry = member_sym_tab(lxr->lexer_buf, alias_tab);
                if (entry && entry->alias) {
                    if (idnum > 1) {
                        ds_cat_printf(&scratch, "%s ", entry->alias);
                    } else if (idnum == 1) {
                        if (starts_with_temp) {
                            skip = TRUE;
                        }
                    }
                } else {
                    ds_cat_printf(&scratch, "%s ", lxr->lexer_buf);
                    if (tok_count == 1) {
                        ds_clear(&non_tmp_name);
                        if (!find_temp_begin(lxr->lexer_buf, &find_str))
                            ds_cat_str(&non_tmp_name, lxr->lexer_buf);
                    } else if (tok_count == 3) {
                        if (ds_get_length(&non_tmp_name) > 0) {
                            char *str1 = NULL;
                            str1 = find_temp_begin(lxr->lexer_buf, &find_str);
                            if (str1) {
                                ds_clear(&tmp_name);
                                ds_cat_str(&tmp_name, lxr->lexer_buf);
                            }
                        }
                    }
                }

                if (idnum > 2) {
                    ds_clear(&non_tmp_name);
                    ds_clear(&tmp_name);
                }
            } else {
                if (val == LEX_OTHER) {
                    delete_parse_table(new_gen);
                    new_gen = NULL;
                    goto quick_return;
                }
                ds_cat_printf(&scratch, "%c ", val);
            }
            val = lexer_scan(lxr);
        }
        t = t->next;
        if (t) {
            delete_lexer(lxr);
            lxr = new_lexer(t->line);
        }
        if (!skip) {
            TLINE tnamel = NULL;
            char *p = NULL;
            DS_CREATE(d_buf, 128);
            BOOL ignore_lhs = FALSE;

            ds_clear(&d_buf);
            if (ds_get_length(&tmp_name) > 0)
                tnamel = tab_find(new_gen, ds_get_buf(&tmp_name), TRUE);
            if (ds_get_length(&non_tmp_name) > 0 && tnamel) {
                ignore_lhs = TRUE;

                ds_clear(&d_buf);
                p = strstr(tnamel->line, " = ");
                if (p) {
                    ds_cat_str(&d_buf, ds_get_buf(&non_tmp_name));
                    ds_cat_str(&d_buf, p);
                    tfree(tnamel->line);
                    tnamel->line = TMALLOC(char, ds_get_length(&d_buf) + 1);
                    strcpy(tnamel->line, ds_get_buf(&d_buf));
                }
            }
            if (!ignore_lhs) {
                (void) add_to_parse_table(new_gen,
                    ds_get_buf(&scratch), TRUE);
            }
            ds_free(&d_buf);
        }
    } // end of while (t) second pass

quick_return:
    if (new_gen && new_gen->entry_count == 0) {
        delete_parse_table(new_gen);
        new_gen = NULL;
    }
    ds_free(&alias);
    ds_free(&scratch);
    ds_free(&non_tmp_name);
    ds_free(&tmp_name);
    ds_free(&find_str);
    delete_lexer(lxr);
    delete_sym_tab(alias_tab);

    return new_gen;
}

static BOOL gen_gates(PTABLE gate_tab, SYM_TAB parser_symbols)
{
    /* gen_gates is called with PTABLE gate_tab being the final
       PTABLE produced by optimize_gen_tab(,..) calls.
       If gate tab is the orignal uncompacted gen_tab, then extra
       redundant intermediate gates will be created.
    */
    TLINE t;
    LEXER lxr = NULL;
    int val, tok_count = 0, gate_op = 0, idnum = 0, in_count = 0;
    BOOL found_tilde = FALSE;
    BOOL prit = PRINT_ALL;
    DS_CREATE(out_name, 64);
    DS_CREATE(in_names, 64);
    DS_CREATE(gate_name, 64);
    DS_CREATE(instance, 128);

    if (!gate_tab || !gate_tab->first) {
        ds_free(&out_name);
        ds_free(&in_names);
        ds_free(&gate_name);
        ds_free(&instance);
        return FALSE;
    }
    t = gate_tab->first;
    lxr = new_lexer(t->line);
    while (t) { // while t loop
        ds_clear(&out_name);
        ds_clear(&in_names);
        ds_clear(&gate_name);
        ds_clear(&instance);
        idnum = 0;
        val = lexer_scan(lxr);
        found_tilde = FALSE;
        tok_count = 0;
        gate_op = 0;
        in_count = 0;
        while (val != '\0') {  // while val loop
            tok_count++;
            if (val == LEX_ID) {
                idnum++;
                if (idnum == 1) { //output name
                    ds_cat_str(&out_name, lxr->lexer_buf);
                } else { // input name
                    char *tail = NULL;
                    DS_CREATE(dstr, 64);
                    in_count++;
                    tail = get_inv_tail(lxr->lexer_buf, &dstr);
                    if (tail && strlen(tail) > 0) {
                        ds_cat_printf(&in_names, " ~%s", tail);
                        if (prit) {
                            printf(
                            "change input name \"%s\" tail \"~%s\"\n",
                            lxr->lexer_buf, tail);
                        }
                    } else {
                        ds_cat_printf(&in_names, " %s", lxr->lexer_buf);
                    }
                    ds_free(&dstr);
                }
            } else if (val == '~') {
                found_tilde = TRUE;
                if (tok_count != 3) goto gen_error;
            } else if (val == '=') {
                if (tok_count != 2) goto gen_error;
            } else if (lex_gate_op(val)) {
                if (gate_op != 0) {
                   if (val != gate_op) goto gen_error;
                }
                gate_op = val;
            } else {
                goto gen_error;
            }
            val = lexer_scan(lxr);
        }  // end while val loop

        if (in_count == 1) { // buffer or inverter
            if (gate_op != 0) goto gen_error;
            gate_op = '~'; // found_tilde specifies inverter or buffer
        } else if (in_count >= 2) { // AND, OR. XOR and inverses
            if (gate_op == 0) goto gen_error;
        } else {
            goto gen_error;
        }

        if (use_tmodel_delays) {
            /* This is the case when logicexp has a UGATE
               timing model (not d0_gate) and no pindly.
            */
            SYM_TAB entry = NULL;
            char *nm1 = 0;
            entry = member_sym_tab(ds_get_buf(&out_name), parser_symbols);
            if (entry && (entry->attribute & SYM_OUTPUT)) {
                nm1 = tmodel_gate_name(gate_op, found_tilde);
                if (nm1) {
                    ds_cat_str(&gate_name, nm1);
                }
            }
            if (!nm1) {
                nm1 = lex_gate_name(gate_op, found_tilde);
                ds_cat_str(&gate_name, nm1);
            }
        } else {
            ds_cat_str(&gate_name, lex_gate_name(gate_op, found_tilde));
        }

        ds_cat_printf(&instance, "%s ", get_inst_name());
        if (in_count == 1) {
            ds_cat_printf(&instance, "%s %s ", ds_get_buf(&in_names),
                ds_get_buf(&out_name));
        } else {
            ds_cat_printf(&instance, "[%s ] %s ", ds_get_buf(&in_names),
                ds_get_buf(&out_name));
        }
        ds_cat_printf(&instance, "%s", ds_get_buf(&gate_name));
        t = t->next;
        if (t) {
            delete_lexer(lxr);
            lxr = new_lexer(t->line);
        }
        if (ds_get_length(&instance) > 0) {
            u_add_instance(ds_get_buf(&instance));
        }
    } // end while t loop

    delete_lexer(lxr);
    ds_free(&out_name);
    ds_free(&in_names);
    ds_free(&gate_name);
    ds_free(&instance);
    return TRUE;

gen_error:
    delete_lexer(lxr);
    ds_free(&out_name);
    ds_free(&in_names);
    ds_free(&gate_name);
    ds_free(&instance);
    return FALSE;
}

/*
    gen_tab lines format:
        name1 = [~] name2 [op name3 {op namei}+]
    [] means optional, {}+ means zero or more times.
    op is gate type (&, |, ^), ~ means invert output.
    name1 is the gate output, and name2,... are inputs.
    & is AND, | is OR, ^ is XOR.
    ~ & is NAND, ~ | is NOR, ~ ^ is XNOR.
    In any given line, all the op values are the same, and don't change.
    AND and OR can have >= 2 inputs, XOR can have only 2 inputs.
    If there is only a single input, then the gate is BUF or INV(~).
*/
static void bevaluate(TLINE t, int deep)
{
    /* TLINE t is the entry in the parse_tab and deep is the call depth
       where the parse_tab is transformed into the gen_tab. The deeper
       calls are evaluated first, bottom-up, as determined by beval_order.
       The tokens in the parse_tab are reassembled into gen_tab lines
       as described above.
    */
    char *s;
    int down = 0;
    DS_CREATE(this, 64);
    DS_CREATE(other, 64);
    DS_CREATE(new_line, LEX_BUF_SZ);
    DS_CREATE(find_str, 128);

    s = find_temp_begin(t->line, &find_str);
    if (!s) {
        ds_free(&find_str);
        return;
    }
    ds_clear(&other);
    ds_clear(&new_line);
    ds_clear(&this);
    ds_cat_str(&this, s);
    if (strstr(t->line + ds_get_length(&this), " ~ ")) {
        ds_cat_printf(&new_line, "%s =  ~ ", ds_get_buf(&this));
    } else {
        if (deep == 1) {
            ds_cat_printf(&new_line, "%s ", parse_tab->first->line);
        } else {
            ds_cat_printf(&new_line, "%s = ", ds_get_buf(&this));
        }
    }
    t = t->next;
    while (t) {
        s = find_temp_anywhere(t->line, &find_str);
        if (s) {
            if (eq(ds_get_buf(&this), s)) {
                break;
            } else {
                if (down == 0) {
                    s = find_temp_begin(t->line, &find_str);
                    ds_clear(&other);
                    ds_cat_str(&other, s);
                    down = 1;
                    ds_cat_printf(&new_line, " %s", ds_get_buf(&other));
                } else if (down == 1) {
                    s = find_temp_anywhere(t->line, &find_str);
                    if (eq(ds_get_buf(&other), s)) {
                        down = 0;
                        ds_clear(&other);
                    }
                }
            }
        } else if (down == 0) {
            s = find_temp_anywhere(t->line, &find_str);
            if (!s) {
                ds_cat_printf(&new_line, " %s", t->line);
            }
        }
        t = t->next;
    }
    (void) gen_tab_add_line(ds_get_buf(&new_line), TRUE);
    ds_free(&this);
    ds_free(&other);
    ds_free(&new_line);
    ds_free(&find_str);
    return;
}

static void beval_order(void)
{
    /* The parser is top-down recursive descent. The depth is used
       so that the parsed data is evaluated bottom-up. Then the
       tmp.. regions can be evaluated before they are referenced.
    */
    int i, depth;
    TLINE t;
    size_t slen;

    if (!parse_tab || !parse_tab->first)
        return;
    slen = strlen("tmp");
    for (i = max_adepth; i > 0; i--) {
        t = parse_tab->first;
        while (t) {
            char *q;
            int cmp = 0;
            cmp = strncmp(t->line, "tmp", slen);
            if (cmp == 0 && ((q = strstr(t->line, " <- ")) != NULL)) {
                depth = t->depth;
                if (depth > 0) {
                    if (i == depth) {
                        bevaluate(t, i);
                    }
                }
            }
            t = t->next;
        }
    }
    return;
}

static BOOL bparse(char *line, BOOL new_lexer)
{
    int stmt_num = 0;
    BOOL ret_val = TRUE, prit = PRINT_ALL;
    PTABLE opt_tab1 = NULL, opt_tab2 = NULL;
    DS_CREATE(stmt, LEX_BUF_SZ);
    char *seed_buf;

    seed_buf = TMALLOC(char, LEX_BUF_SZ);
    (void) memcpy(seed_buf, "seed", strlen("seed"));

    ds_init(&d_curr_line, seed_buf, strlen("seed"),
        LEX_BUF_SZ, ds_buf_type_heap);
    ds_clear(&d_curr_line);

    if (new_lexer)
        lex_init(line);
    if (!parse_lexer) return FALSE;
    lookahead = lex_set_start("logic:");
    lookahead = lex_scan(); // "logic"
    lookahead = lex_scan(); // ':'
    lookahead = lex_scan();
    while (lookahead != '\0') { // while lookahead loop
        unsigned int last_count = 0, curr_count = 0;
        init_parse_tables();
        adepth = max_adepth = 0;
        stmt_num++;
        ds_clear(&stmt);
        ds_cat_str(&stmt, parse_lexer->lexer_buf);
        if (!bstmt()) {
            cleanup_parser();
            ret_val= FALSE;
            break;
        }

        if (prit) {
            printf("START parse_tab\n");
            ptable_print(parse_tab);
            printf("END parse_tab\n");
        }

        beval_order();

        /* generate gates only when optimizations are successful */
        if (prit) {
            printf("gen_tab ");
            ptable_print(gen_tab);
        }
        last_count = gen_tab->entry_count;
        if (last_count == 1) {
            ret_val = gen_gates(gen_tab, parse_lexer->lexer_sym_tab);
            if (!ret_val) {
                fprintf(stderr, "ERROR generating gates for logicexp\n");
            }
        } else if (last_count > 1) {
            opt_tab1 = optimize_gen_tab(gen_tab);
            if (prit) {
                printf("opt_tab1 ");
                ptable_print(opt_tab1);
            }
            if (opt_tab1) {
                curr_count = opt_tab1->entry_count;
                opt_tab2 = opt_tab1;
                while (curr_count > 1 && curr_count < last_count) {
                    last_count = curr_count;
                    opt_tab2 = optimize_gen_tab(opt_tab1);
                    if (prit) {
                        printf("opt_tab2 ");
                        ptable_print(opt_tab2);
                    }
                    delete_parse_table(opt_tab1);
                    if (!opt_tab2) {
                        ret_val = FALSE;
                        break;
                    }
                    opt_tab1 = opt_tab2;
                    curr_count = opt_tab2->entry_count;
                }
                if (opt_tab2) {
                    ret_val = gen_gates(opt_tab2, parse_lexer->lexer_sym_tab);
                    if (!ret_val) {
                        fprintf(stderr,
                            "ERROR generating gates for logicexp\n");
                    }
                    delete_parse_table(opt_tab2);
                }
            } else {
                ret_val = FALSE;
            }
        } else {
            ret_val = FALSE;
        }
        delete_parse_gen_tables();
        if (!ret_val) {
            break;
        }
    } // end while lookahead loop

    ds_free(&d_curr_line);
    if (ret_val)
        gen_models();
    ds_free(&stmt);
    cleanup_parser();
    return ret_val;
}
/* End of logicexp parser */

/* Start of f_logicexp which is called from udevices.c 
   See the PSpice reference which describes the LOGICEXP statement syntax.

   NOTE: Combinational gates are generated and usually have zero delays.
   In XSPICE, the shortest delays are 1.0e-12 secs, not actually zero.

   Timing delays for LOGICEXP come from an associated PINDLY instance
   when the timing model is d0_gate. Otherwise the timing model is used
   for the delay estimates (see f_logicexp).

   The PINDLY statements generate buffers and tristate buffers
   which drive the primary outputs from the LOGICEXP outputs.
   These buffers and tristates have estimated typical delays.
*/
static LEXER current_lexer = NULL;

static BOOL expect_token(
    int tok, int expected_tok, char *expected_str, BOOL msg, int loc)
{
    if (tok != expected_tok) {
        if (msg) {
            fprintf(stderr,
                "ERROR expect_token failed tok %d expected_tok %d loc %d\n",
                tok, expected_tok, loc);
        }
        return FALSE;
    }
    if (tok == LEX_ID) {
        if (expected_str) {
            LEXER lx = current_lexer;
            if (eq(expected_str, lx->lexer_buf))
                return TRUE;
            else {
                if (msg) {
                    fprintf(stderr,
                    "ERROR expect_token failed lexer_buf %s expected_str %s loc %d\n",
                        lx->lexer_buf, expected_str, loc);
                }
                return FALSE;
            }
        } else { // Any LEX_ID string matches
            return TRUE;
        }
    }
    return TRUE;
}

BOOL f_logicexp(char *line)
{
    int t, num_ins = 0, num_outs = 0, i;
    char *endp;
    BOOL ret_val = TRUE;

    lex_init(line);
    current_lexer = parse_lexer;
    (void) add_sym_tab_entry("logic", SYM_KEY_WORD,
        &parse_lexer->lexer_sym_tab);
    t = lex_scan(); // U*
    if (!expect_token(t, LEX_ID, NULL, TRUE, 1)) goto error_return;
    /* logicexp ( int , int ) */
    t = lex_scan();
    if (!expect_token(t, LEX_ID, "logicexp", TRUE, 2)) goto error_return;
    t = lex_scan();
    if (!expect_token(t, '(', NULL, TRUE, 3)) goto error_return;
    t = lex_scan();
    if (!expect_token(t, LEX_ID, NULL, TRUE, 4)) goto error_return;
    if (lex_all_digits(parse_lexer->lexer_buf)) {
        num_ins = (int) strtol(parse_lexer->lexer_buf, &endp, 10);
    } else {
        fprintf(stderr, "ERROR logicexp input count is not an integer\n");
        goto error_return;
    }
    t = lex_scan();
    if (!expect_token(t, ',', NULL, TRUE, 5)) goto error_return;
    t = lex_scan();
    if (!expect_token(t, LEX_ID, NULL, TRUE, 6)) goto error_return;
    if (lex_all_digits(parse_lexer->lexer_buf)) {
        num_outs = (int) strtol(parse_lexer->lexer_buf, &endp, 10);
    } else {
        fprintf(stderr, "ERROR logicexp output count is not an integer\n");
        goto error_return;
    }
    t = lex_scan();
    if (!expect_token(t, ')', NULL, TRUE, 7)) goto error_return;
    t = lex_scan(); // pwr
    if (!expect_token(t, LEX_ID, NULL, TRUE, 8)) goto error_return;
    t = lex_scan(); // gnd
    if (!expect_token(t, LEX_ID, NULL, TRUE, 9)) goto error_return;
    /* num_ins input ids */
    for (i = 0; i < num_ins; i++) {
        t = lex_scan();
        if (!expect_token(t, LEX_ID, NULL, TRUE, 10)) goto error_return;
        (void) add_sym_tab_entry(parse_lexer->lexer_buf,
            SYM_INPUT, &parse_lexer->lexer_sym_tab);
        u_remember_pin(parse_lexer->lexer_buf, 1);
    }
    /* num_outs output ids */
    for (i = 0; i < num_outs; i++) {
        t = lex_scan();
        if (!expect_token(t, LEX_ID, NULL, TRUE, 11)) goto error_return;
        (void) add_sym_tab_entry(parse_lexer->lexer_buf,
            SYM_OUTPUT, &parse_lexer->lexer_sym_tab);
        u_remember_pin(parse_lexer->lexer_buf, 2);
    }
    /* timing model */
    t = lex_scan();
    if (!expect_token(t, LEX_ID, NULL, TRUE, 12)) goto error_return;
    if (!eq(parse_lexer->lexer_buf, "d0_gate")) {
        u_add_logicexp_model(parse_lexer->lexer_buf,
            "d_and", "dxspice_dly_and");
        u_add_logicexp_model(parse_lexer->lexer_buf,
            "d_nand", "dxspice_dly_nand");
        u_add_logicexp_model(parse_lexer->lexer_buf,
            "d_or", "dxspice_dly_or");
        u_add_logicexp_model(parse_lexer->lexer_buf,
            "d_nor", "dxspice_dly_nor");
        u_add_logicexp_model(parse_lexer->lexer_buf,
            "d_xor", "dxspice_dly_xor");
        u_add_logicexp_model(parse_lexer->lexer_buf,
            "d_xnor", "dxspice_dly_xnor");
        u_add_logicexp_model(parse_lexer->lexer_buf,
            "d_buffer", "dxspice_dly_buffer");
        u_add_logicexp_model(parse_lexer->lexer_buf,
            "d_inverter", "dxspice_dly_inverter");
        use_tmodel_delays = TRUE;
    } else {
        use_tmodel_delays = FALSE;
    }
    (void) add_sym_tab_entry(parse_lexer->lexer_buf,
        SYM_TMODEL, &parse_lexer->lexer_sym_tab);
    ret_val = bparse(line, FALSE);

    current_lexer = NULL;
    if (!ret_val) {
        fprintf(stderr, "ERROR parsing logicexp\n");
        fprintf(stderr, "ERROR in \"%s\"\n", line);
        cleanup_parser();
    }
    return ret_val;

error_return:
    delete_lexer(parse_lexer);
    current_lexer = NULL;
    return FALSE;
}

/* Start of f_pindly which is called from udevices.c 
   See the PSpice reference which describes the PINDLY statement syntax.

   NOTE that only two sections, PINDLY: and TRISTATE:, are considered.
   Typical delays are estimated from the DELAY(...) functions.
   XSPICE does not have the variety of delays that PSpice supports.
   Output buffers and tristate buffers are generated.
*/
/* C++ with templates would generalize the different TABLEs */
typedef struct pindly_line *PLINE;
struct pindly_line {
    char *in_name;
    char *out_name;
    char *ena_name;
    char *delays;
    PLINE next;
};

typedef struct pindly_table *PINTABLE;
struct pindly_table {
    PLINE first;
    PLINE last;
    int num_entries;
};

static PINTABLE new_pindly_table(void)
{
    PINTABLE pint;
    pint = TMALLOC(struct pindly_table, 1);
    pint->first = pint->last = NULL;
    pint->num_entries = 0;
    return pint;
}

static int num_pindly_entries(PINTABLE pint)
{
    if (!pint)
        return 0;
    else
        return pint->num_entries;
}

static PLINE new_pindly_line(void)
{
    PLINE p = NULL;
    p = TMALLOC(struct pindly_line, 1);
    p->in_name = p->out_name = p->ena_name = p->delays = NULL;
    return p;
}

static PLINE add_new_pindly_line(PINTABLE pint)
{
    PLINE p;
    p = new_pindly_line();
    p->next = NULL;
    if (!pint->first) {
        pint->first = pint->last = p;
    } else {
        pint->last->next = p;
        pint->last = p;
    }
    pint->num_entries++;
    return p;
}

static PLINE set_in_name(char *s, PLINE p)
{
    if (p->in_name) tfree(p->in_name);
    p->in_name = TMALLOC(char, (strlen(s) + 1));
    strcpy(p->in_name, s);
    return p;
}

static PLINE set_out_name(char *s, PLINE p)
{
    if (p->out_name) tfree(p->out_name);
    p->out_name = TMALLOC(char, (strlen(s) + 1));
    strcpy(p->out_name, s);
    return p;
}

static PLINE set_ena_name(char *s, PLINE p)
{
    if (p->ena_name) tfree(p->ena_name);
    p->ena_name = TMALLOC(char, (strlen(s) + 1));
    strcpy(p->ena_name, s);
    return p;
}

static PLINE set_delays(char *s, PLINE p)
{
    if (p->delays) tfree(p->delays);
    p->delays = TMALLOC(char, (strlen(s) + 1));
    strcpy(p->delays, s);
    return p;
}

static void delete_pindly_table(PINTABLE pint)
{
    PLINE p, next;
    if (!pint)
        return;
    next = pint->first;
    while (next) {
        p = next;
        if (p->in_name) tfree(p->in_name);
        if (p->out_name) tfree(p->out_name);
        if (p->ena_name) tfree(p->ena_name);
        if (p->delays) tfree(p->delays);
        next = p->next;
        tfree(p);
    }
    tfree(pint);
}

static void print_pindly_table(PINTABLE pint)
{
    PLINE p, next;
    if (!pint)
        return;
    printf("num_entries %d\n", pint->num_entries);
    next = pint->first;
    while (next) {
        p = next;
        printf("in_name \"%s\"", p->in_name);
        printf(" out_name \"%s\"", p->out_name);
        printf(" ena_name \"%s\"", p->ena_name);
        printf(" delays \"%s\"\n", p->delays);
        next = p->next;
    }
}

static PLINE nth_pindly_entry(PINTABLE pint, int n)
{
    /* Entries are from 0 to num_entries - 1 */
    PLINE p, next;
    int count = 0;
    if (n < 0) return NULL;
    if (n > pint->num_entries - 1) return NULL;
    next = pint->first;
    while (next) {
        p = next;
        if (count == n) return p;
        count++;
        next = p->next;
    }
    return NULL;
}

static PLINE find_pindly_out_name(PINTABLE pint, char *name)
{
    PLINE p, next;
    if (!pint) return NULL;
    next = pint->first;
    while (next) {
        p = next;
        if (eq(p->out_name, name)) return p;
        next = p->next;
    }
    return NULL;
}

static PINTABLE pindly_tab = NULL;

static void init_pindly_tab(void)
{
    pindly_tab = new_pindly_table();
}

static void cleanup_pindly_tab(void)
{
    delete_pindly_table(pindly_tab);
    pindly_tab = NULL;
}

static void gen_pindly_buffers(void)
{
    DS_CREATE(dbuf, 128);
    PLINE pline = NULL;
    BOOL prit = PRINT_ALL;

    if (prit) { print_pindly_table(pindly_tab); }
    pline = pindly_tab->first;
    while (pline) {
        char *iname = NULL;
        ds_clear(&dbuf);
        iname = get_inst_name();
        if (pline->ena_name && strlen(pline->ena_name) > 0) {
            ds_cat_printf(&dbuf, "%s %s %s %s d_tristate_buf_%s", iname,
                pline->in_name, pline->ena_name, pline->out_name, iname);
        } else {
            ds_cat_printf(&dbuf, "%s %s %s d_pindly_buf_%s", iname,
                pline->in_name, pline->out_name, iname);
        }
        u_add_instance(ds_get_buf(&dbuf));
        ds_clear(&dbuf);
        if (pline->ena_name && strlen(pline->ena_name) > 0) {
            ds_cat_printf(&dbuf, ".model d_tristate_buf_%s d_tristate%s",
                iname, pline->delays);
        } else {
            ds_cat_printf(&dbuf, ".model d_pindly_buf_%s d_buffer%s",
                iname, pline->delays);
        }
        u_add_instance(ds_get_buf(&dbuf));
        pline = pline->next;
    }
    ds_free(&dbuf);
}

static char *get_typ_estimate(char *min, char *typ, char *max, DSTRING *pds)
{
    char *tmpmax = NULL, *tmpmin = NULL;
    float valmin, valmax, average;
    char *units1, *units2;

    ds_clear(pds);
    if (typ && strlen(typ) > 0 && typ[0] != '-') {
        ds_cat_str(pds, typ);
        return ds_get_buf(pds);
    }
    if (max && strlen(max) > 0 && max[0] != '-') {
        tmpmax = max;
    }
    if (min && strlen(min) > 0 && min[0] != '-') {
        tmpmin = min;
    }
    if (tmpmin && tmpmax) {
        if (strlen(tmpmin) > 0 && strlen(tmpmax) > 0) {
            valmin = strtof(tmpmin, &units1);
            valmax = strtof(tmpmax, &units2);
            average = (valmin + valmax) / (float)2.0;
            ds_cat_printf(pds, "%.2f%s", average, units2);
            if (!eq(units1, units2)) {
                printf("WARNING units do not match\n");
            }
            return ds_get_buf(pds);
        }
    } else if (tmpmax && strlen(tmpmax) > 0) {
        ds_cat_str(pds, tmpmax);
        return ds_get_buf(pds);
    } else if (tmpmin && strlen(tmpmin) > 0) {
        ds_cat_str(pds, tmpmin);
        return ds_get_buf(pds);
    } else {
        return NULL;
    }
    return NULL;
}

static char *typical_estimate(char *delay_str, DSTRING *pds)
{
    /* Input string (t1,t2,t2) */
    int which = 0;
    size_t i, slen;
    char *s;
    DS_CREATE(dmin, 32);
    DS_CREATE(dtyp, 32);
    DS_CREATE(dmax, 32);

    ds_clear(&dmin);
    ds_clear(&dtyp);
    ds_clear(&dmax);
    slen = strlen(delay_str) - 1;
    for (i = 1; i < slen; i++) {
        if (delay_str[i] == ',') {
            which++;
            continue;
        }
        switch (which) {
        case 0:
            ds_cat_char(&dmin, delay_str[i]);
            break;
        case 1:
            ds_cat_char(&dtyp, delay_str[i]);
            break;
        case 2:
            ds_cat_char(&dmax, delay_str[i]);
            break;
        default:
            break;
        }
    }
    s = get_typ_estimate(ds_get_buf(&dmin), ds_get_buf(&dtyp),
        ds_get_buf(&dmax), pds);
    ds_free(&dmin);
    ds_free(&dtyp);
    ds_free(&dmax);
    return s;
}

static BOOL extract_delay(
    LEXER lx, int val, PLINE *pline_arr, int idx, BOOL tri)
{
    /* NOTE: The delays are specified in a DELAY(t1,t2,t3) function.
       Beware if the format of t1, t2, t3 changes!
       Expect t1, t2, t3:
          -1 or x.y[time_unit] or w[time_unit]
       where the time_unit is ns, ps, etc. and the same for t1, t2, t3;
       x.y represents a decimal number; w is an integer.
       Either numbers can have more that one digit.
    */
    BOOL in_delay = FALSE, ret_val = TRUE;
    int i;
    BOOL prit = PRINT_ALL;
    float typ_max_val = 0.0, typ_val = 0.0;
    char *units;
    DS_CREATE(dly, 64);
    DS_CREATE(dtyp_max_str, 16);
    DS_CREATE(tmp_ds, 128);

    if (val != '=') {
        ds_free(&dly);
        ds_free(&dtyp_max_str);
        ds_free(&tmp_ds);
        return FALSE;
    }
    val = lexer_scan(lx);
    if (val != '{') {
        ds_free(&dly);
        ds_free(&dtyp_max_str);
        ds_free(&tmp_ds);
        return FALSE;
    }
    val = lexer_scan(lx);
    while (val != '}') {
        if (val == LEX_ID) {
            if (eq(lx->lexer_buf, "delay")) {
                in_delay = TRUE;
                ds_clear(&dly);
            } else {
                if (in_delay) {
                    ds_cat_printf(&dly, "%s", lx->lexer_buf);
                }
            }
        } else {
            if (in_delay) {
                DS_CREATE(delay_string, 64);
                ds_cat_printf(&dly, "%c", val);
                if (val == ')') {
                    char *tmps;
                    ds_clear(&tmp_ds);
                    in_delay = FALSE;
                    tmps = typical_estimate(ds_get_buf(&dly), &tmp_ds);
                    if (!tmps) {
                        ret_val = FALSE;
                        ds_clear(&tmp_ds);
                        break;
                    }
                    if (prit) {
                        printf("%s\n", ds_get_buf(&dly));
                        printf("estimate \"%s\"\n", tmps);
                    }
                    typ_val = strtof(tmps, &units);
                    if (typ_val > typ_max_val) {
                        ds_clear(&delay_string);
                        ds_clear(&dtyp_max_str);
                        ds_cat_str(&dtyp_max_str, tmps);
                        typ_max_val = typ_val;
                        if (ds_get_length(&dtyp_max_str) > 0) {
                            if (tri) {
                                ds_cat_printf(&delay_string,
                                    "(inertial_delay=true delay=%s)",
                                    ds_get_buf(&dtyp_max_str));
                            } else {
                                ds_cat_printf(&delay_string,
                                    "(inertial_delay=true rise_delay=%s fall_delay=%s)",
                                    ds_get_buf(&dtyp_max_str),
                                    ds_get_buf(&dtyp_max_str));
                            }
                        } else {
                            printf("WARNING pindly DELAY not found\n");
                            if (tri) {
                                ds_cat_printf(&delay_string,
                                    "(inertial_delay=true delay=10ns)");
                            } else {
                                ds_cat_printf(&delay_string,
                                    "(inertial_delay=true rise_delay=10ns fall_delay=10ns)");
                            }
                        }
                        for (i = 0; i < idx; i++) {
                            (void) set_delays(
                                ds_get_buf(&delay_string),
                                pline_arr[i]);
                        }
                    }
                }
                ds_free(&delay_string);
            } // end if in_delay
        }
        val = lexer_scan(lx);
    } // end while != '}'
    ds_free(&dly);
    ds_free(&dtyp_max_str);
    ds_free(&tmp_ds);
    return ret_val;
}

static BOOL new_gen_output_models(LEXER lx)
{
    int val, arrlen = 0, idx = 0, i;
    BOOL in_pindly = FALSE, in_tristate = FALSE;
    BOOL prit = PRINT_ALL;
    DS_CREATE(enable_name, 64);
    DS_CREATE(last_enable, 64);
    PLINE pline = NULL;
    PLINE *pline_arr = NULL;

    arrlen = num_pindly_entries(pindly_tab);
    if (arrlen <= 0) {
        ds_free(&enable_name);
        ds_free(&last_enable);
        return FALSE;
    }
    pline_arr = TMALLOC(PLINE, arrlen);
    ds_clear(&last_enable);
    val = lexer_scan(lx);
    while (val != 0) { // Outer while loop
        if (val == LEX_ID) {
            if (eq(lx->lexer_buf, "pindly")) {
                in_pindly = TRUE;
                in_tristate = FALSE;
                val = lexer_scan(lx);
                if (val != ':') {
                    goto err_return;
                }
            } else if (eq(lx->lexer_buf, "tristate")) {
                in_pindly = FALSE;
                in_tristate = TRUE;
                val = lexer_scan(lx);
                if (val != ':') {
                    goto err_return;
                }
            } else if (eq(lx->lexer_buf, "setup_hold")
                || eq(lx->lexer_buf, "width")
                || eq(lx->lexer_buf, "freq")
                || eq(lx->lexer_buf, "general")) {
                in_pindly = FALSE;
                in_tristate = FALSE;
            }
        }
        if (in_pindly && val == LEX_ID) { // start in_pindly and LEX_ID
            while (val == LEX_ID) {
                if (prit) { printf("pindly out \"%s\"\n", lx->lexer_buf); }
                pline = find_pindly_out_name(pindly_tab, lx->lexer_buf);
                if (pline) {
                    pline_arr[idx++] = pline;
                } else {
                    goto err_return;
                }
                val = lexer_scan(lx);
            }
            if (!extract_delay(lx, val, pline_arr, idx, FALSE)) goto err_return;
            for (i = 0; i < arrlen; i++) {
                pline_arr[i] = NULL;
            }
            idx = 0; // end in_pindly and LEX_ID
        } else if (in_tristate && val == LEX_ID) {
        // start in_tristate and LEX_ID
            if (eq(lx->lexer_buf, "enable")) {
                val = lexer_scan(lx);
                if (val == LEX_ID && (eq(lx->lexer_buf, "hi")
                        || eq(lx->lexer_buf, "lo"))) {
                    BOOL invert = FALSE;
                    if (eq(lx->lexer_buf, "lo"))
                        invert = TRUE;
                    if (prit) { printf("tristate enable %s ", lx->lexer_buf); }
                    val = lexer_scan(lx);
                    if (val != '=') {
                        goto err_return;
                    }
                    val = lexer_scan(lx);
                    if (val != LEX_ID) {
                        goto err_return;
                    }
                    if (prit) { printf("ena \"%s\"\n", lx->lexer_buf); }
                    ds_clear(&enable_name);
                    if (invert)
                        ds_cat_char(&enable_name, '~');
                    ds_cat_str(&enable_name, lx->lexer_buf);
                } else {
                    goto err_return;
                }
                ds_clear(&last_enable);
                ds_cat_ds(&last_enable, &enable_name);
                val = lexer_scan(lx);
                if (val != LEX_ID) {
                    goto err_return;
                }
            } else if (ds_get_length(&last_enable) > 0) {
                ds_clear(&enable_name);
                ds_cat_ds(&enable_name, &last_enable);
            } else {
                goto err_return;
            }
            while (val == LEX_ID) {
                if (prit) { printf("tristate out \"%s\"\n", lx->lexer_buf); }
                pline = find_pindly_out_name(pindly_tab, lx->lexer_buf);
                if (pline) {
                    pline_arr[idx++] = pline;
                    (void) set_ena_name(ds_get_buf(&enable_name), pline);
                    u_remember_pin(lx->lexer_buf, 3);
                } else {
                    goto err_return;
                }
                val = lexer_scan(lx);
            }
            if (!extract_delay(lx, val, pline_arr, idx, TRUE)) goto err_return;
            for (i = 0; i < arrlen; i++) {
                pline_arr[i] = NULL;
            }
            idx = 0; // end of in_tristate and LEX_ID
        }
        val = lexer_scan(lx);
    } // end of outer while loop
    ds_free(&enable_name);
    ds_free(&last_enable);
    tfree(pline_arr);
    return TRUE;

err_return:
    ds_free(&enable_name);
    ds_free(&last_enable);
    tfree(pline_arr);
    return FALSE;
}

BOOL f_pindly(char *line)
{
    int t, num_ios = 0, num_ena = 0, num_refs = 0, i;
    char *endp;
    LEXER lxr;
    PLINE pline = NULL;

    init_pindly_tab();

    lxr = new_lexer(line);
    current_lexer = lxr;
    t = lexer_scan(lxr); // U*
    if (!expect_token(t, LEX_ID, NULL, TRUE, 50)) goto error_return;

    /* pindly ( int , int, int ) */
    t = lexer_scan(lxr);
    if (!expect_token(t, LEX_ID, "pindly", TRUE, 51)) goto error_return;

    t = lexer_scan(lxr);
    if (!expect_token(t, '(', NULL, TRUE, 52)) goto error_return;

    t = lexer_scan(lxr);
    if (!expect_token(t, LEX_ID, NULL, TRUE, 53)) goto error_return;
    if (lex_all_digits(lxr->lexer_buf)) {
        num_ios = (int) strtol(lxr->lexer_buf, &endp, 10);
    } else {
        fprintf(stderr, "ERROR pindly io count is not an integer\n");
        goto error_return;
    }

    t = lexer_scan(lxr);
    if (!expect_token(t, ',', NULL, TRUE, 54)) goto error_return;

    t = lexer_scan(lxr);
    if (!expect_token(t, LEX_ID, NULL, TRUE, 55)) goto error_return;
    if (lex_all_digits(lxr->lexer_buf)) {
        num_ena = (int) strtol(lxr->lexer_buf, &endp, 10);
    } else {
        fprintf(stderr, "ERROR pindly enable count is not an integer\n");
        goto error_return;
    }

    t = lexer_scan(lxr);
    if (!expect_token(t, ',', NULL, TRUE, 56)) goto error_return;

    t = lexer_scan(lxr);
    if (!expect_token(t, LEX_ID, NULL, TRUE, 57)) goto error_return;
    if (lex_all_digits(lxr->lexer_buf)) {
        num_refs = (int) strtol(lxr->lexer_buf, &endp, 10);
    } else {
        fprintf(stderr, "ERROR pindly refs count is not an integer\n");
        goto error_return;
    }

    t = lexer_scan(lxr);
    if (!expect_token(t, ')', NULL, TRUE, 58)) goto error_return;

    t = lexer_scan(lxr); // pwr
    if (!expect_token(t, LEX_ID, NULL, TRUE, 59)) goto error_return;
    t = lexer_scan(lxr); // gnd
    if (!expect_token(t, LEX_ID, NULL, TRUE, 60)) goto error_return;

    /* num_ios input ids */
    for (i = 0; i < num_ios; i++) {
        t = lexer_scan(lxr);
        if (!expect_token(t, LEX_ID, NULL, TRUE, 61)) goto error_return;
        pline = add_new_pindly_line(pindly_tab);
        (void) set_in_name(lxr->lexer_buf, pline);
        u_remember_pin(lxr->lexer_buf, 1);
    }

    /* num_ena enable nodes which are ignored */
    /* num_refs reference nodes which are ignored */
    for (i = 0; i < num_ena + num_refs; i++) {
        t = lexer_scan(lxr);
        if (!expect_token(t, LEX_ID, NULL, TRUE, 62)) goto error_return;
        if (i < num_ena) {
            u_remember_pin(lxr->lexer_buf, 1);
        }
    }
    /* num_ios output ids */
    pline = NULL;
    for (i = 0; i < num_ios; i++) {
        t = lexer_scan(lxr);
        if (!expect_token(t, LEX_ID, NULL, TRUE, 63)) goto error_return;
        if (i == 0)
            pline = nth_pindly_entry(pindly_tab, i);
        else
            pline = pline->next;
        (void) set_out_name(lxr->lexer_buf, pline);
        u_remember_pin(lxr->lexer_buf, 2);
    }

    if (!new_gen_output_models(lxr)) {
        fprintf(stderr, "ERROR generating models for pindly\n");
        fprintf(stderr, "ERROR in \"%s\"\n", line);
        goto error_return;;
    }
    gen_pindly_buffers();
    delete_lexer(lxr);
    cleanup_pindly_tab();
    current_lexer = NULL;
    return TRUE;

error_return:
    delete_lexer(lxr);
    cleanup_pindly_tab();
    current_lexer = NULL;
    return FALSE;
}

