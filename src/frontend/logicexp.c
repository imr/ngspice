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
        printf("%p --> \n", t);
    printf("\"%s\"    %d  ref_count=%d", t->name, t->attribute, t->ref_count);
    if (t->alias)
        printf("  alias = \"%s\"", t->alias);
    printf("\n");
    print_sym_tab(t->right, with_addr);
}
/* End of btree symbol table */

/* Start of lexical scanner */
#include <assert.h>
#define LEX_ID 256
#define LEX_OTHER 257
#define LEX_BUF_SZ 512

typedef struct lexer *LEXER;
struct lexer {
    char lexer_buf[LEX_BUF_SZ];
    char *lexer_line;
    int lexer_pos;
    int lexer_back;
    SYM_TAB lexer_sym_tab;
};

static LEXER parse_lexer = NULL;

static LEXER new_lexer(char *line)
{
    LEXER lx;
    lx = TMALLOC(struct lexer, 1);
    lx->lexer_line = TMALLOC(char, (strlen(line) + 1));
    strcpy(lx->lexer_line, line);
    lx->lexer_pos = lx->lexer_back = 0;
    lx->lexer_buf[0] = '\0';
    lx->lexer_sym_tab = NULL;
    return lx;
}

static void delete_lexer(LEXER lx)
{
    if (!lx)
        return;
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
    lx->lexer_pos = pos - &lx->lexer_line[0];
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
    if (isalnum(c) || c == '_' || c == '/' || c == '-')
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
            int i = 0;
            while (lex_ident(c)) {
                lx->lexer_buf[i] = c;
                assert(i < LEX_BUF_SZ);
                i++;
                c = lexer_getchar(lx);
            }
            assert(i < LEX_BUF_SZ);
            lx->lexer_buf[i] = '\0';
            if (c != '\0')
                lexer_putback(lx);
            return LEX_ID;
        } else {
            lx->lexer_buf[0] = c;
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
    int i;
    if (!str || strlen(str) < 1)
        return FALSE;
    for (i = 0; i < strlen(str); i++) {
        if (!isdigit(str[i])) return FALSE;
    }
    return TRUE;
}
/* End of lexical scanner */

/* Start parse table */
typedef struct table_line *TLINE;
struct table_line {
    char *line;
    TLINE next;
};

typedef struct parse_table *PTABLE;
struct parse_table {
    TLINE first;
    TLINE last;
};

static PTABLE parse_tab = NULL;
static PTABLE gen_tab = NULL;

static PTABLE new_parse_table(void)
{
    PTABLE pt;
    pt = TMALLOC(struct parse_table, 1);
    pt->first = pt->last = NULL;
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
    return t;
}

static TLINE ptab_add_line(char *line, BOOL ignore_blank)
{
    TLINE t;
    t = add_to_parse_table(parse_tab, line, ignore_blank);
    return t;
}

static TLINE gen_tab_add_line(char *line, BOOL ignore_blank)
{
    TLINE t;
    t = add_to_parse_table(gen_tab, line, ignore_blank);
    return t;
}

static char *get_temp_from_line(char *line, BOOL begin)
{
    /* First occurrence of "tmp" on the line */
    /* If begin is TRUE then "tmp" must be at the start of line */
    static char lbuf[64];
    char *p, *q;
    int j;
    p = strstr(line, "tmp");
    if (!p)
        return NULL;
    if (begin && p != line)
        return NULL;
    for (q = p, j = 0; isalnum(q[j]) || q[j] == '_'; j++) {
        if (j >= 63)
            return NULL;
        lbuf[j] = q[j];
    }
    lbuf[j] = '\0';
    return lbuf;
}

static char *find_temp_begin(char *line)
{
    return get_temp_from_line(line, TRUE);
}

static char *find_temp_anywhere(char *line)
{
    return get_temp_from_line(line, FALSE);
}

static int get_temp_depth(char *line)
{
    char buf[64];
    char *p, *endp;
    int depth;
    p = find_temp_anywhere(line);
    if (p) {
        strcpy(buf, p);
        p = strstr(buf + strlen("tmp"), "__");
        if (p) {
            p = p + 2;
            depth = (int) strtol(p, &endp, 10);
            return depth;
        }
    }
    return -1;
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
/* End parse table */

/* Start of logicexp parser */
static char *get_inst_name(void);
static char *get_inverter_output_name(char *input);
static void gen_inverters(SYM_TAB t);
static void aerror(char *s);
static void amatch(int t);
static void bexpr(void);
static void bfactor(void);
static void bparse(char *line, BOOL new_lexer);

static int lookahead = 0;
static int adepth = 0;
static int max_adepth = 0;
static DSTRING d_curr_line;

static char *get_inst_name(void)
{
    static char name[64];
    static int number = 0;
    number++;
    (void) sprintf(name, "a_%d", number);
    return name;
}

static char *get_inverter_output_name(char *input)
{
    static char buf[LEX_BUF_SZ];
    LEXER lx = parse_lexer;
    // FIX ME keep this name in the symbol table to ensure uniqueness
    (void) sprintf(buf, "inv_out__%s", input);
    if (member_sym_tab(buf, lx->lexer_sym_tab))
        printf("ERROR %s is already in use\n", buf);
    return buf;
}

static char *get_inv_tail(char *str)
{
    static char lbuf[64];
    char *p = NULL, *q = NULL;
    int j;
    size_t slen = strlen("inv_out__");

    p = strstr(str, "inv_out__");
    if (!p)
        return NULL;
    for (q = p + slen, j = 0; q[j] != '\0' && !isspace(q[j]); j++) {
        if (j >= 63)
            return NULL;
        lbuf[j] = q[j];
    }
    lbuf[j] = '\0';
    return lbuf;
}

static void gen_inverters(SYM_TAB t)
{
    DS_CREATE(instance, 128);
    if (t == NULL)
        return;
    gen_inverters(t->left);
    if (t->attribute & SYM_INVERTER) {
        if (t->ref_count >= 1) {
            printf("%s %s %s d_inv_zero_delay\n", get_inst_name(),
                t->name, get_inverter_output_name(t->name));

            ds_clear(&instance);
            ds_cat_printf(&instance, "%s %s %s d_inv_zero_delay",
                get_inst_name(), t->name, get_inverter_output_name(t->name));
            u_add_instance(ds_get_buf(&instance));
        }
    }
    ds_free(&instance);
    gen_inverters(t->right);
}

static void gen_models(void)
{
    DS_CREATE(model, 64);

    printf(".model d_inv_zero_delay d_inverter\n");
    printf(".model d__inverter__1 d_inverter\n");
    printf(".model d__buffer__1 d_buffer\n");
    printf(".model d__nand__1 d_nand\n");
    printf(".model d__and__1 d_and\n");
    printf(".model d__xnor__1 d_xnor\n");
    printf(".model d__xor__1 d_xor\n");
    printf(".model d__nor__1 d_nor\n");
    printf(".model d__or__1 d_or\n");

    ds_clear(&model);
    ds_cat_printf(&model, ".model d_inv_zero_delay d_inverter");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model, ".model d__inverter__1 d_inverter");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model, ".model d__buffer__1 d_buffer");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model, ".model d__nand__1 d_nand");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model, ".model d__and__1 d_and");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model, ".model d__xnor__1 d_xnor");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model, ".model d__xor__1 d_xor");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model, ".model d__nor__1 d_nor");
    u_add_instance(ds_get_buf(&model));

    ds_clear(&model);
    ds_cat_printf(&model, ".model d__or__1 d_or");
    u_add_instance(ds_get_buf(&model));

    ds_free(&model);
}

static void aerror(char *s)
{
    LEXER lx = parse_lexer;
    printf("%s [%s]\n", s, lx->lexer_line + lx->lexer_pos);
    exit(1);
}

char *get_temp_name(void)
{
    static char name[64];
    static int number = 0;
    number++;
    (void) sprintf(name, "tmp%d", number);
    return name;
}

static void amatch(int t)
{
    LEXER lx = parse_lexer;
    if (lookahead == t) {
        lookahead = lex_scan();
    } else {
        printf("t = '%c' [%d] lookahead = '%c' [%d] lexer_buf %s\n",
            t, t, lookahead, lookahead, lx->lexer_buf);
        aerror("amatch: syntax error");
    }
}

static void bfactor(void)
{
    /* factor is : [~] (optional) rest
       where rest is: id (input_name) | ( expr ) | error
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
            ds_cat_printf(&d_curr_line, "%s ",
                get_inverter_output_name(lx->lexer_buf));
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
        (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE);
        ds_clear(&d_curr_line);
        ds_cat_printf(&d_curr_line, "%s__%d <- ", ds_get_buf(&tmpnam), adepth);

        if (is_not) {
            ds_cat_printf(&d_curr_line, "~ %c", lookahead);
        } else {
            ds_cat_printf(&d_curr_line, "%c", lookahead);
        }
        (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE);
        ds_clear(&d_curr_line);

        lookahead = lex_scan();
        bexpr();

        (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE);
        ds_clear(&d_curr_line);

        ds_cat_printf(&d_curr_line, "%c -> %s__%d", lookahead,
            ds_get_buf(&tmpnam), adepth);
        (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE);
        ds_clear(&d_curr_line);

        amatch(')');
        ds_free(&tmpnam);

    } else {
        aerror("bfactor: syntax error");
    }
    adepth--;
}

static void bexpr(void)
{
    /* expr is: factor { gate_op factor } (0 or more times). */
    bfactor();

    while (lex_gate_op(lookahead)) {
        ds_cat_printf(&d_curr_line, "%c ", lookahead);

        lookahead = lex_scan();
        bfactor();
    }
}

static int bstmt(void)
{
    /* A stmt is: output_name = { expr } */
    int end_pos;
    SYM_TAB entry = NULL;
    LEXER lx = parse_lexer;
    DS_CREATE(tname, 64);
    DS_CREATE(assign, LEX_BUF_SZ);

    if (lookahead == LEX_ID) {
        entry = add_sym_tab_entry(lx->lexer_buf, SYM_ID, &lx->lexer_sym_tab);
    } else {
        aerror("bstmt: syntax error");
    }

    adepth++;
    if (adepth > max_adepth)
        max_adepth = adepth;

    amatch(LEX_ID);
    amatch('=');

    ds_clear(&assign);
    ds_cat_printf(&assign, "%s =", entry->name);
    (void) ptab_add_line(ds_get_buf(&assign), TRUE);

    amatch('{');

    ds_clear(&tname);
    ds_cat_str(&tname, get_temp_name());
    ds_cat_printf(&d_curr_line, "%s__%d <- (", ds_get_buf(&tname), adepth);
    (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE);
    ds_clear(&d_curr_line);

    bexpr();

    end_pos = lx->lexer_pos;
    if (ds_get_length(&d_curr_line) > 0) {
        (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE);
    }
    ds_clear(&d_curr_line);
    ds_cat_printf(&d_curr_line, ") -> %s__%d", ds_get_buf(&tname), adepth);
    (void) ptab_add_line(ds_get_buf(&d_curr_line), TRUE);
    ds_clear(&d_curr_line);

    amatch('}');

    ds_free(&assign);
    ds_free(&tname);
    adepth--;
    return end_pos;
}

static PTABLE optimize_gen_tab(PTABLE pt)
{
    /* This function compacts the gen_tab, returning a new PTABLE.
       Aliases are transformed and removed as described below.
       Usually, optimized_gen_tab is called a second time on the
       PTABLE created by the first call. The algorithm here will
       only transform one level of aliases.
    */
    TLINE t = NULL;
    LEXER lxr = NULL;
    int val, idnum = 0, tok_count = 0;
    SYM_TAB entry = NULL, alias_tab = NULL;
    BOOL found_tilde = FALSE, starts_with_temp = FALSE;
    PTABLE new_gen = NULL;
    DS_CREATE(scratch, LEX_BUF_SZ);
    DS_CREATE(alias, 64);
    DS_CREATE(non_tmp_name, 64);
    DS_CREATE(tmp_name, 64);

    if (!pt || !pt->first)
        return NULL;
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
        if (find_temp_begin(t->line))
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
                assert(tok_count == 3);
            } else if (val == '=') {
                assert(tok_count == 2);
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
    ds_free(&alias);
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
        if (find_temp_begin(t->line))
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
                        if (!find_temp_begin(lxr->lexer_buf))
                            ds_cat_str(&non_tmp_name, lxr->lexer_buf);
                    } else if (tok_count == 3) {
                        if (ds_get_length(&non_tmp_name) > 0) {
                            char *str1 = NULL;
                            str1 = find_temp_begin(lxr->lexer_buf);
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
                assert(val != LEX_OTHER);
                if (val == '~') 
                    found_tilde = TRUE;
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
    ds_free(&scratch);
    ds_free(&non_tmp_name);
    ds_free(&tmp_name);
    delete_lexer(lxr);
    {
        int print_it = 0;
        if (print_it)
            print_sym_tab(alias_tab, FALSE);
    }
    delete_sym_tab(alias_tab);

    return new_gen;
}

static void gen_gates(PTABLE gate_tab, SYM_TAB parser_symbols)
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
    DS_CREATE(out_name, 64);
    DS_CREATE(in_names, 64);
    DS_CREATE(gate_name, 64);
    DS_CREATE(instance, 128);

    if (!gate_tab || !gate_tab->first)
        return;
    t = gate_tab->first;
    lxr = new_lexer(t->line);
    while (t) {
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
        while (val != '\0') {
            tok_count++;
            if (val == LEX_ID) {
                idnum++;
                if (idnum == 1) { //output name
                    ds_cat_str(&out_name, lxr->lexer_buf);
                } else { // input name
                    in_count++;
                    ds_cat_printf(&in_names, " %s", lxr->lexer_buf);
                }
            } else if (val == '~') {
                found_tilde = TRUE;
                assert(tok_count == 3);
            } else if (val == '=') {
                assert(tok_count == 2);
            } else if (lex_gate_op(val)) {
                if (gate_op != 0)
                   assert(val == gate_op);
                gate_op = val;
            } else {
                assert(FALSE);
            }
            val = lexer_scan(lxr);
        }
        if (in_count == 1) { // buffer or inverter
            assert(gate_op == 0);
            ds_cat_str(&gate_name, lex_gate_name('~', found_tilde));
        } else if (in_count >= 2) { // AND, OR. XOR and inverses
            assert(gate_op != 0);
            ds_cat_str(&gate_name, lex_gate_name(gate_op, found_tilde));
        } else {
            assert(FALSE);
        }
        ds_cat_printf(&instance, "%s ", get_inst_name());
        if (in_count == 1) {
            /* If the input name is inv_out_<tail> use the <tail>
               and instantiate an inverter to avoid an extra buffer.
            */
            char *tail = NULL;
            SYM_TAB ent;
            tail = get_inv_tail(ds_get_buf(&in_names));
            if (tail && strlen(tail) > 0) {
                ds_clear(&gate_name);
                ds_cat_str(&gate_name, lex_gate_name('~', TRUE));
                ds_cat_printf(&instance, "%s %s ", tail,
                    ds_get_buf(&out_name));
                ent = member_sym_tab(tail, parser_symbols);
                assert(ent);
                assert(ent->attribute & SYM_INVERTER);
                ent->ref_count--;
            } else {
                ds_cat_printf(&instance, "%s %s ", ds_get_buf(&in_names),
                    ds_get_buf(&out_name));
            }

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

        printf("%s\n", ds_get_buf(&instance));

        u_add_instance(ds_get_buf(&instance));
    }
    delete_lexer(lxr);
    ds_free(&out_name);
    ds_free(&in_names);
    ds_free(&gate_name);
    ds_free(&instance);
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

    s = find_temp_begin(t->line);
    if (!s)
        return;
    ds_clear(&other);
    ds_clear(&new_line);
    ds_clear(&this);
    ds_cat_str(&this, s);
    if (strstr(t->line + ds_get_length(&this), " ~ ")) {
        ds_cat_printf(&new_line, "%s =  ~ ", ds_get_buf(&this));
    } else {
        if (deep == 1)
            ds_cat_printf(&new_line, "%s ", parse_tab->first->line);
        else
            ds_cat_printf(&new_line, "%s = ", ds_get_buf(&this));
    }
    t = t->next;
    while (t) {
        s = find_temp_anywhere(t->line);
        if (s) {
            if (strcmp(ds_get_buf(&this), s) == 0) {
                break;
            } else {
                if (down == 0) {
                    s = find_temp_begin(t->line);
                    ds_clear(&other);
                    ds_cat_str(&other, s);
                    down = 1;
                    ds_cat_printf(&new_line, " %s", ds_get_buf(&other));
                } else if (down == 1) {
                    s = find_temp_anywhere(t->line);
                    if (strcmp(ds_get_buf(&other), s) == 0) {
                        down = 0;
                        ds_clear(&other);
                    }
                }
            }
        } else if (down == 0) {
            s = find_temp_anywhere(t->line);
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
                depth = get_temp_depth(t->line);
                if (depth >= 0) {
                    if (i == depth) {
                        bevaluate(t, i);
                    }
                }
            }
            t = t->next;
        }
    }
}

static void bparse(char *line, BOOL new_lexer)
{
    int start_pos = 0, end_pos = 0, stmt_num = 0;
    LEXER lx;
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
    assert(parse_lexer);
    lx = parse_lexer;
    lookahead = lex_set_start("logic:");
    lookahead = lex_scan(); // "logic"
    lookahead = lex_scan(); // ':'
    lookahead = lex_scan();
    while (lookahead != '\0') {
        init_parse_tables();
        adepth = max_adepth = 0;
        stmt_num++;
        start_pos = lx->lexer_pos;
        ds_clear(&stmt);
        ds_cat_str(&stmt, lx->lexer_buf);
        end_pos = bstmt();

        ds_cat_mem(&stmt, &lx->lexer_line[start_pos], end_pos - start_pos);
        printf("\n* Stmt(%d): %s\n\n", stmt_num, ds_get_buf(&stmt));

        beval_order();

        opt_tab1 = optimize_gen_tab(gen_tab);
        opt_tab2 = optimize_gen_tab(opt_tab1);
        if (opt_tab2) {
            gen_gates(opt_tab2, parse_lexer->lexer_sym_tab);
        }
        delete_parse_table(opt_tab1);
        delete_parse_table(opt_tab2);
        delete_parse_gen_tables();
    }

    ds_free(&d_curr_line);
    gen_inverters(lx->lexer_sym_tab);
    gen_models();
    ds_free(&stmt);
#define TRACE
#ifdef TRACE
    if (!new_lexer)
        print_sym_tab(lx->lexer_sym_tab, FALSE);
#endif
    delete_lexer(lx);
}
/* End of logicexp parser */

static BOOL expect_token(
    int tok, int expected_tok, char *expected_str, BOOL msg)
{
    if (tok != expected_tok) {
        if (msg) {
            printf("ERROR expect_token failed tok %d expected_tok %d\n",
                tok, expected_tok);
        }
        delete_lexer(parse_lexer);
        return FALSE;
    }
    if (tok == LEX_ID) {
        if (expected_str) {
            if (eq(expected_str, parse_lexer->lexer_buf))
                return TRUE;
            else {
                if (msg) {
                    printf(
                    "ERROR expect_token failed lexer_buf %s expected_str %s\n",
                        parse_lexer->lexer_buf, expected_str);
                }
                delete_lexer(parse_lexer);
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

    printf("\nf_logicexp: %s\n", line);
    lex_init(line);
    (void) add_sym_tab_entry("logic", SYM_KEY_WORD,
        &parse_lexer->lexer_sym_tab);
    t = lex_scan(); // U*
    if (!expect_token(t, LEX_ID, NULL, TRUE)) return FALSE;
    /* logicexp ( int , int ) */
    t = lex_scan();
    if (!expect_token(t, LEX_ID, "logicexp", TRUE)) return FALSE;
    t = lex_scan();
    if (!expect_token(t, '(', NULL, TRUE)) return FALSE;
    t = lex_scan();
    if (!expect_token(t, LEX_ID, NULL, TRUE)) return FALSE;
    if (lex_all_digits(parse_lexer->lexer_buf)) {
        num_ins = (int) strtol(parse_lexer->lexer_buf, &endp, 10);
    } else {
        printf("ERROR logicexp input count is not an integer\n");
        delete_lexer(parse_lexer);
        return FALSE;
    }
    t = lex_scan();
    if (!expect_token(t, ',', NULL, TRUE)) return FALSE;
    t = lex_scan();
    if (!expect_token(t, LEX_ID, NULL, TRUE)) return FALSE;
    if (lex_all_digits(parse_lexer->lexer_buf)) {
        num_outs = (int) strtol(parse_lexer->lexer_buf, &endp, 10);
    } else {
        printf("ERROR logicexp output count is not an integer\n");
        delete_lexer(parse_lexer);
        return FALSE;
    }
    num_outs = (int) strtol(parse_lexer->lexer_buf, &endp, 10);
    t = lex_scan();
    if (!expect_token(t, ')', NULL, TRUE)) return FALSE;
    t = lex_scan(); // pwr
    if (!expect_token(t, LEX_ID, NULL, TRUE)) return FALSE;
    t = lex_scan(); // gnd
    if (!expect_token(t, LEX_ID, NULL, TRUE)) return FALSE;
    /* num_ins input ids */
    for (i = 0; i < num_ins; i++) {
        t = lex_scan();
        if (!expect_token(t, LEX_ID, NULL, TRUE)) {
            return FALSE;
        }
        (void) add_sym_tab_entry(parse_lexer->lexer_buf,
            SYM_INPUT, &parse_lexer->lexer_sym_tab);
    }
    /* num_outs output ids */
    for (i = 0; i < num_outs; i++) {
        t = lex_scan();
        if (!expect_token(t, LEX_ID, NULL, TRUE)) {
            return FALSE;
        }
        (void) add_sym_tab_entry(parse_lexer->lexer_buf,
            SYM_OUTPUT, &parse_lexer->lexer_sym_tab);
    }
    /* timing model */
    t = lex_scan();
    if (!expect_token(t, LEX_ID, NULL, TRUE)) return FALSE;
    printf("TMODEL: %s\n", parse_lexer->lexer_buf);
    (void) add_sym_tab_entry(parse_lexer->lexer_buf,
        SYM_TMODEL, &parse_lexer->lexer_sym_tab);
    bparse(line, FALSE);

    return TRUE;
}

BOOL f_pindly(char *line)
{
    //printf("\nf_pindly: %s\n", line);
    return TRUE;
}

