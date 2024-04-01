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
#include <assert.h>

#include "ngspice/memory.h"
#include "ngspice/macros.h"
#include "ngspice/bool.h"
#include "ngspice/ngspice.h"
#include "ngspice/stringskip.h"
#include "ngspice/stringutil.h"
#include "ngspice/dstring.h"
#include "ngspice/logicexp.h"
#include "ngspice/udevices.h"

static char *get_pindly_instance_name(void);
static char *get_inst_name(void);
static char *get_logicexp_tmodel_delays(
    char *out_name, int gate_op, BOOL isnot, DSTRING *mname);

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

static void delete_sym_tab(SYM_TAB t)
{
    if (t == NULL) { return; }
    delete_sym_tab(t->left);
    delete_sym_tab(t->right);
    if (t->name)
        tfree(t->name);
    tfree(t);
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
    int lexer_last_pos;
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
    lx->lexer_pos = lx->lexer_last_pos = lx->lexer_back = 0;
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
    lx->lexer_last_pos = lx->lexer_pos;
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

static void lexer_back_one(LEXER lx)
{
    lx->lexer_pos = lx->lexer_last_pos;
}

static int lexer_scan(LEXER lx)
{
    int c;
    lx->lexer_last_pos = lx->lexer_pos;
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
            BOOL need_gt1 = FALSE;
            if (c == '+') { // an identifier does not begin with '+'
                lx->lexer_buf[0] = (char) c;
                lx->lexer_buf[1] = '\0';
                return LEX_OTHER;
            } else if (c == '_' || c == '/' || c == '-') {
                // these need to be followed by at least one more ident
                need_gt1 = TRUE;
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
            if (i == 1 && need_gt1) {
                lx->lexer_buf[1] = '\0';
                return LEX_OTHER;
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

/* Start of name entries */
typedef struct name_entry *NAME_ENTRY;
struct name_entry {
    char *name;
    NAME_ENTRY next;
};

static NAME_ENTRY new_name_entry(char *name)
{
    NAME_ENTRY newp;
    newp = TMALLOC(struct name_entry, 1);
    newp->next = NULL;
    newp->name = TMALLOC(char, strlen(name) + 1);
    strcpy(newp->name, name);
    return newp;
}

static NAME_ENTRY add_name_entry(char *name, NAME_ENTRY nelist)
{
    NAME_ENTRY newlist = NULL, x = NULL, last = NULL;

    if (nelist == NULL) {
        newlist = new_name_entry(name);
        return newlist;
    }
    for (x = nelist; x; x = x->next) {
        /* No duplicates */
        if (eq(x->name, name)) {
            //printf("\tFound entry %s\n", x->name);
            return x;
        }
        last = x;
    }
    x = new_name_entry(name);
    last->next = x;
    //printf("\tAdd entry %s\n", x->name);
    return x;
}

static void delete_name_entry(NAME_ENTRY entry)
{
    if (!entry) return;
    if (entry->name) tfree(entry->name);
    tfree(entry);
}

static void clear_name_list(NAME_ENTRY nelist)
{
    NAME_ENTRY x = NULL, next = NULL;
    if (!nelist) { return; }
    for (x = nelist; x; x = next) {
        next = x->next;
        delete_name_entry(x);
    }
}
/* End of name entries */

/* Start of infix to posfix */
#define STACK_SIZE 100
#define PUSH_ERROR 1
#define POP_ERROR  2
#define TMP_PREFIX  "tmp__"
#define TMP_LEN     (strlen(TMP_PREFIX))

struct Stack {
    int top;
    char *array[STACK_SIZE];
};

struct gate_data {
    int type;
    BOOL finished;
    BOOL is_not;
    BOOL is_possible;
    char *outp;
    NAME_ENTRY ins;
    NAME_ENTRY last_input;
    struct gate_data *nxt;
    struct gate_data *prev;
};

static struct gate_data *first_gate = NULL;
static struct gate_data *last_gate = NULL;

static struct gate_data *new_gate(int c, char *out, char *i1, char *i2)
{
    NAME_ENTRY np;
    struct gate_data *gdp = TMALLOC(struct gate_data, 1);
    gdp->type = c;
    gdp->finished = gdp->is_possible = FALSE;
    if (c == '~') {
        gdp->is_not = TRUE;
    } else {
        gdp->is_not = FALSE;
    }
    gdp->nxt = gdp->prev = NULL;
    if (out) {
        gdp->outp = TMALLOC(char, strlen(out) + 1);
        strcpy(gdp->outp, out);
    } else {
        gdp->outp = NULL;
    }
    if (i1) { // Only have second input if there is a first
        np = new_name_entry(i1);
        gdp->ins = np;
        if (i2) {
            assert(c != '~'); // inverters have only one input
            np = new_name_entry(i2);
            gdp->ins->next = np;
            if (strncmp(i1, TMP_PREFIX, TMP_LEN) == 0
            && strncmp(i2, TMP_PREFIX, TMP_LEN) != 0) {
                gdp->is_possible = TRUE;
            }
        }
        gdp->last_input = np;
    } else {
        gdp->ins = NULL;
        gdp->last_input = NULL;
    }
    return gdp;
}

static struct gate_data *insert_gate(struct gate_data *gp)
{
    if (!first_gate) {
        first_gate = last_gate = gp;
        gp->nxt = gp->prev = NULL;
    } else {
        last_gate->nxt = gp;
        gp->nxt = NULL;
        gp->prev = last_gate;
        last_gate = gp;
    }
    return last_gate;
}

static char *tilde_tail(char *s, DSTRING *ds)
{
    ds_clear(ds);
    if (strncmp(s, "tilde_", 6) == 0) {
        ds_cat_printf(ds, "~%s", s + 6);
        return ds_get_buf(ds);
    } else {
        return s;
    }
}

static void move_inputs(struct gate_data *curr, struct gate_data *prev)
{
    if (curr == NULL || prev == NULL) return;
    if (prev->finished) return;
    delete_name_entry(curr->ins);
    curr->ins = prev->ins;
    prev->last_input->next = curr->last_input;
    prev->ins = prev->last_input = NULL;
    prev->finished = TRUE;
}

static void scan_gates(DSTRING *lhs, int optimize)
{
    struct gate_data *current = NULL, *previous = NULL, *last_curr = NULL;
    struct gate_data *prev = NULL;

    if (optimize < 1) {
        current = last_gate;
        if (ds_get_length(lhs) > 0) {
            assert(current->finished == FALSE);
            tfree(current->outp);
            current->outp = TMALLOC(char, ds_get_length(lhs) + 1);
            strcpy(current->outp, ds_get_buf(lhs));
        }
        return;
    }

    current = first_gate;
    while (current) {
        int is_gate = (current->type == '&'
                       || current->type == '^'
                       || current->type == '|');
        previous = current->prev;
        if (is_gate && current->is_possible) {
            if (previous && previous->type == current->type
            && previous->is_not == current->is_not) {
                if (eq(current->ins->name, previous->outp)) {
                    move_inputs(current, previous);
                }
            }
        } else if (current->type == '~') {
            if (previous
            && (previous->type == '&' || previous->type == '|'
               || previous->type == '^')) {

                if (strncmp(current->ins->name, TMP_PREFIX, TMP_LEN) == 0
                && strncmp(previous->outp, TMP_PREFIX, TMP_LEN) == 0) {
                    if (eq(current->ins->name, previous->outp)) {
                        tfree(previous->outp);
                        previous->outp = TMALLOC(char, strlen(current->outp) + 1);
                        strcpy(previous->outp, current->outp);
                        previous->is_not = TRUE;
                        current->finished = TRUE;
                    }
                }
            }
        } else if (is_gate) {
            if (current->finished == FALSE
            && strncmp(current->ins->name, TMP_PREFIX, TMP_LEN) == 0) {
                prev = current->prev;
                while (prev) {
                    if (prev->type == current->type
                    && prev->is_not == current->is_not
                    && prev->finished == FALSE
                    && strncmp(prev->outp, TMP_PREFIX, TMP_LEN) == 0
                    && eq(current->ins->name, prev->outp)) {
                        move_inputs(current, prev);
                        break;
                    }
                    prev = prev->prev;
                }
            }
        }
        last_curr = current;
        current = current->nxt;
    }
    if (ds_get_length(lhs) > 0 && last_curr) {
        previous = last_curr;
        while (previous && previous->finished) {
            previous = previous->prev;
        }
        if (previous) {
            assert(previous->outp != NULL);
            assert(previous->finished == FALSE);
            tfree(previous->outp);
            previous->outp = TMALLOC(char, ds_get_length(lhs) + 1);
            strcpy(previous->outp, ds_get_buf(lhs));
        }
    }
}

static void gen_scanned_gates(struct gate_data *gp)
{
    DS_CREATE(instance, 64);
    DS_CREATE(ds, 32);
    DS_CREATE(mname, 32);
    NAME_ENTRY nm = NULL;
    if (!gp) return;
    while (gp) {
        if (gp->finished) {
            gp = gp->nxt;
            continue;
        }
        ds_clear(&instance);
        ds_cat_printf(&instance, "%s ", get_inst_name());
        (void) get_logicexp_tmodel_delays(gp->outp, gp->type, gp->is_not, &mname);
        if (gp->type == '&' || gp->type == '^' || gp->type == '|') {
            nm = gp->ins;
            ds_cat_str(&instance, "[");
            while (nm) {
                ds_cat_printf(&instance, " %s", tilde_tail(nm->name, &ds));
                nm = nm->next;
            }
            ds_cat_printf(&instance, " ] %s %s", gp->outp, ds_get_buf(&mname));
        } else if (gp->type == '~') {
            ds_cat_printf(&instance, "%s %s %s", tilde_tail(gp->ins->name, &ds),
                gp->outp, ds_get_buf(&mname));
        }

        u_add_instance(ds_get_buf(&instance));
        gp = gp->nxt;
    }
    ds_free(&instance);
    ds_free(&mname);
}

static void delete_gates(void)
{
    struct gate_data *g1, *g2;
    NAME_ENTRY n1, n2;
    g1 = first_gate;
    while (g1) {
        g2 = g1;
        if (g1->outp) tfree(g1->outp);
        n1 = g1->ins;
        while (n1) {
            n2 = n1;
            n1 = n1->next;
            delete_name_entry(n2);
        }
        g1 = g1->nxt;
        tfree(g2);
    }
    first_gate = last_gate = NULL;
}

static int get_precedence(char  * s) {
    switch (s[0]) {
    case '~':
        return 4;
    case '&':
        return 3;
    case '^':
        return 2;
    case '|':
        return 1;
    default:
        return 0;
    }
}

static int push(struct Stack* stack, char * item)
{
    if (stack->top == STACK_SIZE - 1) {
        fprintf(stderr, "ERROR Postfix stack Overflow\n");
        return PUSH_ERROR;
    }
    stack->array[++stack->top] = item;
    return 0;
}

static char * pop(struct Stack* stack, int *status)
{
    if (stack->top == -1) {
        fprintf(stderr, "ERROR Postfix stack Underflow\n");
        *status = POP_ERROR;
        return "";
    }
    *status = 0;
    return stack->array[stack->top--];
}

static char *makestr(int c)
{
    static char buf[32];
    sprintf(buf, "%c", c);
    return buf;
}

static int infix_to_postfix(char* infix, DSTRING * postfix_p)
{
    struct Stack stack;
    int ltok, last_tok = -1;
    LEXER lx;
    NAME_ENTRY nlist = NULL, entry = NULL;
    int status = 0;
    int lparen_count = 0, rparen_count = 0;

    lx = new_lexer(infix);
    stack.top = -1;
    nlist = add_name_entry("first", NULL);
    ds_clear(postfix_p);
    while ( ( ltok = lexer_scan(lx) ) != 0 ) { // start while ltok loop
        if (last_tok == -1) { // check starting token
            if (!(ltok == LEX_ID || ltok == '~' || ltok == '(')) {
                fprintf(stderr, "ERROR (1) invalid starting token\n");
                status = 1;
                goto err_return;
            }
        } else {
            if (last_tok == LEX_ID) { // check follower of an ID
                if (!(lex_gate_op(ltok) || ltok == ')')) {
                    fprintf(stderr, "ERROR (2) incorrect token after ID\n");
                    status = 1;
                    goto err_return;
                }
            } else if (lex_gate_op(last_tok)) { // check follower of a gate op
                if (!(ltok == LEX_ID || ltok == '~' || ltok == '(')) {
                    fprintf(stderr, "ERROR (3) incorrect token after gate op\n");
                    status = 1;
                    goto err_return;
                }
            } else if (last_tok == '~') { // check follower of ~
                if (!(ltok == LEX_ID || ltok == '(')) {
                    fprintf(stderr, "ERROR (4) incorrect token after \'~\'\n");
                    status = 1;
                    goto err_return;
                }
            } else if (last_tok == '(') { // check follower of lparen
                if (!(ltok == LEX_ID || ltok == '~' || ltok == '(')) {
                    fprintf(stderr, "ERROR (5) incorrect token after lparen\n");
                    status = 1;
                    goto err_return;
                }
            } else if (last_tok == ')') { // check follower of rparen
                if (!(ltok == ')' || lex_gate_op(ltok))) {
                    fprintf(stderr, "ERROR (6) incorrect token after rparen\n");
                    status = 1;
                    goto err_return;
                }
            }
        }

        last_tok = ltok;
        if (ltok == LEX_ID) {
            ds_cat_printf(postfix_p, " %s", lx->lexer_buf);
            if (strncmp(lx->lexer_buf, TMP_PREFIX, TMP_LEN) == 0) {
                printf("WARNING potential name collision %s in logicexp\n",
                    lx->lexer_buf);
                fflush(stdout);
            }
        } else if (ltok == '(') {
            lparen_count++;
            entry = add_name_entry(makestr(ltok), nlist);
            status = push(&stack, entry->name);
            if (status) {
                goto err_return;
            }
        } else if (ltok == ')') {
            rparen_count++;
            if (rparen_count > lparen_count) {
                fprintf(stderr, "ERROR (9a) mismatched rparen\n");
                status = 1;
                goto err_return;
            }
            while ( stack.top != -1 && !eq(stack.array[stack.top], "(") ) {
                ds_cat_printf(postfix_p, " %s", pop(&stack, &status));
                if (status) {
                    goto err_return;
                }
            }
            pop(&stack, &status);
            if (status) {
                goto err_return;
            }
        } else if (lex_gate_op(ltok) || ltok == '~') {
            char *tokstr = makestr(ltok);
            if (ltok == '~') {  // change ~ id --> tilde_id and continue
                int next_tok;
                next_tok = lexer_scan(lx);
                if (next_tok == LEX_ID) {
                    ds_cat_printf(postfix_p, " tilde_%s", lx->lexer_buf);
                    last_tok = next_tok;
                    continue;  // while ltok loop
                } else {
                    lexer_back_one(lx);
                }
            }
            while ( stack.top != -1 && !eq(stack.array[stack.top], "(") && get_precedence(stack.array[stack.top]) >= get_precedence(tokstr) ) {
                ds_cat_printf(postfix_p, " %s", pop(&stack, &status));
                if (status) {
                    goto err_return;
                }
            }
            entry = add_name_entry(tokstr, nlist);
            status = push(&stack, entry->name);
            if (status) {
                goto err_return;
            }
        } else {
            fprintf(stderr, "ERROR (7) unexpected infix token %d \'%s\'\n",
                ltok, lx->lexer_buf);
            status = 1;
            goto err_return;
        }
    } // end while ltok loop
    if (lex_gate_op(last_tok) || last_tok == '~') {
        fprintf(stderr, "ERROR (8) incomplete infix expression\n");
        status = 1;
        goto err_return;
    }
    if (lparen_count != rparen_count) {
        fprintf(stderr, "ERROR (9) mismatched parentheses\n");
        status = 1;
        goto err_return;
    }
    while (stack.top != -1) {
        ds_cat_printf(postfix_p, " %s", pop(&stack, &status));
        if (status) {
            goto err_return;
        }
    }
err_return:
    if (status) {
        fprintf(stderr, "ERROR invalid infix expression: %s\n", infix);
    }
    delete_lexer(lx);
    clear_name_list(nlist);
    return status;
}

static int evaluate_postfix(char* postfix)
{
    static int count = 1;
    struct Stack stack;
    stack.top = -1;
    char *operand1, *operand2;
    char tmp[32];
    int ltok, prevtok = 0;
    LEXER lx;
    NAME_ENTRY nlist = NULL, entry = NULL;
    struct gate_data *gp = NULL;
    int status = 0;
    int skip = 1;

#ifdef PFX_USE_INVERTERS
    if (getenv("PFX_USE_INVERTERS")) {
        skip = 0;
    } else {
        skip = 1;
    }
#endif

    lx = new_lexer(postfix);
    nlist = add_name_entry("first", NULL);
    tmp[0] = '\0';

    while ( ( ltok = lexer_scan(lx) ) != 0 ) { // while ltok loop
        if (ltok == LEX_ID) {
            entry = add_name_entry(lx->lexer_buf, nlist);
            status = push(&stack, entry->name);
            if (status) {
                goto err_return;
            }
        } else if (ltok == '~') {
            operand1 = pop(&stack, &status);
            if (status) {
                goto err_return;
            }
            sprintf(tmp, "%s%d", TMP_PREFIX, count);
            count++;
            gp = new_gate('~', tmp, operand1, NULL);
            gp = insert_gate(gp);
            entry = add_name_entry(tmp, nlist);
            status = push(&stack, entry->name);
            if (status) {
                goto err_return;
            }
        } else {
            operand2 = pop(&stack, &status);
            if (status) {
                goto err_return;
            }
            operand1 = pop(&stack, &status);
            if (status) {
                goto err_return;
            }
            if (lex_gate_op(ltok)) {
                sprintf(tmp, "%s%d", TMP_PREFIX, count);
                count++;
                gp = new_gate(ltok, tmp, operand1, operand2);
                gp = insert_gate(gp);
                entry = add_name_entry(tmp, nlist);
                status = push(&stack, entry->name);
                if (status) {
                    goto err_return;
                }
            }
        }
        prevtok = ltok;
    }  // end while ltok loop
    if (prevtok == LEX_ID) {
        char *n1 = NULL;
        DS_CREATE(ds1, 32);
        sprintf(tmp, "%s%d", TMP_PREFIX, count);
        count++;
        n1 = tilde_tail(pop(&stack, &status), &ds1);
        if (status) {
            goto err_return;
        }
        if (!skip && n1[0] == '~') {
            gp = new_gate('~', tmp, n1 + 1, NULL);
            gp->is_not = TRUE;
        } else {
            gp = new_gate('~', tmp, n1, NULL);
            gp->is_not = FALSE;
        }
        gp = insert_gate(gp);
        ds_free(&ds1);
    }
err_return:
    if (status) {
        fprintf(stderr, "ERROR invalid postfix expression: %s\n", postfix);
    }
    delete_lexer(lx);
    clear_name_list(nlist);
    return status;
}

/* End of infix to posfix */

/* Start of logicexp parser */
static void aerror(char *s);
static BOOL amatch(int t);
static BOOL bparse(char *line, BOOL new_lexer, int optimize);

static int lookahead = 0;
static int number_of_instances = 0;
static BOOL use_tmodel_delays = FALSE;

static void cleanup_parser(void)
{
    delete_lexer(parse_lexer);
    parse_lexer = NULL;
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

static BOOL bstmt_postfix(int optimize)
{
    /* A stmt is: output_name_id = '{' expr '}' */
    DS_CREATE(lhs, 32);
    DS_CREATE(postfix, 1024);
    DS_CREATE(infix, 1024);
    char *right_bracket = NULL, *rest = NULL;
    BOOL retval = TRUE;

    if (lookahead == LEX_ID) {
        ds_clear(&lhs);
        ds_cat_str(&lhs, parse_lexer->lexer_buf);
        if (strncmp(ds_get_buf(&lhs), TMP_PREFIX, TMP_LEN) == 0) {
            printf("WARNING potential name collision %s in logicexp\n",
                ds_get_buf(&lhs));
            fflush(stdout);
        }
        lookahead = lex_scan();
    } else {
        aerror("bstmt_postfix: syntax error");
        retval = FALSE;
        goto bail_out;
    }
    if (!amatch(('='))) {
        retval = FALSE;
        goto bail_out;
    }
    if (lookahead != '{') {
        printf("ERROR in bstmt_postfix \'{\' was expected\n");
        aerror("bstmt_postfix: syntax error");
        retval = FALSE;
        goto bail_out;
    }

    rest = parse_lexer->lexer_line + parse_lexer->lexer_pos;
    right_bracket = strstr(rest, "}");
    if (!right_bracket) {
        printf("ERROR in bstmt_postfix \'}\' was not found\n");
        aerror("bstmt_postfix: syntax error");
        retval = FALSE;
        goto bail_out;
    }
    ds_clear(&infix);
    ds_cat_mem(&infix, rest, right_bracket - rest);
    if (infix_to_postfix(ds_get_buf(&infix), &postfix)) {
        retval = FALSE;
        goto bail_out;
    }
    if (evaluate_postfix(ds_get_buf(&postfix))) {
        retval = FALSE;
        goto bail_out;
    }
    scan_gates(&lhs, optimize);
    gen_scanned_gates(first_gate);
    lookahead = lex_scan();
    while (lookahead != '}') {
        lookahead = lex_scan();
    }
    lookahead = lex_scan();

bail_out:
    delete_gates();
    ds_free(&lhs);
    ds_free(&postfix);
    ds_free(&infix);
    return retval;
}

static char *get_logicexp_tmodel_delays(
    char *out_name, int gate_op, BOOL isnot, DSTRING *mname)
{
    ds_clear(mname);
    if (use_tmodel_delays) {
        /* This is the case when logicexp has a UGATE
           timing model (not d0_gate) and no pindly.
        */
        SYM_TAB entry = NULL;
        char *nm1 = 0;
        entry = member_sym_tab(out_name, parse_lexer->lexer_sym_tab);
        if (entry && (entry->attribute & SYM_OUTPUT)) {
            nm1 = tmodel_gate_name(gate_op, isnot);
            if (nm1) {
                ds_cat_str(mname, nm1);
            }
        }
        if (!nm1) {
            nm1 = lex_gate_name(gate_op, isnot);
            ds_cat_str(mname, nm1);
        }
    } else {
        ds_cat_str(mname, lex_gate_name(gate_op, isnot));
    }
    return ds_get_buf(mname);
}

static BOOL bparse(char *line, BOOL new_lexer, int optimize)
{
    BOOL ret_val = TRUE;
    DS_CREATE(stmt, LEX_BUF_SZ);

    if (new_lexer)
        lex_init(line);
    if (!parse_lexer) return FALSE;
    lookahead = lex_set_start("logic:");
    lookahead = lex_scan(); // "logic"
    lookahead = lex_scan(); // ':'
    lookahead = lex_scan();
    while (lookahead != '\0') {
        ds_clear(&stmt);
        ds_cat_str(&stmt, parse_lexer->lexer_buf);
        if (!bstmt_postfix(optimize)) {
            cleanup_parser();
            ret_val= FALSE;
            break;
        }
    }

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

BOOL f_logicexp(char *line, int optimize)
{
    /* If optimize > 0 then perform optimizations in scan_gates */
    int t, num_ins = 0, num_outs = 0, i;
    char *endp;
    BOOL ret_val = TRUE;
    char *uname = NULL;

    lex_init(line);
    current_lexer = parse_lexer;
    (void) add_sym_tab_entry("logic", SYM_KEY_WORD,
        &parse_lexer->lexer_sym_tab);
    t = lex_scan(); // U*
    if (!expect_token(t, LEX_ID, NULL, TRUE, 1)) goto error_return;
    uname = (char *)TMALLOC(char, strlen(parse_lexer->lexer_buf) + 1);
    strcpy(uname, parse_lexer->lexer_buf);
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
    ret_val = bparse(line, FALSE, optimize);

    current_lexer = NULL;
    if (!ret_val) {
        fprintf(stderr, "ERROR parsing logicexp\n");
        fprintf(stderr, "ERROR in instance %s\n", uname);
        cleanup_parser();
    }
    if (uname) tfree(uname);
    return ret_val;

error_return:
    delete_lexer(parse_lexer);
    current_lexer = NULL;
    if (uname) tfree(uname);
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
    char *unitsmin, *unitsmax;
    char *instance = NULL;

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
            valmin = strtof(tmpmin, &unitsmin);
            valmax = strtof(tmpmax, &unitsmax);
            if (!eq(unitsmin, unitsmax)) {
                printf("WARNING typ_estimate units do not match"
                       " min %s max %s", tmpmin, tmpmax);
                fflush(stdout);
                if (unitsmin[0] == unitsmax[0]) {
                    average = (valmin + valmax) / (float)2.0;
                    ds_cat_printf(pds, "%.2f%cs", average, unitsmin[0]);
                } else if (unitsmin[0] == 'p' && unitsmax[0] == 'n') {
                    valmax = (float)1000.0 * valmax;
                    average = (valmin + valmax) / (float)2.0;
                    ds_cat_printf(pds, "%.2fps", average);
                } else if (unitsmin[0] == 'n' && unitsmax[0] == 'p') {
                    ds_cat_printf(pds, "%.2fns", valmin);
                } else {
                    ds_cat_printf(pds, "%.2f%s", valmin, unitsmin);
                }
                instance = get_pindly_instance_name();
                printf(" using delay %s", ds_get_buf(pds));
                if (instance) {
                    printf(" pindly %s\n", instance);
                } else {
                    printf("\n");
                }
            } else {
                average = (valmin + valmax) / (float)2.0;
                ds_cat_printf(pds, "%.2f%s", average, unitsmax);
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

static char *get_one_estimate(char *s, DSTRING *pds)
{
    ds_clear(pds);
    if (s && strlen(s) > 0 && s[0] != '-') {
        ds_cat_str(pds, s);
        return ds_get_buf(pds);
    } else {
        return NULL;
    }
}

static char *get_delay_estimate(char *min, char *typ, char *max, DSTRING *pds)
{
    char *one = NULL;
    struct udevices_info info = u_get_udevices_info();
    int delay_type = info.mntymx;
    if (delay_type == 1) { // min
        one = get_one_estimate(min, pds);
        if (one) {
            return one;
        }
    } else if (delay_type == 2) { // max
        one = get_one_estimate(max, pds);
        if (one) {
            return one;
        }
    }
    // typ
    return get_typ_estimate(min, typ, max, pds);
}

static char *mntymx_estimate(char *delay_str, DSTRING *pds)
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
    s = get_delay_estimate(ds_get_buf(&dmin), ds_get_buf(&dtyp),
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
    BOOL shorter = FALSE, update_val = FALSE;
    struct udevices_info info = u_get_udevices_info();
    float del_max_val = 0.0, del_val = 0.0, del_min_val = FLT_MAX;
    char *units;
    shorter = info.shorter_delays;
    DS_CREATE(dly, 64);
    DS_CREATE(ddel_str, 16);
    DS_CREATE(tmp_ds, 128);

    if (val != '=') {
        ds_free(&dly);
        ds_free(&ddel_str);
        ds_free(&tmp_ds);
        return FALSE;
    }
    val = lexer_scan(lx);
    if (val != '{') {
        ds_free(&dly);
        ds_free(&ddel_str);
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
                    tmps = mntymx_estimate(ds_get_buf(&dly), &tmp_ds);
                    if (!tmps) {
                        ret_val = FALSE;
                        ds_clear(&tmp_ds);
                        break;
                    }
                    del_val = strtof(tmps, &units);
                    update_val = FALSE;
                    if (shorter) {
                        if (del_val < del_min_val) {
                            update_val = TRUE;
                        }
                    } else if (del_val > del_max_val) {
                        update_val = TRUE;
                    }
                    if (update_val) {
                        ds_clear(&delay_string);
                        ds_clear(&ddel_str);
                        ds_cat_str(&ddel_str, tmps);
                        if (shorter) {
                           del_min_val = del_val;
                        } else {
                            del_max_val = del_val;
                        }
                        if (ds_get_length(&ddel_str) > 0) {
                            if (tri) {
                                ds_cat_printf(&delay_string,
                                    "(inertial_delay=true delay=%s)",
                                    ds_get_buf(&ddel_str));
                            } else {
                                ds_cat_printf(&delay_string,
                                    "(inertial_delay=true rise_delay=%s fall_delay=%s)",
                                    ds_get_buf(&ddel_str),
                                    ds_get_buf(&ddel_str));
                            }
                        } else {
                            printf("WARNING pindly DELAY not found\n");
                            fflush(stdout);
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
    ds_free(&ddel_str);
    ds_free(&tmp_ds);
    return ret_val;
}

static BOOL new_gen_output_models(LEXER lx)
{
    int val, arrlen = 0, idx = 0, i;
    BOOL in_pindly = FALSE, in_tristate = FALSE;
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
                || eq(lx->lexer_buf, "boolean")
                || eq(lx->lexer_buf, "general")) {
                in_pindly = FALSE;
                in_tristate = FALSE;
            }
        }
        if (in_pindly && val == LEX_ID) { // start in_pindly and LEX_ID
            while (val == LEX_ID) {
                pline = find_pindly_out_name(pindly_tab, lx->lexer_buf);
                if (pline) {
                    pline_arr[idx++] = pline;
                } else {
                    goto err_return;
                }
                val = lexer_scan(lx);
                if (val == ',') {
                    val = lexer_scan(lx);
                }
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
                    val = lexer_scan(lx);
                    if (val != '=') {
                        // if there is no '=' it must be an enable id
                        if (val != LEX_ID) {
                            goto err_return;
                        }
                    } else { // enable id follows '='
                        val = lexer_scan(lx);
                        if (val != LEX_ID) {
                            goto err_return;
                        }
                    }
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
                pline = find_pindly_out_name(pindly_tab, lx->lexer_buf);
                if (pline) {
                    pline_arr[idx++] = pline;
                    (void) set_ena_name(ds_get_buf(&enable_name), pline);
                    u_remember_pin(lx->lexer_buf, 3);
                } else {
                    goto err_return;
                }
                val = lexer_scan(lx);
                if (val == ',') {
                    val = lexer_scan(lx);
                }
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

static char *pindly_instance_name = NULL;
static void set_pindly_instance_name(char *name)
{
    if (pindly_instance_name) {
        tfree(pindly_instance_name);
        pindly_instance_name = NULL;
    }
    if (name) {
        pindly_instance_name = (char *)TMALLOC(char, strlen(name) + 1);
        strcpy(pindly_instance_name, name);
    }
}

static char *get_pindly_instance_name(void)
{
    return pindly_instance_name;
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
    set_pindly_instance_name(lxr->lexer_buf);

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
        char *i_name = get_pindly_instance_name();
        fprintf(stderr, "ERROR generating models for pindly\n");
        if (i_name) {
            fprintf(stderr, "ERROR in instance %s\n", i_name);
        }
        goto error_return;;
    }
    gen_pindly_buffers();
    delete_lexer(lxr);
    cleanup_pindly_tab();
    current_lexer = NULL;
    set_pindly_instance_name(NULL);
    return TRUE;

error_return:
    delete_lexer(lxr);
    cleanup_pindly_tab();
    current_lexer = NULL;
    set_pindly_instance_name(NULL);
    return FALSE;
}

