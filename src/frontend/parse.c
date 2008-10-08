/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
$Id$
**********/

/*
 * A simple operator-precedence parser for algebraic expressions.
 * This also handles relational and logical expressions.
 */

#include <ngspice.h>
#include <bool.h>
#include <fteparse.h>
#include <fteext.h>
#include <sim.h>

#include "parse.h"


/* static declarations */
static bool checkvalid(struct pnode *pn);
static struct element * lexer(void);
static struct pnode * parse(void);
static struct pnode * makepnode(struct element *elem);
static struct pnode * mkbnode(int opnum, struct pnode *arg1, struct pnode *arg2);
static struct pnode * mkunode(int op, struct pnode *arg);
static struct pnode * mkfnode(char *func, struct pnode *arg);
static struct pnode * mknnode(double number);
static struct pnode * mksnode(char *string);
/*static void   print_elem(struct element *elem); / va: for debugging /
static char * get_token_name(int e_token); / va, for debugging */


static int lasttoken = END, lasttype;
static char *sbuf;

struct pnode *
ft_getpnames(wordlist *wl, bool check)
{
    struct pnode *pn = NULL, *lpn = NULL, *p;
    char *xsbuf;
    char buf[BSIZE_SP], *thisone, *s;

    if (!wl) {
        fprintf(cp_err, "Warning: NULL arithmetic expression\n");
        return (NULL);
    }

    lasttoken = END;
    xsbuf = sbuf = wl_flatten(wl);
    thisone = sbuf;
    while (*sbuf != '\0') {
        if (!(p = parse())) {
	    tfree(xsbuf);
            return (NULL);
	}

        /* Now snag the name... Much trouble... */
        while (isspace(*thisone))
            thisone++;
        for (s = buf; thisone < sbuf; s++, thisone++)
            *s = *thisone;
        *s = '\0';
        p->pn_name = copy(buf);

        if (pn) {
            lpn->pn_next = p;
	    pn->pn_next->pn_use++;
            lpn = p;
        } else
            pn = lpn = p;
    }
    tfree(xsbuf);
    if (check)
        if (!checkvalid(pn))
            return (NULL);
    return (pn);
}

/* See if there are any variables around which have length 0 and are
 * not named 'list'. There should really be another flag for this...
 */

static bool
checkvalid(struct pnode *pn)
{
    while (pn) {
        if (pn->pn_value) {
            if ((pn->pn_value->v_length == 0) && 
                !eq(pn->pn_value->v_name, "list")) {
                if (eq(pn->pn_value->v_name, "all"))
                    fprintf(cp_err,
                    "Error: %s: no matching vectors.\n",
                        pn->pn_value->v_name);
                else
                    fprintf(cp_err,
                        "Error(parse.c--checkvalid): %s: no such vector.\n",
                        pn->pn_value->v_name);
                return (FALSE);
            }
        } else if (pn->pn_func || 
                (pn->pn_op && (pn->pn_op->op_arity == 1))) {
            if (!checkvalid(pn->pn_left))
                return (FALSE);
        } else if (pn->pn_op && (pn->pn_op->op_arity == 2)) {
            if (!checkvalid(pn->pn_left))
                return (FALSE);
            if (!checkvalid(pn->pn_right))
                return (FALSE);
        } else
            fprintf(cp_err, 
                "checkvalid: Internal Error: bad node\n");
        pn = pn->pn_next;
    }
    return (TRUE);
}

/* Everything else is a string or a number. Quoted strings are kept in 
 * the form "string", and the lexer strips off the quotes...
 */
/* va: the structure returned is static, e_string is a copy 
       (in case of e_token==VALUE,e_type==STRING) */
static struct element *
lexer(void)
{
    double *td;
    int j = 0;
    static struct element el;
    static struct element end = { END };
    static char *specials = " \t%()-^+*,/|&<>~=";
    static bool bracflag = FALSE;
    char *ss, *s;
    int atsign;

    if (bracflag) {
        bracflag = FALSE;
        el.e_token = LPAREN;
        goto done;
    }

    el.e_token = END;
    while ((*sbuf == ' ') || (*sbuf == '\t'))
        sbuf++;
    if (*sbuf == '\0')
        goto done;

    switch (*sbuf) {

        case '-':
        if ((lasttoken == VALUE) || (lasttoken == RPAREN))
            el.e_token = MINUS;
        else
            el.e_token = UMINUS;
        sbuf++;
        break;

        case '+':
        el.e_token = PLUS; 
        sbuf++;
        break;

        case ',':
        el.e_token = COMMA;
        sbuf++;
        break;

        case '*':
        el.e_token = TIMES; 
        sbuf++;
        break;

        case '%':
        el.e_token = MOD; 
        sbuf++;
        break;

        case '/':
        el.e_token = DIVIDE; 
        sbuf++;
        break;

        case '^':
        el.e_token = POWER; 
        sbuf++;
        break;

        case '[':
        if (sbuf[1] == '[') {
            el.e_token = RANGE;
            sbuf += 2;
        } else {
            el.e_token = INDX;
            sbuf++;
        }
        bracflag = TRUE;
        break;

        case '(':
        if (((lasttoken == VALUE) && ((lasttype == NUM))) || (lasttoken
                == RPAREN)) {
            el = end;
            goto done;
        } else {
            el.e_token = LPAREN; 
            sbuf++;
            break;
        }

        case ']':
        el.e_token = RPAREN; 
        if (sbuf[1] == ']')
            sbuf += 2;
        else
            sbuf++;
        break;

        case ')':
        el.e_token = RPAREN; 
        sbuf++;
        break;

        case '=':
        el.e_token = EQ;
        sbuf++;
        break;

        case '>':
        case '<':
        for (j = 0; isspace(sbuf[j]); j++)
            ; /* The lexer makes <> into < > */
        if (((sbuf[j] == '<') || (sbuf[j] == '>')) &&
                (sbuf[0] != sbuf[j])) {
            /* Allow both <> and >< for NE. */
            el.e_token = NE;
            sbuf += 2 + j;
        } else if (sbuf[1] == '=') {
            if (sbuf[0] == '>')
                el.e_token = GE;
            else
                el.e_token = LE;
            sbuf += 2;
        } else {
            if (sbuf[0] == '>')
                el.e_token = GT;
            else
                el.e_token = LT;
            sbuf++;
        }
        break;

        case '&':
        el.e_token = AND;
        sbuf++;
        break;

        case '|':
        el.e_token = OR;
        sbuf++;
        break;

        case '~':
        el.e_token = NOT;
        sbuf++;
        break;

        case '"':
        if ((lasttoken == VALUE) || (lasttoken == RPAREN)) {
            el = end;
            goto done;
        }
        el.e_token = VALUE;
        el.e_type = STRING;
        el.e_string = copy(++sbuf);
        for (s = el.e_string; *s && (*s != '"'); s++, sbuf++)
            ;
        *s = '\0';
        sbuf++;
        break;
    }

    if (el.e_token != END)
        goto done;

    ss = sbuf;
    td = ft_numparse(&ss, FALSE);
    if ((!ss || *ss != ':') && td) {
        if ((lasttoken == VALUE) || (lasttoken == RPAREN)) {
            el = end;
            goto done;
        }
        el.e_double = *td;
        el.e_type = NUM;
        el.e_token = VALUE;
        sbuf = ss;
        if (ft_parsedb)
            fprintf(stderr, "lexer: double %G\n",
                    el.e_double);
    } else {
        /* First, let's check for eq, ne, and so on. */
        if ((sbuf[0] == 'g') && (sbuf[1] == 't') && 
               strchr(specials, sbuf[2])) {
            el.e_token = GT;
            sbuf += 2;
        } else if ((sbuf[0] == 'l') && (sbuf[1] == 't') && 
               strchr(specials, sbuf[2])) {
            el.e_token = LT;
            sbuf += 2;
        } else if ((sbuf[0] == 'g') && (sbuf[1] == 'e') && 
               strchr(specials, sbuf[2])) {
            el.e_token = GE;
            sbuf += 2;
        } else if ((sbuf[0] == 'l') && (sbuf[1] == 'e') && 
               strchr(specials, sbuf[2])) {
            el.e_token = LE;
            sbuf += 2;
        } else if ((sbuf[0] == 'n') && (sbuf[1] == 'e') && 
               strchr(specials, sbuf[2])) {
            el.e_token = NE;
            sbuf += 2;
        } else if ((sbuf[0] == 'e') && (sbuf[1] == 'q') && 
               strchr(specials, sbuf[2])) {
            el.e_token = EQ;
            sbuf += 2;
        } else if ((sbuf[0] == 'o') && (sbuf[1] == 'r') && 
               strchr(specials, sbuf[2])) {
            el.e_token = OR;
            sbuf += 2;
        } else if ((sbuf[0] == 'a') && (sbuf[1] == 'n') && 
                (sbuf[2] == 'd') &&strchr(specials, sbuf[3])) {
            el.e_token = AND;
            sbuf += 3;
        } else if ((sbuf[0] == 'n') && (sbuf[1] == 'o') && 
                (sbuf[2] == 't') &&strchr(specials, sbuf[3])) {
            el.e_token = NOT;
            sbuf += 3;
        } else {
            if ((lasttoken == VALUE) || (lasttoken == RPAREN)) {
                el = end;
                goto done;
            }
            el.e_string = copy(sbuf);	/* XXXX !!!! */
            /* It is bad how we have to recognise '[' -- sometimes
             * it is part of a word, when it defines a parameter
             * name, and otherwise it isn't.
	     * va, ']' too
             */
	    atsign = 0;
            for (s = el.e_string; *s && !index(specials, *s); s++, sbuf++) {
                if (*s == '@')
		    atsign = 1;
                else if (!atsign && ( *s == '[' || *s == ']' ) )
                    break;
	    }
            if (*s)
                *s = '\0';
            el.e_type = STRING;
            el.e_token = VALUE;
            if (ft_parsedb)
                fprintf(stderr, "lexer: string %s\n",
                        el.e_string);
        }
    }
done:
    lasttoken = el.e_token;
    lasttype = el.e_type;
    if (ft_parsedb)
        fprintf(stderr, "lexer: token %d\n", el.e_token);
    return (&el);
}

/* The operator-precedence parser. */

#define G 1 /* Greater than. */
#define L 2 /* Less than. */
#define E 3 /* Equal. */
#define R 4 /* Error. */

#define STACKSIZE 200

static char prectable[23][23] = {
       /* $  +  -  *  %  /  ^  u- (  )  ,  v  =  >  <  >= <= <> &  |  ~ IDX R */
/* $ */ { R, L, L, L, L, L, L, L, L, R, L, L, L, L, L, L, L, L, L, L, L, L, L },
/* + */ { G, G, G, L, L, L, L, L, L, G, G, L, G, G, G, G, G, G, G, G, G, L, L },
/* - */ { G, G, G, L, L, L, L, L, L, G, G, L, G, G, G, G, G, G, G, G, G, L, L },
/* * */ { G, G, G, G, G, G, L, L, L, G, G, L, G, G, G, G, G, G, G, G, G, L, L },
/* % */ { G, G, G, G, G, G, L, L, L, G, G, L, G, G, G, G, G, G, G, G, G, L, L },
/* / */ { G, G, G, G, G, G, L, L, L, G, G, L, G, G, G, G, G, G, G, G, G, L, L },
/* ^ */ { G, G, G, G, G, G, L, L, L, G, G, L, G, G, G, G, G, G, G, G, G, L, L },
/* u-*/ { G, G, G, G, G, G, G, G, L, G, G, L, G, G, G, G, G, G, G, G, G, L, L },
/* ( */ { R, L, L, L, L, L, L, L, L, E, L, L, L, L, L, L, L, L, L, L, L, L, L },
/* ) */ { G, G, G, G, G, G, G, G, R, G, G, R, G, G, G, G, G, G, G, G, G, G, G },
/* , */ { G, L, L, L, L, L, L, L, L, G, L, L, G, G, G, G, G, G, G, G, G, L, L },
/* v */ { G, G, G, G, G, G, G, G, G, G, G, R, G, G, G, G, G, G, G, G, G, G, G },
/* = */ { G, L, L, L, L, L, L, L, L, G, L, L, G, G, G, G, G, G, G, G, L, L, L },
/* > */ { G, L, L, L, L, L, L, L, L, G, L, L, G, G, G, G, G, G, G, G, L, L, L },
/* < */ { G, L, L, L, L, L, L, L, L, G, L, L, G, G, G, G, G, G, G, G, L, L, L },
/* >=*/ { G, L, L, L, L, L, L, L, L, G, L, L, G, G, G, G, G, G, G, G, L, L, L },
/* <=*/ { G, L, L, L, L, L, L, L, L, G, L, L, G, G, G, G, G, G, G, G, L, L, L },
/* <>*/ { G, L, L, L, L, L, L, L, L, G, L, L, G, G, G, G, G, G, G, G, L, L, L },
/* & */ { G, L, L, L, L, L, L, L, L, G, L, L, L, L, L, L, L, L, G, G, L, L, L },
/* | */ { G, L, L, L, L, L, L, L, L, G, L, L, L, L, L, L, L, L, L, G, L, L, L },
/* ~ */ { G, L, L, L, L, L, L, L, L, G, L, L, G, G, G, G, G, G, G, G, G, L, L },
/*INDX*/{ G, G, G, G, G, G, G, G, L, G, G, L, G, G, G, G, G, G, G, G, G, G, L },
/*RAN*/ { G, G, G, G, G, G, G, G, L, G, G, L, G, G, G, G, G, G, G, G, G, G, G }
} ;

/* Return an expr. */
static struct pnode *
parse(void)
{
    struct element stack[STACKSIZE];
    int sp = 0, st, i, spmax=0; /* va: spmax = maximal used stack */
    struct element *top, *next;
    struct pnode *pn, *lpn, *rpn;
    char rel;
    char * parse_string=sbuf; /* va, duplicate sbuf's pointer for error message only, no tfree */

    stack[0].e_token = END;
    next = lexer();

    while ((sp > 1) || (next->e_token != END)) {
        /* Find the top-most terminal. */
        /* va: no stack understepping, because stack[0].e_token==END */
        i = sp;
        do {
            top = &stack[i--];
        } while (top->e_token == VALUE && i>=0); /* va: do not understep stack */
        if (top->e_token == VALUE) {
            fprintf(cp_err, "Error: in parse.c(parse) stack understep.\n");
            return (NULL);
        }
/*for (i=0; i<=sp; i++) print_elem(stack+i); printf("next: "); print_elem(next); printf("\n");*/ 

        rel = prectable[top->e_token][next->e_token];
        switch (rel) {
            case L:
            case E:
            /* Push the token read. */
            if (sp == (STACKSIZE - 1)) {
                fprintf(cp_err, "Error: stack overflow\n");
                return (NULL);
            }
            bcopy((char *) next, (char *) &stack[++sp],
                    sizeof (struct element));
            if (spmax<sp) spmax=sp; /* va: maximal used stack increased */
            next = lexer();
            continue;

            case R:
            fprintf(cp_err, "Syntax error: parsing expression '%s'.\n", parse_string);
            return (NULL);

            case G:
            /* Reduce. Make st and sp point to the elts on the
             * stack at the end and beginning of the junk to
             * reduce, then try and do some stuff. When scanning
             * back for a <, ignore VALUES.
             */

            st = sp;
            if (stack[sp].e_token == VALUE)
                sp--;
            while (sp > 0) {
                if (stack[sp - 1].e_token == VALUE)
                    i = 2;  /* No 2 pnodes together... */
                else
                    i = 1;
                if (prectable[stack[sp - i].e_token]
                         [stack[sp].e_token] == L)
                    break;
                else
                    sp = sp - i;
            }
            if (stack[sp - 1].e_token == VALUE)
                sp--;
            /* Now try and see what we can make of this.
             * The possibilities are: unop node
             *            node op node
             *            ( node )
             *            func ( node )
             *            node
             *  node [ node ] is considered node op node.
             */
            if (st == sp) {
                pn = makepnode(&stack[st]);
                if (pn == NULL)
                    goto err;
            } else if (((stack[sp].e_token == UMINUS) ||
                    (stack[sp].e_token == NOT)) && 
                    (st == sp + 1)) {
                lpn = makepnode(&stack[st]);
                if (lpn == NULL)
                        goto err;
                pn = mkunode(stack[sp].e_token, lpn);
            } else if ((stack[sp].e_token == LPAREN) &&
                       (stack[st].e_token == RPAREN)) {
                pn = makepnode(&stack[sp + 1]);
                if (pn == NULL)
                    goto err;
            } else if ((stack[sp + 1].e_token == LPAREN) &&
                       (stack[st].e_token == RPAREN)) {
                lpn = makepnode(&stack[sp + 2]);
                if ((lpn == NULL) || (stack[sp].e_type !=
                        STRING))
                    goto err;
                if (!(pn = mkfnode(stack[sp].e_string, lpn)))
                    return (NULL);
                /* va: avoid memory leakage: 
                   in case of variablenames (i.e. i(vd)) mkfnode makes in 
                   reality a snode, the old lpn (and its plotless vector) is 
                   then a memory leak */
                if (pn->pn_func==NULL && pn->pn_value!=NULL) /* a snode */
                {
                   if (lpn->pn_value && lpn->pn_value->v_plot==NULL)
                   {
                       tfree(lpn->pn_value->v_name);
                       tfree(lpn->pn_value);
                   }
                   free_pnode(lpn);
                }
            } else { /* node op node */
                lpn = makepnode(&stack[sp]);
                rpn = makepnode(&stack[st]);
                if ((lpn == NULL) || (rpn == NULL))
                    goto err;
                pn = mkbnode(stack[sp + 1].e_token, 
                    lpn, rpn);
            }
            /* va: avoid memory leakage: tfree all old strings on stack,
                   copied up to now within lexer */
            for (i=sp; i<=spmax; i++) {
                if (stack[i].e_token==VALUE && stack[i].e_type==STRING) {
                    tfree(stack[i].e_string);
                }
            }
            spmax=sp; /* up to there stack is now clean */

            stack[sp].e_token = VALUE;
            stack[sp].e_type = PNODE;
            stack[sp].e_pnode = pn;
            continue;
        }
    }
    pn = makepnode(&stack[1]);

    /* va: avoid memory leakage: tfree all remaining strings,
       copied within lexer */
    for (i=0; i<=spmax; i++) {
        if (stack[i].e_token == VALUE && stack[i].e_type == STRING) {
            tfree(stack[i].e_string);
        }
    }
    if (next->e_token == VALUE && next->e_type == STRING) {
        tfree(next->e_string);
    }

    if (pn)
        return (pn);
err:
    fprintf(cp_err, "Syntax error: expression not understood '%s'.\n", parse_string);
    return (NULL);
}

/* Given a pointer to an element, make a pnode out of it (if it already
 * is one, return a pointer to it). If it isn't of type VALUE, then return
 * NULL.
 */

static struct pnode *
makepnode(struct element *elem)
{
    if (elem->e_token != VALUE)
        return (NULL);
    switch (elem->e_type) {
        case STRING:
            return (mksnode(elem->e_string));
        case NUM:
            return (mknnode(elem->e_double));
        case PNODE:
            return (elem->e_pnode);
        default:
            return (NULL);
    }   
}

/*
static char * get_token_name(int e_token)
{
  / see include/fteparse.h /
    switch (e_token) {
    case   0: return "END   ";
    case   1: return "PLUS  ";
    case   2: return "MINUS ";
    case   3: return "TIMES ";
    case   4: return "MOD   ";
    case   5: return "DIVIDE";
    case   6: return "POWER ";
    case   7: return "UMINUS";
    case   8: return "LPAREN";
    case   9: return "RPAREN";
    case  10: return "COMMA ";
    case  11: return "VALUE ";
    case  12: return "EQ    ";
    case  13: return "GT    ";
    case  14: return "LT    ";
    case  15: return "GE    ";
    case  16: return "LE    ";
    case  17: return "NE    ";
    case  18: return "AND   ";
    case  19: return "OR    ";
    case  20: return "NOT   ";
    case  21: return "INDX  ";
    case  22: return "RANGE ";
    default : return "UNKNOWN";
    }
}

static void print_elem(struct element *elem)
{
    printf("e_token = %d(%s)", elem->e_token, get_token_name(elem->e_token)); 
    if (elem->e_token == VALUE) {
        printf(", e_type  = %d", elem->e_type); 
        switch (elem->e_type) {
            case STRING:
                printf(", e_string = %s(%p)", elem->e_string,elem->e_string); 
                break; 
            case NUM:
                printf(", e_double = %g", elem->e_double); break; 
            case PNODE:
                printf(", e_pnode  = %p", elem->e_pnode);  break; 
            default:
                break;
        }
    }   
    printf("\n");
}
*/


/* Some auxiliary functions for building the parse tree. */

static
struct op ops[] = { 
        { PLUS, "+", 2, op_plus } ,
        { MINUS, "-", 2, op_minus } ,
        { TIMES, "*", 2, op_times } ,
        { MOD, "%", 2, op_mod } ,
        { DIVIDE, "/", 2, op_divide } ,
        { COMMA, ",", 2, op_comma } ,
        { POWER, "^", 2, op_power } ,
        { EQ, "=", 2, op_eq } ,
        { GT, ">", 2, op_gt } ,
        { LT, "<", 2, op_lt } ,
        { GE, ">=", 2, op_ge } ,
        { LE, "<=", 2, op_le } ,
        { NE, "<>", 2, op_ne } ,
        { AND, "&", 2, op_and } ,
        { OR, "|", 2, op_or } ,
        { INDX, "[", 2, op_ind } ,
        { RANGE, "[[", 2, op_range } ,
        { 0, NULL, 0, NULL }
} ;

static
struct op uops[] = {
    { UMINUS, "-", 1, op_uminus } ,
    { NOT, "~", 1, op_not } ,
    { 0, NULL, 0, NULL }
} ;

/* We have 'v' declared as a function, because if we don't then the defines
 * we do for vm(), etc won't work. This is caught in evaluate(). Bad kludge.
 */

struct func ft_funcs[] = {
        { "mag",    cx_mag } ,
        { "magnitude",  cx_mag } ,
        { "ph",     cx_ph } ,
        { "phase",  cx_ph } ,
        { "j",      cx_j } ,
        { "real",   cx_real } ,
        { "re",     cx_real } ,
        { "imag",   cx_imag } ,
        { "im",     cx_imag } ,
        { "db",     cx_db } ,
        { "log",    cx_log } ,
        { "log10",  cx_log } ,
        { "ln",     cx_ln } ,
        { "exp",    cx_exp } ,
        { "abs",    cx_mag } ,
        { "sqrt",   cx_sqrt } ,
        { "sin",    cx_sin } ,
        { "cos",    cx_cos } ,
        { "tan",    cx_tan } ,
        { "atan",   cx_atan } ,
        { "norm",   cx_norm } ,
        { "rnd",    cx_rnd } ,
        { "pos",    cx_pos } ,
        { "mean",   cx_mean } ,
        { "avg",   cx_avg } ,     //A.Rroldan 03/06/05 incremental average  new function
        { "group_delay",  cx_group_delay } , //A.Rroldan 10/06/05 group delay new function
        { "vector", cx_vector } ,
        { "unitvec",    cx_unitvec } ,
        { "length", cx_length } ,
        { "vecmin", cx_min } ,
        { "vecmax", cx_max } ,
        { "vecd", cx_d } ,
        { "interpolate", cx_interpolate } ,
        { "deriv",       cx_deriv } ,
        { "v",      NULL } ,
        { NULL,     NULL }
} ;

struct func func_uminus = { "minus", cx_uminus };

struct func func_not = { "not", cx_not };

/* Binary operator node. */

static struct pnode *
mkbnode(int opnum, struct pnode *arg1, struct pnode *arg2)
{
    struct op *o;
    struct pnode *p;

    for (o = &ops[0]; o->op_name; o++)
        if (o->op_num == opnum)
            break;
    if (!o->op_name)
        fprintf(cp_err, "mkbnode: Internal Error: no such op num %d\n",
                    opnum);
    p = alloc(struct pnode);
    p->pn_use = 0;
    p->pn_value = NULL;
    p->pn_name = NULL;	/* sjb */
    p->pn_func = NULL;
    p->pn_op = o;
    p->pn_left = arg1;
    if(p->pn_left) p->pn_left->pn_use++;
    p->pn_right = arg2;
    if(p->pn_right) p->pn_right->pn_use++;
    p->pn_next = NULL;
    return (p);
}

/* Unary operator node. */

static struct pnode *
mkunode(int op, struct pnode *arg)
{
    struct pnode *p;
    struct op *o;

    p = alloc(struct pnode);
    for (o = uops; o->op_name; o++)
        if (o->op_num == op)
            break;
    if (!o->op_name)
        fprintf(cp_err, "mkunode: Internal Error: no such op num %d\n",
                op);

    p->pn_op = o;
    p->pn_use = 0;
    p->pn_value = NULL;
    p->pn_name = NULL;	/* sjb */
    p->pn_func = NULL;
    p->pn_left = arg;
    if(p->pn_left) p->pn_left->pn_use++;
    p->pn_right = NULL;
    p->pn_next = NULL;
    return (p);
}

/* Function node. We have to worry about a lot of things here. Something
 * like f(a) could be three things -- a call to a standard function, which
 * is easiest to deal with, a variable name, in which case we do the
 * kludge with 0-length lists, or it could be a user-defined function,
 * in which case we have to figure out which one it is, substitute for
 * the arguments, and then return a copy of the expression that it was
 * defined to be.
 */

static struct pnode *
mkfnode(char *func, struct pnode *arg)
{
    struct func *f;
    struct pnode *p, *q;
    struct dvec *d;
    char buf[BSIZE_SP], *s;

    (void) strcpy(buf, func);
    for (s = buf; *s; s++)      /* Make sure the case is ok. */
        if (isupper(*s))
            *s = tolower(*s);
    for (f = &ft_funcs[0]; f->fu_name; f++)
        if (eq(f->fu_name, buf))
            break;
    if (f->fu_name == NULL) {
        /* Give the user-defined functions a try. */
        q = ft_substdef(func, arg);
        if (q)
            return (q);
    }
    if ((f->fu_name == NULL) && arg->pn_value) {
        /* Kludge -- maybe it is really a variable name. */
        (void) sprintf(buf, "%s(%s)", func, arg->pn_value->v_name);
        d = vec_get(buf);
        if (d == NULL) {
            /* Well, too bad. */
            fprintf(cp_err, "Error: no such function as %s.\n", 
                    func);
            return (NULL);
        }
        /* (void) strcpy(buf, d->v_name); XXX */
        return (mksnode(buf));
    } else if (f->fu_name == NULL) {
        fprintf(cp_err, "Error: no function as %s with that arity.\n",
                func);
            return (NULL);
    }

    if (!f->fu_func && arg->pn_op && arg->pn_op->op_num == COMMA) {
	p = mkbnode(MINUS, mkfnode(func, arg->pn_left),
		mkfnode(func, arg->pn_right));
	tfree(arg);
	return p;
    }

    p = alloc(struct pnode);
    p->pn_use = 0;
    p->pn_name = NULL;
    p->pn_value = NULL;
    p->pn_func = f;
    p->pn_op = NULL;
    p->pn_left = arg;
    if(p->pn_left) p->pn_left->pn_use++;
    p->pn_right = NULL;
    p->pn_next = NULL;
    return (p);
}

/* Number node. */

static struct pnode *
mknnode(double number)
{
    struct pnode *p;
    struct dvec *v;
    char buf[BSIZE_SP];

    p = alloc(struct pnode);
    v = alloc(struct dvec);
    ZERO(v, struct dvec);
    p->pn_use = 0;
    p->pn_name = NULL;
    p->pn_value = v;
    p->pn_func = NULL;
    p->pn_op = NULL;
    p->pn_left = p->pn_right = NULL;
    p->pn_next = NULL;

    /* We don't use printnum because it screws up mkfnode above. We have
     * to be careful to deal properly with node numbers that are quite
     * large...
     */
    if (number < MAXPOSINT)
        (void) sprintf(buf, "%d", (int) number);
    else
        (void) sprintf(buf, "%G", number);
    v->v_name = copy(buf);
    v->v_type = SV_NOTYPE;
    v->v_flags = VF_REAL;
    v->v_realdata = (double *) tmalloc(sizeof (double));
    *v->v_realdata = number;
    v->v_length = 1;
    v->v_plot = NULL;
    vec_new(v);
    return (p);
}

/* String node. */

static struct pnode *
mksnode(char *string)
{
    struct dvec *v, *nv, *vs, *newv = NULL, *end = NULL;
    struct pnode *p;

    p = alloc(struct pnode);
    p->pn_use = 0;
    p->pn_name = NULL;
    p->pn_func = NULL;
    p->pn_op = NULL;
    p->pn_left = p->pn_right = NULL;
    p->pn_next = NULL;
    v = vec_get(string);
    if (v == NULL) {
        nv = alloc(struct dvec);
	ZERO(nv, struct dvec);
        p->pn_value = nv;
        nv->v_name = copy(string);
        return (p);
    }
    p->pn_value = NULL;

    /* It's not obvious that we should be doing this, but... */
    for (vs = v; vs; vs = vs->v_link2) {
        nv = vec_copy(vs);
        vec_new(nv);
        if (end)
            end->v_link2 = nv;
        else
            newv = end = nv;
        end = nv;
    }
    p->pn_value = newv;
    
    /* va: tfree v in case of @xxx[par], because vec_get created a new vec and
       nobody will free it elsewhere */
    if (v && v->v_name && *v->v_name=='@' && isreal(v) && v->v_realdata) {
    	vec_free(v);
    }
    return (p);
}

/* Don't call this directly, always use the free_pnode() macro. 
   The linked pnodes do not necessarily form a perfect tree as some nodes get
   reused.  Hence, in this recursive walk trough the 'tree' we only free node
   that have their pn_use value at zero. Nodes that have pn_use values above
   zero have the link severed and their pn_use value decremented.
   In addition, we don't walk past nodes with pn_use values avoid zero, just
   in case we have a circular reference (this probable does not happen in
  practice, but it does no harm playing safe) */
void
free_pnode_x(struct pnode *t)
{
    if (!t)
	return;
    
    /* don't walk past nodes used elsewhere. We decrement the pn_use value here,
       but the link gets severed by the action of the free_pnode() macro */
    if(t->pn_use>1)
	t->pn_use--;
    else {
	/* pn_use is now 1, so its safe to free the pnode */
	free_pnode(t->pn_left);
	free_pnode(t->pn_right);
	free_pnode(t->pn_next);
	tfree(t->pn_name); /* va: it is a copy() of original string, can be free'd */
	if (t->pn_value)
	    vec_free(t->pn_value); /* patch by Stefan Jones */
	tfree(t);
    }
}

