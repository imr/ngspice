/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Do variable substitution.
 */

#include "ngspice.h"
#include "cpdefs.h"
#include "var2.h"

/* Print the values of currently defined variables. */


extern struct variable *variables;

/* A variable substitution is
 * indicated by a $, and the variable name is the following string of
 * non-special characters. All variable values are inserted as a single
 * word, except for lists, which are a list of words.
 * A routine cp_usrset must be supplied by the host program to deal 
 * with variables that aren't used by cshpar -- it should be 
 * cp_usrset(var, isset), where var is a variable *, and isset is 
 * TRUE if the variable is being set, FALSE if unset.
 * Also required is a routine cp_enqvar(name) which returns a struct
 * variable *, which allows the host program to provide values for
 * non-cshpar variables.
 */

char cp_dol = '$';

/* Non-alphanumeric characters that may appear in variable names. < is very
 * special...
 */

#define VALIDCHARS "$-_<#?@.()[]&"

wordlist *
cp_variablesubst(wordlist *wlist)
{
    wordlist *wl, *nwl;
    char *s, *t, buf[BSIZE_SP], wbuf[BSIZE_SP], tbuf[BSIZE_SP];
	/* MW. tbuf holds curret word after wl_splice() calls free() on it */
    int i;

    for (wl = wlist; wl; wl = wl->wl_next) {
        t = wl->wl_word;
        i = 0;
        while ((s =strchr(t, cp_dol))) {
            while (t < s)
                wbuf[i++] = *t++;
            wbuf[i] = '\0';
            (void) strcpy(buf, ++s);
            s = buf;
            t++;
            while (*s && (isalphanum(*s) ||
                   strchr(VALIDCHARS, *s))) {
                /* Get s and t past the end of the var name. */
                t++;
                s++;
            }
            *s = '\0';
            nwl = vareval(buf);
            if (i) {
                (void) strcpy(buf, wbuf);
                if (nwl) {
                    (void) strcat(buf, nwl->wl_word);
                    tfree(nwl->wl_word);
                } else {
                    nwl = alloc(struct wordlist);
		    nwl->wl_next = nwl->wl_prev = NULL;
                }
                nwl->wl_word = copy(buf);
            }
	    
            	(void) strcpy(tbuf, t); /* MW. Save t*/
	    if (!(wl = wl_splice(wl, nwl)))
                return (NULL);
            /* This is bad... */
            for (wlist = wl; wlist->wl_prev; wlist = wlist->wl_prev)
                ;
            (void) strcpy(buf, wl->wl_word);
            i = strlen(buf);
            (void) strcat(buf, tbuf); /* MW. tbuf is used here only */
             
            tfree(wl->wl_word);
            wl->wl_word = copy(buf);
            t = &wl->wl_word[i];
            s = wl->wl_word;
            for (i = 0; s < t; s++)
                wbuf[i++] = *s;
        }
    }
    return (wlist);
}

/* Evaluate a variable. */

wordlist *
vareval(char *string)
{
    struct variable *v;
    wordlist *wl;
    char buf[BSIZE_SP], *s;
    char *oldstring = copy(string);
    char *range = NULL;
    int i, up, low;

    cp_wstrip(string);
    if ((s =strchr(string, '['))) {
        *s = '\0';
        range = s + 1;
    }

    switch (*string) {

        case '$':
        wl = alloc(struct wordlist);
        wl->wl_next = wl->wl_prev = NULL;


        (void) sprintf(buf, "%d", getpid());

        wl->wl_word = copy(buf);
        return (wl);

        case '<':
        (void) fflush(cp_out);
        if (!fgets(buf, BSIZE_SP, cp_in)) {
            clearerr(cp_in);
            (void) strcpy(buf, "EOF");
        }
        for (s = buf; *s && (*s != '\n'); s++)
            ;
        *s = '\0';
        wl = cp_lexer(buf);
        /* This is a hack. */
        if (!wl->wl_word)
            wl->wl_word = copy("");
        return (wl);
    
        case '?':
        wl = alloc(struct wordlist);
        wl->wl_next = wl->wl_prev = NULL;
        string++;
        for (v = variables; v; v = v->va_next)
            if (eq(v->va_name, string))
                break;
        if (!v)
            v = cp_enqvar(string);
        wl->wl_word = copy(v ? "1" : "0");
        return (wl);
        
        case '#':
        wl = alloc(struct wordlist);
        wl->wl_next = wl->wl_prev = NULL;
        string++;
        for (v = variables; v; v = v->va_next)
            if (eq(v->va_name, string))
                break;
        if (!v)
            v = cp_enqvar(string);
        if (!v) {
            fprintf(cp_err, "Error: %s: no such variable.\n",
                    string);
            return (NULL);
        }
        if (v->va_type == VT_LIST)
            for (v = v->va_vlist, i = 0; v; v = v->va_next)
                i++;
        else
            i = (v->va_type != VT_BOOL);
        (void) sprintf(buf, "%d", i);
        wl->wl_word = copy(buf);
        return (wl);

        case '\0':
        wl = alloc(struct wordlist);
        wl->wl_next = wl->wl_prev = NULL;
        wl->wl_word = copy("$");
        return (wl);
    }

    /* The notation var[stuff] has two meanings...  If this is a real
     * variable, then the [] denotes range, but if this is a strange
     * (e.g, device parameter) variable, it could be anything...
     */
    for (v = variables; v; v = v->va_next)
        if (eq(v->va_name, string))
            break;
    if (!v && isdigit(*string)) {
        for (v = variables; v; v = v->va_next)
            if (eq(v->va_name, "argv"))
                break;
        range = string;
    }
    if (!v) {
        range = NULL;
        string = oldstring;
        v = cp_enqvar(string);
    }
    if (!v && (s = getenv(string))) {
        wl = alloc(struct wordlist);
        wl->wl_next = wl->wl_prev = NULL;
        wl->wl_word = copy(s);
        return (wl);
    }
    if (!v) {
        fprintf(cp_err, "Error: %s: no such variable.\n", string);
        return (NULL);
    }
    wl = cp_varwl(v);

    /* Now parse and deal with 'range' ... */
    if (range) {
        for (low = 0; isdigit(*range); range++)
            low = low * 10 + *range - '0';
        if ((*range == '-') && isdigit(range[1]))
            for (up = 0, range++; isdigit(*range); range++)
                up = up * 10 + *range - '0';
        else if (*range == '-')
            up = wl_length(wl);
        else
            up = low;
        up--, low--;
        wl = wl_range(wl, low, up);
    }

    return (wl);
}


static int
vcmp(const void *a, const void *b)
{
    int i;
    struct xxx *v1 = (struct xxx *) a;
    struct xxx *v2 = (struct xxx *) b;
    if ((i = strcmp(v1->x_v->va_name, v2->x_v->va_name)))
        return (i);
    else
        return (v1->x_char - v2->x_char);
}



void
cp_vprint(void)
{
    struct variable *v;
    struct variable *uv1, *uv2;
    wordlist *wl;
    int i, j;
    char *s;
    struct xxx *vars;

    cp_usrvars(&uv1, &uv2);

    for (v = uv1, i = 0; v; v = v->va_next)
        i++;
    for (v = uv2; v; v = v->va_next)
        i++;
    for (v = variables; v; v = v->va_next)
        i++;
    
    vars = (struct xxx *) tmalloc(sizeof (struct xxx) * i);

    out_init();
    for (v = variables, i = 0; v; v = v->va_next, i++) {
        vars[i].x_v = v;
        vars[i].x_char = ' ';
    }
    for (v = uv1; v; v = v->va_next, i++) {
        vars[i].x_v = v;
        vars[i].x_char = '*';
    }
    for (v = uv2; v; v = v->va_next, i++) {
        vars[i].x_v = v;
        vars[i].x_char = '+';
    }

    qsort((char *) vars, i, sizeof (struct xxx), vcmp);

    for (j = 0; j < i; j++) {
        if (j && eq(vars[j].x_v->va_name, vars[j - 1].x_v->va_name))
            continue;
        v = vars[j].x_v;
        if (v->va_type == VT_BOOL) {
/*             out_printf("%c %s\n", vars[j].x_char, v->va_name); */
             sprintf(out_pbuf, "%c %s\n", vars[j].x_char, v->va_name);
         out_send(out_pbuf);
        } else {
            out_printf("%c %s\t", vars[j].x_char, v->va_name);
            wl = vareval(v->va_name);
            s = wl_flatten(wl);
            if (v->va_type == VT_LIST) {
                out_printf("( %s )\n", s);
            } else
                out_printf("%s\n", s);
        }
    }

    tfree(vars);
    return;
}

/* The set command. Syntax is 
 * set [opt ...] [opt = val ...]. Val may be a string, an int, a float,
 * or a list of the form (elt1 elt2 ...).
 */


void
com_set(wordlist *wl)
{
    struct variable *vars;
    char *s;

    if (wl == NULL) {
        cp_vprint();
        return;
    }
    vars = cp_setparse(wl);

    /* This is sort of a hassle... */
    while (vars) { 
        switch (vars->va_type) {
	case VT_BOOL:
            s = (char *) &vars->va_bool;
            break;
	case VT_NUM:
            s = (char *) &vars->va_num;
            break;
	case VT_REAL:
            s = (char *) &vars->va_real;
            break;
	case VT_STRING:
            s = vars->va_string;
            break;
	case VT_LIST:
            s = (char *) vars->va_vlist;
            break;
	default:
	    s = (char *) NULL;
        }
        cp_vset(vars->va_name, vars->va_type, s);
        vars = vars->va_next;
    }
    return;
}

void
com_unset(wordlist *wl)
{
    register char *name;
    struct variable *var, *nv;

    if (eq(wl->wl_word, "*")) {
        for (var = variables; var; var = nv) {
            nv = var->va_next;
            cp_remvar(var->va_name);
        }
        wl = wl->wl_next;
    }
    while (wl != NULL) {
        name = wl->wl_word;
        cp_remvar(name);
        wl = wl->wl_next;
    }
    return;
}

/* Shift a list variable, by default argv, one to the left (or more if a
 * second argument is given.
 */

void
com_shift(wordlist *wl)
{
    struct variable *v, *vv;
    char *n = "argv";
    int num = 1;

    if (wl) {
        n = wl->wl_word;
        wl = wl->wl_next;
    }
    if (wl)
        num = scannum(wl->wl_word);
    
    for (v = variables; v; v = v->va_next)
        if (eq(v->va_name, n))
            break;
    if (!v) {
        fprintf(cp_err, "Error: %s: no such variable\n", n);
        return;
    }
    if (v->va_type != VT_LIST) {
        fprintf(cp_err, "Error: %s not of type list\n", n);
        return;
    }
    for (vv = v->va_vlist; vv && (num > 0); num--)
        vv = vv->va_next;
    if (num) {
        fprintf(cp_err, "Error: variable %s not long enough\n", n);
        return;
    }

    v->va_vlist = vv;
    return;
}

/* Determine the value of a variable.  Fail if the variable is unset,
 * and if the type doesn't match, try and make it work...
 */

bool
cp_getvar(char *name, int type, char *retval)
{
    struct variable *v;

    for (v = variables; v; v = v->va_next)
        if (eq(name, v->va_name))
            break;
    if (v == NULL) {
        if (type == VT_BOOL)
            * (bool *) retval = FALSE; 
        return (FALSE);
    }
    if (v->va_type == type) {
        switch (type) {
            case VT_BOOL:
                * (bool *) retval = TRUE; 
                break;
            case VT_NUM: {
                int *i;
                i = (int *) retval;
                *i = v->va_num;
                break;
            }
            case VT_REAL: {
                double *d;
                d = (double *) retval;
                *d = v->va_real;
                break;
            }
            case VT_STRING: { /* Gotta be careful to have room. */
                char *s;
                s = cp_unquote(v->va_string);
                cp_wstrip(s);
                (void) strcpy(retval, s);
                break;
            }
            case VT_LIST: { /* Funny case... */
                struct variable **tv;
                tv = (struct variable **) retval;
                *tv = v->va_vlist;
                break;
            }
            default:
                fprintf(cp_err, 
                "cp_getvar: Internal Error: bad var type %d.\n",
                        type);
                break;
        }
        return (TRUE);
    } else {
        /* Try to coerce it.. */
        if ((type == VT_NUM) && (v->va_type == VT_REAL)) {
            int *i;
            i = (int *) retval;
            *i = (int) v->va_real;
            return (TRUE);
        } else if ((type == VT_REAL) && (v->va_type == VT_NUM)) {
            double *d;
            d = (double *) retval;
            *d = (double) v->va_num;
            return (TRUE);
        } else if ((type == VT_STRING) && (v->va_type == VT_NUM)) {
            (void) sprintf(retval, "%d", v->va_num);
            return (TRUE);
        } else if ((type == VT_STRING) && (v->va_type == VT_REAL)) {
            (void) sprintf(retval, "%f", v->va_real);
            return (TRUE);
        }
        return (FALSE);
    }
}

