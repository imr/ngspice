/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/wordlist.h"
#include "ngspice/defines.h"
#include "ngspice/macros.h"
#include "ngspice/cpdefs.h"
#include "ngspice/memory.h"
#include "ngspice/inpdefs.h"
#include "ngspice/fteext.h"

#include "circuits.h"
#include "com_history.h"
#include "quote.h"
#include "ngspice/cpextern.h"
#include "variable.h"


bool cp_noglob = TRUE;
bool cp_nonomatch = FALSE;
bool cp_noclobber = FALSE;
bool cp_ignoreeof = FALSE;
bool cp_echo = FALSE;   /* CDHW */

struct variable *variables = NULL;


wordlist *
cp_varwl(struct variable *var)
{
    wordlist *wl = NULL, *w, *wx = NULL;
    char *buf;
    struct variable *vt;

    switch (var->va_type) {
    case CP_BOOL:
        /* Can't ever be FALSE. */
        buf = copy(var->va_bool ? "TRUE" : "FALSE");
        break;
    case CP_NUM:
        buf = tprintf("%d", var->va_num);
        break;
    case CP_REAL:
        /* This is a case where printnum isn't too good... */
        buf = tprintf("%G", var->va_real);
        break;
    case CP_STRING:
        buf = cp_unquote(var->va_string);
        break;
    case CP_LIST:   /* The tricky case. */
        for (vt = var->va_vlist; vt; vt = vt->va_next) {
            w = cp_varwl(vt);
            if (wl == NULL) {
                wl = wx = w;
            } else {
                wx->wl_next = w;
                w->wl_prev = wx;
                wx = w;
            }
        }
        return (wl);
    default:
        fprintf(cp_err,
                "cp_varwl: Internal Error: bad variable type %d\n",
                var->va_type);
        return (NULL);
    }

    return wl_cons(buf, NULL);
}


/* Set a variable. */
void
cp_vset(char *varname, enum cp_types type, void *value)
{
    struct variable *v, *u, *w;
    int i;
    bool alreadythere = FALSE, v_free = FALSE;
    char *copyvarname;

    /* varname = cp_unquote(varname);  DG: Memory leak old varname is lost*/

    copyvarname = cp_unquote(varname);

    w = NULL;
    for (v = variables; v; v = v->va_next) {
        if (eq(copyvarname, v->va_name)) {
            alreadythere = TRUE;
            break;
        }
        w = v;
    }

    if (alreadythere) {
        if (v->va_type == CP_LIST)
            free_struct_variable(v->va_vlist);
        if (v->va_type == CP_STRING)
            tfree(v->va_string);
    }

    if (!v) {
        v = var_alloc(copy(copyvarname), NULL);
        v_free = TRUE;
    }

    switch (type) {
    case CP_BOOL:
        if (* ((bool *) value) == FALSE) {
            cp_remvar(copyvarname);
            if (v_free) {
                tfree(v->va_name);
                tfree(v);
            }
            tfree(copyvarname);
            return;
        } else {
            var_set_bool(v, TRUE);
        }
        break;

    case CP_NUM:
        var_set_num(v, * (int *) value);
        break;

    case CP_REAL:
        var_set_real(v, * (double *) value);
        break;

    case CP_STRING:
        var_set_string(v, copy((char*) value));
        break;

    case CP_LIST:
        var_set_vlist(v, (struct variable *) value);
        break;

    default:
        fprintf(cp_err,
                "cp_vset: Internal Error: bad variable type %d.\n",
                type);
        tfree(copyvarname);
        return;
    }

    /* Now, see if there is anything interesting going on. We
     * recognise these special variables: noglob, nonomatch, history,
     * echo, noclobber, prompt, and verbose. cp_remvar looks for these
     * variables too. The host program will get any others.  */
    if (eq(copyvarname, "noglob"))
        cp_noglob = TRUE;
    else if (eq(copyvarname, "nonomatch"))
        cp_nonomatch = TRUE;
    else if (eq(copyvarname, "history") && (type == CP_NUM))
        cp_maxhistlength = v->va_num;
    else if (eq(copyvarname, "history") && (type == CP_REAL))
        cp_maxhistlength = (int)floor(v->va_real + 0.5);
    else if (eq(copyvarname, "noclobber"))
        cp_noclobber = TRUE;
    else if (eq(varname, "echo"))   /*CDHW*/
        cp_echo = TRUE;             /*CDHW*/
    else if (eq(copyvarname, "prompt") && (type == CP_STRING))
        cp_promptstring = v->va_string;
    else if (eq(copyvarname, "ignoreeof"))
        cp_ignoreeof = TRUE;
    else if (eq(copyvarname, "cpdebug")) {
        cp_debug = TRUE;
#ifndef CPDEBUG
        fprintf(cp_err,
                "Warning: program not compiled with cshpar debug messages\n");
#endif
    }

    switch (i = cp_usrset(v, TRUE)) {

    case US_OK:
        /* Normal case. */
        if (!alreadythere) {
            v->va_next = variables;
            variables = v;
        }
        else if (v_free)
            free_struct_variable(v);
        break;

    case US_DONTRECORD:
        /* 'curplot' 'curplotname' 'curplottitle' 'curplotdate' */
        /* Do nothing... */
        if (alreadythere) {
            fprintf(cp_err, "cp_vset: Internal Error: "
                    "%s already there, but 'dont record'\n", v->va_name);
        }
        if (v_free)
            free_struct_variable(v);
        break;

    case US_READONLY:
        /* 'plots' and any var in plot_cur->pl_env */
        fprintf(cp_err, "Error: %s is a read-only variable.\n", v->va_name);
        if (alreadythere)
            fprintf(cp_err, "cp_vset: Internal Error: "
                    "it was already there too!!\n");
        break;

    case US_SIMVAR:
        /* variables processed by if_option(ft_curckt->ci_ckt, ...) */
        if (alreadythere) {
            /* somehow it got into the front-end list of variables */
            if (w) {
                w->va_next = v->va_next;
            } else {
                variables = v->va_next;
            }
        }
        alreadythere = FALSE;
        if (ft_curckt) {
            for (u = ft_curckt->ci_vars; u; u = u->va_next)
                if (eq(copyvarname, u->va_name)) {
                    alreadythere = TRUE;
                    break;
                }
            if (!alreadythere) {
                v->va_next = ft_curckt->ci_vars;
                ft_curckt->ci_vars = v;
            } else {
                if (u->va_type == CP_STRING)
                    tfree(u->va_string);
                else if (u->va_type == CP_LIST)
                    tfree(u->va_vlist);
                u->va_V = v->va_V;
                u->va_type = v->va_type;
                /* va_name is the same string */
                tfree(u->va_name);
                u->va_name = v->va_name;
                /* va_next left unchanged */
                tfree(v);
            }
        }
        break;

    case US_NOSIMVAR:
        /* variables processed by if_option(NULL, ...) */
        /* What do you do? */
        free_struct_variable(v);
        break;

    default:
        fprintf(cp_err, "cp_vset: Internal Error: bad US val %d\n", i);
        break;
    }

    tfree(copyvarname);
}


/*CDHW This needs leak checking carefully CDHW*/
struct variable *
cp_setparse(wordlist *wl)
{
    char *name = NULL, *val, *copyval, *s, *ss;
    double *td;
    struct variable *listv = NULL, *vv, *lv = NULL;
    struct variable *vars = NULL;
    int balance;

    while (wl) {

        if (name)
            tfree(name);

        name = cp_unquote(wl->wl_word);

        wl = wl->wl_next;
        if ((!wl || (*wl->wl_word != '=')) && !strchr(name, '=')) {
            vars = var_alloc_bool(copy(name), TRUE, vars);
            tfree(name);        /*DG: cp_unquote Memory leak*/
            continue;
        }

        if (wl && eq(wl->wl_word, "=")) {
            wl = wl->wl_next;
            if (wl == NULL) {
                fprintf(cp_err, "Error: bad set form.\n");
                tfree(name);    /*DG: cp_unquote Memory leak*/
                if (ft_stricterror)
                    controlled_exit(EXIT_BAD);
                return (NULL);
            }
            val = wl->wl_word;
            wl = wl->wl_next;
        } else if (wl && (*wl->wl_word == '=')) {
            val = wl->wl_word + 1;
            wl = wl->wl_next;
        } else if ((s = strchr(name, '=')) != NULL) {
            val = s + 1;
            *s = '\0';
            if (*val == '\0') {
                if (!wl) {
                    fprintf(cp_err, "Error:  %s equals what?.\n", name);
                    tfree(name); /*DG: cp_unquote Memory leak: free name before exiting*/
                    if (ft_stricterror)
                        controlled_exit(EXIT_BAD);
                    return (NULL);
                } else {
                    val = wl->wl_word;
                    wl = wl->wl_next;
                }
            }
        } else {
            fprintf(cp_err, "Error: bad set form.\n");
            tfree(name); /*DG: cp_unquote Memory leak: free name befor exiting */
            if (ft_stricterror)
                controlled_exit(EXIT_BAD);
            return (NULL);
        }

        /*   val = cp_unquote(val);  DG: bad   old val is lost*/
        copyval = cp_unquote(val); /*DG*/
        strcpy(val, copyval);
        tfree(copyval);

        if (eq(val, "(")) {
            /* The beginning of a list... We have to walk down the
             * list until we find a close paren... If there are nested
             * ()'s, treat them as tokens...  */
            balance = 1;
            while (wl && wl->wl_word) {
                if (eq(wl->wl_word, "(")) {
                    balance++;
                } else if (eq(wl->wl_word, ")")) {
                    if (!--balance)
                        break;
                }
                copyval = ss = cp_unquote(wl->wl_word);
                td = ft_numparse(&ss, FALSE);
                if (td)
                    vv = var_alloc_real(NULL, *td, NULL);
                else
                    vv = var_alloc_string(NULL, copy(ss), NULL);
                tfree(copyval); /*DG: must free ss any way to avoid cp_unquote memory leak*/
                if (listv) {
                    lv->va_next = vv;
                    lv = vv;
                } else {
                    listv = lv = vv;
                }
                wl = wl->wl_next;
            }
            if (balance && !wl) {
                fprintf(cp_err, "Error: bad set form.\n");
                tfree(name); /* va: cp_unquote memory leak: free name before exiting */
                if (ft_stricterror)
                    controlled_exit(EXIT_BAD);
                return (NULL);
            }

            vars = var_alloc_vlist(copy(name), listv, vars);

            wl = wl->wl_next;
            continue;
        }

        copyval = ss = cp_unquote(val);
        td = ft_numparse(&ss, FALSE);
        if (td) {
            /*** We should try to get CP_NUM's... */
            vars = var_alloc_real(copy(name), *td, vars);
        } else {
            vars = var_alloc_string(copy(name), copy(val), vars);
        }
        tfree(copyval); /*DG: must free ss any way to avoid cp_unquote memory leak */
        tfree(name);  /* va: cp_unquote memory leak: free name for every loop */
    }

    if (name)
        tfree(name);
    return (vars);
}


/* free the struct variable. The type of the union is given by va_type */
void
free_struct_variable(struct variable *v)
{
    while (v) {
        struct variable *next_v = v->va_next;
        if (v->va_name)
            tfree(v->va_name);
        if (v->va_type == CP_LIST)
            free_struct_variable(v->va_vlist);
        if (v->va_type == CP_STRING)
            tfree(v->va_string);
        tfree(v);
        v = next_v;
    }
}


void
cp_remvar(char *varname)
{
    struct variable *v, **p;
    struct variable *uv1;
    int i;

    uv1 = cp_usrvars();

    for (p = &variables; *p; p = &(*p)->va_next)
        if (eq((*p)->va_name, varname))
            break;

    if (*p == NULL)
        for (p = &uv1; *p; p = &(*p)->va_next)
            if (eq((*p)->va_name, varname))
                break;

    if (*p == NULL && plot_cur)
        for (p = &plot_cur->pl_env; *p; p = &(*p)->va_next)
            if (eq((*p)->va_name, varname))
                break;

    if (*p == NULL && ft_curckt)
        for (p = &ft_curckt->ci_vars; *p; p = &(*p)->va_next)
            if (eq((*p)->va_name, varname))
                break;

    v = *p;

    /* make up an auxiliary struct variable for cp_usrset() */
    if (!v)
        v = var_alloc_num(copy(varname), 0, NULL);

    /* Note that 'unset history' doesn't do anything here... Causes
     * trouble...  */
    if (eq(varname, "noglob"))
        cp_noglob = FALSE;
    else if (eq(varname, "nonomatch"))
        cp_nonomatch = FALSE;
    else if (eq(varname, "noclobber"))
        cp_noclobber = FALSE;
    else if (eq(varname, "echo")) /*CDHW*/
        cp_echo = FALSE;          /*CDHW*/
    else if (eq(varname, "prompt"))
        cp_promptstring = NULL;
    else if (eq(varname, "cpdebug"))
        cp_debug = FALSE;
    else if (eq(varname, "ignoreeof"))
        cp_ignoreeof = FALSE;
    else if (eq(varname, "program"))
        cp_program = "";

    switch (i = cp_usrset(v, FALSE)) {

    case US_OK:
        /* Normal case. */
        if (*p) {
            *p = v->va_next;
        }
        break;

    case US_DONTRECORD:
        /* 'curplot' 'curplotname' 'curplottitle' 'curplotdate' */
        /* Do nothing... */
        if (*p)
            fprintf(cp_err, "cp_remvar: Internal Error: var %d\n", *varname);
        break;

    case US_READONLY:
        /* 'plots' and any var in plot_cur->pl_env */
        /* Badness... */
        fprintf(cp_err, "Error: %s is read-only.\n", v->va_name);
        if (*p)
            fprintf(cp_err, "cp_remvar: Internal Error: var %d\n", *varname);
        break;

    case US_SIMVAR:
        /* variables processed by if_option(ft_curckt->ci_ckt, ...) */
        fprintf(stderr, "it's a US_SIMVAR!\n");
        if (ft_curckt) {
            for (p = &ft_curckt->ci_vars; *p; p = &(*p)->va_next)
                if (eq(varname, (*p)->va_name))
                    break;
            if (*p) {
                struct variable *u = *p;
                *p = u->va_next;
                tfree(u);
            }
        }
        break;

    case US_NOSIMVAR:
    default:
        /* variables processed by if_option(NULL, ...) */
        fprintf(cp_err, "cp_remvar: Internal Error: US val %d\n", i);
        break;
    }

    v->va_next = NULL;
    free_struct_variable(v);

    free_struct_variable(uv1);
}


/* Determine the value of a variable.  Fail if the variable is unset,
 * and if the type doesn't match, try and make it work...  */
bool
cp_getvar(char *name, enum cp_types type, void *retval)
{
    struct variable *v;
    struct variable *uv1;

    uv1 = cp_usrvars();

#ifdef TRACE
    /* SDB debug statement */
    fprintf(stderr, "in cp_getvar, trying to get value of variable %s.\n", name);
#endif

    for (v = variables; v; v = v->va_next)
        if (eq(name, v->va_name))
            break;

    if (!v)
        for (v = uv1; v; v = v->va_next)
            if (eq(name, v->va_name))
                break;

    if (!v && plot_cur)
        for (v = plot_cur->pl_env; v; v = v->va_next)
            if (eq(name, v->va_name))
                break;

    if (!v && ft_curckt)
        for (v = ft_curckt->ci_vars; v; v = v->va_next)
            if (eq(name, v->va_name))
                break;

    if (!v) {
        if (type == CP_BOOL && retval)
            *(bool *) retval = FALSE;
        free_struct_variable(uv1);
        return (FALSE);
    }

    if (v->va_type == type) {

        if (retval)
            switch (type) {
            case CP_BOOL:
                *(bool *) retval = TRUE;
                break;
            case CP_NUM:
                *(int *) retval = v->va_num;
                break;
            case CP_REAL:
                *(double *) retval = v->va_real;
                break;
            case CP_STRING: {   /* Gotta be careful to have room. */
                char *s = cp_unquote(v->va_string);
                cp_wstrip(s);
                strcpy((char*) retval, s);
                tfree(s);
                break;
            }
            case CP_LIST:       /* Funny case... */
                *(struct variable **) retval = v->va_vlist;
                break;
            default:
                fprintf(cp_err,
                        "cp_getvar: Internal Error: bad var type %d.\n", type);
                break;
            }

        free_struct_variable(uv1);
        return (TRUE);
    }

    /* Try to coerce it.. */
    if ((type == CP_NUM) && (v->va_type == CP_REAL)) {
        *(int *) retval = (int) v->va_real;
    } else if ((type == CP_REAL) && (v->va_type == CP_NUM)) {
        *(double *) retval = (double) v->va_num;
    } else if ((type == CP_STRING) && (v->va_type == CP_NUM)) {
        sprintf((char*) retval, "%d", v->va_num);
    } else if ((type == CP_STRING) && (v->va_type == CP_REAL)) {
        sprintf((char*) retval, "%f", v->va_real);
    } else {
        free_struct_variable(uv1);
        return (FALSE);
    }

    free_struct_variable(uv1);
    return (TRUE);
}


/* A variable substitution is indicated by a $, and the variable name
 * is the following string of non-special characters. All variable
 * values are inserted as a single word, except for lists, which are a
 * list of words.  A routine cp_usrset must be supplied by the host
 * program to deal with variables that aren't used by cshpar -- it
 * should be cp_usrset(var, isset), where var is a variable *, and
 * isset is TRUE if the variable is being set, FALSE if unset.  Also
 * required is a routine cp_enqvar(name) which returns a struct
 * variable *, which allows the host program to provide values for
 * non-cshpar variables.  */

char cp_dol = '$';

/* Non-alphanumeric characters that may appear in variable names. < is very
 * special...
 */

#define VALIDCHARS "$-_<#?@.()[]&"

char *
span_var_expr(char *t)
{
    int parenthesis = 0;
    int brackets = 0;

    while (*t && (isalnum_c(*t) || strchr(VALIDCHARS, *t)))
        switch (*t++)
        {
        case '[':
            brackets++;
            break;
        case '(':
            parenthesis++;
            break;
        case ']':
            if (brackets <= 0)
                return t-1;
            if (--brackets <= 0)
                return t;
            break;
        case ')':
            if (parenthesis <= 0)
                return t-1;
            if (--parenthesis <= 0)
                return t;
            break;
        default:
            break;
        }

    return t;
}


/* Substitute variable name by its value and restore to wordlist */
wordlist *
cp_variablesubst(wordlist *wlist)
{
    wordlist *wl;

    for (wl = wlist; wl; wl = wl->wl_next) {

        char *s_dollar;
        int i = 0;

        while ((s_dollar = strchr(wl->wl_word + i, cp_dol)) != NULL) {

            int prefix_len = (int) (s_dollar - wl->wl_word);

            char *tail = span_var_expr(s_dollar + 1);
            char *var = copy_substring(s_dollar + 1, tail);

            wordlist *nwl = vareval(var);
            tfree(var);

            if (nwl) {
                char *x = nwl->wl_word;
                char *tail_ = copy(tail);
                nwl->wl_word = tprintf("%.*s%s", prefix_len, wl->wl_word, nwl->wl_word);
                free(x);
                if (wlist == wl)
                    wlist = nwl;
                wl = wl_splice(wl, nwl);
                i = (int) strlen(wl->wl_word);
                x = wl->wl_word;
                wl->wl_word = tprintf("%s%s", wl->wl_word, tail_);
                free(x);
                free(tail_);
            } else if (prefix_len || *tail) {
                char *x = wl->wl_word;
                wl->wl_word = tprintf("%.*s%s", prefix_len, wl->wl_word, tail);
                i = prefix_len;
                free(x);
            } else {
                wordlist *next = wl->wl_next;
                if (wlist == wl)
                    wlist = next;
                wl_delete_slice(wl, next);
                if (!next)
                    return wlist;
                wl = next;
                i = 0;
            }
        }
    }

    return (wlist);
}


/* Evaluate a variable. */
wordlist *
vareval(char *string)
{
    struct variable *v, *vfree = NULL;
    wordlist *wl;
    char buf[BSIZE_SP], *s;
    char *oldstring = copy(string);
    char *range = NULL;
    int i, up, low;

    cp_wstrip(string);
    if ((s = strchr(string, '[')) != NULL) {
        *s = '\0';
        range = s + 1;
    }

    switch (*string) {

    case '$':
        wl = wl_cons(tprintf("%d", getpid()), NULL);
        tfree(oldstring);
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
        tfree(oldstring);
        return (wl);

    case '?':
        string++;
        for (v = variables; v; v = v->va_next)
            if (eq(v->va_name, string))
                break;
        if (!v)
            vfree = v = cp_enqvar(string);
        wl = wl_cons(copy(v ? "1" : "0"), NULL);
        tfree(oldstring);
        free_struct_variable(vfree); /* bug: free still used struct variable */
        return (wl);

    case '#':
        string++;
        for (v = variables; v; v = v->va_next)
            if (eq(v->va_name, string))
                break;
        if (!v)
            vfree = v = cp_enqvar(string);
        if (!v) {
            fprintf(cp_err, "Error: %s: no such variable.\n", string);
            tfree(oldstring);
            return (NULL);
        }
        if (v->va_type == CP_LIST)
            for (v = v->va_vlist, i = 0; v; v = v->va_next)
                i++;
        else
            i = (v->va_type != CP_BOOL);
        wl = wl_cons(tprintf("%d", i), NULL);
        tfree(oldstring);
        free_struct_variable(vfree); /* bug: free still used struct variable */
        return (wl);

    case '\0':
        wl = wl_cons(copy("$"), NULL);
        tfree(oldstring);
        return (wl);
    }

    /* The notation var[stuff] has two meanings...  If this is a real
     * variable, then the [] denotes range, but if this is a strange
     * (e.g, device parameter) variable, it could be anything...
     */
    for (v = variables; v; v = v->va_next)
        if (eq(v->va_name, string))
            break;
    if (!v && isdigit_c(*string)) {
        for (v = variables; v; v = v->va_next)
            if (eq(v->va_name, "argv"))
                break;
        range = string;
    }
    if (!v) {
        range = NULL;
        string = oldstring;
        vfree = v = cp_enqvar(string);
    }
    if (!v && (s = getenv(string)) != NULL) {
        wl = wl_cons(copy(s), NULL);
        tfree(oldstring);
        return (wl);
    }
    if (!v) {
        fprintf(cp_err, "Error: %s: no such variable.\n", string);
        tfree(oldstring);
        return (NULL);
    }
    wl = cp_varwl(v);
    free_struct_variable(vfree); /* bug: free still used struct variable */

    /* Now parse and deal with 'range' ... */
    if (range) {
        /* rather crude fix when range itself is a $expression */
        wordlist *r = NULL;
        if (*range == '$') {
            char *t = ++range;
            if (*t == '&')
                t++;
            while (isalnum_c(*t))
                t++;
            *t = '\0';
            r = vareval(range);
            if (!r || r->wl_next) {
                fprintf(cp_err, "Error: %s: illegal index.\n", string);
                tfree(oldstring);
                wl_free(r);
                return NULL;
            }
            range = r->wl_word;
        }
        for (low = 0; isdigit_c(*range); range++)
            low = low * 10 + *range - '0';
        if ((*range == '-') && isdigit_c(range[1]))
            for (up = 0, range++; isdigit_c(*range); range++)
                up = up * 10 + *range - '0';
        else if (*range == '-')
            up = wl_length(wl);
        else
            up = low;
        up--, low--;
        wl = wl_range(wl, low, up);
        wl_free(r);
    }
    tfree(oldstring);
    return (wl);
}


struct xxx {
    struct variable *x_v;
    char x_char;
};


static int
vcmp(const void *a, const void *b)
{
    int i;
    struct xxx *v1 = (struct xxx *) a;
    struct xxx *v2 = (struct xxx *) b;

    if ((i = strcmp(v1->x_v->va_name, v2->x_v->va_name)) != 0)
        return (i);
    else
        return (v1->x_char - v2->x_char);
}


/* Print the values of currently defined variables. */
void
cp_vprint(void)
{
    struct variable *v;
    struct variable *uv1;
    wordlist *wl;
    int i, j;
    char *s;
    struct xxx *vars;

    uv1 = cp_usrvars();

    for (v = variables, i = 0; v; v = v->va_next)
        i++;
    for (v = uv1; v; v = v->va_next)
        i++;
    if (plot_cur)
        for (v = plot_cur->pl_env; v; v = v->va_next)
            i++;
    if (ft_curckt)
        for (v = ft_curckt->ci_vars; v; v = v->va_next)
            i++;

    vars = TMALLOC(struct xxx, i);

    out_init();
    for (v = variables, i = 0; v; v = v->va_next, i++) {
        vars[i].x_v = v;
        vars[i].x_char = ' ';
    }
    for (v = uv1; v; v = v->va_next, i++) {
        vars[i].x_v = v;
        vars[i].x_char = '*';
    }
    if (plot_cur)
        for (v = plot_cur->pl_env; v; v = v->va_next, i++) {
            vars[i].x_v = v;
            vars[i].x_char = '*';
        }
    if (ft_curckt)
        for (v = ft_curckt->ci_vars; v; v = v->va_next, i++) {
            vars[i].x_v = v;
            vars[i].x_char = '+';
        }

    qsort(vars, (size_t) i, sizeof(*vars), vcmp);

    for (j = 0; j < i; j++) {
        if (j && eq(vars[j].x_v->va_name, vars[j-1].x_v->va_name))
            continue;
        v = vars[j].x_v;
        if (v->va_type == CP_BOOL) {
            out_printf("%c %s\n", vars[j].x_char, v->va_name);
        } else {
            out_printf("%c %s\t", vars[j].x_char, v->va_name);
            wl = vareval(v->va_name);
            s = wl_flatten(wl);
            if (v->va_type == CP_LIST)
                out_printf("( %s )\n", s);
            else
                out_printf("%s\n", s);
        }
    }

    free_struct_variable(uv1);
    tfree(vars);
}

struct variable *
var_alloc(char *name, struct variable *next)
{
  struct variable *v = TMALLOC(struct variable, 1);
  ZERO(v, struct variable);
  v -> va_name = name;
  v -> va_next = next;
  return v;
}

struct variable *
var_alloc_bool(char *name, bool value, struct variable *next)
{
    struct variable *v = var_alloc(name, next);
    var_set_bool(v, value);
    return v;
}

struct variable *
var_alloc_num(char *name, int value, struct variable *next)
{
    struct variable *v = var_alloc(name, next);
    var_set_num(v, value);
    return v;
}

struct variable *
var_alloc_real(char *name, double value, struct variable *next)
{
    struct variable *v = var_alloc(name, next);
    var_set_real(v, value);
    return v;
}

struct variable *
var_alloc_string(char *name, char * value, struct variable *next)
{
    struct variable *v = var_alloc(name, next);
    var_set_string(v, value);
    return v;
}

struct variable *
var_alloc_vlist(char *name, struct variable * value, struct variable *next)
{
    struct variable *v = var_alloc(name, next);
    var_set_vlist(v, value);
    return v;
}

void
var_set_bool(struct variable *v, bool value)
{
  v->va_type = CP_BOOL;
  v->va_bool = value;
}

void
var_set_num(struct variable *v, int value)
{
  v->va_type = CP_NUM;
  v->va_num = value;
}

void
var_set_real(struct variable *v, double value)
{
  v->va_type = CP_REAL;
  v->va_real = value;
}

void
var_set_string(struct variable *v, char *value)
{
  v->va_type = CP_STRING;
  v->va_string = value;
}

void
var_set_vlist(struct variable *v, struct variable *value)
{
  v->va_type = CP_LIST;
  v->va_vlist = value;
}

void
cp_remvar_all(void)
{
    free_struct_variable(variables);
    variables = NULL;
}
