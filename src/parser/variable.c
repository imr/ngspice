/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cpdefs.h"
#include "fteext.h"
#include "ftedefs.h"
#include "variable.h"


bool cp_noglob = TRUE;
bool cp_nonomatch = FALSE;
bool cp_noclobber = FALSE;
bool cp_ignoreeof = FALSE;

struct variable *variables = NULL;

wordlist *
cp_varwl(struct variable *var)
{
    wordlist *wl = NULL, *w, *wx = NULL;
    char buf[BSIZE_SP];
    struct variable *vt;

    switch(var->va_type) {
        case VT_BOOL:
            /* Can't ever be FALSE. */
            (void) sprintf(buf, "%s", var->va_bool ? "TRUE" :
                    "FALSE");
            break;
        case VT_NUM:
            (void) sprintf(buf, "%d", var->va_num);
            break;
        case VT_REAL:
            /* This is a case where printnum isn't too good... */
            (void) sprintf(buf, "%G", var->va_real);
            break;
        case VT_STRING:
            (void) strcpy(buf, cp_unquote(var->va_string));
            break;
        case VT_LIST:   /* The tricky case. */
            for (vt = var->va_vlist; vt; vt = vt->va_next) {
                w = cp_varwl(vt);
                if (wl == NULL)
                    wl = wx = w;
                else {
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
    wl = alloc(struct wordlist);
    wl->wl_next = wl->wl_prev = NULL;
    wl->wl_word = copy(buf);
    return (wl);
}

/* Set a variable. */

void
cp_vset(char *varname, char type, char *value)
{
    struct variable *v, *u, *w;
    int i;
    bool alreadythere = FALSE;

/* for (v = variables; v; v = v->va_next) ; printf("ok while setting %s\n", 
        varname);*/
    varname = cp_unquote(varname);
    w = NULL;
    for (v = variables; v; v = v->va_next) {
        if (eq(varname, v->va_name)) {
            alreadythere = TRUE;
            break;
        }
	w = v;
    }
    if (!v) {
        v = alloc(struct variable);
        v->va_name = copy(varname);
        v->va_next = NULL;
    }
    switch (type) {
        case VT_BOOL:
        if (* ((bool *) value) == FALSE) {
            cp_remvar(varname);
            return;
        } else
            v->va_bool = TRUE;
        break;

        case VT_NUM:
        v->va_num = * (int *) value;
        break;

        case VT_REAL:
        v->va_real = * (double *) value;
        break;

        case VT_STRING:
        v->va_string = copy(value);
        break;

        case VT_LIST:
        v->va_vlist = (struct variable *) value;
        break;

        default:
        fprintf(cp_err, 
            "cp_vset: Internal Error: bad variable type %d.\n", 
                type);
        return;
    }
    v->va_type = type;

    /* Now, see if there is anything interesting going on. We recognise
     * these special variables: noglob, nonomatch, history, echo,
     * noclobber, prompt, and verbose. cp_remvar looks for these variables
     * too. The host program will get any others.
     */

    if (eq(varname, "noglob"))
        cp_noglob = TRUE;
    else if (eq(varname, "nonomatch"))
        cp_nonomatch = TRUE;
    else if (eq(varname, "history") && (type == VT_NUM))
        cp_maxhistlength = v->va_num;
    else if (eq(varname, "history") && (type == VT_REAL))
        cp_maxhistlength = v->va_real;
    else if (eq(varname, "noclobber"))
        cp_noclobber = TRUE;
    else if (eq(varname, "prompt") && (type == VT_STRING))
        cp_promptstring = copy(v->va_string);
    else if (eq(varname, "ignoreeof"))
        cp_ignoreeof = TRUE;
    else if (eq(varname, "cpdebug")) {
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
        break;

        case US_DONTRECORD:
        /* Do nothing... */
        if (alreadythere) {
            fprintf(cp_err, 
    "cp_vset: Internal Error: %s already there, but 'dont record'\n",
                    v->va_name);
        }
        break;
    
        case US_READONLY:
        fprintf(cp_err, "Error: %s is a read-only variable.\n",
                v->va_name);
        if (alreadythere)
            fprintf(cp_err, 
        "cp_vset: Internal Error: it was already there too!!\n");
        break;

        case US_SIMVAR:
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
		if (eq(varname, u->va_name)) {
		    alreadythere = TRUE;
		    break;
		}
	    if (!alreadythere) {
		v->va_next = ft_curckt->ci_vars;
		ft_curckt->ci_vars = v;
	    } else {
		w = u->va_next;
		bcopy(v, u, sizeof(*u));
		u->va_next = w;
	    }
	}
	break;

        case US_NOSIMVAR:
	/* What do you do? */
	tfree(v);
	break;

        default:
        fprintf(cp_err, "cp_vset: Internal Error: bad US val %d\n", i);
        break;
    }

    return;
}

struct variable *
cp_setparse(wordlist *wl)
{
    char *name, *val, *s, *ss;
    double *td;
    struct variable *listv = NULL, *vv, *lv = NULL;
    struct variable *vars = NULL;
    int balance;

    while (wl) {
        name = cp_unquote(wl->wl_word);
        wl = wl->wl_next;
        if (((wl == NULL) || (*wl->wl_word != '=')) && 
               strchr(name, '=') == NULL) {
            vv = alloc(struct variable);
            vv->va_name = copy(name);
            vv->va_type = VT_BOOL;
            vv->va_bool = TRUE;
            vv->va_next = vars;
            vars = vv;
            continue;
        }
        if (wl && eq(wl->wl_word, "=")) {
            wl = wl->wl_next;
            if (wl == NULL) {
                fprintf(cp_err, "Error: bad set form.\n");
                return (NULL);
            }
            val = wl->wl_word;
            wl = wl->wl_next;
        } else if (wl && (*wl->wl_word == '=')) {
            val = wl->wl_word + 1;
            wl = wl->wl_next;
        } else if ((s =strchr(name, '='))) {
            val = s + 1;
            *s = '\0';
            if (*val == '\0') {
                if (!wl) {
                    fprintf(cp_err,
                        "Error:  %s equals what?.\n",
                        name);
                    return (NULL);
                } else {
                    val = wl->wl_word;
                    wl = wl->wl_next;
                }
            }
        } else {
            fprintf(cp_err, "Error: bad set form.\n");
            return (NULL);
        }
        val = cp_unquote(val);
        if (eq(val, "(")) { /* ) */
            /* The beginning of a list... We have to walk down
             * the list until we find a close paren... If there
             * are nested ()'s, treat them as tokens...
             */
            balance = 1;
            while (wl && wl->wl_word) {
                if (eq(wl->wl_word, "(")) { /* ) ( */
                    balance++;
                } else if (eq(wl->wl_word, ")")) {
                    if (!--balance)
                        break;
                }
                vv = alloc(struct variable);
		vv->va_next = NULL;
                ss = cp_unquote(wl->wl_word);
                td = ft_numparse(&ss, FALSE);
                if (td) {
                    vv->va_type = VT_REAL;
                    vv->va_real = *td;
                } else {
                    vv->va_type = VT_STRING;
                    vv->va_string = copy(ss);
                }
                if (listv) {
                    lv->va_next = vv;
                    lv = vv;
                } else
                    listv = lv = vv;
                wl = wl->wl_next;
            }
            if (balance && !wl) {
                fprintf(cp_err, "Error: bad set form.\n");
                return (NULL);
            }
            
            vv = alloc(struct variable);
            vv->va_name = copy(name);
            vv->va_type = VT_LIST;
            vv->va_vlist = listv;
            vv->va_next = vars;
            vars = vv;

            wl = wl->wl_next;
            continue;
        }

        ss = cp_unquote(val);
        td = ft_numparse(&ss, FALSE);
        vv = alloc(struct variable);
        vv->va_name = copy(name);
        vv->va_next = vars;
        vars = vv;
        if (td) {
            /*** We should try to get VT_NUM's... */
            vv->va_type = VT_REAL;
            vv->va_real = *td;
        } else {
            vv->va_type = VT_STRING;
            vv->va_string = copy(val);
        }
    }
    return (vars);
}

void
cp_remvar(char *varname)
{
    struct variable *v, *u, *lv = NULL;
    bool found = TRUE;
    int i;

    for (v = variables; v; v = v->va_next) {
        if (eq(v->va_name, varname))
            break;
        lv = v;
    }
    if (!v) {
        /* Gotta make up a var struct for cp_usrset()... */
        v = alloc(struct variable);
	ZERO(v, struct variable);
        v->va_name = varname;
        v->va_type = VT_NUM;
        v->va_bool = 0;
        found = FALSE;
    }

    /* Note that 'unset history' doesn't do anything here... Causes
     * trouble...
     */
    if (eq(varname, "noglob"))
        cp_noglob = FALSE;
    else if (eq(varname, "nonomatch"))
        cp_nonomatch = FALSE;
    else if (eq(varname, "noclobber"))
        cp_noclobber = FALSE;
    else if (eq(varname, "prompt"))
        cp_promptstring = "";
    else if (eq(varname, "cpdebug"))
        cp_debug = FALSE;
    else if (eq(varname, "ignoreeof"))
        cp_ignoreeof = FALSE;

    switch (i = cp_usrset(v, FALSE)) {

        case US_OK:
        /* Normal case. */
        if (found) {
            if (lv)
                lv->va_next = v->va_next;
            else
                variables = v->va_next;
        }
        break;

        case US_DONTRECORD:
        /* Do nothing... */
        if (found)
            fprintf(cp_err,
        "cp_remvar: Internal Error: var %d\n", *varname);
        break;
    
        case US_READONLY:
        /* Badness... */
        fprintf(cp_err, "Error: %s is read-only.\n",
                v->va_name);
        if (found)
            fprintf(cp_err,
        "cp_remvar: Internal Error: var %d\n", *varname);
        break;

        case US_SIMVAR:
	lv = NULL;
	if (ft_curckt) {
	    for (u = ft_curckt->ci_vars; u; u = u->va_next) {
		if (eq(varname, u->va_name)) {
		    break;
		}
		lv = u;
	    }
	    if (u) {
		if (lv)
		    lv->va_next = u->va_next;
		else
		    ft_curckt->ci_vars = u->va_next;
		tfree(u);
	    }
	}
	break;

        default:
        fprintf(cp_err, "cp_remvar: Internal Error: US val %d\n", i);
        break;
    }

    tfree(v);
    return;
}
