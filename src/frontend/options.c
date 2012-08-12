/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * The user-supplied routine to deal with variables. Most variables we
 * don't use often, so just call cp_getvar when they are needed. Spice
 * variables, though, and a few commonly used ones are dealt with here.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteinp.h"

#include "circuits.h"
#include "options.h"
#include "variable.h"
#include "control.h"
#include "spiceif.h"


static void setdb(char *str);

bool ft_acctprint = FALSE, ft_noacctprint = FALSE, ft_listprint = FALSE;
bool ft_nodesprint = FALSE, ft_optsprint = FALSE, ft_noinitprint = FALSE;
bool ft_ngdebug = FALSE, ft_stricterror = FALSE;


/* The user-supplied routine to query the values of variables. This
 * recognises the $&varname notation, and also searches the values of
 * plot and circuit environment variables.
 */

struct variable *
cp_enqvar(char *word)
{
    struct dvec *d;
    struct variable *vv, *tv;
    struct plot *pl;
    int i;

    if (*word == '&') {

        word++;

        d = vec_get(word);
        if (!d)
            return (NULL);

        if (d->v_length == 1) {
            vv = alloc(struct variable);
            vv->va_next = NULL;
            vv->va_name = copy(word);
            vv->va_type = CP_REAL;
            if (isreal(d))
                vv->va_real = d->v_realdata[0];
            else
                vv->va_real = realpart(d->v_compdata[0]);
        } else {
            vv = alloc(struct variable);
            vv->va_next = NULL;
            vv->va_name = copy(word);
            vv->va_type = CP_LIST;
            vv->va_vlist = NULL;
            for (i = d->v_length - 1; i >= 0; i--) {
                tv = alloc(struct variable);
                tv->va_type = CP_REAL;
                if (isreal(d))
                    tv->va_real = d->v_realdata[i];
                else
                    tv->va_real = realpart(d->v_compdata[i]);
                tv->va_next = vv->va_vlist;
                vv->va_vlist = tv;
            }
        }

        if (d->v_link2)
            fprintf(cp_err,
                    "Warning: only one vector may be accessed with the $& notation.\n");
        return (vv);

    }

    if (plot_cur) {
        for (vv = plot_cur->pl_env; vv; vv = vv->va_next)
            if (eq(vv->va_name, word))
                return (vv);
        if (eq(word, "curplotname")) {
            vv = alloc(struct variable);
            vv->va_next = NULL;
            vv->va_name = word;
            vv->va_type = CP_STRING;
            vv->va_string = copy(plot_cur->pl_name);
        } else if (eq(word, "curplottitle")) {
            vv = alloc(struct variable);
            vv->va_next = NULL;
            vv->va_name = word;
            vv->va_type = CP_STRING;
            vv->va_string = copy(plot_cur->pl_title);
        } else if (eq(word, "curplotdate")) {
            vv = alloc(struct variable);
            vv->va_next = NULL;
            vv->va_name = word;
            vv->va_type = CP_STRING;
            vv->va_string = copy(plot_cur->pl_date);
        } else if (eq(word, "curplot")) {
            vv = alloc(struct variable);
            vv->va_next = NULL;
            vv->va_name = word;
            vv->va_type = CP_STRING;
            vv->va_string = copy(plot_cur->pl_typename);
        } else if (eq(word, "plots")) {
            vv = alloc(struct variable);
            vv->va_next = NULL;
            vv->va_name = word;
            vv->va_type = CP_LIST;
            vv->va_vlist = NULL;
            for (pl = plot_list; pl; pl = pl->pl_next) {
                tv = alloc(struct variable);
                tv->va_type = CP_STRING;
                tv->va_string = copy(pl->pl_typename);
                tv->va_next = vv->va_vlist;
                vv->va_vlist = tv;
            }
        }
        if (vv)
            return (vv);
    }

    if (ft_curckt)
        for (vv = ft_curckt->ci_vars; vv; vv = vv->va_next)
            if (eq(vv->va_name, word))
                return (vv);

    return (NULL);
}


/* Return the plot and ckt env vars, $plots, and $curplot{name,title,date,} */

void
cp_usrvars(struct variable **v1, struct variable **v2)
{
    struct variable *v, *tv;

    v =  plot_cur ? plot_cur->pl_env : NULL;

    if ((tv = cp_enqvar("plots")) != NULL) {
        tv->va_next = v;
        v = tv;
    }
    if ((tv = cp_enqvar("curplot")) != NULL) {
        tv->va_next = v;
        v = tv;
    }
    if ((tv = cp_enqvar("curplottitle")) != NULL) {
        tv->va_next = v;
        v = tv;
    }
    if ((tv = cp_enqvar("curplotname")) != NULL) {
        tv->va_next = v;
        v = tv;
    }
    if ((tv = cp_enqvar("curplotdate")) != NULL) {
        tv->va_next = v;
        v = tv;
    }

    *v1 = v;
    *v2 = ft_curckt ? ft_curckt->ci_vars : NULL;
}


/* Extract the .option lines from the deck */

struct line *
inp_getopts(struct line *deck)
{
    struct line *last = NULL, *opts = NULL, *dd, *next = NULL;

    for (dd = deck->li_next; dd; dd = next) {
        next = dd->li_next;
        if (ciprefix(".opt", dd->li_line)) {
            inp_casefix(dd->li_line);
            if (last)
                last->li_next = dd->li_next;
            else
                deck->li_next = dd->li_next;
            dd->li_next = opts;
            opts = dd;
        } else
            last = dd;
    }

    return (opts);
}


/* Extract the option lines from a comfile (spinit, .spiceinit) */
struct line *
inp_getoptsc(char *in_line, struct line *com_options)
{
    struct line *next = NULL;
    char* line;

    line = TMALLOC(char, strlen(in_line) + 3);
    /* option -> .options */
    /* skip option */
    gettok(&in_line);
    sprintf(line, ".options %s", in_line);

    next = TMALLOC(struct line, 1);
    next->li_line    = line;
    next->li_linenum = 0;
    next->li_error   = NULL;
    next->li_actual  = NULL;

    /* put new line in front */
    if (com_options)
        next->li_next = com_options;

    return next;
}


/* The one variable that we consider read-only so far is plots.  The ones
 * that are 'dontrecord' are curplottitle, curplotname, and curplotdate.
 * Also things already in the plot env are 'dontrecord'.
 */

int
cp_usrset(struct variable *var, bool isset)
{
    void *vv;
    struct variable *tv;
    int iv;
    double dv;
    bool bv;

    if (eq(var->va_name, "debug")) {
        if (var->va_type == CP_BOOL) {
            cp_debug = ft_simdb = ft_parsedb = ft_evdb = ft_vecdb =
                ft_grdb = ft_gidb = ft_controldb = isset;
        } else if (var->va_type == CP_LIST) {
            for (tv = var->va_vlist; tv; tv = tv->va_next)
                if (var->va_type == CP_STRING)
                    setdb(tv->va_string);
                else
                    fprintf(cp_err, "Error: bad type for debug var\n");
        } else if (var->va_type == CP_STRING) {
            setdb(var->va_string);
        } else
            fprintf(cp_err, "Error: bad type for debug var\n");
#ifndef FTEDEBUG
        fprintf(cp_err, "Warning: %s compiled without debug messages\n",
                cp_program);
#endif
    } else if (eq(var->va_name, "program")) {
        cp_program = copy(var->va_string);
    } else if (eq(var->va_name, "rawfile")) {
        ft_rawfile = copy(var->va_string);
    } else if (eq(var->va_name, "acct")) {
        ft_acctprint = isset;
    } else if (eq(var->va_name, "noacct")) {
        ft_noacctprint = isset;
    } else if (eq(var->va_name, "ngdebug")) {
        ft_ngdebug = isset;
    } else if (eq(var->va_name, "noinit")) {
        ft_noinitprint = isset;
    } else if (eq(var->va_name, "list")) {
        ft_listprint = isset;
    } else if (eq(var->va_name, "nopage")) {
        ft_nopage = isset;
    } else if (eq(var->va_name, "nomod")) {
        ft_nomod = isset;
    } else if (eq(var->va_name, "node")) {
        ft_nodesprint = isset;
    } else if (eq(var->va_name, "opts")) {
        ft_optsprint = isset;
    } else if (eq(var->va_name, "strictnumparse")) {
        ft_strictnumparse = isset;
    } else if (eq(var->va_name, "strict_errorhandling")) {
        ft_stricterror = isset;
    } else if (eq(var->va_name, "rawfileprec")) {
        if ((var->va_type == CP_BOOL) && (isset == FALSE))
            raw_prec = -1;
        else if (var->va_type == CP_REAL)
            raw_prec = (int)floor(var->va_real + 0.5);
        else if (var->va_type == CP_NUM)
            raw_prec = var->va_num;
        else
            fprintf(cp_err, "Bad 'rawfileprec' \"%s\"\n", var->va_name);
    } else if (eq(var->va_name, "numdgt")) {
        if ((var->va_type == CP_BOOL) && (isset == FALSE))
            cp_numdgt = -1;
        else if (var->va_type == CP_REAL)
            cp_numdgt = (int)floor(var->va_real + 0.5);
        else if (var->va_type == CP_NUM)
            cp_numdgt = var->va_num;
        else
            fprintf(cp_err, "Excuse me??\n");
    } else if (eq(var->va_name, "unixcom")) {
        cp_dounixcom = isset;
        if (isset) {
            char *s = getenv("PATH");
            if (s)
                cp_rehash(s, TRUE);
            else
                fprintf(cp_err, "Warning: no PATH in environment.\n");
        }
    } else if (eq(var->va_name, "units") && (var->va_type == CP_STRING)) {
        if (isset && ((*var->va_string == 'd') || (*var->va_string == 'D')))
            cx_degrees = TRUE;
        else
            cx_degrees = FALSE;
    } else if (eq(var->va_name, "curplot")) {
        if (var->va_type == CP_STRING)
            plot_setcur(var->va_string);
        else
            fprintf(cp_err, "Error: plot name not a string\n");
        return (US_DONTRECORD);
    } else if (eq(var->va_name, "curplotname")) {
        if (plot_cur && (var->va_type == CP_STRING))
            plot_cur->pl_name = copy(var->va_string);
        else
            fprintf(cp_err, "Error: can't set plot name\n");
        return (US_DONTRECORD);
    } else if (eq(var->va_name, "curplottitle")) {
        if (plot_cur && (var->va_type == CP_STRING))
            plot_cur->pl_title = copy(var->va_string);
        else
            fprintf(cp_err, "Error: can't set plot title\n");
        return (US_DONTRECORD);
    } else if (eq(var->va_name, "curplotdate")) {
        if (plot_cur && (var->va_type == CP_STRING))
            plot_cur->pl_date = copy(var->va_string);
        else
            fprintf(cp_err, "Error: can't set plot date\n");
        return (US_DONTRECORD);
    } else if (eq(var->va_name, "plots")) {
        return (US_READONLY);
    }

    if (plot_cur)
        for (tv = plot_cur->pl_env; tv; tv = tv->va_next)
            if (eq(tv->va_name, var->va_name))
                return (US_READONLY);

    /*
      if (ft_curckt)
      for (tv = ft_curckt->ci_vars; tv; tv = tv->va_next)
      if (eq(tv->va_name, var->va_name))
      return (US_READONLY);
    */

    if (ft_nutmeg)
        return (US_OK);

    /* Now call the interface option routine. */
    switch (var->va_type) {
    case CP_BOOL:
        bv = (var->va_bool) ? TRUE : FALSE;
        vv = &bv;
        break;
    case CP_STRING:
        vv = var->va_string;
        break;
    case CP_NUM:
        iv = var->va_num;
        vv = &iv;
        break;
    case CP_REAL:
        dv = var->va_real;
        vv = &dv;
        break;
    case CP_LIST:
        /* if_option can't handle lists anyway. */
        vv = NULL;
        break;
    default:
        fprintf(cp_err,
                "cp_usrset: Internal Error: Bad var type %d\n", var->va_type);
        return (0);
    }

    if (ft_curckt) {
        if (if_option(ft_curckt->ci_ckt, var->va_name, var->va_type, vv))
            return US_SIMVAR;
    } else {
        if (if_option(NULL, var->va_name, var->va_type, vv))
            return US_NOSIMVAR;
    }

    return (US_OK);
}


static void
setdb(char *str)
{
    if (eq(str, "siminterface"))
        ft_simdb = TRUE;
    else if (eq(str, "cshpar"))
        cp_debug = TRUE;
    else if (eq(str, "parser"))
        ft_parsedb = TRUE;
    else if (eq(str, "eval"))
        ft_evdb = TRUE;
    else if (eq(str, "vecdb"))
        ft_vecdb = TRUE;
    else if (eq(str, "graf"))
        ft_grdb = TRUE;
    else if (eq(str, "ginterface"))
        ft_gidb = TRUE;
    else if (eq(str, "control"))
        ft_controldb = TRUE;
    else if (eq(str, "async"))
        ft_asyncdb = TRUE;
    else
        fprintf(cp_err, "Warning: no such debug class %s\n", str);
}
