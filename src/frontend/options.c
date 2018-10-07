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


/* The user-supplied routine to query the address of a variable, if its
 * name is given. This recognises the $&varname notation, and also
 * searches the address of plot and circuit environment variables.
 * tbfreed is set to 1, if the variable is malloced here and may safely
 * be freed, and is set to 0 if plot and circuit environment variables
 * are returned.
 */

struct variable *
cp_enqvar(char *word, int *tbfreed)
{
    struct dvec *d;
    struct variable *vv;

    if (*word == '&') {

        word++;
        *tbfreed = 1;

        d = vec_get(word);
        if (!d)
            return (NULL);

        if (d->v_link2)
            fprintf(cp_err,
                    "Warning: only one vector may be accessed with the $& notation.\n");

        if (d->v_length == 1) {
            double value = isreal(d)
                ? d->v_realdata[0]
                : realpart(d->v_compdata[0]);
            return var_alloc_real(copy(word), value, NULL);
        } else {
            struct variable *list = NULL;
            int i;
            for (i = d->v_length; --i >= 0;) {
                double value = isreal(d)
                    ? d->v_realdata[i]
                    : realpart(d->v_compdata[i]);
                list = var_alloc_real(NULL, value, list);
            }
            return var_alloc_vlist(copy(word), list, NULL);
        }
    }

    if (plot_cur) {
        *tbfreed = 0;
        for (vv = plot_cur->pl_env; vv; vv = vv->va_next)
            if (eq(vv->va_name, word))
                return (vv);
        *tbfreed = 1;
        if (eq(word, "curplotname"))
            return var_alloc_string(copy(word), copy(plot_cur->pl_name), NULL);
        if (eq(word, "curplottitle"))
            return var_alloc_string(copy(word), copy(plot_cur->pl_title), NULL);
        if (eq(word, "curplotdate"))
            return var_alloc_string(copy(word), copy(plot_cur->pl_date), NULL);
        if (eq(word, "curplot"))
            return var_alloc_string(copy(word), copy(plot_cur->pl_typename), NULL);
        if (eq(word, "plots")) {
            struct variable *list = NULL;
            struct plot *pl;
            for (pl = plot_list; pl; pl = pl->pl_next)
                list = var_alloc_string(NULL, copy(pl->pl_typename), list);
            return var_alloc_vlist(copy(word), list, NULL);
        }
    }

    *tbfreed = 0;
    if (ft_curckt)
        for (vv = ft_curckt->ci_vars; vv; vv = vv->va_next)
            if (eq(vv->va_name, word))
                return (vv);

    return (NULL);
}


/* Return $plots, $curplot, $curplottitle, $curplotname, $curplotdate */

struct variable *
cp_usrvars(void)
{
    struct variable *v, *tv;
    int tbfreed;

    v = NULL;

    if ((tv = cp_enqvar("plots", &tbfreed)) != NULL) {
        tv->va_next = v;
        v = tv;
    }
    if ((tv = cp_enqvar("curplot", &tbfreed)) != NULL) {
        tv->va_next = v;
        v = tv;
    }
    if ((tv = cp_enqvar("curplottitle", &tbfreed)) != NULL) {
        tv->va_next = v;
        v = tv;
    }
    if ((tv = cp_enqvar("curplotname", &tbfreed)) != NULL) {
        tv->va_next = v;
        v = tv;
    }
    if ((tv = cp_enqvar("curplotdate", &tbfreed)) != NULL) {
        tv->va_next = v;
        v = tv;
    }

    return v;
}


/* Extract the .option lines from the deck */

struct card *
inp_getopts(struct card *deck)
{
    struct card *last = NULL, *opts = NULL, *dd, *next = NULL;

    for (dd = deck->nextcard; dd; dd = next) {
        next = dd->nextcard;
        if (ciprefix(".opt", dd->line)) {
            inp_casefix(dd->line);
            if (last)
                last->nextcard = dd->nextcard;
            else
                deck->nextcard = dd->nextcard;
            dd->nextcard = opts;
            opts = dd;
        } else {
            last = dd;
        }
    }

    return (opts);
}


/* copy the given option line,
 *   (presumably from a comfile, e.g. spinit or .spiceinit)
 * substitute '.options' for 'option'
 * then put it in front of the given 'options' list */

struct card *
inp_getoptsc(char *line, struct card *options)
{
    line = nexttok(line);           /* skip option */

    struct card *next = TMALLOC(struct card, 1);

    next->line    = tprintf(".options %s", line);
    next->linenum = 0;
    next->error   = NULL;
    next->actualLine  = NULL;

    /* put new line in front */
    next->nextcard = options;

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
        } else {
            fprintf(cp_err, "Error: bad type for debug var\n");
        }
#ifndef FTEDEBUG
        fprintf(cp_err, "Warning: %s compiled without debug messages\n",
                cp_program);
#endif
    } else if (eq(var->va_name, "program")) {
        cp_program = var->va_string;
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
        if (plot_cur && (var->va_type == CP_STRING)) {
            FREE(plot_cur->pl_name);
            plot_cur->pl_name = copy(var->va_string);
        }
        else
            fprintf(cp_err, "Error: can't set plot name\n");
        return (US_DONTRECORD);
    } else if (eq(var->va_name, "curplottitle")) {
        if (plot_cur && (var->va_type == CP_STRING)) {
            FREE(plot_cur->pl_title);
            plot_cur->pl_title = copy(var->va_string);
        }
        else
            fprintf(cp_err, "Error: can't set plot title\n");
        return (US_DONTRECORD);
    } else if (eq(var->va_name, "curplotdate")) {
        if (plot_cur && (var->va_type == CP_STRING)) {
            FREE(plot_cur->pl_date);
            plot_cur->pl_date = copy(var->va_string);
        }
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
