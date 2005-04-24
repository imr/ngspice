/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000 AlansFixes
**********/

/*
 * Routines to query and alter devices.
 */

#include "ngspice.h"
#include "gendefs.h"
#include "cktdefs.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "dgen.h"

#include "circuits.h"
#include "device.h"
#include "variable.h"

#include "gens.h" /* wl_forall */

static wordlist *devexpand(char *name);
static void all_show(wordlist *wl, int mode);

/*
 *	show: list device operating point info
 *		show
 *		show devs : params
 *		show devs : params ; devs : params
 *		show dev dev dev : param param param , dev dev : param param
 *		show t : param param param, t : param param
 *
 */


static	int	count;


void
com_showmod(wordlist *wl)
{
    all_show(wl, 1);
}

void
com_show(wordlist *wl)
{
    all_show(wl, 0);
}

static void
all_show(wordlist *wl, int mode)
{
    wordlist	*params, *nextgroup, *thisgroup;
    wordlist	*prev, *next, *w;
    int		screen_width;
    dgen	*dg, *listdg;
    int		instances;
    int		i, j, n;
    int		param_flag, dev_flag;

    if (!ft_curckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
    }

    if (wl && wl->wl_word && eq(wl->wl_word, "-v")) {
	old_show(wl->wl_next);
	return;
    }

    if (!cp_getvar("width", VT_NUM, (char *) &screen_width))
	    screen_width = DEF_WIDTH;
    count = (screen_width - LEFT_WIDTH) / (DEV_WIDTH + 1);

    n = 0;
    do {
	prev = NULL;
	params = NULL;
	nextgroup = NULL;
	thisgroup = wl;
	param_flag = 0;
	dev_flag = 0;

	/* find the parameter list and the nextgroup */
	for (w = wl; w && !nextgroup; w = next) {
	    next = w->wl_next;
	    if (eq(w->wl_word, "++") || eq(w->wl_word, "all")) {
		if (params) {
			param_flag = DGEN_ALLPARAMS;
			if (prev)
				prev->wl_next = w->wl_next;
			else
				params = next;
		} else {
			dev_flag = DGEN_ALLDEVS;
			if (prev)
				prev->wl_next = w->wl_next;
			else
				thisgroup = next;
		}
		/* w must not be freed here */
		w = NULL;
	    } else if (eq(w->wl_word, "+")) {
		if (params) {
			param_flag = DGEN_DEFPARAMS;
			if (prev)
				prev->wl_next = w->wl_next;
			else
				params = next;
		} else {
			dev_flag = DGEN_DEFDEVS;
			if (prev)
				prev->wl_next = w->wl_next;
			else
				thisgroup = next;
		}
		/* w must not be freed here */
		w = NULL;
	    } else if (eq(w->wl_word, ":")) {
		/* w must not be freed here */
		w = NULL;
		if (!params) {
		    params = next;
		    if (prev)
			    prev->wl_next = NULL;
		    else
			    thisgroup = NULL;
		} else {
		    if (prev)
			prev->wl_next = next;
		    else
			params = next;
		}
	    } else if (eq(w->wl_word, ";") || eq(w->wl_word, ",")) {
		    nextgroup = next;
		    /* w must not be freed here */
		    w = NULL;
		    if (prev)
			prev->wl_next = NULL;
		    break;
	    }
	    prev = w;
	}

	instances = 0;
	for (dg = dgen_init(ft_curckt->ci_ckt, thisgroup, 1, dev_flag, mode);
		dg; dgen_nth_next(&dg, count))
	{
	    instances = 1;
	    if (dg->flags & DGEN_INSTANCE) {
		instances = 2;
		printf(" %s: %s\n",
			ft_sim->devices[dg->dev_type_no]->name,
			ft_sim->devices[dg->dev_type_no]->description);
		n += 1;

		i = 0;
		do {
		printf("%*s", LEFT_WIDTH, "device");
			j = dgen_for_n(dg, count, printstr, "n", i);
			i += 1;
			printf("\n");
		} while (j);

		if (ft_sim->devices[dg->dev_type_no]->numModelParms) {
			i = 0;
			do {
				printf("%*s", LEFT_WIDTH, "model");
				j = dgen_for_n(dg, count, printstr, "m", i);
				i += 1;
				printf("\n");
			} while (j);
		}
		listdg = dg;

		if (param_flag)
		    param_forall(dg, param_flag);
		else if (!params)
		    param_forall(dg, DGEN_DEFPARAMS);
		if (params)
		    wl_forall(params, listparam, dg);
		printf("\n");

	    } else if (ft_sim->devices[dg->dev_type_no]->numModelParms) {
		printf(" %s models (%s)\n",
			ft_sim->devices[dg->dev_type_no]->name,
			ft_sim->devices[dg->dev_type_no]->description);
		n += 1;
		i = 0;
		do {
		printf("%*s", LEFT_WIDTH, "model");
			j = dgen_for_n(dg, count, printstr, "m", i);
			i += 1;
			printf("\n");
		} while (j);
		printf("\n");

		if (param_flag)
		    param_forall(dg, param_flag);
		else if (!params)
		    param_forall(dg, DGEN_DEFPARAMS);
		if (params)
		    wl_forall(params, listparam, dg);
		printf("\n");
	    }
	}

	wl = nextgroup;

    } while (wl);

    if (!n) {
	    if (instances == 0)
		printf("No matching instances or models\n");
	    else if (instances == 1)
		printf("No matching models\n");
	    else
		printf("No matching elements\n");
    }
}

int
printstr(dgen *dg, char *name)
{
    /* va: ' ' is no flag for %s; \? avoids trigraph warning */
    if (*name == 'n') {
	if (dg->instance)
	   printf(" %*.*s", DEV_WIDTH, DEV_WIDTH, dg->instance->GENname);
	else
	   printf(" %*s", DEV_WIDTH, "<\?\?\?\?\?\?\?>");
    } else if (*name == 'm') {
	if (dg->model)
	   printf(" %*.*s", DEV_WIDTH, DEV_WIDTH, dg->model->GENmodName);
	else
	   printf(" %*s", DEV_WIDTH, "<\?\?\?\?\?\?\?>");
    } else
	printf(" %*s", DEV_WIDTH, "<error>");

    return 0;
}

void
param_forall(dgen *dg, int flags)
{
    int	i, j, k, found;
    int xcount;
    IFparm *plist;

    found = 0;

    if (dg->flags & DGEN_INSTANCE) {
	xcount = *ft_sim->devices[dg->dev_type_no]->numInstanceParms;
	plist = ft_sim->devices[dg->dev_type_no]->instanceParms;
    } else {
	xcount = *ft_sim->devices[dg->dev_type_no]->numModelParms;
	plist = ft_sim->devices[dg->dev_type_no]->modelParms;
    }

    for (i = 0; i < xcount; i++) {
	if (plist[i].dataType & IF_ASK) {
	    if ((((CKTcircuit *) (dg->ckt))->CKTrhsOld
		|| (plist[i].dataType & IF_SET))
		&& (!(plist[i].dataType & (IF_REDUNDANT | IF_UNINTERESTING))
		|| (flags == DGEN_ALLPARAMS
		&& !(plist[i].dataType & IF_REDUNDANT))))
	    {
		j = 0;
		do {
			if (!j)
			   printf("%*.*s", LEFT_WIDTH, LEFT_WIDTH,
                                           plist[i].keyword);
			else
			   printf("%*.*s", LEFT_WIDTH, LEFT_WIDTH, " ");
			k = dgen_for_n(dg, count, printvals, (plist + i), j);
			printf("\n");
			j += 1;
		} while (k);
	    }
	}
    }
}

void
listparam(wordlist *p, dgen *dg)
{
    int	i, j, k, found;
    int	xcount;
    IFparm *plist;

    found = 0;

    if (dg->flags & DGEN_INSTANCE) {
	xcount = *ft_sim->devices[dg->dev_type_no]->numInstanceParms;
	plist = ft_sim->devices[dg->dev_type_no]->instanceParms;
    } else {
	xcount = *ft_sim->devices[dg->dev_type_no]->numModelParms;
	plist = ft_sim->devices[dg->dev_type_no]->modelParms;
    }

    for (i = 0; i < xcount; i++) {
	if (eqc(p->wl_word, plist[i].keyword) && (plist[i].dataType & IF_ASK))
	{
	    found = 1;
	    break;
	}
    }

    if (found) {
	if ((((CKTcircuit *) (dg->ckt))->CKTrhsOld
	    || (plist[i].dataType & IF_SET)))
	{
	    j = 0;
	    do {
		if (!j)
		   printf("%*.*s", LEFT_WIDTH, LEFT_WIDTH, p->wl_word);
		else
		   printf("%*.*s", LEFT_WIDTH, LEFT_WIDTH, " ");
		k = dgen_for_n(dg, count, printvals, (plist + i), j);
		printf("\n");
		j += 1;
	    } while (k > 0);
	} else {
	    j = 0;
	    do {
		if (!j)
		   printf("%*.*s", LEFT_WIDTH, LEFT_WIDTH, p->wl_word);
		else
		   printf("%*s", LEFT_WIDTH, " ");
		k = dgen_for_n(dg, count, bogus1, 0, j);
		printf("\n");
		j += 1;
	    } while (k > 0);
	}
    } else {
	j = 0;
	do {
	    if (!j)
		printf("%*.*s", LEFT_WIDTH, LEFT_WIDTH, p->wl_word);
	    else
		printf("%*s", LEFT_WIDTH, " ");
	    k = dgen_for_n(dg, count, bogus2, 0, j);
	    printf("\n");
	    j += 1;
	} while (k > 0);
    }
}

int bogus1(dgen *dg)
{
   printf(" %*s", DEV_WIDTH, "---------");
    return 0;
}

int bogus2(dgen *dg)
{
    printf(" %*s", DEV_WIDTH, "?????????");
    return 0;
}

int
printvals(dgen *dg, IFparm *p, int i)
{
    IFvalue	val;
    int		n;

    if (dg->flags & DGEN_INSTANCE)
	(*ft_sim->askInstanceQuest)(ft_curckt->ci_ckt, dg->instance,
	    p->id, &val, &val);
    else
	(*ft_sim->askModelQuest)(ft_curckt->ci_ckt, dg->model,
	    p->id, &val, &val);

    if (p->dataType & IF_VECTOR)
	n = val.v.numValue;
    else
	n = 1;

    if (((p->dataType & IF_VARTYPES) & ~IF_VECTOR) == IF_COMPLEX)
	n *= 2;

    if (i >= n) {
	if (i == 0)
	    printf("         -");
	else
	    printf("          ");
	return 0;
    }

    if (p->dataType & IF_VECTOR) {
        /* va: ' ' is no flag for %s */
	switch ((p->dataType & IF_VARTYPES) & ~IF_VECTOR) {
	    case IF_FLAG:
		    printf(" % *d", DEV_WIDTH, val.v.vec.iVec[i]);
		    break;
	    case IF_INTEGER:
		    printf(" % *d", DEV_WIDTH, val.v.vec.iVec[i]);
		    break;
	    case IF_REAL:
		    printf(" % *.6g", DEV_WIDTH, val.v.vec.rVec[i]);
		    break;
	    case IF_COMPLEX:
		    if (!(i % 2))
			   printf(" % *.6g", DEV_WIDTH, val.v.vec.cVec[i / 2].real);
		    else
			   printf(" % *.6g", DEV_WIDTH, val.v.vec.cVec[i / 2].imag);
		    break;
	    case IF_STRING:
		    printf(" %*.*s", DEV_WIDTH, DEV_WIDTH, val.v.vec.sVec[i]);
		    break;
	    case IF_INSTANCE:
		    printf(" %*.*s", DEV_WIDTH, DEV_WIDTH, val.v.vec.uVec[i]);
		    break;
	    default:
		    printf(" %*.*s", DEV_WIDTH, DEV_WIDTH, " ******** ");
	}
    } else {
	switch ((p->dataType & IF_VARTYPES) & ~IF_VECTOR) {
	    case IF_FLAG:
		    printf(" % *d", DEV_WIDTH, val.iValue);
		    break;
	    case IF_INTEGER:
		    printf(" % *d", DEV_WIDTH, val.iValue);
		    break;
	    case IF_REAL:
		    printf(" % *.6g", DEV_WIDTH, val.rValue);
		    break;
	    case IF_COMPLEX:
		    if (i % 2)
			   printf(" % *.6g", DEV_WIDTH, val.cValue.real);
		    else
			   printf(" % *.6g", DEV_WIDTH, val.cValue.imag);
		    break;
	    case IF_STRING:
		    printf(" %*.*s", DEV_WIDTH, DEV_WIDTH, val.sValue);
		    break;
	    case IF_INSTANCE:
		    printf(" %*.*s", DEV_WIDTH, DEV_WIDTH, val.uValue);
		    break;
	    default:
		    printf(" %*.*s", DEV_WIDTH, DEV_WIDTH, " ******** ");
	}
    }

    return n - 1;
}


/* (old "show" command)
 * Display various device parameters.  The syntax of this command is
 *   show devicelist : parmlist
 * where devicelist can be "all", the name of a device, a string like r*,
 * which means all devices with names that begin with 'r', repeated one
 * or more times.   The parms are names of parameters that are (hopefully)
 * valid for all the named devices, or "all".
 */

void
old_show(wordlist *wl)
{
    wordlist *devs, *parms, *tw, *ww;
    struct variable *v;
    char *nn;

    devs = wl;
    while (wl && !eq(wl->wl_word, ":"))
        wl = wl->wl_next;
    if (!wl)
        parms = NULL;
    else {
        if (wl->wl_prev)
            wl->wl_prev->wl_next = NULL;
        parms = wl->wl_next;
        if (parms)
            parms->wl_prev = NULL;
    }

    /* Now expand the devicelist... */
    for (tw = NULL; devs; devs = devs->wl_next) {
        inp_casefix(devs->wl_word);
        tw = wl_append(tw, devexpand(devs->wl_word));
    }

    devs = tw;
    for (tw = parms; tw; tw = tw->wl_next)
        if (eq(tw->wl_word, "all"))
            break;
    if (tw)
        parms = NULL;

    /* This is a crock... */
    if (!devs)
        devs = cp_cctowl(ft_curckt->ci_devices);

    out_init();

    while (devs) {
        out_printf("%s:\n", devs->wl_word);
        if (parms) {
            for (tw = parms; tw; tw = tw->wl_next) {
                nn = copy(devs->wl_word);
                v = (*if_getparam)(ft_curckt->ci_ckt,
			&nn, tw->wl_word, 0, 0);
		if (!v)
		    v = (*if_getparam)(ft_curckt->ci_ckt,
			    &nn, tw->wl_word, 0, 1);
                if (v) {
                    out_printf("\t%s =", tw->wl_word);
                    for (ww = cp_varwl(v); ww; ww =
                            ww->wl_next)
                        out_printf(" %s", ww->wl_word);
                    out_send("\n");
                }
            }
        } else {
            nn = copy(devs->wl_word);
            v = (*if_getparam)(ft_curckt->ci_ckt, &nn, "all", 0, 0);
	    if (!v)
		v = (*if_getparam)(ft_curckt->ci_ckt, &nn, "all", 0, 1);
            while (v) {
                out_printf("\t%s =", v->va_name);
                for (ww = cp_varwl(v); ww; ww = ww->wl_next)
                    out_printf(" %s", ww->wl_word);
                out_send("\n");
                v = v->va_next;
            }
        }
        devs = devs->wl_next;
    }
    return;
}

/* Alter a device parameter.  The new syntax here is
 *	alter @device[parameter] = expr
 *	alter device = expr
 *	alter device parameter = expr
 * expr must be real (complex isn't handled right now, integer is fine though,
 * but no strings ... for booleans, use 0/1).
 */

void
com_alter(wordlist *wl)
{
    if (!wl) {
	fprintf(cp_err, "usage: alter dev param = expression\n");
	fprintf(cp_err, "  or   alter @dev[param] = expression\n");
	fprintf(cp_err, "  or   alter dev = expression\n");
	return;
    }
    com_alter_common(wl, 0);
}

void
com_altermod(wordlist *wl)
{
    com_alter_common(wl, 1);
}

void
com_alter_common(wordlist *wl, int do_model)
{
    wordlist *eqword, *words;
    char *dev, *p;
    char *param;
    struct dvec *dv;
    struct pnode *names;

    if (!ft_curckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
    }

    words = wl;
    while (words) {
	p = words->wl_word;
	eqword = words;
	words = words->wl_next;
	if (eq(p, "=")) {
	    break;
	}
    }
    if (!words) {
	fprintf(cp_err, "Error: no assignment found.\n");
	return;
    }

    /* device parameter = expr
       device = expr
       @dev[param] = expr
     */

    dev = NULL;
    param = NULL;
    words = wl;
    while (words != eqword) {
	p = words->wl_word;
	if (param) {
	    fprintf(cp_err, "Error: excess parameter name \"%s\" ignored.\n",
		p);
	} else if (dev) {
	    param = words->wl_word;
	} else if (*p == '@' || *p == '#') {
	    dev = p + 1;
	    p =strchr(p, '[');
	    if (p) {
		*p++ = 0;
		param = p;
		p =strchr(p, ']');
		if (p)
		    *p = 0;
	    }
	} else {
	    dev = p;
	}
	words = words->wl_next;
    }
    if (!dev) {
	fprintf(cp_err, "Error: no model or device name provided.\n" );
	return;
    }

    words = eqword->wl_next;
    names = ft_getpnames(words, FALSE);
    if (!names) {
	fprintf(cp_err, "Error: cannot parse new parameter value.\n");
	return;
    }
    dv = ft_evaluate(names);
    if (!dv)
	return;
    if (dv->v_length < 1) {
	fprintf(cp_err, "Error: cannot evaluate new parameter value.\n");
	return;
    }

    if_setparam(ft_curckt->ci_ckt, &dev, param, dv, do_model);

    /* va: garbage collection for dv, if pnode names is no simple value */
    if (names->pn_value==NULL && dv!=NULL) vec_free(dv);
    free_pnode(names); /* free also dv, if pnode names is simple value */
    return;
}

/* Given a device name, possibly with wildcards, return the matches. */

static wordlist *
devexpand(char *name)
{
    wordlist *wl, *devices, *tw;

    if (index(name, '*') ||strchr(name, '[') ||strchr(name, '?')) {
        devices = cp_cctowl(ft_curckt->ci_devices);
        for (wl = NULL; devices; devices = devices->wl_next)
            if (cp_globmatch(name, devices->wl_word)) {
                tw = alloc(struct wordlist);
                if (wl) {
                    wl->wl_prev = tw;
                    tw->wl_next = wl;
                    wl = tw;
                } else
                    wl = tw;
                wl->wl_word = devices->wl_word;
            }
    } else if (cieq(name, "all")) {
        wl = cp_cctowl(ft_curckt->ci_devices);
    } else {
        wl = alloc(struct wordlist);
        wl->wl_word = name;
    }
    wl_sort(wl);
    return (wl);
}

