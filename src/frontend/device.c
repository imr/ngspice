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
#include "com_commands.h"

#include "gens.h" /* wl_forall */

static wordlist *devexpand(char *name);
static void all_show(wordlist *wl, int mode);
static void all_show_old(wordlist *wl, int mode);

/*
 *      show: list device operating point info
 *              show
 *              show devs : params
 *              show devs : params ; devs : params
 *              show dev dev dev : param param param , dev dev : param param
 *              show t : param param param, t : param param
 *
 */


static  int     count;


void
com_showmod(wordlist *wl)
{
    if (cp_getvar("altshow", CP_BOOL, NULL))
        all_show(wl, 1);
        else
        all_show_old(wl, 1);
}

void
com_show(wordlist *wl)
{
    if (cp_getvar("altshow", CP_BOOL, NULL))
        all_show(wl, 0);
        else
        all_show_old(wl, 0);
}

static void
all_show(wordlist *wl, int mode)
{
    wordlist    *params, *nextgroup, *thisgroup;
    wordlist    *prev, *next, *w;
    int         screen_width;
    dgen        *dg, *listdg;
    int         instances;
    int         i, j, n;
    int         param_flag, dev_flag;

    if (!ft_curckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
    }

    if (wl && wl->wl_word && eq(wl->wl_word, "-v")) {
        old_show(wl->wl_next);
        return;
    }

    if (!cp_getvar("width", CP_NUM, (char *) &screen_width))
            screen_width = DEF_WIDTH;
    count = (screen_width - LEFT_WIDTH) / (DEV_WIDTH + 1);
    count = 1;

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

            if ( eq(w->wl_word, "*") ) {
              tfree(w->wl_word);
              w->wl_word = strdup("all");
            }
 
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
                n += 1;

                fprintf(cp_out,"%s:\n", dg->instance->GENname);
                fprintf(cp_out,"    %-19s= %s\n", "model", dg->model->GENmodName);

                listdg = dg;

                if (param_flag) {
                    param_forall(dg, param_flag);
                }
                else if (!params) {
                    param_forall(dg, DGEN_DEFPARAMS);
                }
                if (params) {
                    wl_forall(params, listparam, dg);
                }

            } else if (ft_sim->devices[dg->dev_type_no]->numModelParms) {
                fprintf(cp_out," %s models (%s)\n",
                        ft_sim->devices[dg->dev_type_no]->name,
                        ft_sim->devices[dg->dev_type_no]->description);
                n += 1;
                i = 0;
                do {
                  fprintf(cp_out,"%*s", LEFT_WIDTH, "model");
                  j = dgen_for_n(dg, count, printstr_m, NULL, i);
                  i += 1;
                  fprintf(cp_out,"\n");
                } while (j);
                fprintf(cp_out,"\n");

                if (param_flag)
                    param_forall(dg, param_flag);
                else if (!params)
                    param_forall(dg, DGEN_DEFPARAMS);
                if (params)
                    wl_forall(params, listparam, dg);
                fprintf(cp_out,"\n");
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

static void
all_show_old(wordlist *wl, int mode)
{
    wordlist    *params, *nextgroup, *thisgroup;
    wordlist    *prev, *next, *w;
    int         screen_width;
    dgen        *dg, *listdg;
    int         instances;
    int         i, j, n;
    int         param_flag, dev_flag;

    if (!ft_curckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
    }

    if (wl && wl->wl_word && eq(wl->wl_word, "-v")) {
        old_show(wl->wl_next);
        return;
    }

    if (!cp_getvar("width", CP_NUM, (char *) &screen_width))
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

            if ( eq(w->wl_word, "*") ) {
              tfree(w->wl_word);
              w->wl_word = strdup("all");
            }
 
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
                fprintf(cp_out," %s: %s\n",
                        ft_sim->devices[dg->dev_type_no]->name,
                        ft_sim->devices[dg->dev_type_no]->description);
                n += 1;

                i = 0;
                do {
                  fprintf(cp_out,"%*s", LEFT_WIDTH, "device");
                  j = dgen_for_n(dg, count, printstr_n, NULL, i);
                  i += 1;
                  fprintf(cp_out,"\n");
                } while (j);

                if (ft_sim->devices[dg->dev_type_no]->numModelParms) {
                        i = 0;
                        do {
                                fprintf(cp_out,"%*s", LEFT_WIDTH, "model");
                                j = dgen_for_n(dg, count, printstr_m, NULL, i);
                                i += 1;
                                fprintf(cp_out,"\n");
                        } while (j);
                }
                listdg = dg;

                if (param_flag)
                    param_forall_old(dg, param_flag);
                else if (!params)
                    param_forall_old(dg, DGEN_DEFPARAMS);
                if (params)
                    wl_forall(params, listparam, dg);
                fprintf(cp_out,"\n");

            } else if (ft_sim->devices[dg->dev_type_no]->numModelParms) {
                fprintf(cp_out," %s models (%s)\n",
                        ft_sim->devices[dg->dev_type_no]->name,
                        ft_sim->devices[dg->dev_type_no]->description);
                n += 1;
                i = 0;
                do {
                  fprintf(cp_out,"%*s", LEFT_WIDTH, "model");
                  j = dgen_for_n(dg, count, printstr_m, NULL, i);
                  i += 1;
                  fprintf(cp_out,"\n");
                } while (j);
                fprintf(cp_out,"\n");

                if (param_flag)
                    param_forall_old(dg, param_flag);
                else if (!params)
                    param_forall_old(dg, DGEN_DEFPARAMS);
                if (params)
                    wl_forall(params, listparam, dg);
                fprintf(cp_out,"\n");
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
printstr_n(dgen *dg, IFparm *p, int i)
{
    NG_IGNORE(p);
    NG_IGNORE(i);

    if (dg->instance)
        fprintf(cp_out," %*.*s", DEV_WIDTH, DEV_WIDTH, dg->instance->GENname);
    else
        fprintf(cp_out," %*s", DEV_WIDTH, "<\?\?\?\?\?\?\?>");
    return 0;
}

int
printstr_m(dgen *dg, IFparm *p, int i)
{
    NG_IGNORE(p);
    NG_IGNORE(i);

    if (dg->model)
        fprintf(cp_out," %*.*s", DEV_WIDTH, DEV_WIDTH, dg->model->GENmodName);
    else
        fprintf(cp_out," %*s", DEV_WIDTH, "<\?\?\?\?\?\?\?>");
    return 0;
}

void
param_forall(dgen *dg, int flags)
{
    int i, j, k, found;
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
                  fprintf(cp_out,"    %-19s=", plist[i].keyword);

                  k = dgen_for_n(dg, count, printvals, (plist + i), j);
                  fprintf(cp_out,"\n");
                  j += 1;

                } while (k);
            }
        }
    }
}

void
param_forall_old(dgen *dg, int flags)
{
    int i, j, k, found;
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
                           fprintf(cp_out,"%*.*s", LEFT_WIDTH, LEFT_WIDTH,
                                   plist[i].keyword);
                        else
                           fprintf(cp_out,"%*.*s", LEFT_WIDTH, LEFT_WIDTH, " ");
                        k = dgen_for_n(dg, count, printvals_old, (plist + i), j);
                        fprintf(cp_out,"\n");
                        j += 1;
                } while (k);
            }
        }
    }
}

void
listparam(wordlist *p, dgen *dg)
{
    int i, j, k, found;
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
                   fprintf(cp_out,"%*.*s", LEFT_WIDTH, LEFT_WIDTH, p->wl_word);
                else
                   fprintf(cp_out,"%*.*s", LEFT_WIDTH, LEFT_WIDTH, " ");
                k = dgen_for_n(dg, count, printvals_old, (plist + i), j);
                printf("\n");
                j += 1;
            } while (k > 0);
        } else {
            j = 0;
            do {
                if (!j)
                   fprintf(cp_out,"%*.*s", LEFT_WIDTH, LEFT_WIDTH, p->wl_word);
                else
                   fprintf(cp_out,"%*s", LEFT_WIDTH, " ");
                k = dgen_for_n(dg, count, bogus1, 0, j);
                fprintf(cp_out,"\n");
                j += 1;
            } while (k > 0);
        }
    } else {
        j = 0;
        do {
            if (!j)
                fprintf(cp_out,"%*.*s", LEFT_WIDTH, LEFT_WIDTH, p->wl_word);
            else
                fprintf(cp_out,"%*s", LEFT_WIDTH, " ");
            k = dgen_for_n(dg, count, bogus2, 0, j);
            fprintf(cp_out,"\n");
            j += 1;
        } while (k > 0);
    }
}

int bogus1(dgen *dg, IFparm *p, int i)
{
    NG_IGNORE(dg);
    NG_IGNORE(p);
    NG_IGNORE(i);

    fprintf(cp_out," %*s", DEV_WIDTH, "---------");
    return 0;
}

int bogus2(dgen *dg, IFparm *p, int i)
{
    NG_IGNORE(dg);
    NG_IGNORE(p);
    NG_IGNORE(i);

    fprintf(cp_out," %*s", DEV_WIDTH, "?????????");
    return 0;
}

int
printvals(dgen *dg, IFparm *p, int i)
{
    IFvalue     val;
    int         n;

    if (dg->flags & DGEN_INSTANCE)
        ft_sim->askInstanceQuest (ft_curckt->ci_ckt, dg->instance,
            p->id, &val, &val);
    else
        ft_sim->askModelQuest (ft_curckt->ci_ckt, dg->model,
            p->id, &val, &val);

    if (p->dataType & IF_VECTOR)
        n = val.v.numValue;
    else
        n = 1;

    if (((p->dataType & IF_VARTYPES) & ~IF_VECTOR) == IF_COMPLEX)
        n *= 2;

    if (i >= n) {
        if (i == 0)
            fprintf(cp_out,"         -");
        else
            fprintf(cp_out,"          ");
        return 0;
    }

    if (p->dataType & IF_VECTOR) {
        /* va: ' ' is no flag for %s */
        switch ((p->dataType & IF_VARTYPES) & ~IF_VECTOR) {
            case IF_FLAG:
                    fprintf(cp_out," %d",  val.v.vec.iVec[i]);
                    break;
            case IF_INTEGER:
                    fprintf(cp_out," %d",  val.v.vec.iVec[i]);
                    break;
            case IF_REAL:
                    fprintf(cp_out," %.6g",  val.v.vec.rVec[i]);
                    break;
            case IF_COMPLEX:
                    if (!(i % 2))
                           fprintf(cp_out," %.6g",  val.v.vec.cVec[i / 2].real);
                    else
                           fprintf(cp_out," %.6g",  val.v.vec.cVec[i / 2].imag);
                    break;
            case IF_STRING:
                    fprintf(cp_out," %s",   val.v.vec.sVec[i]);
                    break;
            case IF_INSTANCE:
                    fprintf(cp_out," %s",   val.v.vec.uVec[i]);
                    break;
            default:
                    fprintf(cp_out," %s",   " ******** ");
        }
    } else {
        switch ((p->dataType & IF_VARTYPES) & ~IF_VECTOR) {
            case IF_FLAG:
                    fprintf(cp_out," %d",  val.iValue);
                    break;
            case IF_INTEGER:
                    fprintf(cp_out," %d",  val.iValue);
                    break;
            case IF_REAL:
                    fprintf(cp_out," %.6g",  val.rValue);
                    break;
            case IF_COMPLEX:
                    if (i % 2)
                           fprintf(cp_out," %.6g",  val.cValue.real);
                    else
                           fprintf(cp_out," %.6g",  val.cValue.imag);
                    break;
            case IF_STRING:
                    fprintf(cp_out," %s",   val.sValue);
                    break;
            case IF_INSTANCE:
                    fprintf(cp_out," %s",   val.uValue);
                    break;
            default:
                    fprintf(cp_out," %s",   " ******** ");
        }
    }

    return n - 1;
}

int
printvals_old(dgen *dg, IFparm *p, int i)
{
    IFvalue     val;
    int         n, error;

    if (dg->flags & DGEN_INSTANCE)
        error = ft_sim->askInstanceQuest (ft_curckt->ci_ckt, dg->instance,
            p->id, &val, &val);
    else
        error = ft_sim->askModelQuest (ft_curckt->ci_ckt, dg->model,
            p->id, &val, &val);

    if (p->dataType & IF_VECTOR)
        n = val.v.numValue;
    else
        n = 1;

    if (((p->dataType & IF_VARTYPES) & ~IF_VECTOR) == IF_COMPLEX)
        n *= 2;

    if (i >= n) {
        if (i == 0)
            fprintf(cp_out,"         -");
        else
            fprintf(cp_out,"          ");
        return 0;
    }

    if(error) {
        fprintf(cp_out," <<NAN, error = %d>>", error);
    } else
    if (p->dataType & IF_VECTOR) {
        /* va: ' ' is no flag for %s */
        switch ((p->dataType & IF_VARTYPES) & ~IF_VECTOR) {
            case IF_FLAG:
                    fprintf(cp_out," % *d", DEV_WIDTH, val.v.vec.iVec[i]);
                    break;
            case IF_INTEGER:
                    fprintf(cp_out," % *d", DEV_WIDTH, val.v.vec.iVec[i]);
                    break;
            case IF_REAL:
                    fprintf(cp_out," % *.6g", DEV_WIDTH, val.v.vec.rVec[i]);
                    break;
            case IF_COMPLEX:
                    if (!(i % 2))
                           fprintf(cp_out," % *.6g", DEV_WIDTH, val.v.vec.cVec[i / 2].real);
                    else
                           fprintf(cp_out," % *.6g", DEV_WIDTH, val.v.vec.cVec[i / 2].imag);
                    break;
            case IF_STRING:
                    fprintf(cp_out," %*.*s", DEV_WIDTH, DEV_WIDTH, val.v.vec.sVec[i]);
                    break;
            case IF_INSTANCE:
                    fprintf(cp_out," %*.*s", DEV_WIDTH, DEV_WIDTH, val.v.vec.uVec[i]);
                    break;
            default:
                    fprintf(cp_out," %*.*s", DEV_WIDTH, DEV_WIDTH, " ******** ");
        }
    } else {
        switch ((p->dataType & IF_VARTYPES) & ~IF_VECTOR) {
            case IF_FLAG:
                    fprintf(cp_out," % *d", DEV_WIDTH, val.iValue);
                    break;
            case IF_INTEGER:
                    fprintf(cp_out," % *d", DEV_WIDTH, val.iValue);
                    break;
            case IF_REAL:
                    fprintf(cp_out," % *.6g", DEV_WIDTH, val.rValue);
                    break;
            case IF_COMPLEX:
                    if (i % 2)
                           fprintf(cp_out," % *.6g", DEV_WIDTH, val.cValue.real);
                    else
                           fprintf(cp_out," % *.6g", DEV_WIDTH, val.cValue.imag);
                    break;
            case IF_STRING:
                    fprintf(cp_out," %*.*s", DEV_WIDTH, DEV_WIDTH, val.sValue);
                    break;
            case IF_INSTANCE:
                    fprintf(cp_out," %*.*s", DEV_WIDTH, DEV_WIDTH, val.uValue);
                    break;
            default:
                    fprintf(cp_out," %*.*s", DEV_WIDTH, DEV_WIDTH, " ******** ");
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
                v = if_getparam (ft_curckt->ci_ckt,
                        &nn, tw->wl_word, 0, 0);
                if (!v)
                    v = if_getparam (ft_curckt->ci_ckt,
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
            v = if_getparam (ft_curckt->ci_ckt, &nn, "all", 0, 0);
            if (!v)
                v = if_getparam (ft_curckt->ci_ckt, &nn, "all", 0, 1);
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
 *      alter @device[parameter] = expr
 *      alter device = expr
 *      alter device parameter = expr
 * expr must be real (complex isn't handled right now, integer is fine though,
 * but no strings ... for booleans, use 0/1).
 */

static void com_alter_common(wordlist *wl, int do_model);

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

static void
com_alter_common(wordlist *wl, int do_model)
{
    wordlist *eqword = NULL, *words;
    char *dev, *p;
    char *param;
    struct dvec *dv;
    struct pnode *names;
        
    /* DIE 2009_02_06 */
    char *argument;
    char **arglist;
    int i=0, step=0, n, wlen, maxelem=3;
    wordlist *wl2 = NULL, *wlin, *wleq;
    bool eqfound = FALSE, vecfound = FALSE;

    if (!ft_curckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
    }

    /* 
    wordlist 'wl' will be splitted into a wordlist wl2 with three elements, 
    containing
    1) '@dev[param]' string (i.e.: the substring before '=' char);
    2) '=' string;
    3) 'expression' string.

    Spaces around the '=' sign have to be removed. This is provided 
    by inp_remove_excess_ws().
    
    If the 'altermod' argument is 'altermod m1 vth0=0.7', 'm1' has to be kept as the 
    element in wl2 before splitting inserts the three new elements. 
    If 'expression' is a vector (e.g. [ 1.0 1.2 1.4 ] ), its elements
    in wl2 have to follow the splitting. wl_splice() will take care of this.
    */
    wlin = wl;
    while(wl){         
            argument = wl->wl_word;
            /* searching for '=' ... */
            i = 0;
            while(argument[i]!='=' && argument[i]!='\0'){
                    i++;
            }
            /* ...and if found split argument into three chars and make a new wordlist */
            if(argument[i]!='\0'){
                    /* We found '=' */
                    eqfound = TRUE;
                    arglist = TMALLOC(char*, 4);
                    arglist[3] = NULL;
                    arglist[0] = TMALLOC(char, i + 1);
                    arglist[2] = TMALLOC(char, strlen(&argument[i + 1]) + 1);
                    /* copy argument */
                    strncpy(arglist[0], argument, (size_t) i);
                    arglist[0][i] = '\0';
                    /* copy equal sign */
                    arglist[1] = copy("=");
                    /* copy expression */
                    strncpy(arglist[2],&argument[i+1],strlen(&argument[i+1])+1);

                    /* create a new wordlist from array arglist */
                    wl2 = wl_build(arglist);
                    /* combine wordlists into wl2, free wl */
                    wl_splice(wl, wl2);
                    wl = NULL;
                    /* free arglist */
                    for (n=0; n < 3; n++) tfree(arglist[n]);
                    tfree(arglist);
            } else {
                    /* deal with 'altermod m1 vth0=0.7' by moving
                    forward beyond 'm1' */
                    wl = wl->wl_next;
                    step++;
            }
    }

    if(eqfound) {
        /* step back in the wordlist, if we have moved forward, to catch 'm1' */
        for(n=step;n>0;n--) wl2 = wl2->wl_prev;
    } else {
        /* no equal sign found, probably a pre3f4 input format 
           'alter device value'
           'alter device parameter value'
           are supported, 
           'alter device parameter value parameter value [ parameter value ]'
           multiple param value pairs are not supported!
        */
        wl2 = wlin;
        wlen = wl_length(wlin);
        /* Return the last element of wlin */
        wlin = wl_nthelem(100, wlin); /* no more than 100 vector elements */

        if (eq(wlin->wl_word, "]"))/* we have a vector */ {
           for (n=0;n<100;n++) {/* no more than 100 vector elements */
              wlin=wlin->wl_prev;
              maxelem++;
              if (eq(wlin->wl_word, "[")) {
                 vecfound = TRUE; 
                 break;
              }
              if(wlin->wl_prev==NULL) {
                 fprintf(cp_err, "Error: '[' is missing.\n");
                 fprintf(cp_err, "Cannot alter parameters.\n");
                  return;
              }
           }
        }
        if(wlen > maxelem) {
            fprintf(cp_err, "Error: Only a single param - value pair supported.\n");
            fprintf(cp_err, "Cannot alter parameters.\n");
            return;
        }
        /* add the '=' */
        /* create wordlist with '=' */
        wleq = TMALLOC(wordlist, 1);
        wleq->wl_word = copy("=");
        /* add the last element (the value of the param - value pair) */
        wleq->wl_next = wlin;
        /* move back one element to place equal sign */
        wlin = wlin->wl_prev;
        /* add ' = value' */
        wlin->wl_next = wleq;
        /* step back until 'alter' or 'altermod' is found, 
        then move one step forward */
        while (!ciprefix(wlin->wl_word,"alter"))
            wlin = wlin->wl_prev;
        wlin = wlin->wl_next;
        wl2 = wlin;
    }

    /* Everything is ready, parsing of the wordlist starts here. */
    words = wl2; 
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
        fprintf(cp_err, "Cannot alter parameters.\n");
        return;
    }

    /* device parameter = expr
       device = expr
       @dev[param] = expr
     */

    dev = NULL;
    param = NULL;
    words = wl2;
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
        fprintf(cp_err, "Cannot alter parameters.\n");
        return;
    }

    words = eqword->wl_next;
    /* skip next line if words is a vector */
    if(!eq(words->wl_word, "["))
        names = ft_getpnames(words, FALSE);
    else names = NULL;
    if (!names) {
       /* Put this to try to resolve the case of 
       alter @vin[pulse] = [ 0 5 10n 10n 10n 50n 100n ]
       */
       char *xsbuf;
       int type = IF_REALVEC,i=0;

       double *list;
       double tmp;
       int error;
       /* move beyond '[' to allow INPevaluate() */
       if(eq(words->wl_word, "[")) words = words->wl_next;
       xsbuf = wl_flatten(words);
       /* fprintf(cp_err, "Chain    converted  %s \n",xsbuf); */
       dv = TMALLOC(struct dvec, 1);
       dv->v_name = copy("real vector");
       type &= IF_VARTYPES;
       if (type == IF_REALVEC) {
           list = TMALLOC(double, 1);
           tmp = INPevaluate(&xsbuf,&error,1);
           while (error == 0)
           {
               /*printf(" returning vector value %g\n",tmp); */
               i++;
               list=TREALLOC(double, list, i);
               *(list+i-1) = tmp;
               tmp = INPevaluate(&xsbuf,&error,1);
           }
           dv->v_realdata=list;
       }
       dv->v_length=i;

       if (!dv)
           return;
       if (dv->v_length < 1)
       {
           fprintf(cp_err, "Error: cannot evaluate new parameter value.\n");
           return;
       }

       /*       Here I was, to change the inclusion in the circuit.
        * will have to revise that dv is right for its insertion.
        */
       if_setparam(ft_curckt->ci_ckt, &dev, param, dv, do_model);

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
