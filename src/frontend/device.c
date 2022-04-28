/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000 AlansFixes
**********/

/*
 * Routines to query and alter devices.
 */

#include "ngspice/ngspice.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dgen.h"
#include "ngspice/sim.h"

#include "circuits.h"
#include "device.h"
#include "variable.h"
#include "com_commands.h"
#include "../misc/util.h" /* ngdirname() */

#include "gens.h" /* wl_forall */


static wordlist *devexpand(char *name);
static void all_show(wordlist *wl, int mode);
static void all_show_old(wordlist *wl, int mode);
static void com_alter_mod(wordlist *wl);
static void if_set_binned_model(CKTcircuit *, char *, char *, struct dvec *);


/*
 * devhelp: lists available devices and information on parameters
 *   devhelp                 : shows all available devices
 *   devhelp devname         : shows all parameters of that model/instance
 *   devhelp devname parname : shows parameter meaning
 *   Options: -csv (comma separated value for generating docs)
 *            -type (show parameter types)
 *            -flags (show parameter flags)
 */

void
com_devhelp(wordlist *wl)
{
    /* Just a simple driver now */
    devhelp(wl);
}


void
devhelp(wordlist *wl)
{
    int i, k = 0;
    int devindex = -1, devInstParNo = 0, devModParNo = 0;
    bool found = FALSE;
    bool print_type = FALSE;
    bool print_flags = FALSE;
    bool print_csv = FALSE;
    wordlist *wlist;
    IFparm *plist;

    /*First copy the base pointer */
    wlist = wl;

    /* If there are no arguments output the list of available devices */
    if (!wlist) {
        out_init();
        out_printf("\nDevices available in the simulator\n\n");
        for (k = 0; k < ft_sim->numDevices; k++)
            if (ft_sim->devices[k])
                out_printf("%-*s:\t%s\n",
                           DEV_WIDTH, ft_sim->devices[k]->name,
                           ft_sim->devices[k]->description);
        out_send("\n");
        return;
    }

    while (TRUE) {
        /* -type, -csv, -flags options can be passed as the initial arguments */
        if (wlist && wlist->wl_word && eq(wlist->wl_word, "-type")) {
            print_type = TRUE;
        } else if (wlist && wlist->wl_word && eq(wlist->wl_word, "-flags")) {
            print_flags = TRUE;
        } else if (wlist && wlist->wl_word && eq(wlist->wl_word, "-csv")) {
            print_csv = TRUE;
        } else
            break;

        if (wlist->wl_next)
            wlist = wlist->wl_next;
        else
            return;
    }

    /* This argument, if exists, must be the device name */
    if (wlist && wlist->wl_word) {
        while (k < ft_sim->numDevices && !found) {
            if (ft_sim->devices[k])
                if (strcasecmp(ft_sim->devices[k]->name, wlist->wl_word) == 0) {
                    devindex = k;
                    if (ft_sim->devices[devindex]->numInstanceParms)
                        devInstParNo = *(ft_sim->devices[devindex]->numInstanceParms);
                    else
                        devInstParNo = 0;

                    if (ft_sim->devices[devindex]->numModelParms)
                        devModParNo = *(ft_sim->devices[devindex]->numModelParms);
                    else
                        devModParNo = 0;

                    wlist = wlist->wl_next;
                    found = TRUE;
                }
            k++;
        }

        if (!found) {
            fprintf(cp_out, "Error: Device %s not found\n", wlist->wl_word);
            return;
        }
    }

    /* At this point, found is TRUE and we have found the device.
     * Now we have to scan the model and instance parameters to print
     * the string
     */
    found = FALSE;
    if (wlist && wlist->wl_word) {
        plist = ft_sim->devices[devindex]->modelParms;
        for (i = 0; i < devModParNo; i++) { /* Scan model parameters first */
            if (strcasecmp(plist[i].keyword, wlist->wl_word) == 0) {
                found = TRUE;
                out_init();
                out_printf("Model Parameters\n");
                printheaders(print_type, print_flags, print_csv);
                printdesc(plist[i], print_type, print_flags, print_csv);
                out_send("\n");
            }
        }

        if (!found) {
            plist = ft_sim->devices[devindex]->instanceParms;
            for (i = 0; i < devInstParNo; i++) { /* Scan instance parameters then */
                if (strcasecmp(plist[i].keyword, wlist->wl_word) == 0) {
                    found = TRUE;
                    out_init();
                    out_printf("Instance Parameters\n");
                    printdesc(plist[i], print_type, print_flags, print_csv);
                    out_send("\n");
                }
            }
        }

        if (!found)
            fprintf(cp_out, "Error: Parameter %s not found\n", wlist->wl_word);
        return;

    }

    /* No arguments - we want all the parameters*/
    out_init();
    out_printf("%s - %s\n\n", ft_sim->devices[devindex]->name, ft_sim->devices[devindex]->description);
    out_printf("Model Parameters\n");
    printheaders(print_type, print_flags, print_csv);

    plist = ft_sim->devices[devindex]->modelParms;
    for (i = 0; i < devModParNo; i++)
        printdesc(plist[i], print_type, print_flags, print_csv);
    out_printf("\n");
    out_printf("Instance Parameters\n");
    printheaders(print_type, print_flags, print_csv);

    plist = ft_sim->devices[devindex]->instanceParms;
    for (i = 0; i < devInstParNo; i++)
        printdesc(plist[i], print_type, print_flags, print_csv);

    out_send("\n");
}


/*
 * Print headers for printdesc()
 */

void
printheaders(bool print_type, bool print_flags, bool csv)
{
    if (csv)
        out_printf("id#, Name, Dir, ");
    else
        out_printf("%5s\t %-10s\t Dir\t ", "id#", "Name");

    if (print_type) {
        if (csv)
            out_printf("Type, ");
        else
            out_printf("%-10s\t ", "Type");
    }

    if (print_flags) {
        if (csv)
            out_printf("Flags, ");
        else
            out_printf("%-6s\t ", "Flags");
    }

    out_printf("Description\n");
}


/*
 * Pretty print parameter descriptions
 * This function prints description of device parameters
 */

void
printdesc(IFparm p, bool print_type, bool print_flags, bool csv)
{
    char sep;
    int id_spacer, keyword_spacer, type_spacer, flags_spacer;

    /* First we indentify the separator */
    if (csv) {
        sep = ',';
        id_spacer = 0;
        keyword_spacer = 0;
        type_spacer = 0;
        flags_spacer = 0;
    } else {
        sep = '\t';
        id_spacer = 5;
        keyword_spacer = 10;
        type_spacer = 10;
        flags_spacer = 5;
    }

    out_printf("%*d%c %-*s%c ", id_spacer, p.id, sep, keyword_spacer, p.keyword, sep);

    if (p.dataType & IF_SET)
        if (p.dataType & IF_ASK)
            out_printf("inout%c ", sep);
        else
            out_printf("in%c ", sep);
    else
        out_printf("out%c ", sep);

    if (print_type) {
        switch (p.dataType & IF_VARTYPES) {
        case IF_FLAG:
            out_printf("%-*s%c ", type_spacer, "flag", sep);
            break;
        case IF_INTEGER:
            out_printf("%-*s%c ", type_spacer, "integer", sep);
            break;
        case IF_REAL:
            out_printf("%-*s%c ", type_spacer, "real", sep);
            break;
        case IF_COMPLEX:
            out_printf("%-*s%c ", type_spacer, "complex", sep);
            break;
        case IF_NODE:
            out_printf("%-*s%c ", type_spacer, "node", sep);
            break;
        case IF_INSTANCE:
            out_printf("%-*s%c ", type_spacer, "instance", sep);
            break;
        case IF_STRING:
            out_printf("%-*s%c ", type_spacer, "string", sep);
            break;
        case IF_PARSETREE:
            out_printf("%-*s%c ", type_spacer, "parsetree", sep);
            break;
        case IF_VECTOR: /* A few variables have only the vector vartype bit set */
            out_printf("%-*s%c ", type_spacer, "vector", sep);
            break;
        case IF_FLAGVEC:
            out_printf("%-*s%c ", type_spacer, "flagvec", sep);
            break;
        case IF_INTVEC:
            out_printf("%-*s%c ", type_spacer, "intvec", sep);
            break;
        case IF_REALVEC:
            out_printf("%-*s%c ", type_spacer, "realvec", sep);
            break;
        case IF_CPLXVEC:
            out_printf("%-*s%c ", type_spacer, "cplxvec", sep);
            break;
        case IF_NODEVEC:
            out_printf("%-*s%c ", type_spacer, "nodevec", sep);
            break;
        case IF_INSTVEC:
            out_printf("%-*s%c ", type_spacer, "instvec", sep);
            break;
        case IF_STRINGVEC:
            out_printf("%-*s%c ", type_spacer, "stringvec", sep);
            break;
        default:
            out_printf("%-*s%c ", type_spacer, "?????????", sep);
        }
    }

    if (print_flags) {
        char flags_str[20 + 1] = "";

        if (p.dataType & IF_NONSENSE)
            strncat(flags_str, "X", 20);

        if (p.dataType & IF_SETQUERY)
            strncat(flags_str, "Q", 20);

        if (p.dataType & IF_CHKQUERY)
            strncat(flags_str, "Z", 20);

        if (p.dataType & IF_ORQUERY)
            strncat(flags_str, "QO", 20);

        if (p.dataType & IF_AC)
            strncat(flags_str, "A", 20);

        if (p.dataType & IF_PRINCIPAL)
            strncat(flags_str, "P", 20);

        if (p.dataType & IF_AC_ONLY)
            strncat(flags_str, "AA", 20);

        if (p.dataType & IF_NOISE)
            strncat(flags_str, "N", 20);

        if (p.dataType & IF_UNINTERESTING)
            strncat(flags_str, "U", 20);

        if (p.dataType & IF_REDUNDANT)
            strncat(flags_str, "R", 20);

        // Is empty?
        if (flags_str[0] == '\0')
            strncat(flags_str, "-", 20);

        out_printf("%-*s%c ", flags_spacer, flags_str, sep);
    }

    if (p.description)
        out_printf("%s\n", p.description);
    else
        out_printf("n.a.\n");
}


/*
 * show: list device operating point info
 *   show
 *   show devs : params
 *   show devs : params ; devs : params
 *   show dev dev dev : param param param , dev dev : param param
 *   show t : param param param, t : param param
 */

static int count;

void
com_showmod(wordlist *wl)
{
    if (cp_getvar("altshow", CP_BOOL, NULL, 0))
        all_show(wl, 1);
    else
        all_show_old(wl, 1);
}


void
com_show(wordlist *wl)
{
    if (cp_getvar("altshow", CP_BOOL, NULL, 0))
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
    dgen        *dg;
    int         instances;
    int         i, j, n;
    int         param_flag, dev_flag;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
    }

    if (wl && wl->wl_word && eq(wl->wl_word, "-v")) {
        old_show(wl->wl_next);
        return;
    }

    if (!cp_getvar("width", CP_NUM, &screen_width, 0))
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

            if (eq(w->wl_word, "*")) {
                tfree(w->wl_word);
                w->wl_word = copy("all");
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

                fprintf(cp_out, "%s:\n", dg->instance->GENname);
                fprintf(cp_out, "    %-19s= %s\n", "model", dg->model->GENmodName);

                if (param_flag)
                    param_forall(dg, param_flag);
                else if (!params)
                    param_forall(dg, DGEN_DEFPARAMS);

                if (params)
                    wl_forall(params, listparam, dg);

            } else if (ft_sim->devices[dg->dev_type_no]->numModelParms) {
                fprintf(cp_out, " %s models (%s)\n",
                        ft_sim->devices[dg->dev_type_no]->name,
                        ft_sim->devices[dg->dev_type_no]->description);
                n += 1;
                i = 0;
                do {
                    fprintf(cp_out, "%*s", LEFT_WIDTH, "model");
                    j = dgen_for_n(dg, count, printstr_m, NULL, i);
                    i += 1;
                    fprintf(cp_out, "\n");
                } while (j);
                fprintf(cp_out, "\n");

                if (param_flag)
                    param_forall(dg, param_flag);
                else if (!params)
                    param_forall(dg, DGEN_DEFPARAMS);

                if (params)
                    wl_forall(params, listparam, dg);
                fprintf(cp_out, "\n");
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
    dgen        *dg;
    int         instances;
    int         i, j, n;
    int         param_flag, dev_flag;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
    }

    if (wl && wl->wl_word && eq(wl->wl_word, "-v")) {
        old_show(wl->wl_next);
        return;
    }

    if (!cp_getvar("width", CP_NUM, &screen_width, 0))
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

            if (eq(w->wl_word, "*")) {
                tfree(w->wl_word);
                w->wl_word = copy("all");
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
                fprintf(cp_out, " %s: %s\n",
                        ft_sim->devices[dg->dev_type_no]->name,
                        ft_sim->devices[dg->dev_type_no]->description);
                n += 1;

                i = 0;
                do {
                    fprintf(cp_out, "%*s", LEFT_WIDTH, "device");
                    j = dgen_for_n(dg, count, printstr_n, NULL, i);
                    i += 1;
                    fprintf(cp_out, "\n");
                } while (j);

                if (ft_sim->devices[dg->dev_type_no]->numModelParms) {
                    i = 0;
                    do {
                        fprintf(cp_out, "%*s", LEFT_WIDTH, "model");
                        j = dgen_for_n(dg, count, printstr_m, NULL, i);
                        i += 1;
                        fprintf(cp_out, "\n");
                    } while (j);
                }

                if (param_flag)
                    param_forall_old(dg, param_flag);
                else if (!params)
                    param_forall_old(dg, DGEN_DEFPARAMS);

                if (params)
                    wl_forall(params, listparam, dg);
                fprintf(cp_out, "\n");

            } else if (ft_sim->devices[dg->dev_type_no]->numModelParms) {
                fprintf(cp_out, " %s models (%s)\n",
                        ft_sim->devices[dg->dev_type_no]->name,
                        ft_sim->devices[dg->dev_type_no]->description);
                n += 1;
                i = 0;
                do {
                    fprintf(cp_out, "%*s", LEFT_WIDTH, "model");
                    j = dgen_for_n(dg, count, printstr_m, NULL, i);
                    i += 1;
                    fprintf(cp_out, "\n");
                } while (j);
                fprintf(cp_out, "\n");

                if (param_flag)
                    param_forall_old(dg, param_flag);
                else if (!params)
                    param_forall_old(dg, DGEN_DEFPARAMS);

                if (params)
                    wl_forall(params, listparam, dg);
                fprintf(cp_out, "\n");
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
        fprintf(cp_out, " %*.*s", DEV_WIDTH, DEV_WIDTH, dg->instance->GENname);
    else
        fprintf(cp_out, " %*s", DEV_WIDTH, "<\?\?\?\?\?\?\?>");
    return 0;
}


int
printstr_m(dgen *dg, IFparm *p, int i)
{
    NG_IGNORE(p);
    NG_IGNORE(i);

    if (dg->model)
        fprintf(cp_out, " %*.*s", DEV_WIDTH, DEV_WIDTH, dg->model->GENmodName);
    else
        fprintf(cp_out, " %*s", DEV_WIDTH, "<\?\?\?\?\?\?\?>");
    return 0;
}


void
param_forall(dgen *dg, int flags)
{
    int i, j, k;
    int xcount;
    IFparm *plist;

    if (dg->flags & DGEN_INSTANCE) {
        xcount = *ft_sim->devices[dg->dev_type_no]->numInstanceParms;
        plist = ft_sim->devices[dg->dev_type_no]->instanceParms;
    } else {
        xcount = *ft_sim->devices[dg->dev_type_no]->numModelParms;
        plist = ft_sim->devices[dg->dev_type_no]->modelParms;
    }

    for (i = 0; i < xcount; i++)
        if ((plist[i].dataType & IF_ASK)
            && !(plist[i].dataType & IF_REDUNDANT)
            && ((plist[i].dataType & IF_SET) || dg->ckt->CKTrhsOld)
            && (!(plist[i].dataType & IF_UNINTERESTING) || (flags == DGEN_ALLPARAMS)))
        {
            j = 0;
            do {
                fprintf(cp_out, "    %-19s=", plist[i].keyword);

                k = dgen_for_n(dg, count, printvals, (plist + i), j);
                fprintf(cp_out, "\n");
                j += 1;

            } while (k);
        }
}


void
param_forall_old(dgen *dg, int flags)
{
    int i, j, k;
    int xcount;
    IFparm *plist;

    if (dg->flags & DGEN_INSTANCE) {
        xcount = *ft_sim->devices[dg->dev_type_no]->numInstanceParms;
        plist = ft_sim->devices[dg->dev_type_no]->instanceParms;
    } else {
        xcount = *ft_sim->devices[dg->dev_type_no]->numModelParms;
        plist = ft_sim->devices[dg->dev_type_no]->modelParms;
    }

    for (i = 0; i < xcount; i++)
        if ((plist[i].dataType & IF_ASK)
            && !(plist[i].dataType & IF_REDUNDANT)
            && ((plist[i].dataType & IF_SET) || dg->ckt->CKTrhsOld)
            && (!(plist[i].dataType & IF_UNINTERESTING) || (flags == DGEN_ALLPARAMS)))
        {
            j = 0;
            do {
                if (!j)
                    fprintf(cp_out, "%*.*s", LEFT_WIDTH, LEFT_WIDTH,
                            plist[i].keyword);
                else
                    fprintf(cp_out, "%*.*s", LEFT_WIDTH, LEFT_WIDTH, " ");
                k = dgen_for_n(dg, count, printvals_old, (plist + i), j);
                fprintf(cp_out, "\n");
                j += 1;
            } while (k);
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

    for (i = 0; i < xcount; i++)
        if (eqc(p->wl_word, plist[i].keyword) && (plist[i].dataType & IF_ASK)) {
            found = 1;
            break;
        }

    if (found) {
        if (dg->ckt->CKTrhsOld ||
            (plist[i].dataType & IF_SET))
        {
            j = 0;
            do {
                if (!j)
                    fprintf(cp_out, "%*.*s", LEFT_WIDTH, LEFT_WIDTH, p->wl_word);
                else
                    fprintf(cp_out, "%*.*s", LEFT_WIDTH, LEFT_WIDTH, " ");
                k = dgen_for_n(dg, count, printvals_old, (plist + i), j);
                printf("\n");
                j += 1;
            } while (k > 0);
        } else {
            j = 0;
            do {
                if (!j)
                    fprintf(cp_out, "%*.*s", LEFT_WIDTH, LEFT_WIDTH, p->wl_word);
                else
                    fprintf(cp_out, "%*s", LEFT_WIDTH, " ");
                k = dgen_for_n(dg, count, bogus1, NULL, j);
                fprintf(cp_out, "\n");
                j += 1;
            } while (k > 0);
        }
    } else {
        j = 0;
        do {
            if (!j)
                fprintf(cp_out, "%*.*s", LEFT_WIDTH, LEFT_WIDTH, p->wl_word);
            else
                fprintf(cp_out, "%*s", LEFT_WIDTH, " ");
            k = dgen_for_n(dg, count, bogus2, NULL, j);
            fprintf(cp_out, "\n");
            j += 1;
        } while (k > 0);
    }
}


int
bogus1(dgen *dg, IFparm *p, int i)
{
    NG_IGNORE(dg);
    NG_IGNORE(p);
    NG_IGNORE(i);

    fprintf(cp_out, " %*s", DEV_WIDTH, "---------");
    return 0;
}


int
bogus2(dgen *dg, IFparm *p, int i)
{
    NG_IGNORE(dg);
    NG_IGNORE(p);
    NG_IGNORE(i);

    fprintf(cp_out, " %*s", DEV_WIDTH, "?????????");
    return 0;
}


int
printvals(dgen *dg, IFparm *p, int i)
{
    IFvalue     val;
    int         n;

    if (dg->flags & DGEN_INSTANCE)
        ft_sim->askInstanceQuest
            (ft_curckt->ci_ckt, dg->instance, p->id, &val, &val);
    else
        ft_sim->askModelQuest
            (ft_curckt->ci_ckt, dg->model, p->id, &val, &val);

    if (p->dataType & IF_VECTOR)
        n = val.v.numValue;
    else
        n = 1;

    if (((p->dataType & IF_VARTYPES) & ~IF_VECTOR) == IF_COMPLEX)
        n *= 2;

    if (i >= n) {
        if (i == 0)
            fprintf(cp_out, "         -");
        else
            fprintf(cp_out, "          ");
        return 0;
    }

    if (p->dataType & IF_VECTOR) {
        /* va: ' ' is no flag for %s */
        switch ((p->dataType & IF_VARTYPES) & ~IF_VECTOR) {
        case IF_FLAG:
            fprintf(cp_out, " %d", val.v.vec.iVec[i]);
            break;
        case IF_INTEGER:
            fprintf(cp_out, " %d", val.v.vec.iVec[i]);
            break;
        case IF_REAL:
            fprintf(cp_out, " %.6g", val.v.vec.rVec[i]);
            break;
        case IF_COMPLEX:
            if (!(i % 2))
                fprintf(cp_out, " %.6g", val.v.vec.cVec[i / 2].real);
            else
                fprintf(cp_out, " %.6g", val.v.vec.cVec[i / 2].imag);
            break;
        case IF_STRING:
            fprintf(cp_out, " %s", val.v.vec.sVec[i]);
            break;
        case IF_INSTANCE:
            fprintf(cp_out, " %s", val.v.vec.uVec[i]);
            break;
        default:
            fprintf(cp_out, " %s", " ******** ");
        }
    } else {
        switch ((p->dataType & IF_VARTYPES) & ~IF_VECTOR) {
        case IF_FLAG:
            fprintf(cp_out, " %d", val.iValue);
            break;
        case IF_INTEGER:
            fprintf(cp_out, " %d", val.iValue);
            break;
        case IF_REAL:
            fprintf(cp_out, " %.6g", val.rValue);
            break;
        case IF_COMPLEX:
            if (i % 2)
                fprintf(cp_out, " %.6g", val.cValue.real);
            else
                fprintf(cp_out, " %.6g", val.cValue.imag);
            break;
        case IF_STRING:
            fprintf(cp_out, " %s", val.sValue);
            break;
        case IF_INSTANCE:
            fprintf(cp_out, " %s", val.uValue);
            break;
        default:
            fprintf(cp_out, " %s", " ******** ");
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
        error = ft_sim->askInstanceQuest
            (ft_curckt->ci_ckt, dg->instance, p->id, &val, &val);
    else
        error = ft_sim->askModelQuest
            (ft_curckt->ci_ckt, dg->model, p->id, &val, &val);

    if (p->dataType & IF_VECTOR)
        n = val.v.numValue;
    else
        n = 1;

    if (((p->dataType & IF_VARTYPES) & ~IF_VECTOR) == IF_COMPLEX)
        n *= 2;

    if (i >= n) {
        if (i == 0)
            fprintf(cp_out, "         -");
        else
            fprintf(cp_out, "          ");
        return 0;
    }

    if (error) {
        fprintf(cp_out, " <<NAN, error = %d>>", error);
    } else if (p->dataType & IF_VECTOR) {
        /* va: ' ' is no flag for %s */
        switch ((p->dataType & IF_VARTYPES) & ~IF_VECTOR) {
        case IF_FLAG:
            fprintf(cp_out, " % *d", DEV_WIDTH, val.v.vec.iVec[i]);
            break;
        case IF_INTEGER:
            fprintf(cp_out, " % *d", DEV_WIDTH, val.v.vec.iVec[i]);
            break;
        case IF_REAL:
            fprintf(cp_out, " % *.6g", DEV_WIDTH, val.v.vec.rVec[i]);
            break;
        case IF_COMPLEX:
            if (!(i % 2))
                fprintf(cp_out, " % *.6g", DEV_WIDTH, val.v.vec.cVec[i / 2].real);
            else
                fprintf(cp_out, " % *.6g", DEV_WIDTH, val.v.vec.cVec[i / 2].imag);
            break;
        case IF_STRING:
            fprintf(cp_out, " %*.*s", DEV_WIDTH, DEV_WIDTH, val.v.vec.sVec[i]);
            break;
        case IF_INSTANCE:
            fprintf(cp_out, " %*.*s", DEV_WIDTH, DEV_WIDTH, val.v.vec.uVec[i]);
            break;
        default:
            fprintf(cp_out, " %*.*s", DEV_WIDTH, DEV_WIDTH, " ******** ");
        }
    } else {
        switch ((p->dataType & IF_VARTYPES) & ~IF_VECTOR) {
        case IF_FLAG:
            fprintf(cp_out, " % *d", DEV_WIDTH, val.iValue);
            break;
        case IF_INTEGER:
            fprintf(cp_out, " % *d", DEV_WIDTH, val.iValue);
            break;
        case IF_REAL:
            fprintf(cp_out, " % *.6g", DEV_WIDTH, val.rValue);
            break;
        case IF_COMPLEX:
            if (i % 2)
                fprintf(cp_out, " % *.6g", DEV_WIDTH, val.cValue.real);
            else
                fprintf(cp_out, " % *.6g", DEV_WIDTH, val.cValue.imag);
            break;
        case IF_STRING:
            fprintf(cp_out, " %*.*s", DEV_WIDTH, DEV_WIDTH, val.sValue);
            break;
        case IF_INSTANCE:
            fprintf(cp_out, " %*.*s", DEV_WIDTH, DEV_WIDTH, val.uValue);
            break;
        default:
            fprintf(cp_out, " %*.*s", DEV_WIDTH, DEV_WIDTH, " ******** ");
        }
    }

    return n - 1;
}


/*
 * (old "show" command)
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
    wl = wl_find(":", wl);
    if (!wl) {
        parms = NULL;
    } else {
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
    tw = wl_find("all", parms);
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
                v = if_getparam(ft_curckt->ci_ckt, &nn, tw->wl_word, 0, 0);
                if (!v)
                    v = if_getparam(ft_curckt->ci_ckt, &nn, tw->wl_word, 0, 1);
                if (v) {
                    out_printf("\t%s =", tw->wl_word);
                    for (ww = cp_varwl(v); ww; ww = ww->wl_next)
                        out_printf(" %s", ww->wl_word);
                    out_send("\n");
                }
            }
        } else {
            nn = copy(devs->wl_word);
            v = if_getparam(ft_curckt->ci_ckt, &nn, "all", 0, 0);
            if (!v)
                v = if_getparam(ft_curckt->ci_ckt, &nn, "all", 0, 1);
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
}


/*
 * Alter a device parameter.  The new syntax here is
 *   alter @device[parameter] = expr
 *   alter device = expr
 *   alter device parameter = expr
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
    wordlist *fileword;
    bool newfile = FALSE;

    fileword = wl;
    while (fileword) {
        if (ciprefix("file", fileword->wl_word))
            newfile = TRUE;
        fileword = fileword->wl_next;
    }

    if (newfile)
        com_alter_mod(wl);
    else
        com_alter_common(wl, 1);
}


static void
if_set_binned_model(CKTcircuit *ckt, char *devname, char *param, struct dvec *val)
{
    char *width_length;
    double w = 0.0, l = 0.0;
    struct variable *v;

    v = if_getparam(ckt, &devname, "w", 0, 0);
    if (!v) {
        fprintf(cp_err, "Error: Can't access width instance parameter.\n");
        return;
    }
    w = v->va_V.vV_real;
    free_struct_variable(v);

    v = if_getparam(ckt, &devname, "l", 0, 0);
    if (!v) {
        fprintf(cp_err, "Error: Can't access length instance parameter.\n");
        return;
    }
    l = v->va_V.vV_real;
    free_struct_variable(v);

    if (param[0] == 'w')
        w = *val->v_realdata; /* overwrite the width with the alter param */
    else
        l = *val->v_realdata; /* overwrite the length with the alter param */

    width_length = tprintf("w=%15.7e l=%15.7e", w, l);

    if_setparam_model(ft_curckt->ci_ckt, &devname, width_length);
    FREE(width_length);
}


static void
com_alter_common(wordlist *wl, int do_model)
{
    wordlist *wl_head = wl;
    wordlist *eqword, *words;
    char *dev, *param;
    struct dvec *dv;
    struct pnode *names;

    int i;

    if (!ft_curckt) {
        fprintf(cp_err, "Error: no circuit loaded\n");
        return;
    }

    /*
     * when the assignment operator '=' is embedded in a wl_word
     *  then split the word into several words
     *
     * Spaces around the '=' sign have to be removed. This is provided
     * by inp_remove_excess_ws(). But take care if command is entered manually!
     */
    for (; wl; wl = wl->wl_next) {
        char *argument = wl->wl_word;
        char *eqptr = strchr(argument, '=');
        if (eqptr) {
            if (strlen(argument) > 1) {
                wordlist *wn = NULL;
                if (eqptr[1])
                    wn = wl_cons(copy(eqptr + 1), wn);
                wn = wl_cons(copy("="), wn);
                if (eqptr > argument)
                    wn = wl_cons(copy_substring(argument, eqptr), wn);
                wl_splice(wl, wn);
                if (wl_head == wl)
                    wl_head = wn;
            }
            break;
        }
    }

    if (!wl) {
        /* no equal sign found, probably a pre3f4 input format
         *   'alter device value'
         *   'alter device parameter value'
         * are supported,
         *   'alter device parameter value parameter value [ parameter value ]'
         * with multiple param value pairs are not supported!
         */
        wordlist *wlin = wl_head;
        int wlen = wl_length(wlin);
        int maxelem = 3;
        /* Return the last element of wlin */
        wlin = wl_nthelem(100, wlin); /* no more than 100 vector elements */

        if (eq(wlin->wl_word, "]"))     /* we have a vector */
            for (i = 0; i < 100; i++) { /* no more than 100 vector elements */
                wlin = wlin->wl_prev;
                maxelem++;
                if (eq(wlin->wl_word, "["))
                    break;
                if (wlin->wl_prev == NULL) {
                    fprintf(cp_err, "Error: '[' is missing.\n");
                    fprintf(cp_err, "Cannot alter parameters.\n");
                    return;
                }
            }

        if (wlen > maxelem) {
            fprintf(cp_err, "Error: Only a single param - value pair supported.\n");
            fprintf(cp_err, "Cannot alter parameters.\n");
            return;
        }
        /* add the '=' */
        wlin = wlin->wl_prev;
        wlin = wl_append(wlin, wl_cons(copy("="), wl_chop_rest(wlin)));
    }

    wl = wl_head;

    /* Everything is ready, parsing of the wordlist starts here. */
    eqword = wl_find("=", wl);
    if (!eqword || !eqword->wl_next) {
        fprintf(cp_err, "Error: no assignment found.\n");
        fprintf(cp_err, "Cannot alter parameters.\n");
        return;
    }

    /*
     * device parameter = expr
     * device = expr
     * @dev[param] = expr
     */

    dev = NULL;
    param = NULL;
    words = wl;
    while (words != eqword) {
        char *p = words->wl_word;
        if (param) {
            fprintf(cp_err, "Warning: excess parameter name \"%s\" ignored.\n", p);
        } else if (dev) {
            param = words->wl_word;
        } else if (*p == '@' || *p == '#') {
            dev = p + 1;
            p = strchr(p, '[');
            if (p) {
                *p++ = '\0';
                param = p;
                p = strchr(p, ']');
                if (p)
                    *p = '\0';
            }
        } else {
            dev = p;
        }
        words = words->wl_next;
    }

    if (!dev) {
        fprintf(cp_err, "Error: no model or device name provided.\n");
        fprintf(cp_err, "Cannot alter parameters.\n");
        return;
    }

    /* in case the altermod command comes from commandline or
       over shared library we have to provide lowercase */
    strtolower(param);
    strtolower(dev);

    words = eqword->wl_next;
    /* skip next line if words is a vector */
    if (!eq(words->wl_word, "["))
        names = ft_getpnames_quotes(words, FALSE);
    else
        names = NULL;

    if (!names) {
        /* Put this to try to resolve the case of
         *   alter @vin[pulse] = [ 0 5 10n 10n 10n 50n 100n ]
         */
        char *xsbuf, *rem_xsbuf;

        double *list;
        double tmp;
        int error;
        /* move beyond '[' to allow INPevaluate() */
        if (eq(words->wl_word, "["))
            words = words->wl_next;
        xsbuf = rem_xsbuf = wl_flatten(words);
        /* fprintf(cp_err, "Chain    converted  %s \n", xsbuf); */

        for (i = 0, list = NULL;;) {
            tmp = INPevaluate(&xsbuf, &error, 1);
            if (error)
                break;
            /* printf(" returning vector value %g\n", tmp); */
            list = TREALLOC(double, list, i + 1);
            list[i++] = tmp;
        }

        if (i < 1) {
            fprintf(cp_err, "Error: cannot evaluate new parameter value.\n");
            return;
        }

        dv = dvec_alloc(copy("real vector"),
                        SV_NOTYPE,
                        VF_REAL,
                        i, list);
        if (!dv)
            return;

        /* Here I was, to change the inclusion in the circuit.
         * will have to revise that dv is right for its insertion.
         */
        if_setparam(ft_curckt->ci_ckt, &dev, param, dv, do_model);

        tfree(rem_xsbuf);
        vec_free(dv);
        return;
    }

    dv = ft_evaluate(names);
    if (!dv)
        goto done;

    if (dv->v_length < 1) {
        fprintf(cp_err, "Error: cannot evaluate new parameter value.\n");
        goto done;
    }

    /* If we want alter the geometry of a MOS device
       we have to ensure that we are in the valid model bin. */
    if ((dev[0] == 'm') && (eq(param, "w") || eq(param, "l")))
        if_set_binned_model(ft_curckt->ci_ckt, dev, param, dv);

    if_setparam(ft_curckt->ci_ckt, &dev, param, dv, do_model);

 done:
    /* va: garbage collection for dv, if pnode names is no simple value */
    if (names && !names->pn_value && dv)
        vec_free(dv);
    free_pnode(names); /* free also dv, if pnode names is simple value */
}


/* Given a device name, possibly with wildcards, return the matches. */

static wordlist *
devexpand(char *name)
{
    wordlist *wl, *devices;

    if (strchr(name, '*') || strchr(name, '[') || strchr(name, '?')) {
        devices = cp_cctowl(ft_curckt->ci_devices);
        for (wl = NULL; devices; devices = devices->wl_next)
            if (!strcmp(name, devices->wl_word))
                wl = wl_cons(devices->wl_word, wl);
    } else if (cieq(name, "all")) {
        wl = cp_cctowl(ft_curckt->ci_devices);
    } else {
        wl = wl_cons(name, NULL);
    }

    wl_sort(wl);
    return wl;
}


/* altermod mod_1 [mod_nn] file=modelparam.mod
   load model file and overwrite models mod_1 till mod_nn with
   all new parameters (limited to 16 models) */

static void
com_alter_mod(wordlist *wl)
{
#define MODLIM 16 /* max number of models */
    FILE *modfile;
    char *modellist[MODLIM] = {NULL}, *modellines[MODLIM] = {NULL}, *newmodelname, *newmodelline;
    char *filename = NULL, *eqword, *input, *modelline = NULL, *inptoken;
    int modno = 0, molineno = 0, i, j;
    wordlist *newcommand;
    struct card *modeldeck, *tmpdeck;
    char *readmode = "r";
    char **arglist;
    bool modelfound = FALSE;
    int ij[MODLIM];

    /* initialize */
    for (i = 0; i < MODLIM; i++)
        ij[i] = -1;

    /* read all model names */
    while (!ciprefix("file", wl->wl_word)) {
        if (modno == MODLIM) {
            fprintf(cp_err, "Error: too many model names in altermod command\n");
            controlled_exit(1);
        }
        modellist[modno] = copy(wl->wl_word);
        modno++;
        wl = wl->wl_next;
    }
    input = wl_flatten(wl);
    /* get the file name */
    eqword = strchr(input, '=');
    if (eqword) {
        eqword++;
        while (*eqword == ' ')
            eqword++;
        if (*eqword == '\0') {
            fprintf(cp_err, "Error: no filename given\n");
            controlled_exit(1);
        }
        filename = copy(eqword);
    } else {
        eqword = strstr(input, "file");
        eqword += 4;
        while (*eqword == ' ')
            eqword++;
        if (*eqword == '\0') {
            fprintf(cp_err, "Error: no filename given\n");
            controlled_exit(1);
        }
        filename = copy(eqword);
    }

    modfile = inp_pathopen(filename, readmode);

    if (modfile == NULL) {
        fprintf(cp_err, "Warning: Could not open file %s, altermod ignored\n", filename);
        tfree(input);
        tfree(filename);
        return;
    }
    {
        char *dir_name = ngdirname(filename);
        modeldeck = inp_readall(modfile, dir_name, 0, 0, NULL);
        tfree(dir_name);
    }
    tfree(input);
    tfree(filename);
    /* get all lines starting with *model */
    for (tmpdeck = modeldeck; tmpdeck; tmpdeck = tmpdeck->nextcard)
        /* We are looking for *model because the input paerser has
           invalidated all unused models by replacing '.' by '*'. */
        if (ciprefix("*model", tmpdeck->line)) {
            if (molineno == MODLIM) {
                fprintf(cp_err, "Error: more than %d models in deck, rest ignored\n", molineno);
                break;
            }
            modellines[molineno] = tmpdeck->line;
            molineno++;
        }
    /* Check if all models named in altermod command are to be found in input deck.
       Exit if not successfull */
    for (i = 0; i < modno; i++) {
        for (j = 0; j < molineno; j++) {
            newmodelline = modellines[j];
            /* get model name from model line */
            inptoken = gettok(&newmodelline); /* *model */
            tfree(inptoken);
            newmodelname = gettok(&newmodelline); /* modelname */
            if (cieq(newmodelname, modellist[i])) {
                modelfound = TRUE;
                tfree(newmodelname);
                break;
            }
            tfree(newmodelname);
        }
        if (modelfound) {
            modelfound = FALSE;
            ij[i] = j; /* model in altermod, found in model line */
            continue;
        } else {
            fprintf(cp_err, "Error: could not find model %s in input deck\n", modellist[i]);
            controlled_exit(1);
        }
    }
    /* read the model line, generate the altermod commands as a wordlist,
       and call com_alter_common() */
    arglist = TMALLOC(char *, 4);
    arglist[0] = copy("altermod");
    arglist[3] = NULL;
    /* for each model name of altermod command */
    for (i = 0; i < modno; i++) {
        /* model name */
        arglist[1] = copy(modellist[i]);
        /* parse model line from deck */
        modelline = modellines[ij[i]];
        inptoken = gettok(&modelline); /* skip *model */
        tfree(inptoken);
        inptoken = gettok(&modelline); /* skip modelname */
        tfree(inptoken);
        inptoken = gettok(&modelline); /* skip model type */
        tfree(inptoken);
        while ((inptoken = gettok_node(&modelline)) != NULL) {
            /* exclude level, version, mfg, and type */
            if (ciprefix("version", inptoken) || ciprefix("level", inptoken) ||
                ciprefix("mfg", inptoken) || ciprefix("type", inptoken) ) {
                tfree(inptoken);
                continue;
            }
            arglist[2] = inptoken;
            /* create a new wordlist from array arglist */
            newcommand = wl_build((const char * const *) arglist);
            com_alter_common(newcommand->wl_next, 1);
            wl_free(newcommand);
            tfree(inptoken);
        }
        tfree(arglist[1]);
    }
    tfree(arglist[0]);
    tfree(arglist[3]);
}


#ifdef HAVE_TSEARCH

#include <search.h>

static int
check_ifparm_compare(const void *a, const void *b)
{
    IFparm *pa = (IFparm *) a;
    IFparm *pb = (IFparm *) b;
    return pa->id - pb->id;
}


static void
check_ifparm_freenode(void *node)
{
    NG_IGNORE(node);
}


static void
check_ifparm(IFdevice *device, int instance_flag)
{
    int i, xcount;
    IFparm *plist;

    if (instance_flag) {
        plist = device->instanceParms;
        if (!plist)
            return;
        fprintf(stderr, " checking %s instanceParams\n", device->name);
        xcount = *device->numInstanceParms;
    } else {
        plist = device->modelParms;
        if (!plist)
            return;
        fprintf(stderr, " checking %s modelParams\n", device->name);
        xcount = *device->numModelParms;
    }

    void *root = NULL;

    for (i = 0; i < xcount; i++) {

        IFparm *psearch = *(IFparm **) tsearch(plist + i, &root,
                                               check_ifparm_compare);

        int type_err = (psearch->dataType ^ plist[i].dataType) & ~IF_REDUNDANT;
        if (type_err)
            fprintf(stderr,
                    " ERROR, dataType mismatch \"%s\" \"%s\" %08x\n",
                    psearch->keyword, plist[i].keyword, type_err);

        if ((plist[i].dataType & IF_REDUNDANT) &&
            (i == 0 || plist[i-1].id != plist[i].id)) {
            fprintf(stderr,
                    "ERROR, alias \"%s\" has non matching predecessor \"%s\"\n",
                    plist[i].keyword, plist[i-1].keyword);
        }

        if (i == 0)
            continue;

        if (plist[i-1].id != plist[i].id) {
            if (psearch != plist + i)
                fprintf(stderr,
                        "ERROR: non neighbored duplicate id: \"%s\" \"%s\"\n",
                        psearch->keyword, plist[i].keyword);
        } else if (!(plist[i].dataType & IF_REDUNDANT)) {
            fprintf(stderr,
                    "ERROR: non R duplicate id: \"%s\" \"%s\"\n",
                    plist[i-1].keyword, plist[i].keyword);
        }
    }

#ifdef HAVE_TDESTROY
    tdestroy (root, check_ifparm_freenode);
#endif
}


void
com_check_ifparm(wordlist *wl)
{
    NG_IGNORE(wl);

    int k;

    for (k = 0; k < ft_sim->numDevices; k++)
        if (ft_sim->devices[k]) {
            check_ifparm(ft_sim->devices[k], 0);
            check_ifparm(ft_sim->devices[k], 1);
        }
}

#endif
