/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000 AlansFixes
**********/

/*
 * Interface routines. These are specific to spice. The only changes to FTE
 * that should be needed to make FTE work with a different simulator is
 * to rewrite this file. What each routine is expected to do can be
 * found in the programmer's manual. This file should be the only one
 * that includes ngspice.header files.
 */

/*CDHW Notes:

I have never really understood the way Berkeley intended the six pointers
to default values (ci_defOpt/Task  ci_specOpt/Task ci_curOpt/Task) to work,
as there only see to be two data blocks to point at, or I've missed something
clever elsewhere.

Anyway, in the original 3f4 the interactive command 'set temp = 10'
set temp for its current task and clobbered the default values as a side
effect. When an interactive is run it created specTask using the spice
application default values, not the circuit defaults affected
by 'set temp = 10'.

The fix involves two changes

  1. Make 'set temp = 10' change the values in the 'default' block, not whatever
     the 'current' pointer happens to be pointing at (which is usually the
     default block except when one interactive is run immediately
after another).

  2. Hack CKTnewTask() so that it looks to see whether it is creating
a 'special'
     task, in which case it copies the values from
ft_curckt->ci_defTask providing
     everything looks sane, otherwise it uses the hard-coded
'application defaults'.

These are fairly minor changes, and as they don't change the data structures
they should be fairly 'safe'. However, ...


CDHW*/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/cpdefs.h"
#include "ngspice/tskdefs.h" /* Is really needed ? */
#include "ngspice/ftedefs.h"
#include "ngspice/fteinp.h"
#include "ngspice/inpdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/ifsim.h"

#include "circuits.h"
#include "spiceif.h"
#include "variable.h"


#ifdef XSPICE
#include "ngspice/evt.h"
#include "ngspice/enh.h"
/* gtri - add - wbk - 11/9/90 - include MIF function prototypes */
#include "ngspice/mifproto.h"
/* gtri - end - wbk - 11/9/90 */

/* gtri - evt - wbk - 5/20/91 - Add stuff for user-defined nodes */
#include "ngspice/evtproto.h"
#include "ngspice/evtudn.h"
/* gtri - end - wbk - 5/20/91 - Add stuff for user-defined nodes */
#include "ngspice/mif.h"
#endif

extern INPmodel *modtab;
extern bool ft_batchmode;

static struct variable *parmtovar(IFvalue *pv, IFparm *opt);
static IFparm *parmlookup(IFdevice *dev, GENinstance **inptr, char *param,
                           int do_model, int inout);
static IFvalue *doask(CKTcircuit *ckt, int typecode, GENinstance *dev, GENmodel *mod,
                       IFparm *opt, int ind);
static int doset(CKTcircuit *ckt, int typecode, GENinstance *dev, GENmodel *mod,
                 IFparm *opt, struct dvec *val);
static int finddev(CKTcircuit *ckt, char *name, GENinstance **devptr, GENmodel **modptr);

/* espice fix integration */
static int finddev_special(CKTcircuit *ckt, char *name, GENinstance **devptr, GENmodel **modptr, int *device_or_model);

/* Input a single deck, and return a pointer to the circuit. */

CKTcircuit *
if_inpdeck(struct card *deck, INPtables **tab)
{
    CKTcircuit *ckt;
    int err, i;
    struct card *ll;
    IFuid taskUid;
    IFuid optUid;
    int which = -1;

    for (i = 0, ll = deck; ll; ll = ll->nextcard)
        i++;
    *tab = INPtabInit(i);
    ft_curckt->ci_symtab = *tab;

    err = ft_sim->newCircuit (&ckt);
    if (err != OK) {
        ft_sperror(err, "CKTinit");
        return (NULL);
    }

    /*CDHW Create a task DDD with a new UID. ci_defTask will point to it CDHW*/

    err = IFnewUid(ckt, &taskUid, NULL, "default", UID_TASK, NULL);
    if (err) {
        ft_sperror(err, "newUid");
        return (NULL);
    }

#if (0)
    err = ft_sim->newTask (ckt, &(ft_curckt->ci_defTask), taskUid);
#else /*CDHW*/
    err = ft_sim->newTask (ckt, &(ft_curckt->ci_defTask), taskUid, NULL);
#endif
    if (err) {
        ft_sperror(err, "newTask");
        return (NULL);
    }

    /*CDHW which options available for this simulator? CDHW*/

    which = ft_find_analysis("options");

    if (which != -1) {
        err = IFnewUid(ckt, &optUid, NULL, "options", UID_ANALYSIS, NULL);
        if (err) {
            ft_sperror(err, "newUid");
            return (NULL);
        }

        err = ft_sim->newAnalysis (ft_curckt->ci_ckt, which, optUid,
                                   &(ft_curckt->ci_defOpt),
                                   ft_curckt->ci_defTask);

        /*CDHW ci_defTask and ci_defOpt point to parameters DDD CDHW*/

        if (err) {
            ft_sperror(err, "createOptions");
            return (NULL);
        }

        ft_curckt->ci_curOpt  = ft_curckt->ci_defOpt;
        /*CDHW ci_curOpt and ci_defOpt point to DDD CDHW*/
    }

    ft_curckt->ci_curTask = ft_curckt->ci_defTask;

    /* Parse the .model lines. Enter the model into the global model table modtab. */
    modtab = NULL;
    INPpas1(ckt, deck->nextcard, *tab);
    /* store the new model table in the current circuit */
    ft_curckt->ci_modtab = modtab;

    /* Scan through the instance lines and parse the circuit. */
    INPpas2(ckt, deck->nextcard, *tab, ft_curckt->ci_defTask);
#ifdef XSPICE
    if (!Evtcheck_nodes(ckt, *tab)) {
        ft_sperror(E_PRIVATE, "Evtcheck_nodes");
        return NULL;
    }
#endif

    /* If option cshunt is given, add capacitors to each voltage node */
    INPpas4(ckt, *tab);

    /* Fill in .NODESET and .IC data.
     * nodeset/ic of non-existent nodes is rejected.  */
    INPpas3(ckt, deck->nextcard,
            *tab, ft_curckt->ci_defTask, ft_sim->nodeParms,
            ft_sim->numNodeParms);

#ifdef XSPICE
    /* gtri - begin - wbk - 6/6/91 - Finish initialization of event driven structures */
    err = EVTinit(ckt);
    if (err) {
        ft_sperror(err, "EVTinit");
        return (NULL);
    }
    /* gtri - end - wbk - 6/6/91 - Finish initialization of event driven structures */
#endif

    return (ckt);
}


/* Do a run of the circuit, of the given type. Type "resume" is
 * special -- it means to resume whatever simulation that was in
 * progress. The return value of this routine is 0 if the exit was ok,
 * and 1 if there was a reason to interrupt the circuit (interrupt
 * typed at the keyboard, error in the simulation, etc). args should
 * be the entire command line, e.g. "tran 1 10 20 uic" */
int
if_run(CKTcircuit *ckt, char *what, wordlist *args, INPtables *tab)
{
    int err;
    struct card deck;
    char buf[BSIZE_SP];
    int which = -1;
    IFuid specUid, optUid;
    char *s;


    /* First parse the line... */
    /*CDHW Look for an interactive task CDHW*/
    if (eq(what, "tran") ||
        eq(what, "ac") ||
        eq(what, "dc") ||
        eq(what, "op") ||
        eq(what, "pz") ||
        eq(what, "disto") ||
        eq(what, "adjsen") ||
        eq(what, "sens") ||
        eq(what, "tf") ||
        eq(what, "noise")
#ifdef WITH_PSS
        /* Steady State Analysis */
        || eq(what, "pss")
#endif
#ifdef RFSPICE
        || eq(what, "sp")
#ifdef WITH_HB
        || eq(what, "hb")
#endif
#endif
        )
    {
        s = wl_flatten(args); /* va: tfree char's tmalloc'ed in wl_flatten */
        (void) sprintf(buf, ".%s", s);
        tfree(s);
        deck.nextcard = deck.actualLine = NULL;
        deck.error = NULL;
        deck.linenum = 0;
        deck.line = buf;

        /*CDHW Delete any previous special task CDHW*/

        if (ft_curckt->ci_specTask) {
            if (ft_curckt->ci_specTask == ft_curckt->ci_defTask)   /*CDHW*/
                printf("Oh dear...something bad has happened to the options.\n");

            err = ft_sim->deleteTask (ft_curckt->ci_ckt, ft_curckt->ci_specTask);
            if (err) {
                ft_sperror(err, "deleteTask");
                return (2);
            }

            ft_curckt->ci_specTask = NULL;
            ft_curckt->ci_specOpt  = NULL; /*CDHW*/
        }
        /*CDHW Create an interactive task AAA with a new UID.
          ci_specTask will point to it CDHW*/

        err = IFnewUid(ft_curckt->ci_ckt, &specUid, NULL, "special", UID_TASK, NULL);
        if (err) {
            ft_sperror(err, "newUid");
            return (2);
        }
#if (0)
        err = ft_sim->newTask (ft_curckt->ci_ckt,
                               &(ft_curckt->ci_specTask), specUid);
#else /*CDHW*/

        err = ft_sim->newTask (ft_curckt->ci_ckt,
                               &(ft_curckt->ci_specTask),
                               specUid, &(ft_curckt->ci_defTask));
#endif
        if (err) {
            ft_sperror(err, "newTask");
            return (2);
        }

        /*CDHW which options available for this simulator? CDHW*/

        which = ft_find_analysis("options");

        if (which != -1) { /*CDHW options are available CDHW*/
            err = IFnewUid(ft_curckt->ci_ckt, &optUid, NULL, "options", UID_ANALYSIS, NULL);
            if (err) {
                ft_sperror(err, "newUid");
                return (2);
            }

            err = ft_sim->newAnalysis (ft_curckt->ci_ckt, which, optUid,
                                       &(ft_curckt->ci_specOpt),
                                       ft_curckt->ci_specTask);

            /*CDHW 'options' ci_specOpt points to AAA in this case CDHW*/

            if (err) {
                ft_sperror(err, "createOptions");
                return (2);
            }

            ft_curckt->ci_curOpt  = ft_curckt->ci_specOpt;

            /*CDHW ci_specTask ci_specOpt and ci_curOpt all point to AAA CDHW*/

        }

        ft_curckt->ci_curTask = ft_curckt->ci_specTask;

        /*CDHW ci_curTask and ci_specTask point to the interactive task AAA CDHW*/

        INPpas2(ckt, &deck, tab, ft_curckt->ci_specTask);

        if (deck.error) {
            fprintf(cp_err, "Error: %sin   %s\n\n", deck.error, deck.line);
            return 2;
        }
    }

    /*CDHW
    ** if the task is to 'run' the deck, change ci_curTask and
    ** ci_curOpt to point to DDD
    ** created by if_inpdeck(), otherwise they point to AAA.
    CDHW*/

    if (eq(what, "run")) {
        ft_curckt->ci_curTask = ft_curckt->ci_defTask;
        ft_curckt->ci_curOpt = ft_curckt->ci_defOpt;
        if (ft_curckt->ci_curTask->jobs == NULL) {
            /* nothing to 'run' */
            if (!ft_batchmode) { /* FIXME: This is a hack to re-enable 'make check' */
                fprintf(stderr, "Warning: No job (tran, ac, op etc.) defined:\n");
                return (3);
            }
        }
    }

    /* -- Find out what we are supposed to do.              */

    if ((eq(what, "tran")) ||
        (eq(what, "ac")) ||
        (eq(what, "dc")) ||
        (eq(what, "op")) ||
        (eq(what, "pz")) ||
        (eq(what, "disto")) ||
        (eq(what, "noise")) ||
        (eq(what, "adjsen")) ||
        (eq(what, "sens")) ||
        (eq(what, "tf")) ||
#ifdef WITH_PSS
        /* SP: Steady State Analysis */
        (eq(what, "pss")) ||
        /* SP */
#endif
#ifdef RFSPICE
        (eq(what, "sp")) ||
#ifdef WITH_HB
        (eq(what, "hb")) ||
#endif
#endif
        (eq(what, "run")))
    {
        /*CDHW Run the analysis pointed to by ci_curTask CDHW*/

        ft_curckt->ci_curOpt = ft_curckt->ci_defOpt;
        if ((err = ft_sim->doAnalyses (ckt, 1, ft_curckt->ci_curTask)) != OK) {
            ft_sperror(err, "doAnalyses");
            /* wrd_end(); */
            if (err == E_PAUSE)
                return (1);
            else
                return (2);
        }
    } else if (eq(what, "resume")) {
        if ((err = ft_sim->doAnalyses (ckt, 0, ft_curckt->ci_curTask)) != OK) {
            ft_sperror(err, "doAnalyses");
            /* wrd_end(); */
            if (err == E_PAUSE)
                return (1);
            else
                return (2);
        }
    } else {
        fprintf(cp_err, "if_run: Internal Error: bad run type %s\n", what);
        return (2);
    }

    return (0);
}


/* Set an option in the circuit. Arguments are option name, type, and
 * value (the last a char *), suitable for casting to whatever needed...
 */

static char *unsupported[] = {
    "itl3",
    "itl5",
    "lvltim",
    "maxord",
    "method",
    NULL
};

static char *obsolete[] = {
    "limpts",
    "limtim",
    "lvlcod",
    NULL
};


int
if_option(CKTcircuit *ckt, char *name, enum cp_types type, void *value)
{
    IFvalue pval;
    int err;
    char **vv, *sfree = NULL;
    int which = -1;
    IFparm *if_parm;

    if (eq(name, "acct")) {
        ft_acctprint = TRUE;
        return 0;
    } else if (eq(name, "noacct")) {
        ft_noacctprint = TRUE;
        return 0;
    } else if (eq(name, "noinit")) {
        ft_noinitprint = TRUE;
        return 0;
    } else if (eq(name, "norefvalue")) {
        ft_norefprint = TRUE;
        return 0;
    } else if (eq(name, "list")) {
        ft_listprint = TRUE;
        return 0;
    } else if (eq(name, "node")) {
        ft_nodesprint = TRUE;
        return 0;
    } else if (eq(name, "opts")) {
        ft_optsprint = TRUE;
        return 0;
    } else if (eq(name, "nopage")) {
        ft_nopage = TRUE;
        return 0;
    } else if (eq(name, "nomod")) {
        ft_nomod = TRUE;
        return 0;
    }

    which = ft_find_analysis("options");

    if (which == -1) {
        fprintf(cp_err, "Warning:  .options line unsupported\n");
        return 0;
    }

    if_parm = ft_find_analysis_parm(which, name);

    if (!if_parm || !(if_parm->dataType & IF_SET)) {
        /* See if this is unsupported or obsolete. */
        for (vv = unsupported; *vv; vv++)
            if (eq(name, *vv)) {
                fprintf(cp_err, "Warning: option %s is currently unsupported.\n", name);
                return 1;
            }
        for (vv = obsolete; *vv; vv++)
            if (eq(name, *vv)) {
                fprintf(cp_err, "Warning: option %s is obsolete.\n", name);
                return 1;
            }
        return 0;
    }

    switch (if_parm->dataType & IF_VARTYPES) {
    case IF_REAL:
        if (type == CP_REAL)
            pval.rValue = *((double *) value);
        else if (type == CP_NUM)
            pval.rValue = *((int *) value);
        else
            goto badtype;
        break;
    case IF_INTEGER:
        if (type == CP_NUM)
            pval.iValue = *((int *) value);
        else if (type == CP_REAL)
            pval.iValue = (int)floor((*(double *)value) + 0.5);
        else
            goto badtype;
        break;
    case IF_STRING:
        if (type == CP_STRING)
            sfree = pval.sValue = copy((char*) value);
        else
            goto badtype;
        break;
    case IF_FLAG:
        if (type == CP_BOOL)
            pval.iValue = *((bool *) value) ? 1 : 0;
        else if (type == CP_NUM) /* FIXME, shall we allow this ? */
            pval.iValue = *((int *) value);
        else
            goto badtype;
        break;
    default:
        fprintf(cp_err,
                "if_option: Internal Error: bad option type %d.\n",
                if_parm->dataType);
    }

    if (!ckt) {
        /* XXX No circuit loaded */
        fprintf(cp_err, "Simulation parameter \"%s\" can't be set until\n",
                name);
        fprintf(cp_err, "a circuit has been loaded.\n");
        return 1;
    }

#if (0)
    if ((err = ft_sim->setAnalysisParm (ckt, ft_curckt->ci_curOpt,
                                        if_parm->id, &pval,
                                        NULL)) != OK)
        ft_sperror(err, "setAnalysisParm(options) ci_curOpt");
#else /*CDHW*/
    if ((err = ft_sim->setAnalysisParm (ckt, ft_curckt->ci_defOpt,
                                        if_parm->id, &pval,
                                        NULL)) != OK)
        ft_sperror(err, "setAnalysisParm(options) ci_curOpt");
    tfree(sfree);
    return 1;
#endif

badtype:
    fprintf(cp_err, "Error: bad type given for option %s --\n", name);
    fprintf(cp_err, "\ttype given was ");
    switch (type) {
    case CP_BOOL:
        fputs("boolean", cp_err);
        break;
    case CP_NUM:
        fputs("integer", cp_err);
        break;
    case CP_REAL:
        fputs("real", cp_err);
        break;
    case CP_STRING:
        fputs("string", cp_err);
        break;
    case CP_LIST:
        fputs("list", cp_err);
        break;
    default:
        fputs("something strange", cp_err);
        break;
    }
    fprintf(cp_err, ", type expected was ");
    switch (if_parm->dataType & IF_VARTYPES) {
    case IF_REAL:
        fputs("real.\n", cp_err);
        break;
    case IF_INTEGER:
        fputs("integer.\n", cp_err);
        break;
    case IF_STRING:
        fputs("string.\n", cp_err);
        break;
    case IF_FLAG:
        fputs("flag.\n", cp_err);
        break;
    default:
        fputs("something strange.\n", cp_err);
        break;
    }

    if (type == CP_BOOL)
        fputs("\t(Note that you must use an = to separate option name and value.)\n",
              cp_err);
    return 0;
}


void
if_dump(CKTcircuit *ckt, FILE *file)
{
    NG_IGNORE(ckt);

    fprintf(file, "diagnostic output dump unavailable.");
}


void
if_cktfree(CKTcircuit *ckt, INPtables *tab)
{
    ft_sim->deleteCircuit (ckt);
    INPtabEnd(tab);
}


/* Return a string describing an error code. */

/* BLOW THIS AWAY.... */

char *
if_errstring(int code)
{
    return (INPerror(code));
}


/* Get pointers to a device, its model, and its type number given the name. If
 * there is no such device, try to find a model with that name
 * device_or_model says if we are referencing a device or a model.
 *  finddev_special(ck, name, devptr, modptr, device_or_model):
 *  Introduced to look for correct reference in expression like  print @BC107 [is]
 * and find out  whether a model or a device parameter is referenced and properly
 * call the spif_getparam_special (ckt, name, param, ind, do_model) function in
 * vector.c - A. Roldan (espice).
 */
static int
finddev_special(
    CKTcircuit *ckt,
    char *name,
    GENinstance **devptr,
    GENmodel **modptr,
    int *device_or_model)
{
    *devptr = ft_sim->findInstance (ckt, name);
    if (*devptr) {
        *device_or_model = 0;
        return (*devptr)->GENmodPtr->GENmodType;
    }

    *modptr = ft_sim->findModel (ckt, name);
    if (*modptr) {
        *device_or_model = 1;
        return (*modptr)->GENmodType;
    }

    *device_or_model = 2;
    return (-1);
}


/* Get a parameter value from the circuit. If name is left unspecified,
 * we want a circuit parameter. Now works both for devices and models.
 * A.Roldan (espice)
 */
struct variable *
spif_getparam_special(CKTcircuit *ckt, char **name, char *param, int ind, int do_model)
{
    struct variable *vv = NULL, *tv;
    IFvalue *pv;
    IFparm *opt;
    int typecode, i, modelo_dispositivo;
    GENinstance *dev = NULL;
    GENmodel *mod = NULL;
    IFdevice *device;

    NG_IGNORE(do_model);

    /* fprintf(cp_err, "Calling if_getparam(%s, %s)\n", *name, param); */

    if (!param || (param && eq(param, "all"))) {
        INPretrieve(name, ft_curckt->ci_symtab);
        typecode = finddev_special(ckt, *name, &dev, &mod, &modelo_dispositivo);
        if (typecode == -1) {
            fprintf(cp_err, "Error: no such device or model name %s\n", *name);
            return (NULL);
        }
        device = ft_sim->devices[typecode];
        if (!modelo_dispositivo) {
            /* It is a device */
            for (i = 0; i < *(device->numInstanceParms); i++) {
                opt = &device->instanceParms[i];
                if (opt->dataType & IF_REDUNDANT || !opt->description)
                    continue;
                if (!(opt->dataType & IF_ASK))
                    continue;
                pv = doask(ckt, typecode, dev, mod, opt, ind);
                if (pv) {
                    tv = parmtovar(pv, opt);

                    /* With the following we pack the name and the acronym of the parameter */
                    {
                        char *x = tv->va_name;
                        tv->va_name = tprintf("%s [%s]", tv->va_name, device->instanceParms[i].keyword);
                        tfree(x);
                    }
                    if (vv)
                        tv->va_next = vv;
                    vv = tv;
                } else {
                    fprintf(cp_err,
                            "Internal Error: no parameter '%s' on device '%s'\n",
                            device->instanceParms[i].keyword, device->name);
                }
            }
            return (vv);
        } else { /* Is it a model or a device ? */
            /* It is a model */
            for (i = 0; i < *(device->numModelParms); i++) {
                opt = &device->modelParms[i];
                if (opt->dataType & IF_REDUNDANT || !opt->description)
                    continue;

                /* We check that the parameter is interesting and therefore is
                 * implemented in the corresponding function ModelAsk. Originally
                 * the argument of "if" was: || (opt->dataType & IF_STRING)) continue;
                 * so, a model parameter defined like  OP("type",   MOS_SGT_MOD_TYPE,
                 * IF_STRING, N-channel or P-channel MOS") would not be printed.
                 */

                /* if (!(opt->dataType & IF_ASK) || (opt->dataType & IF_UNINTERESTING) || (opt->dataType & IF_STRING)) continue; */
                if (!(opt->dataType & IF_ASK) || (opt->dataType & IF_UNINTERESTING))
                    continue;
                pv = doask(ckt, typecode, dev, mod, opt, ind);
                if (pv) {
                    tv = parmtovar(pv, opt);
                    /* Inside parmtovar:
                     * 1. tv->va_name = copy(opt->description);
                     * 2. Copy the type of variable of IFparm into a variable (thus parm-to-var)
                     * vv->va_type = opt->dataType
                     * The long description of the parameter:
                     * IFparm MOS_SGTmPTable[] = { // model parameters //
                     * OP("type",   MOS_SGT_MOD_TYPE,  IF_STRING, "N-channel or P-channel MOS")
                     * goes into tv->va_name to put braces around the parameter of the model
                     * tv->va_name += device->modelParms[i].keyword;
                     */
                    {
                        char *x = tv->va_name;
                        tv->va_name = tprintf("%s [%s]", tv->va_name, device->modelParms[i].keyword);
                        tfree(x);
                    }
                    /* tv->va_string = device->modelParms[i].keyword;   Put the name of the variable */
                    if (vv)
                        tv->va_next = vv;
                    vv = tv;
                } else {
                    fprintf(cp_err,
                            "Internal Error: no parameter '%s' on device '%s'\n",
                            device->modelParms[i].keyword, device->name);
                }
            }
            return (vv);
        }
    } else if (param) {
        INPretrieve(name, ft_curckt->ci_symtab);
        typecode = finddev_special(ckt, *name, &dev, &mod, &modelo_dispositivo);
        if (typecode == -1) {
            fprintf(cp_err, "Error: no such device or model name %s\n", *name);
            return (NULL);
        }
        device = ft_sim->devices[typecode];
        opt = parmlookup(device, &dev, param, modelo_dispositivo, 0);
        if (!opt) {
            fprintf(cp_err, "Error: no such parameter %s.\n", param);
            return (NULL);
        }
        pv = doask(ckt, typecode, dev, mod, opt, ind);
        if (pv)
            vv = parmtovar(pv, opt);
        return (vv);
    } else {
        return (if_getstat(ckt, *name));
    }
}


/* Get a parameter value from the circuit. If name is left unspecified,
 * we want a circuit parameter.
 */

struct variable *
spif_getparam(CKTcircuit *ckt, char **name, char *param, int ind, int do_model)
{
    struct variable *vv = NULL, *tv;
    IFvalue *pv;
    IFparm *opt;
    int typecode, i;
    GENinstance *dev = NULL;
    GENmodel *mod = NULL;
    IFdevice *device;

    /* fprintf(cp_err, "Calling if_getparam(%s, %s)\n", *name, param); */

    if (param && eq(param, "all")) {

        /* MW. My "special routine here" */
        INPretrieve(name, ft_curckt->ci_symtab);

        typecode = finddev(ckt, *name, &dev, &mod);
        if (typecode == -1) {
            fprintf(cp_err,
                    "Error: no such device or model name %s\n",
                    *name);
            return (NULL);
        }
        device = ft_sim->devices[typecode];
        for (i = 0; i < *(device->numInstanceParms); i++) {
            opt = &device->instanceParms[i];
            if (opt->dataType & IF_REDUNDANT || !opt->description)
                continue;
            if (!(opt->dataType & IF_ASK))
                continue;
            pv = doask(ckt, typecode, dev, mod, opt, ind);
            if (pv) {
                tv = parmtovar(pv, opt);
                if (vv)
                    tv->va_next = vv;
                vv = tv;
            } else {
                fprintf(cp_err,
                        "Internal Error: no parameter '%s' on device '%s'\n",
                        device->instanceParms[i].keyword,
                        device->name);
            }
        }
        return (vv);
    } else if (param) {

        /* MW.  */
        INPretrieve(name, ft_curckt->ci_symtab);
        typecode = finddev(ckt, *name, &dev, &mod);
        if (typecode == -1) {
            fprintf(cp_err, "Error: no such device or model name %s\n", *name);
            return (NULL);
        }
        device = ft_sim->devices[typecode];
        opt = parmlookup(device, &dev, param, do_model, 0);
        if (!opt) {
            fprintf(cp_err, "Error: no such parameter %s.\n", param);
            return (NULL);
        }
        pv = doask(ckt, typecode, dev, mod, opt, ind);
        if (pv)
            vv = parmtovar(pv, opt);
        return (vv);
    } else {
        return (if_getstat(ckt, *name));
    }
}


/* 9/26/03 PJB : function to allow setting model of device */
void
if_setparam_model(CKTcircuit *ckt, char **name, char *val)
{
    GENinstance *dev     = NULL;
    GENinstance *prevDev = NULL;
    GENmodel    *curMod  = NULL;
    GENmodel    *newMod  = NULL;
    INPmodel    *inpmod  = NULL;
    GENinstance *iter;
    GENmodel    *mods, *prevMod;
    int         typecode;
    char        *modname;

    /* retrieve device name from symbol table */
    INPretrieve(name, ft_curckt->ci_symtab);
    /* find the specified device */
    typecode = finddev(ckt, *name, &dev, &curMod);
    if (typecode == -1) {
        fprintf(cp_err, "Error: no such device name %s\n", *name);
        return;
    }
    curMod = dev->GENmodPtr;
    modname = copy(dev->GENmodPtr->GENmodName);
    modname = strtok(modname, "."); /* want only have the parent model name */
    /*
      retrieve the model from the global model table; also add the model to 'ckt'
      and indicate model is being used
    */
    INPgetMod(ckt, modname, &inpmod, ft_curckt->ci_symtab);
    /* check if using model binning -- pass in line since need 'l' and 'w' */
    if (inpmod == NULL)
        INPgetModBin(ckt, modname, &inpmod, ft_curckt->ci_symtab, val);
    tfree(modname);
    if (inpmod == NULL) {
        fprintf(cp_err, "Error: no model available for %s.\n", val);
        return;
    }
    newMod = inpmod->INPmodfast;

    /* see if new model name same as current model name */
    if (newMod->GENmodName != curMod->GENmodName)
        printf("Notice: model has changed from %s to %s.\n", curMod->GENmodName, newMod->GENmodName);
    if (newMod->GENmodType != curMod->GENmodType) {
        fprintf(cp_err, "Error: new model %s must be same type as current model.\n", val);
        return;
    }

    /* fix current model linked list */
    prevDev = NULL;
    for (iter = curMod->GENinstances; iter; iter = iter->GENnextInstance) {
        if (iter->GENname == dev->GENname) {

            /* see if at beginning of linked list */
            if (prevDev == NULL)
                curMod->GENinstances     = iter->GENnextInstance;
            else
                prevDev->GENnextInstance = iter->GENnextInstance;

            /* update model for device */
            dev->GENmodPtr       = newMod;
            dev->GENnextInstance = newMod->GENinstances;
            newMod->GENinstances = dev;
            break;
        }
        prevDev = iter;
    }
    /* see if any devices remaining that reference current model */
    if (curMod->GENinstances == NULL) {
        prevMod = NULL;
        for (mods = ckt->CKThead[typecode]; mods; mods = mods->GENnextModel) {
            if (mods->GENmodName == curMod->GENmodName) {

                /* see if at beginning of linked list */
                if (prevMod == NULL)
                    ckt->CKThead[typecode] = mods->GENnextModel;
                else
                    prevMod->GENnextModel  = mods->GENnextModel;

                INPgetMod(ckt, mods->GENmodName, &inpmod, ft_curckt->ci_symtab);
                if (curMod != nghash_delete(ckt->MODnameHash, curMod->GENmodName))
                    fprintf(stderr, "ERROR, ouch nasal daemons ...\n");
                GENmodelFree(mods);

                inpmod->INPmodfast = NULL;
                break;
            }
            prevMod = mods;
        }
    }
}


void
if_setparam(CKTcircuit *ckt, char **name, char *param, struct dvec *val, int do_model)
{
    IFparm *opt;
    IFdevice *device;
    GENmodel *mod = NULL;
    GENinstance *dev = NULL;
    int typecode;

    /* PN  */
    INPretrieve(name, ft_curckt->ci_symtab);
    typecode = finddev(ckt, *name, &dev, &mod);
    if (typecode == -1) {
        fprintf(cp_err, "Error: no such device or model name %s\n", *name);
        return;
    }
    device = ft_sim->devices[typecode];
    opt = parmlookup(device, &dev, param, do_model, 1);
    if (!opt) {
        if (param)
            fprintf(cp_err, "Error: no such parameter %s.\n", param);
        else
            fprintf(cp_err, "Error: no default parameter.\n");
        return;
    }
    if (do_model && !mod) {
        mod = dev->GENmodPtr;
        dev = NULL;
    }
    doset(ckt, typecode, dev, mod, opt, val);

    /* Call to CKTtemp(ckt) will be invoked here only by 'altermod' commands,
       to set internal model parameters pParam of each instance for immediate use,
       otherwise e.g. model->BSIM3vth0 will be set, but not pParam of any BSIM3 instance.
       Call only if CKTtime > 0 to avoid conflict with previous 'reset' command.
       May contain side effects because called from many places.  h_vogt 110101
    */
    if (do_model && (ckt->CKTtime > 0)) {
        int error = 0;
        error = CKTtemp(ckt);
        if (error)
            fprintf(stderr, "Error during changing a device model parameter!\n");
        if (error)
            controlled_exit(1);
    }
}


static struct variable *
parmtovar(IFvalue *pv, IFparm *opt)
{
    /* It is not clear whether we want to name the variable
     *   by `keyword' or by `description' */

    switch (opt->dataType & IF_VARTYPES) {
    case IF_INTEGER:
        return var_alloc_num(copy(opt->description), pv->iValue, NULL);
    case IF_REAL:
    case IF_COMPLEX:
        return var_alloc_real(copy(opt->description), pv->rValue, NULL);
    case IF_STRING:
        return var_alloc_string(copy(opt->description), pv->sValue, NULL);
    case IF_FLAG:
        return var_alloc_bool(copy(opt->description), pv->iValue ? TRUE : FALSE, NULL);
    case IF_REALVEC: {
        struct variable *list = NULL;
        int i;
        for (i = pv->v.numValue; --i >= 0;)
            list = var_alloc_real(NULL, pv->v.vec.rVec[i], list);
        return var_alloc_vlist(copy(opt->description), list, NULL);
        /* It is a linked list where the first node is a variable
         * pointing to the different values of the variables.
         *
         * To access the values of the real variable vector must be
         * vv->va_V.vV_real = valor node ppal that is of no use.
         *
         * In the case of Vin_sin 1 0 sin (0 2 2000)
         * and of print @vin_sin[sin]
         *
         * vv->va_V.vV_list->va_V.vV_real = 2000
         * vv->va_V.vV_list->va_next->va_V.vV_real = 2
         * vv->va_V.vV_list->va_next->va_next->va_V.vV_real = 0
         * So the list is starting from behind, but no problem
         * This works fine
         */
    }
    default:
        fprintf(cp_err,
                "parmtovar: Internal Error: bad PARM type %d.\n",
                opt->dataType);
        return (NULL);
    }
}


/* Extract the parameter (IFparm structure) from the device or device's model.
 * If do_mode is TRUE then look in the device's parameters
 * If do_mode is FALSE then look in the device model's parameters
 * If inout equals 1 then look only for parameters with the IF_SET type flag
 * if inout equals 0 then look only for parameters with the IF_ASK type flag
 */

static IFparm *
parmlookup(IFdevice *dev, GENinstance **inptr, char *param, int do_model, int inout)
{
    int i;

    NG_IGNORE(inptr);

    /* First try the device questions... */
    if (!do_model && dev->numInstanceParms) {
        for (i = 0; i < *(dev->numInstanceParms); i++) {
            if (!param && (dev->instanceParms[i].dataType & IF_PRINCIPAL))
                return (&dev->instanceParms[i]);
            else if (!param)
                continue;
            else if ((((dev->instanceParms[i].dataType & IF_SET) && inout == 1) ||
                      ((dev->instanceParms[i].dataType & IF_ASK) && inout == 0)) &&
                     cieq(dev->instanceParms[i].keyword, param))
            {
                while ((dev->instanceParms[i].dataType & IF_REDUNDANT) && (i > 0))
                    i--;
                return (&dev->instanceParms[i]);
            }
        }
        return NULL;
    }

    if (dev->numModelParms)
        for (i = 0; i < *(dev->numModelParms); i++)
            if ((((dev->modelParms[i].dataType & IF_SET) && inout == 1) ||
                 ((dev->modelParms[i].dataType & IF_ASK) && inout == 0)) &&
                eq(dev->modelParms[i].keyword, param))
            {
                while ((dev->modelParms[i].dataType & IF_REDUNDANT) && (i > 0))
                    i--;
                return (&dev->modelParms[i]);
            }

    return (NULL);
}


/* Perform the CKTask call. We have both 'fast' and 'modfast', so the other
 * parameters aren't necessary.
 */

static IFvalue *
doask(CKTcircuit *ckt, int typecode, GENinstance *dev, GENmodel *mod, IFparm *opt, int ind)
{
    static IFvalue pv;
    int err;

    NG_IGNORE(typecode);

    pv.iValue = ind;    /* Sometimes this will be junk and ignored... */

    /* fprintf(cp_err, "Calling doask(%d, %x, %x, %x)\n",
       typecode, dev, mod, opt); */
    if (dev)
        err = ft_sim->askInstanceQuest (ckt, dev, opt->id, &pv, NULL);
    else
        err = ft_sim->askModelQuest (ckt, mod, opt->id, &pv, NULL);

    if (err != OK) {
        ft_sperror(err, "if_getparam");
        return (NULL);
    }

    return (&pv);
}


/* Perform the CKTset call. We have both 'fast' and 'modfast', so the other
 * parameters aren't necessary.
 */

static int
doset(CKTcircuit *ckt, int typecode, GENinstance *dev, GENmodel *mod, IFparm *opt, struct dvec *val)
{
    IFvalue nval;
    int err;
    int n;
    int *iptr;
    double *dptr;
    int i;

    NG_IGNORE(typecode);

    /* Count items */
    if (opt->dataType & IF_VECTOR) {
        n = nval.v.numValue = val->v_length;

        dptr = val->v_realdata;
        /* XXXX compdata!!! */

        switch (opt->dataType & (IF_VARTYPES & ~IF_VECTOR)) {
        case IF_FLAG:
        case IF_INTEGER:
            iptr = nval.v.vec.iVec = TMALLOC(int, n);

            for (i = 0; i < n; i++)
                *iptr++ = (int)floor(*dptr++ + 0.5);
            break;

        case IF_REAL:
            nval.v.vec.rVec = val->v_realdata;
            break;

        default:
            fprintf(cp_err,
                    "Can't assign value to \"%s\" (unsupported vector type)\n",
                    opt->keyword);
            return E_UNSUPP;
        }
    } else {
        switch (opt->dataType & IF_VARTYPES) {
        case IF_FLAG:
        case IF_INTEGER:
            nval.iValue = (int)floor(*val->v_realdata + 0.5);
            break;

        case IF_REAL:
            /*kensmith don't blow up with NULL dereference*/
            if (!val->v_realdata) {
                fprintf(cp_err, "Unable to determine the value\n");
                return E_UNSUPP;
            }

            nval.rValue = *val->v_realdata;
            break;

        default:
            fprintf(cp_err,
                    "Can't assign value to \"%s\" (unsupported type)\n",
                    opt->keyword);
            return E_UNSUPP;
        }
    }

    /* fprintf(cp_err, "Calling doask(%d, %x, %x, %x)\n",
       typecode, dev, mod, opt); */

    if (dev)
        err = ft_sim->setInstanceParm (ckt, dev, opt->id, &nval, NULL);
    else
        err = ft_sim->setModelParm (ckt, mod, opt->id, &nval, NULL);

    return err;
}


/* Get pointers to a device, its model, and its type number given the name. If
 * there is no such device, try to find a model with that name.
 */

static int
finddev(CKTcircuit *ckt, char *name, GENinstance **devptr, GENmodel **modptr)
{
    *devptr = ft_sim->findInstance (ckt, name);
    if (*devptr)
        return (*devptr)->GENmodPtr->GENmodType;

    *modptr = ft_sim->findModel (ckt, name);
    if (*modptr)
        return (*modptr)->GENmodType;

    return (-1);
}


/* get an analysis parameter by name instead of id */

int
if_analQbyName(CKTcircuit *ckt, int which, JOB *anal, char *name, IFvalue *parm)
{
    IFparm *if_parm = ft_find_analysis_parm(which, name);

    if (!if_parm)
        return (E_BADPARM);

    return (ft_sim->askAnalysisQuest (ckt, anal, if_parm->id, parm, NULL));
}


/* Get the parameters tstart, tstop, and tstep from the CKT struct. */

/* BLOW THIS AWAY TOO */

bool
if_tranparams(struct circ *ci, double *start, double *stop, double *step)
{
    IFvalue tmp;
    int err;
    int which = -1;
    JOB *anal;
    IFuid tranUid;

    if (!ci->ci_curTask)
        return (FALSE);

    which = ft_find_analysis("TRAN");

    if (which == -1)
        return (FALSE);

    err = IFnewUid(ci->ci_ckt, &tranUid, NULL, "Transient Analysis", UID_ANALYSIS, NULL);
    if (err != OK)
        return (FALSE);

    err = ft_sim->findAnalysis (ci->ci_ckt, &which, &anal, tranUid,
                                ci->ci_curTask, NULL);
    if (err != OK)
        return (FALSE);

    err = if_analQbyName(ci->ci_ckt, which, anal, "tstart", &tmp);
    if (err != OK)
        return (FALSE);

    *start = tmp.rValue;

    err = if_analQbyName(ci->ci_ckt, which, anal, "tstop", &tmp);
    if (err != OK)
        return (FALSE);

    *stop = tmp.rValue;

    err = if_analQbyName(ci->ci_ckt, which, anal, "tstep", &tmp);
    if (err != OK)
        return (FALSE);

    *step = tmp.rValue;
    return (TRUE);
}


/* Get the statistic called 'name'.  If this is NULL get all statistics
 * available.
 */

struct variable *
if_getstat(CKTcircuit *ckt, char *name)
{
    int         options_idx, i;
    IFanalysis *options;
    IFvalue     parm;
    IFparm     *if_parm;

    options_idx = ft_find_analysis("options");

    if (options_idx == -1) {
        fprintf(cp_err, "Warning:  statistics unsupported\n");
        return (NULL);
    }

    options = ft_sim->analyses[options_idx];

    if (name) {

        if_parm = ft_find_analysis_parm(options_idx, name);

        if (!if_parm)
            return (NULL);

        if (ft_sim->askAnalysisQuest (ckt,
                                      &(ft_curckt->ci_curTask->taskOptions),
                                      if_parm->id, &parm,
                                      NULL) == -1)
        {
            fprintf(cp_err, "if_getstat: Internal Error: can't get %s\n", name);
            return (NULL);
        }

        return (parmtovar(&parm, if_parm));

    } else {

        struct variable *vars = NULL, **v = &vars;

        for (i = 0; i < options->numParms; i++) {

            if_parm = &(options->analysisParms[i]);

            if (!(if_parm->dataType & IF_ASK))
                continue;

            if (ft_sim->askAnalysisQuest (ckt,
                                          &(ft_curckt->ci_curTask->taskOptions),
                                          if_parm->id, &parm,
                                          NULL) == -1)
            {
                fprintf(cp_err, "if_getstat: Internal Error: can't get a name\n");
                return (NULL);
            }

            *v = parmtovar(&parm, if_parm);
            v = &((*v)->va_next);
        }

        return (vars);
    }
}


/* Some small updates to make it work, h_vogt, Feb. 2012
   Still very experimental !
   It is now possible to save a state during transient simulation,
   reload it later into a new ngspice run and resume simulation.
   XSPICE code models probably will not do.
   LTRA transmission line will not do.
   Many others are not tested.
*/

#include "ngspice/cktdefs.h"
#include "ngspice/trandefs.h"

/* arg0: circuit file, arg1: data file */
void com_snload(wordlist *wl)
{
    int error = 0;
    FILE *file;
    int tmpI, i, size;
    CKTcircuit *my_ckt, *ckt;

    /*
      Pseudo code:

      source(file_name);
      This should setup all the device structs, voltage nodes, etc.

      call cktsetup;
      This is needed to setup vector mamory allocation for vectors and branch nodes

      load_binary_data(info);
      Overwrite the allocated numbers, rhs etc, with saved data
    */


    if (ft_curckt && !strstr(ft_curckt->ci_name, "script")) {
        /* Circuit, not a script */
        fprintf(cp_err, "Error: there is already a circuit loaded.\n");
        return;
    }

    /* source the circuit */
    inp_source(wl->wl_word);
    if (!ft_curckt)
        return;

    /* allocate all the vectors, with luck!  */
    if (!error)
        error = CKTsetup(ft_curckt->ci_ckt);
    if (!error)
        error = CKTtemp(ft_curckt->ci_ckt);

    if (error) {
        fprintf(cp_err, "Some error in the CKT setup fncts!\n");
        return;
    }

    /* so it resumes ... */
    ft_curckt->ci_inprogress = TRUE;

    /* now load the binary file */
    ckt = ft_curckt->ci_ckt;

    file = fopen(wl->wl_next->wl_word, "rb");

    if (!file) {
        fprintf(cp_err, "Error: Couldn't open \"%s\" for reading\n", wl->wl_next->wl_word);
        return;
    }

    if (fread(&tmpI, sizeof(int), 1, file) != 1) {
        (void) fprintf(cp_err, "Unable to read spice version from snapshot.\n");
        fclose(file);
        return;
    }
    if (tmpI != sizeof(CKTcircuit)) {
        fprintf(cp_err, "loaded num: %d, expected num: %ld\n", tmpI, (long)sizeof(CKTcircuit));
        fprintf(cp_err, "Error: snapshot saved with different version of spice\n");
        fclose(file);
        return;
    }

    my_ckt = TMALLOC(CKTcircuit, 1);

    if (fread(my_ckt, sizeof(CKTcircuit), 1, file) != 1) {
        (void) fprintf(cp_err, "Unable to read spice circuit from snapshot.\n");
        fclose(file);
        return;
    }

#define _t(name) ckt->name = my_ckt->name
#define _ta(name, size)                                                 \
    do { int __i; for (__i = 0; __i < size; __i++) _t(name[__i]); } while(0)

    _t(CKTtime);
    _t(CKTdelta);
    _ta(CKTdeltaOld, 7);
    _t(CKTtemp);
    _t(CKTnomTemp);
    _t(CKTvt);
    _ta(CKTag, 7);

    _t(CKTorder);
    _t(CKTmaxOrder);
    _t(CKTintegrateMethod);
    _t(CKTxmu);
    _t(CKTindverbosity);
    _t(CKTepsmin);

    _t(CKTniState);

    _t(CKTmaxEqNum);
    _t(CKTcurrentAnalysis);

    _t(CKTnumStates);
    _t(CKTmode);

    _t(CKTbypass);
    _t(CKTdcMaxIter);
    _t(CKTdcTrcvMaxIter);
    _t(CKTtranMaxIter);
    _t(CKTbreakSize);
    _t(CKTbreak);
    _t(CKTsaveDelta);
    _t(CKTminBreak);
    _t(CKTabstol);
    _t(CKTpivotAbsTol);
    _t(CKTpivotRelTol);
    _t(CKTreltol);
    _t(CKTchgtol);
    _t(CKTvoltTol);

    _t(CKTgmin);
    _t(CKTgshunt);
    _t(CKTcshunt);
    _t(CKTdelmin);
    _t(CKTtrtol);
    _t(CKTfinalTime);
    _t(CKTstep);
    _t(CKTmaxStep);
    _t(CKTinitTime);
    _t(CKTomega);
    _t(CKTsrcFact);
    _t(CKTdiagGmin);
    _t(CKTnumSrcSteps);
    _t(CKTnumGminSteps);
    _t(CKTgminFactor);
    _t(CKTnoncon);
    _t(CKTdefaultMosM);
    _t(CKTdefaultMosL);
    _t(CKTdefaultMosW);
    _t(CKTdefaultMosAD);
    _t(CKTdefaultMosAS);
    _t(CKThadNodeset);
    _t(CKTfixLimit);
    _t(CKTnoOpIter);
    _t(CKTisSetup);
#ifdef XSPICE
    _t(CKTadevFlag);
#endif
    _t(CKTtimeListSize);
    _t(CKTtimeIndex);
    _t(CKTsizeIncr);

    _t(CKTtryToCompact);
    _t(CKTbadMos3);
    _t(CKTkeepOpInfo);
    _t(CKTcopyNodesets);
    _t(CKTnodeDamping);
    _t(CKTabsDv);
    _t(CKTrelDv);
    _t(CKTtroubleNode);

#undef _foo
#define _foo(name, type, _size)                                         \
    do {                                                                \
        int __i;                                                        \
        if (fread(&__i, sizeof(int), 1, file) == 1 && __i > 0) {        \
            if (name) {                                                 \
                txfree(name);                                           \
            }                                                           \
            name = (type *) tmalloc((size_t) __i);                      \
            if (fread(name, 1, (size_t) __i, file) != (size_t) __i) {   \
                (void) fprintf(cp_err,                                  \
                        "Unable to read vector " #name "\n");           \
                break;                                                  \
            }                                                           \
        }                                                               \
        else {                                                          \
            fprintf(cp_err, "size for vector " #name " is 0\n");        \
        }                                                               \
        if ((_size) != -1 && __i !=                                     \
                (int) (_size) * (int) sizeof(type)) {                   \
            fprintf(cp_err, "expected %ld, but got %d for "#name"\n",   \
                    (_size)*(long)sizeof(type), __i);                   \
        }                                                               \
    } while(0)


    for (i = 0; i <= ckt->CKTmaxOrder+1; i++)
        _foo(ckt->CKTstates[i], double, ckt->CKTnumStates);

    size = SMPmatSize(ckt->CKTmatrix) + 1;
    _foo(ckt->CKTrhs, double, size);
    _foo(ckt->CKTrhsOld, double, size);
    _foo(ckt->CKTrhsSpare, double, size);
    _foo(ckt->CKTirhs, double, size);
    _foo(ckt->CKTirhsOld, double, size);
    _foo(ckt->CKTirhsSpare, double, size);
//    _foo(ckt->CKTrhsOp, double, size);
//    _foo(ckt->CKTsenRhs, double, size);
//    _foo(ckt->CKTseniRhs, double, size);

//    _foo(ckt->CKTtimePoints, double, -1);
//    _foo(ckt->CKTdeltaList, double, -1);

    _foo(ckt->CKTbreaks, double, ckt->CKTbreakSize);

    {   /* avoid invalid lvalue assignment errors in the macro _foo() */
        TSKtask *lname = NULL;
        _foo(lname, TSKtask, 1);
        ft_curckt->ci_curTask = lname;
    }

    /* To stop the Free */
    ft_curckt->ci_curTask->TSKname = NULL;
    ft_curckt->ci_curTask->jobs = NULL;

    _foo(ft_curckt->ci_curTask->TSKname, char, -1);

    {
        TRANan *lname = NULL;
        _foo(lname, TRANan, -1);
        ft_curckt->ci_curTask->jobs = (JOB *)lname;
    }
    ft_curckt->ci_curTask->jobs->JOBname = NULL;
    _foo(ft_curckt->ci_curTask->jobs->JOBname, char, -1);
    ft_curckt->ci_curTask->jobs->JOBnextJob = NULL;
    ckt->CKTcurJob = ft_curckt->ci_curTask->jobs;
    ((TRANan *)ft_curckt->ci_curTask->jobs)->TRANplot = NULL;

    _foo(ckt->CKTstat, STATistics, 1);
    ckt->CKTstat->STATdevNum = NULL;
    _foo(ckt->CKTstat->STATdevNum, STATdevList, -1);

#ifdef XSPICE
    _foo(ckt->evt, Evt_Ckt_Data_t, 1);
    _foo(ckt->enh, Enh_Ckt_Data_t, 1);
    g_mif_info.breakpoint.current = ckt->enh->breakpoint.current;
    g_mif_info.breakpoint.last = ckt->enh->breakpoint.last;
#endif

    tfree(my_ckt);
    fclose(file);

    /* Finally to resume the plot in some fashion */

    /* a worked out version of this should be enough */
    {
        IFuid *nameList;
        int numNames;
        IFuid timeUid;

        error = CKTnames(ckt, &numNames, &nameList);
        if (error) {
            fprintf(cp_err, "error in CKTnames\n");
            return;
        }
        SPfrontEnd->IFnewUid (ckt, &timeUid, NULL, "time", UID_OTHER, NULL);
        error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                           ckt->CKTcurJob->JOBname,
                                           timeUid, IF_REAL,
                                           numNames, nameList, IF_REAL,
                                           &(((TRANan*)ckt->CKTcurJob)->TRANplot));
        if (error) {
            fprintf(cp_err, "error in CKTnames\n");
            return;
        }
    }
}


void com_snsave(wordlist *wl)
{
    FILE *file;
    int i, size;
    CKTcircuit *ckt;
    TSKtask *task;

    if (!ft_curckt) {
        fprintf(cp_err, "Warning: there is no circuit loaded.\n");
        fprintf(cp_err, "    Command 'snsave' is ignored.\n");
        return;
    } else if (!ft_curckt->ci_ckt) { /* Set noparse? */
        fprintf(cp_err, "Warning: circuit not parsed.\n");
        fprintf(cp_err, "    Command 'snsave' is ignored.\n");
        return;
    }

    /* save the data */

    ckt = ft_curckt->ci_ckt;

#ifdef XSPICE
    if (ckt->CKTadevFlag == 1) {
        fprintf(cp_err, "Warning: snsave not implemented for XSPICE A devices.\n");
        fprintf(cp_err, "    Command 'snsave' will be ingnored!\n");
        return;
    }
#endif

    task = ft_curckt->ci_curTask;

    if (task->jobs->JOBtype != 4) {
        fprintf(cp_err, "Warning: Only saving of tran analysis is implemented\n");
        return;
    }

    file = fopen(wl->wl_word, "wb");

    if (!file) {
        fprintf(cp_err,
                "Error: Couldn't open \"%s\" for writing\n", wl->wl_word);
        return;
    }

#undef _foo
#define _foo(name, type, num)                                           \
    do {                                                                \
        int __i;                                                        \
        if (name) {                                                     \
            __i = (num) * (int)sizeof(type); fwrite(&__i, sizeof(int), 1, file); \
            if ((num))                                                  \
                fwrite(name, sizeof(type), (size_t)(num), file);        \
        } else {                                                        \
            __i = 0;                                                    \
            fprintf(cp_err, #name " is NULL, zero written\n");          \
            fwrite(&__i, sizeof(int), 1, file);                         \
        }                                                               \
    } while(0)


    _foo(ckt, CKTcircuit, 1);

    /* To save list

       double *(CKTstates[8]);
       double *CKTrhs;
       double *CKTrhsOld;
       double *CKTrhsSpare;
       double *CKTirhs;
       double *CKTirhsOld;
       double *CKTirhsSpare;
       double *CKTrhsOp;
       double *CKTsenRhs;
       double *CKTseniRhs;
       double *CKTtimePoints;       list of all accepted timepoints in
       the current transient simulation
       double *CKTdeltaList;        list of all timesteps in the
       current transient simulation

    */


    for (i = 0; i <= ckt->CKTmaxOrder+1; i++)
        _foo(ckt->CKTstates[i], double, ckt->CKTnumStates);


    size = SMPmatSize(ckt->CKTmatrix) + 1;

    _foo(ckt->CKTrhs, double, size);
    _foo(ckt->CKTrhsOld, double, size);
    _foo(ckt->CKTrhsSpare, double, size);
    _foo(ckt->CKTirhs, double, size);
    _foo(ckt->CKTirhsOld, double, size);
    _foo(ckt->CKTirhsSpare, double, size);
//    _foo(ckt->CKTrhsOp, double, size);
//    _foo(ckt->CKTsenRhs, double, size);
//    _foo(ckt->CKTseniRhs, double, size);

//    _foo(ckt->CKTtimePoints, double, ckt->CKTtimeListSize);
//    _foo(ckt->CKTdeltaList, double, ckt->CKTtimeListSize);

    /* need to save the breakpoints, or something */
    _foo(ckt->CKTbreaks, double, ckt->CKTbreakSize);

    /* now save the TSK struct, ft_curckt->ci_curTask*/
    _foo(task, TSKtask, 1);
    _foo(task->TSKname, char, ((int)strlen(task->TSKname)+1));

    /* now save the JOB struct task->jobs */
    /* lol, only allow one job, tough! */
    /* Note that JOB is a base class, need to save actual type!! */
    _foo(task->jobs, TRANan, 1);
    _foo(task->jobs->JOBname, char, ((int)strlen(task->jobs->JOBname)+1));

    /* Finally the stats */
    _foo(ckt->CKTstat, STATistics, 1);
    _foo(ckt->CKTstat->STATdevNum, STATdevList, 1);

#ifdef XSPICE
    /* FIXME struct ckt->evt->data and others are not stored
       thus snsave, snload not compatible with XSPICE code models*/
    _foo(ckt->evt, Evt_Ckt_Data_t, 1);
    _foo(ckt->enh, Enh_Ckt_Data_t, 1);
#endif

    fclose(file);
    fprintf(stdout, "Snapshot saved to %s.\n", wl->wl_word);
}


int
ft_find_analysis(char *name)
{
    int j;
    for (j = 0; j < ft_sim->numAnalyses; j++)
        if (strcmp(ft_sim->analyses[j]->name, name) == 0)
            return j;
    return -1;
}


IFparm *
ft_find_analysis_parm(int which, char *name)
{
    int i;
    for (i = 0; i < ft_sim->analyses[which]->numParms; i++)
        if (!strcmp(ft_sim->analyses[which]->analysisParms[i].keyword, name))
            return &(ft_sim->analyses[which]->analysisParms[i]);
    return NULL;
}
