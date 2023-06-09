/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"
#include "ngspice/cpdefs.h"
#include "ngspice/tskdefs.h"

static int
dot_noise(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
          TSKtask *task, CKTnode *gnode, JOB *foo)
{
    int which;			/* which analysis we are performing */
    int error;			/* error code temporary */
    char *name;			/* the resistor's name */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    CKTnode *node1;		/* the first node's node pointer */
    CKTnode *node2;		/* the second node's node pointer */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    char *steptype;		/* ac analysis, type of stepping function */

    int found;
    char *point;

    /* .noise V(OUTPUT,REF) SRC {DEC OCT LIN} NP FSTART FSTOP <PTSPRSUM> */
    which = ft_find_analysis("NOISE");
    if (which == -1) {
        LITERR("Noise analysis unsupported.\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "Noise Analysis", &foo, task));
    INPgetTok(&line, &name, 1);

    /* Make sure the ".noise" command is followed by V(xxxx).  If it
       is, extract 'xxxx'.  If not, report an error. */

    if (name != NULL) {

        if ((*name == 'V' || *name == 'v') && !name[1]) {

            INPgetNetTok(&line, &nname1, 0);
            INPtermInsert(ckt, &nname1, tab, &node1);
            ptemp.nValue = node1;
            GCA(INPapName, (ckt, which, foo, "output", &ptemp));

            if (*line != ')') {
                INPgetNetTok(&line, &nname2, 1);
                INPtermInsert(ckt, &nname2, tab, &node2);
                ptemp.nValue = node2;
            } else {
                ptemp.nValue = gnode;
            }
            GCA(INPapName, (ckt, which, foo, "outputref", &ptemp));

            tfree(name);
            INPgetTok(&line, &name, 1);
            INPinsert(&name, tab);
            ptemp.uValue = name;
            GCA(INPapName, (ckt, which, foo, "input", &ptemp));

            INPgetTok(&line, &steptype, 1);
            ptemp.iValue = 1;
            error = INPapName(ckt, which, foo, steptype, &ptemp);
            tfree(steptype);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
            parm = INPgetValue(ckt, &line, IF_INTEGER, tab);
            error = INPapName(ckt, which, foo, "numsteps", parm);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
            parm = INPgetValue(ckt, &line, IF_REAL, tab);
            error = INPapName(ckt, which, foo, "start", parm);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
            parm = INPgetValue(ckt, &line, IF_REAL, tab);
            error = INPapName(ckt, which, foo, "stop", parm);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));

            /* now see if "ptspersum" has been specified by the user */

            for (found = 0, point = line; (!found) && (*point != '\0'); found = ((*point != ' ') && (*(point++) != '\t')))
                ;
            if (found) {
                parm = INPgetValue(ckt, &line, IF_INTEGER, tab);
                error = INPapName(ckt, which, foo, "ptspersum", parm);
                if (error)
                    current->error = INPerrCat(current->error, INPerror(error));
            } else {
                ptemp.iValue = 0;
                error = INPapName(ckt, which, foo, "ptspersum", &ptemp);
                if (error)
                    current->error = INPerrCat(current->error, INPerror(error));
            }
        } else
            LITERR("bad syntax "
                   "[.noise v(OUT) SRC {DEC OCT LIN} "
                   "NP FSTART FSTOP <PTSPRSUM>]\n");
    } else {
        LITERR("bad syntax "
               "[.noise v(OUT) SRC {DEC OCT LIN} "
               "NP FSTART FSTOP <PTSPRSUM>]\n");
    }
    return 0;
}


static int
dot_op(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
       TSKtask *task, CKTnode *gnode, JOB *foo)
{
    int which;			/* which analysis we are performing */
    int error;			/* error code temporary */

    NG_IGNORE(line);
    NG_IGNORE(tab);
    NG_IGNORE(gnode);

    /* .op */
    which = ft_find_analysis("OP");
    if (which == -1) {
        LITERR("DC operating point analysis unsupported\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "Operating Point", &foo, task));
    return (0);
}


static int
dot_disto(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
          TSKtask *task, CKTnode *gnode, JOB *foo)
{
    int which;			/* which analysis we are performing */
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    char *steptype;		/* ac analysis, type of stepping function */

    NG_IGNORE(gnode);

    /* .disto {DEC OCT LIN} NP FSTART FSTOP <F2OVERF1> */
    which = ft_find_analysis("DISTO");
    if (which == -1) {
        LITERR("Small signal distortion analysis unsupported.\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "Distortion Analysis", &foo, task));
    INPgetTok(&line, &steptype, 1);	/* get DEC, OCT, or LIN */
    ptemp.iValue = 1;
    GCA(INPapName, (ckt, which, foo, steptype, &ptemp));
    parm = INPgetValue(ckt, &line, IF_INTEGER, tab);	/* number of points */
    GCA(INPapName, (ckt, which, foo, "numsteps", parm));
    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* fstart */
    GCA(INPapName, (ckt, which, foo, "start", parm));
    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* fstop */
    GCA(INPapName, (ckt, which, foo, "stop", parm));
    if (*line) {
        parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* f1phase */
        GCA(INPapName, (ckt, which, foo, "f2overf1", parm));
    }
    return (0);
}


static int
dot_ac(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
       TSKtask *task, CKTnode *gnode, JOB *foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    char *steptype;		/* ac analysis, type of stepping function */
    bool pdef = FALSE;  /* issue a warning if default parameters are used */

    NG_IGNORE(gnode);

    /* .ac {DEC OCT LIN} NP FSTART FSTOP */
    which = ft_find_analysis("AC");
    if (which == -1) {
        LITERR("AC small signal analysis unsupported.\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "AC Analysis", &foo, task));
    INPgetTok(&line, &steptype, 1);	/* get DEC, OCT, or LIN */
    if (!*steptype || (!ciprefix("dec", steptype) && !ciprefix("oct", steptype) && !ciprefix("lin", steptype))) {
        current->error = "Missing DEC, OCT, or LIN\n";
        return (0);
    }
    ptemp.iValue = 1;
    GCA(INPapName, (ckt, which, foo, steptype, &ptemp));
    tfree(steptype);

    parm = INPgetValue(ckt, &line, IF_INTEGER, tab); /* number of points */
    if (parm->iValue == 0)
        pdef = TRUE;
    GCA(INPapName, (ckt, which, foo, "numsteps", parm));

    if(!isdigit(*line))
        pdef = TRUE;
    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* fstart */
    GCA(INPapName, (ckt, which, foo, "start", parm));

    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* fstop */
    if (parm->rValue == 0)
        pdef = TRUE;
    GCA(INPapName, (ckt, which, foo, "stop", parm));

    if (pdef) {
        fprintf(stderr, "Warning, ngspice assumes default parameter(s) for ac simulation\n");
        fprintf(stderr, "    Check your ac or .ac line\n\n");
    }
    return (0);
}

static int
dot_pz(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
       TSKtask *task, CKTnode *gnode, JOB *foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    char *steptype;		/* ac analysis, type of stepping function */

    NG_IGNORE(gnode);

    /* .pz nodeI nodeG nodeJ nodeK {V I} {POL ZER PZ} */
    which = ft_find_analysis("PZ");
    if (which == -1) {
        LITERR("Pole-zero analysis unsupported.\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "Pole-Zero Analysis", &foo, task));
    parm = INPgetValue(ckt, &line, IF_NODE, tab);
    GCA(INPapName, (ckt, which, foo, "nodei", parm));
    parm = INPgetValue(ckt, &line, IF_NODE, tab);
    GCA(INPapName, (ckt, which, foo, "nodeg", parm));
    parm = INPgetValue(ckt, &line, IF_NODE, tab);
    GCA(INPapName, (ckt, which, foo, "nodej", parm));
    parm = INPgetValue(ckt, &line, IF_NODE, tab);
    GCA(INPapName, (ckt, which, foo, "nodek", parm));
    INPgetTok(&line, &steptype, 1);	/* get V or I */
    ptemp.iValue = 1;
    GCA(INPapName, (ckt, which, foo, steptype, &ptemp));
    INPgetTok(&line, &steptype, 1);	/* get POL, ZER, or PZ */
    ptemp.iValue = 1;
    GCA(INPapName, (ckt, which, foo, steptype, &ptemp));
    return (0);
}


static int
dot_dc(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
       TSKtask *task, CKTnode *gnode, JOB *foo)
{
    char *name;			/* the resistor's name */
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */

    NG_IGNORE(gnode);

    /* .dc SRC1NAME Vstart1 Vstop1 Vinc1 [SRC2NAME Vstart2 */
    /*        Vstop2 Vinc2 */
    which = ft_find_analysis("DC");
    if (which == -1) {
        LITERR("DC transfer curve analysis unsupported\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "DC transfer characteristic", &foo, task));
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);
    ptemp.uValue = name;
    GCA(INPapName, (ckt, which, foo, "name1", &ptemp));
    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* vstart1 */
    GCA(INPapName, (ckt, which, foo, "start1", parm));
    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* vstop1 */
    GCA(INPapName, (ckt, which, foo, "stop1", parm));
    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* vinc1 */
    GCA(INPapName, (ckt, which, foo, "step1", parm));
    if (*line) {
        INPgetTok(&line, &name, 1);
        INPinsert(&name, tab);
        ptemp.uValue = name;
        GCA(INPapName, (ckt, which, foo, "name2", &ptemp));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* vstart2 */
        GCA(INPapName, (ckt, which, foo, "start2", parm));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* vstop2 */
        GCA(INPapName, (ckt, which, foo, "stop2", parm));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* vinc2 */
        GCA(INPapName, (ckt, which, foo, "step2", parm));
    }
    return 0;
}


static int
dot_tf(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
       TSKtask *task, CKTnode *gnode, JOB *foo)
{
    char *name;			/* the resistor's name */
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    int which;			/* which analysis we are performing */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    CKTnode *node1;		/* the first node's node pointer */
    CKTnode *node2;		/* the second node's node pointer */

    /* .tf v( node1, node2 ) src */
    /* .tf vsrc2             src */
    which = ft_find_analysis("TF");
    if (which == -1) {
        LITERR("Transfer Function analysis unsupported.\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "Transfer Function", &foo, task));
    INPgetTok(&line, &name, 0);
    /* name is now either V or I or a serious error */
    if (*name == 'v' && strlen(name) == 1) {
        if (*line != '(' ) {
            /* error, bad input format */
        }
        INPgetNetTok(&line, &nname1, 0);
        INPtermInsert(ckt, &nname1, tab, &node1);
        ptemp.nValue = node1;
        GCA(INPapName, (ckt, which, foo, "outpos", &ptemp));
        if (*line != ')') {
            INPgetNetTok(&line, &nname2, 1);
            INPtermInsert(ckt, &nname2, tab, &node2);
            ptemp.nValue = node2;
            GCA(INPapName, (ckt, which, foo, "outneg", &ptemp));
            ptemp.sValue = tprintf("V(%s,%s)", nname1, nname2);
            GCA(INPapName, (ckt, which, foo, "outname", &ptemp));
        } else {
            ptemp.nValue = gnode;
            GCA(INPapName, (ckt, which, foo, "outneg", &ptemp));
            ptemp.sValue = tprintf("V(%s)", nname1);
            GCA(INPapName, (ckt, which, foo, "outname", &ptemp));
        }
    } else if (*name == 'i' && strlen(name) == 1) {
        INPgetTok(&line, &name, 1);
        INPinsert(&name, tab);
        ptemp.uValue = name;
        GCA(INPapName, (ckt, which, foo, "outsrc", &ptemp));
    } else {
        LITERR("Syntax error: voltage or current expected.\n");
        return 0;
    }
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);
    ptemp.uValue = name;
    GCA(INPapName, (ckt, which, foo, "insrc", &ptemp));
    return (0);
}


static int
dot_tran(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
         TSKtask *task, CKTnode *gnode, JOB *foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    double dtemp;		/* random double precision temporary */
    char *word;			/* something to stick a word of input into */

    NG_IGNORE(gnode);

    /* .tran Tstep Tstop <Tstart <Tmax> > <UIC> */
    which = ft_find_analysis("TRAN");
    if (which == -1) {
        LITERR("Transient analysis unsupported.\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "Transient Analysis", &foo, task));
    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* Tstep */
    GCA(INPapName, (ckt, which, foo, "tstep", parm));
    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* Tstop */
    GCA(INPapName, (ckt, which, foo, "tstop", parm));
    if (*line) {
        dtemp = INPevaluate(&line, &error, 1);	/* tstart? */
        if (error == 0) {
            ptemp.rValue = dtemp;
            GCA(INPapName, (ckt, which, foo, "tstart", &ptemp));
            dtemp = INPevaluate(&line, &error, 1);	/* tmax? */
            if (error == 0) {
                ptemp.rValue = dtemp;
                GCA(INPapName, (ckt, which, foo, "tmax", &ptemp));
            }
        }
    }
    if (*line) {
        INPgetTok(&line, &word, 1);	/* uic? */
        if (strcmp(word, "uic") == 0) {
            ptemp.iValue = 1;
            GCA(INPapName, (ckt, which, foo, "uic", &ptemp));
        } else {
            LITERR(" Error: unknown parameter on .tran - ignored\n");
        }
        tfree(word);
    }
    return (0);
}


static int
dot_sens(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
         TSKtask *task, CKTnode *gnode, JOB *foo)
{
    char *name;			/* the resistor's name */
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    CKTnode *node1;		/* the first node's node pointer */
    CKTnode *node2;		/* the second node's node pointer */
    char *steptype;		/* ac analysis, type of stepping function */

    which = ft_find_analysis("SENS");
    if (which == -1) {
        LITERR("Sensitivity unsupported.\n");
        return (0);
    }

    IFC(newAnalysis, (ckt, which, "Sensitivity Analysis", &foo, task));

    /* Format is:
     *      .sens <output>
     *      + [ac [dec|lin|oct] <pts> <low freq> <high freq> | dc ]
     */
    /* Get the output voltage or current */
    INPgetTok(&line, &name, 0);
    /* name is now either V or I or a serious error */
    if (*name == 'v' && strlen(name) == 1) {
        if (*line != '(') {
            LITERR("Syntax error: '(' expected after 'v'\n");
            return 0;
        }
        INPgetNetTok(&line, &nname1, 0);
        INPtermInsert(ckt, &nname1, tab, &node1);
        ptemp.nValue = node1;
        GCA(INPapName, (ckt, which, foo, "outpos", &ptemp));

        if (*line != ')') {
            INPgetNetTok(&line, &nname2, 1);
            INPtermInsert(ckt, &nname2, tab, &node2);
            ptemp.nValue = node2;
            GCA(INPapName, (ckt, which, foo, "outneg", &ptemp));
            ptemp.sValue = tprintf("V(%s,%s)", nname1, nname2);
            GCA(INPapName, (ckt, which, foo, "outname", &ptemp));
        } else {
            ptemp.nValue = gnode;
            GCA(INPapName, (ckt, which, foo, "outneg", &ptemp));
            ptemp.sValue = tprintf("V(%s)", nname1);
            GCA(INPapName, (ckt, which, foo, "outname", &ptemp));
        }
    } else if (*name == 'i' && strlen(name) == 1) {
        INPgetTok(&line, &name, 1);
        INPinsert(&name, tab);
        ptemp.uValue = name;
        GCA(INPapName, (ckt, which, foo, "outsrc", &ptemp));
    } else {
        LITERR("Syntax error: voltage or current expected.\n");
        return 0;
    }

    INPgetTok(&line, &name, 1);
    if (name && !strcmp(name, "pct")) {
        ptemp.iValue = 1;
        GCA(INPapName, (ckt, which, foo, "pct", &ptemp));
        INPgetTok(&line, &name, 1);
    }
    if (name && !strcmp(name, "ac")) {
        INPgetTok(&line, &steptype, 1);	/* get DEC, OCT, or LIN */
        ptemp.iValue = 1;
        GCA(INPapName, (ckt, which, foo, steptype, &ptemp));
        parm = INPgetValue(ckt, &line, IF_INTEGER, tab); /* number of points */
        GCA(INPapName, (ckt, which, foo, "numsteps", parm));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* fstart */
        GCA(INPapName, (ckt, which, foo, "start", parm));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* fstop */
        GCA(INPapName, (ckt, which, foo, "stop", parm));
        return (0);
    } else if (name && *name && strcmp(name, "dc")) {
        /* Bad flag */
        LITERR("Syntax error: 'ac' or 'dc' expected.\n");
        return 0;
    }
    return (0);
}


#ifdef WANT_SENSE2
static int
dot_sens2(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
          TSKtask *task, CKTnode *gnode, JOB *foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    char *token;		/* a token from the line */

    NG_IGNORE(gnode);

    /* .sens {AC} {DC} {TRAN} [dev=nnn parm=nnn]* */
    which = ft_find_analysis("SENS2");
    if (which == -1) {
        LITERR("Sensitivity-2 analysis unsupported\n");
        return (0);
    }

    IFC(newAnalysis, (ckt, which, "Sensitivity-2 Analysis", &foo, task));

    while (*line) {

        IFparm *if_parm;

        /* read the entire line */
        INPgetTok(&line, &token, 1);

        if_parm = ft_find_analysis_parm(which, token);

        if (!if_parm) {
            /* didn't find it! */
            LITERR(" Error: unknown parameter on .sens-ignored \n");
            continue;
        }

        /* found it, analysis which, parameter i */
        if (if_parm->dataType & IF_FLAG) {

            /* one of the keywords! */
            ptemp.iValue = 1;
            error = ft_sim->setAnalysisParm (ckt, foo,
                                             if_parm->id,
                                             &ptemp,
                                             NULL);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));

        } else {

            parm = INPgetValue(ckt, &line, if_parm->dataType, tab);
            error = ft_sim->setAnalysisParm (ckt, foo,
                                             if_parm->id,
                                             parm,
                                             NULL);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
        }
    }

    return (0);
}
#endif

#ifdef WITH_PSS
/*SP: Steady State Analyis */
static int
dot_pss(char *line, void *ckt, INPtables *tab, struct card *current,
        void *task, void *gnode, JOB *foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    char *nname;		/* the oscNode name */
    CKTnode *nnode;		/* the oscNode node */
    int which;			/* which analysis we are performing */
    char *word;			/* something to stick a word of input into */

    NG_IGNORE(gnode);

    /* .pss Fguess StabTime OscNode <UIC>*/
    which = ft_find_analysis("PSS");
    if (which == -1) {
        LITERR("Periodic steady state analysis unsupported.\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "Periodic Steady State Analysis", &foo, task));

    parm = INPgetValue(ckt, &line, IF_REAL, tab);		/* Fguess */
    GCA(INPapName, (ckt, which, foo, "fguess", parm));

    parm = INPgetValue(ckt, &line, IF_REAL, tab);		/* StabTime */
    GCA(INPapName, (ckt, which, foo, "stabtime", parm));

    INPgetNetTok(&line, &nname, 0);
    INPtermInsert(ckt, &nname, tab, &nnode);
    ptemp.nValue = nnode;
    GCA(INPapName, (ckt, which, foo, "oscnode", &ptemp));	/* OscNode given as string */

    parm = INPgetValue(ckt, &line, IF_INTEGER, tab);		/* PSS points */
    GCA(INPapName, (ckt, which, foo, "points", parm));

    parm = INPgetValue(ckt, &line, IF_INTEGER, tab);		/* PSS harmonics */
    GCA(INPapName, (ckt, which, foo, "harmonics", parm));

    parm = INPgetValue(ckt, &line, IF_INTEGER, tab);		/* SC iterations */
    GCA(INPapName, (ckt, which, foo, "sc_iter", parm));

    parm = INPgetValue(ckt, &line, IF_REAL, tab);		/* Steady coefficient */
    GCA(INPapName, (ckt, which, foo, "steady_coeff", parm));

    if (*line) {
        INPgetTok(&line, &word, 1);	/* uic? */
        if (strcmp(word, "uic") == 0) {
            ptemp.iValue = 1;
            GCA(INPapName, (ckt, which, foo, "uic", &ptemp));
        } else {
            fprintf(stderr,"Error: unknown parameter %s on .pss - ignored\n", word);
        }
    }
    return (0);
}
/* SP */
#endif


#ifdef RFSPICE
/* S-Parameter Analyis */
static int
dot_sp(char* line, void* ckt, INPtables* tab, struct card* current,
    void* task, void* gnode, JOB* foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue* parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    char* steptype;		/* ac analysis, type of stepping function */

    NG_IGNORE(gnode);

    /* .ac {DEC OCT LIN} NP FSTART FSTOP */
    which = ft_find_analysis("SP");
    if (which == -1) {
        LITERR("S-Params analysis unsupported.\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "SP Analysis", &foo, task));
    INPgetTok(&line, &steptype, 1);	/* get DEC, OCT, or LIN */
    ptemp.iValue = 1;
    GCA(INPapName, (ckt, which, foo, steptype, &ptemp));
    tfree(steptype);
    parm = INPgetValue(ckt, &line, IF_INTEGER, tab); /* number of points */
    GCA(INPapName, (ckt, which, foo, "numsteps", parm));
    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* fstart */
    GCA(INPapName, (ckt, which, foo, "start", parm));
    parm = INPgetValue(ckt, &line, IF_REAL, tab);	/* fstop */
    GCA(INPapName, (ckt, which, foo, "stop", parm));
    parm = INPgetValue(ckt, &line, IF_INTEGER, tab);	/* fstop */
    GCA(INPapName, (ckt, which, foo, "donoise", parm));
    return (0);
}

#ifdef WITH_HB
/*SP: Steady State Analyis */
static int
dot_hb(char* line, void* ckt, INPtables* tab, struct card* current,
    void* task, void* gnode, JOB* foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue* parm;		/* a pointer to a value struct for function returns */
    char* nname;		/* the oscNode name */
    CKTnode* nnode;		/* the oscNode node */
    int which;			/* which analysis we are performing */
    char* word;			/* something to stick a word of input into */

    NG_IGNORE(gnode);

    /* .pss Fguess StabTime OscNode <UIC>*/
    which = ft_find_analysis("PSS");
    if (which == -1) {
        LITERR("Periodic steady state analysis unsupported.\n");
        return (0);
    }
    IFC(newAnalysis, (ckt, which, "Harmonic Balance State Analysis", &foo, task));

    parm = INPgetValue(ckt, &line, IF_REALVEC, tab);		/* Fguess */
    GCA(INPapName, (ckt, which, foo, "freq", parm));

    parm = INPgetValue(ckt, &line, IF_INTVEC, tab);		/* StabTime */
    GCA(INPapName, (ckt, which, foo, "harmonics", parm));

    INPgetNetTok(&line, &nname, 0);
    INPtermInsert(ckt, &nname, tab, &nnode);
    ptemp.nValue = nnode;
    GCA(INPapName, (ckt, which, foo, "oscnode", &ptemp));	/* OscNode given as string */

    parm = INPgetValue(ckt, &line, IF_INTEGER, tab);		/* PSS points */
    GCA(INPapName, (ckt, which, foo, "points", parm));

    parm = INPgetValue(ckt, &line, IF_INTEGER, tab);		/* PSS harmonics */
    GCA(INPapName, (ckt, which, foo, "harmonics", parm));

    parm = INPgetValue(ckt, &line, IF_INTEGER, tab);		/* SC iterations */
    GCA(INPapName, (ckt, which, foo, "sc_iter", parm));

    parm = INPgetValue(ckt, &line, IF_REAL, tab);		/* Steady coefficient */
    GCA(INPapName, (ckt, which, foo, "steady_coeff", parm));

    if (*line) {
        INPgetTok(&line, &word, 1);	/* uic? */
        if (strcmp(word, "uic") == 0) {
            ptemp.iValue = 1;
            GCA(INPapName, (ckt, which, foo, "uic", &ptemp));
        }
        else {
            fprintf(stderr, "Error: unknown parameter %s on .pss - ignored\n", word);
        }
    }
    return (0);
}
#endif

#endif


static int
dot_options(char *line, CKTcircuit *ckt, INPtables *tab, struct card *current,
            TSKtask *task, CKTnode *gnode, JOB *foo)
{
    NG_IGNORE(line);
    NG_IGNORE(gnode);
    NG_IGNORE(foo);

    /* .option - specify program options - rather complicated */
    /* use a subroutine to handle all of them to keep this    */
    /* subroutine managable.                                  */

    INPdoOpts(ckt, &(task->taskOptions), current, tab);
    return (0);
}


int
INP2dot(CKTcircuit *ckt, INPtables *tab, struct card *current, TSKtask *task, CKTnode *gnode)
{

    /* .<something> Many possibilities */
    char *token;		/* a token from the line, tmalloc'ed */
    JOB *foo = NULL;		/* pointer to analysis */
    /* the part of the current line left to parse */
    char *line = current->line;
    int rtn = 0;

    INPgetTok(&line, &token, 1);
    if (strcmp(token, ".model") == 0) {
        /* don't have to do anything, since models were all done in
         * pass 1 */
        goto quit;
    } else if ((strcmp(token, ".width") == 0) ||
               strcmp(token, ".print") == 0 || strcmp(token, ".plot") == 0) {
        /* obsolete - ignore */
        LITERR(" Warning: obsolete control card - ignored \n");
        goto quit;
    } else if ((strcmp(token, ".temp") == 0)) {
        /* .temp temp1 temp2 temp3 temp4 ..... */
        /* not yet implemented - warn & ignore */
        /*
        LITERR(" Warning: .TEMP card obsolete - use .options TEMP and TNOM\n");
        */
        goto quit;
    } else if ((strcmp(token, ".op") == 0)) {
        rtn = dot_op(line, ckt, tab, current, task, gnode, foo);
        goto quit;
    } else if ((strcmp(token, ".nodeset") == 0)) {
        goto quit;
    } else if ((strcmp(token, ".disto") == 0)) {
        rtn = dot_disto(line, ckt, tab, current, task, gnode, foo);
        goto quit;
    } else if ((strcmp(token, ".noise") == 0)) {
        rtn = dot_noise(line, ckt, tab, current, task, gnode, foo);
        goto quit;
    } else if ((strcmp(token, ".four") == 0)
               || (strcmp(token, ".fourier") == 0)) {
        /* .four */
        /* not implemented - warn & ignore */
        LITERR("Use fourier command to obtain fourier analysis\n");
        goto quit;
    } else if ((strcmp(token, ".ic") == 0)) {
        goto quit;
    } else if ((strcmp(token, ".ac") == 0)) {
        rtn = dot_ac(line, ckt, tab, current, task, gnode, foo);
        goto quit;
    } else if ((strcmp(token, ".pz") == 0)) {
        rtn = dot_pz(line, ckt, tab, current, task, gnode, foo);
        goto quit;
    } else if ((strcmp(token, ".dc") == 0)) {
        rtn = dot_dc(line, ckt, tab, current, task, gnode, foo);
        goto quit;
    } else if ((strcmp(token, ".tf") == 0)) {
        rtn = dot_tf(line, ckt, tab, current, task, gnode, foo);
        goto quit;
    } else if ((strcmp(token, ".tran") == 0)) {
        rtn = dot_tran(line, ckt, tab, current, task, gnode, foo);
        goto quit;
#ifdef WITH_PSS
        /* SP: Steady State Analysis */
    } else if ((strcmp(token, ".pss") == 0)) {
        rtn = dot_pss(line, ckt, tab, current, task, gnode, foo);
        goto quit;
        /* SP */
#endif
#ifdef RFSPICE
    }
    else if ((strcmp(token, ".sp") == 0)) {
        rtn = dot_sp(line, ckt, tab, current, task, gnode, foo);
        goto quit;
        /* SP */
#ifdef WITH_HB
    }
    else if ((strcmp(token, ".hb") == 0)) {
        rtn = dot_hb(line, ckt, tab, current, task, gnode, foo);
        goto quit;
        /* SP */
#endif
#endif
    } else if ((strcmp(token, ".subckt") == 0) ||
               (strcmp(token, ".ends") == 0)) {
        /* not yet implemented - warn & ignore */
        LITERR(" Warning: Subcircuits not yet implemented - ignored \n");
        goto quit;
    } else if ((strcmp(token, ".end") == 0)) {
        /* .end - end of input */
        /* not allowed to pay attention to additional input - return */
        rtn = 1;
        goto quit;
    } else if (strcmp(token, ".sens") == 0) {
        rtn = dot_sens(line, ckt, tab, current, task, gnode, foo);
        goto quit;
    }
#ifdef WANT_SENSE2
    else if ((strcmp(token, ".sens2") == 0)) {
        rtn = dot_sens2(line, ckt, tab, current, task, gnode, foo);
        goto quit;
    }
#endif
    else if ((strcmp(token, ".probe") == 0)) {
        /* Maybe generate a "probe" format file in the future. */
        goto quit;
    } else if ((strcmp(token, ".options") == 0)||
               (strcmp(token,".option")==0) ||
               (strcmp(token,".opt")==0)) {
        rtn = dot_options(line, ckt, tab, current, task, gnode, foo);
        goto quit;
    }
    /* Added by H.Tanaka to find .global option */
    else if (strcmp(token, ".global") == 0) {
        rtn = 0;
        LITERR(" Warning: .global not yet implemented - ignored \n");
        goto quit;
    }
    /* ignore .meas statements -- these will be handled after analysis */
    /* also ignore .param statements */
    /* ignore .prot, .unprot */
    else if (strcmp(token, ".meas") == 0 || ciprefix(".para", token) || strcmp(token, ".measure") == 0 ||
             strcmp(token, ".prot") == 0 || strcmp(token, ".unprot") == 0) {
        rtn = 0;
        goto quit;
    }
    LITERR(" unimplemented control card - error \n");
quit:
    tfree(token);
    return rtn;
}
