/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "inpdefs.h"
#include "inpmacs.h"
#include "fteext.h"
#include "inp.h"
#include "cpdefs.h"


static int
dot_noise(char *line, void *ckt, INPtables *tab, card *current,
	  void *task, void *gnode, void *foo)
{
    int which;			/* which analysis we are performing */
    int i;			/* generic loop variable */
    int error;			/* error code temporary */
    char *name;			/* the resistor's name */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    void *node1;		/* the first node's node pointer */
    void *node2;		/* the second node's node pointer */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    char *steptype;		/* ac analysis, type of stepping function */

    int found;
    char *point;

    /* .noise V(OUTPUT,REF) SRC {DEC OCT LIN} NP FSTART FSTOP <PTSPRSUM> */
    which = -1;
    for (i = 0; i < ft_sim->numAnalyses; i++) {
	if (strcmp(ft_sim->analyses[i]->name, "NOISE") == 0) {
	    which = i;
	    break;
	}
    }
    if (which == -1) {
	LITERR("Noise analysis unsupported.\n");
	return (0);
    }
    IFC(newAnalysis, (ckt, which, "Noise Analysis", &foo, task));
    INPgetTok(&line, &name, 1);

    /* Make sure the ".noise" command is followed by V(xxxx).  If it
       is, extract 'xxxx'.  If not, report an error. */

    if (name != NULL) {
	int length;

	length = strlen(name);
	if (((*name == 'V') || (*name == 'v')) && (length == 1)) {

	    INPgetNetTok(&line, &nname1, 0);
	    INPtermInsert(ckt, &nname1, tab, &node1);
	    ptemp.nValue = (IFnode) node1;
	    GCA(INPapName, (ckt, which, foo, "output", &ptemp))

		if (*line != ')') {
		    INPgetNetTok(&line, &nname2, 1);
		    INPtermInsert(ckt, &nname2, tab, &node2);
		    ptemp.nValue = (IFnode) node2;
		} else {
		    ptemp.nValue = (IFnode) gnode;
		}
	    GCA(INPapName, (ckt, which, foo, "outputref", &ptemp))

		INPgetTok(&line, &name, 1);
	    INPinsert(&name, tab);
	    ptemp.uValue = name;
	    GCA(INPapName, (ckt, which, foo, "input", &ptemp))

		INPgetTok(&line, &steptype, 1);
	    ptemp.iValue = 1;
	    error = INPapName(ckt, which, foo, steptype, &ptemp);
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

	    for (found = 0, point = line; (!found) && (*point != '\0');
		 found = ((*point != ' ') && (*(point++) != '\t')));
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
dot_op(char *line, void *ckt, INPtables *tab, card *current,
       void *task, void *gnode, void *foo)
{
    int which;			/* which analysis we are performing */
    int i;			/* generic loop variable */
    int error;			/* error code temporary */

    /* .op */
    which = -1;
    for (i = 0; i < ft_sim->numAnalyses; i++) {
	if (strcmp(ft_sim->analyses[i]->name, "OP") == 0) {
	    which = i;
	    break;
	}
    }
    if (which == -1) {
	LITERR("DC operating point analysis unsupported\n");
	return (0);
    }
    IFC(newAnalysis, (ckt, which, "Operating Point", &foo, task));
    return (0);
}


static int
dot_disto(char *line, void *ckt, INPtables *tab, card *current,
	  void *task, void *gnode, void *foo)
{
    int which;			/* which analysis we are performing */
    int i;			/* generic loop variable */
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    char *steptype;		/* ac analysis, type of stepping function */

    /* .disto {DEC OCT LIN} NP FSTART FSTOP <F2OVERF1> */
    which = -1;
    for (i = 0; i < ft_sim->numAnalyses; i++) {
	if (strcmp(ft_sim->analyses[i]->name, "DISTO") == 0) {
	    which = i;
	    break;
	}
    }
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
dot_ac(char *line, void *ckt, INPtables *tab, card *current,
       void *task, void *gnode, void *foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    int i;			/* generic loop variable */
    char *steptype;		/* ac analysis, type of stepping function */

    /* .ac {DEC OCT LIN} NP FSTART FSTOP */
    which = -1;
    for (i = 0; i < ft_sim->numAnalyses; i++) {
	if (strcmp(ft_sim->analyses[i]->name, "AC") == 0) {
	    which = i;
	    break;
	}
    }
    if (which == -1) {
	LITERR("AC small signal analysis unsupported.\n");
	return (0);
    }
    IFC(newAnalysis, (ckt, which, "AC Analysis", &foo, task));
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
    return (0);
}

static int
dot_pz(char *line, void *ckt, INPtables *tab, card *current,
       void *task, void *gnode, void *foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    int i;			/* generic loop variable */
    char *steptype;		/* ac analysis, type of stepping function */

    /* .pz nodeI nodeG nodeJ nodeK {V I} {POL ZER PZ} */
    which = -1;
    for (i = 0; i < ft_sim->numAnalyses; i++) {
	if (strcmp(ft_sim->analyses[i]->name, "PZ") == 0) {
	    which = i;
	    break;
	}
    }
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
dot_dc(char *line, void *ckt, INPtables *tab, card *current,
       void *task, void *gnode, void *foo)
{
    char *name;			/* the resistor's name */
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    int i;			/* generic loop variable */

    /* .dc SRC1NAME Vstart1 Vstop1 Vinc1 [SRC2NAME Vstart2 */
    /*        Vstop2 Vinc2 */
    which = -1;
    for (i = 0; i < ft_sim->numAnalyses; i++) {
	if (strcmp(ft_sim->analyses[i]->name, "DC") == 0) {
	    which = i;
	    break;
	}
    }
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
dot_tf(char *line, void *ckt, INPtables *tab, card *current,
       void *task, void *gnode, void *foo)
{
    char *name;			/* the resistor's name */
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    int which;			/* which analysis we are performing */
    int i;			/* generic loop variable */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    void *node1;		/* the first node's node pointer */
    void *node2;		/* the second node's node pointer */

    /* .tf v( node1, node2 ) src */
    /* .tf vsrc2             src */
    which = -1;
    for (i = 0; i < ft_sim->numAnalyses; i++) {
	if (strcmp(ft_sim->analyses[i]->name, "TF") == 0) {
	    which = i;
	    break;
	}
    }
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
	ptemp.nValue = (IFnode) node1;
	GCA(INPapName, (ckt, which, foo, "outpos", &ptemp));
	if (*line != ')') {
	    INPgetNetTok(&line, &nname2, 1);
	    INPtermInsert(ckt, &nname2, tab, &node2);
	    ptemp.nValue = (IFnode) node2;
	    GCA(INPapName, (ckt, which, foo, "outneg", &ptemp));
	    ptemp.sValue =
		(char *) MALLOC(sizeof(char) *
				(5 + strlen(nname1) + strlen(nname2)));
	    (void) sprintf(ptemp.sValue, "V(%s,%s)", nname1, nname2);
	    GCA(INPapName, (ckt, which, foo, "outname", &ptemp));
	} else {
	    ptemp.nValue = (IFnode) gnode;
	    GCA(INPapName, (ckt, which, foo, "outneg", &ptemp));
	    ptemp.sValue =
		(char *) MALLOC(sizeof(char) * (4 + strlen(nname1)));
	    (void) sprintf(ptemp.sValue, "V(%s)", nname1);
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
dot_tran(char *line, void *ckt, INPtables *tab, card *current,
	 void *task, void *gnode, void *foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    int i;			/* generic loop variable */
    double dtemp;		/* random double precision temporary */
    char *word;			/* something to stick a word of input into */

    /* .tran Tstep Tstop <Tstart <Tmax> > <UIC> */
    which = -1;
    for (i = 0; i < ft_sim->numAnalyses; i++) {
	if (strcmp(ft_sim->analyses[i]->name, "TRAN") == 0) {
	    which = i;
	    break;
	}
    }
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
    }
    return (0);
}


static int
dot_sens(char *line, void *ckt, INPtables *tab, card *current,
	 void *task, void *gnode, void *foo)
{
    char *name;			/* the resistor's name */
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    int i;			/* generic loop variable */
    char *nname1;		/* the first node's name */
    char *nname2;		/* the second node's name */
    void *node1;		/* the first node's node pointer */
    void *node2;		/* the second node's node pointer */
    char *steptype;		/* ac analysis, type of stepping function */

    which = -1;			/* Bug fix from Glao Dezai */
    for (i = 0; i < ft_sim->numAnalyses; i++) {
	if (strcmp(ft_sim->analyses[i]->name, "SENS") == 0) {
	    which = i;
	    break;
	}
    }
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
	ptemp.nValue = (IFnode) node1;
	GCA(INPapName, (ckt, which, foo, "outpos", &ptemp))

	    if (*line != ')') {
		INPgetNetTok(&line, &nname2, 1);
		INPtermInsert(ckt, &nname2, tab, &node2);
		ptemp.nValue = (IFnode) node2;
		GCA(INPapName, (ckt, which, foo, "outneg", &ptemp));
		ptemp.sValue = (char *)
		    MALLOC(sizeof(char) *
			   (5 + strlen(nname1) + strlen(nname2)));
		(void) sprintf(ptemp.sValue, "V(%s,%s)", nname1, nname2);
		GCA(INPapName, (ckt, which, foo, "outname", &ptemp));
	    } else {
		ptemp.nValue = (IFnode) gnode;
		GCA(INPapName, (ckt, which, foo, "outneg", &ptemp));
		ptemp.sValue =
		    (char *) MALLOC(sizeof(char) * (4 + strlen(nname1)));
		(void) sprintf(ptemp.sValue, "V(%s)", nname1);
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
	GCA(INPapName, (ckt, which, foo, "pct", &ptemp))
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
dot_sens2(char *line, void *ckt, INPtables *tab, card *current,
	  void *task, void *gnode, void *foo)
{
    int error;			/* error code temporary */
    IFvalue ptemp;		/* a value structure to package resistance into */
    IFvalue *parm;		/* a pointer to a value struct for function returns */
    int which;			/* which analysis we are performing */
    int i;			/* generic loop variable */
    char *token;		/* a token from the line */

    /* .sens {AC} {DC} {TRAN} [dev=nnn parm=nnn]* */
    which = -1;
    for (i = 0; i < ft_sim->numAnalyses; i++) {
	if (strcmp(ft_sim->analyses[i]->name, "SENS2") == 0) {
	    which = i;
	    break;
	}
    }
    if (which == -1) {
	LITERR("Sensitivity-2 analysis unsupported\n");
	return (0);
    }
    IFC(newAnalysis, (ckt, which, "Sensitivity-2 Analysis", &foo, task));
    while (*line) {
	/* read the entire line */
	INPgetTok(&line, &token, 1);
	for (i = 0; i < ft_sim->analyses[which]->numParms; i++) {
	    /* find the parameter */
	    if (0 == strcmp(token,
			    ft_sim->analyses[which]->analysisParms[i].keyword)) {
		/* found it, analysis which, parameter i */
		if (ft_sim->analyses[which]->analysisParms[i].dataType & IF_FLAG) {
		    /* one of the keywords! */
		    ptemp.iValue = 1;
		    error =
			(*(ft_sim->setAnalysisParm)) (ckt, foo,
						      ft_sim->analyses[which]->analysisParms[i].id,
						      &ptemp,
						      (IFvalue *) NULL);
		    if (error)
			current->error =
			    INPerrCat(current->error, INPerror(error));
		} else {
		    parm =
			INPgetValue(ckt, &line,
				    ft_sim->analyses[which]->analysisParms[i].dataType, tab);
		    error =
			(*(ft_sim->setAnalysisParm)) (ckt, foo,
						      ft_sim->
						      analyses
						      [which]->analysisParms
						      [i].id, parm,
						      (IFvalue *)
						      NULL);
		    if (error)
			current->error =
			    INPerrCat(current->error, INPerror(error));

		}
		break;
	    }
	}
	if (i == ft_sim->analyses[which]->numParms) {
	    /* didn't find it! */
	    LITERR(" Error: unknown parameter on .sens-ignored \n");
	}
    }
    return (0);
}
#endif

static int
dot_options(char *line, void *ckt, INPtables *tab, card *current,
	  void *task, void *gnode, void *foo)
{
	/* .option - specify program options - rather complicated */
	/* use a subroutine to handle all of them to keep this    */
	/* subroutine managable.                                  */
    	
    	INPdoOpts(ckt,task,current,tab);	
    	return (0);
}


int
INP2dot(void *ckt, INPtables *tab, card *current, void *task, void *gnode)
{

    /* .<something> Many possibilities */
    char *token;		/* a token from the line, tmalloc'ed */
    void *foo = NULL;		/* pointer to analysis */
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
	LITERR(" Warning: .TEMP card obsolete - use .options TEMP and TNOM\n");
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
    }
    else if ((strcmp(token, ".options") == 0)||
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
    LITERR(" unimplemented control card - error \n");
quit:
    tfree(token);
    return rtn;
}
