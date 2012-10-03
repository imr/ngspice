/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2000 AlansFixes
**********/

/*
 * This module replaces the old "writedata" routines in nutmeg.
 * Unlike the writedata routines, the OUT routines are only called by
 * the simulator routines, and only call routines in nutmeg.  The rest
 * of nutmeg doesn't deal with OUT at all.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/plot.h"
#include "ngspice/sim.h"
#include "ngspice/inpdefs.h"        /* for INPtables */
#include "ngspice/ifsim.h"
#include "ngspice/jobdefs.h"
#include "ngspice/iferrmsg.h"
#include "circuits.h"
#include "outitf.h"
#include "variable.h"
#include <fcntl.h>
#include "ngspice/cktdefs.h"
#include "ngspice/inpdefs.h"
#include "breakp2.h"
#include "runcoms.h"
#include "plotting/graf.h"
#include "../misc/misc_time.h"

extern char *spice_analysis_get_name(int index);
extern char *spice_analysis_get_description(int index);


static int beginPlot(JOB *analysisPtr, CKTcircuit *circuitPtr, char *cktName, char *analName,
                     char *refName, int refType, int numNames, char **dataNames, int dataType,
                     bool windowed, runDesc **runp);
static int addDataDesc(runDesc *run, char *name, int type, int ind);
static int addSpecialDesc(runDesc *run, char *name, char *devname, char *param, int depind);
static void fileInit(runDesc *run);
static void fileInit_pass2(runDesc *run);
static void fileStartPoint(FILE *fp, bool bin, int num);
static void fileAddRealValue(FILE *fp, bool bin, double value);
static void fileAddComplexValue(FILE *fp, bool bin, IFcomplex value);
static void fileEndPoint(FILE *fp, bool bin);
static void fileEnd(runDesc *run);
static void plotInit(runDesc *run);
static void plotAddRealValue(dataDesc *desc, double value);
static void plotAddComplexValue(dataDesc *desc, IFcomplex value);
static void plotEnd(runDesc *run);
static bool parseSpecial(char *name, char *dev, char *param, char *ind);
static bool name_eq(char *n1, char *n2);
static bool getSpecial(dataDesc *desc, runDesc *run, IFvalue *val);
static void freeRun(runDesc *run);

/*Output data to spice module saj*/
#ifdef TCL_MODULE
#include "ngspice/tclspice.h"
#endif
/*saj*/


#define DOUBLE_PRECISION    15


static clock_t lastclock, currclock;
static double *rowbuf;
static size_t column, rowbuflen;

static bool shouldstop = FALSE; /* Tell simulator to stop next time it asks. */


/* The two "begin plot" routines share all their internals... */

int
OUTpBeginPlot(CKTcircuit *circuitPtr, JOB *analysisPtr,
              IFuid analName,
              IFuid refName, int refType,
              int numNames, IFuid *dataNames, int dataType, runDesc **plotPtr)
{
    char *name;

#ifdef PARALLEL_ARCH
    if (ARCHme != 0)
        return (OK);
#endif

    if (ft_curckt->ci_ckt == circuitPtr)
        name = ft_curckt->ci_name;
    else
        name = "circuit name";

    return (beginPlot(analysisPtr, circuitPtr, name,
                      analName, refName, refType, numNames,
                      dataNames, dataType, FALSE,
                      plotPtr));
}


int
OUTwBeginPlot(CKTcircuit *circuitPtr, JOB *analysisPtr,
              IFuid analName,
              IFuid refName, int refType,
              int numNames, IFuid *dataNames, int dataType, runDesc **plotPtr)
{

#ifdef PARALLEL_ARCH
    if (ARCHme != 0)
        return (OK);
#endif

    return (beginPlot(analysisPtr, circuitPtr, "circuit name",
                      analName, refName, refType, numNames,
                      dataNames, dataType, TRUE,
                      plotPtr));
}


static int
beginPlot(JOB *analysisPtr, CKTcircuit *circuitPtr, char *cktName, char *analName, char *refName, int refType, int numNames, char **dataNames, int dataType, bool windowed, runDesc **runp)
{
    runDesc *run;
    struct save_info *saves;
    bool *savesused = NULL;
    int numsaves;
    int i, j, depind = 0;
    char namebuf[BSIZE_SP], parambuf[BSIZE_SP], depbuf[BSIZE_SP];
    char *ch, tmpname[BSIZE_SP];
    bool saveall  = TRUE;
    bool savealli = FALSE;
    char *an_name;
    /*to resume a run saj
     *All it does is reassign the file pointer and return (requires *runp to be NULL if this is not needed)
     */
    if (dataType == 666 && numNames == 666) {
        run = *runp;
        run->writeOut = ft_getOutReq(&run->fp, &run->runPlot, &run->binary,
                                     run->type, run->name);

    } else {
        /*end saj*/

        /* Check to see if we want to print informational data. */
        if (cp_getvar("printinfo", CP_BOOL, NULL))
            fprintf(cp_err, "(debug printing enabled)\n");

        *runp = run = alloc(struct runDesc);

        /* First fill in some general information. */
        run->analysis = analysisPtr;
        run->circuit = circuitPtr;
        run->name = copy(cktName);
        run->type = copy(analName);
        run->windowed = windowed;
        run->numData = 0;

        an_name = spice_analysis_get_name(analysisPtr->JOBtype);
        ft_curckt->ci_last_an = an_name;

        /* Now let's see which of these things we need.  First toss in the
         * reference vector.  Then toss in anything that getSaves() tells
         * us to save that we can find in the name list.  Finally unpack
         * the remaining saves into parameters.
         */
        numsaves = ft_getSaves(&saves);
        if (numsaves) {
            savesused = TMALLOC(bool, numsaves);
            saveall = FALSE;
            for (i = 0; i < numsaves; i++) {
                if (saves[i].analysis && !cieq(saves[i].analysis, an_name)) {
                    /* ignore this one this time around */
                    savesused[i] = TRUE;
                    continue;
                }

                /*  Check for ".save all" and new synonym ".save allv"  */

                if (cieq(saves[i].name, "all") || cieq(saves[i].name, "allv")) {
                    saveall = TRUE;
                    savesused[i] = TRUE;
                    saves[i].used = 1;
                    continue;
                }

                /*  And now for the new ".save alli" option  */

                if (cieq(saves[i].name, "alli")) {
                    savealli = TRUE;
                    savesused[i] = TRUE;
                    saves[i].used = 1;
                    continue;
                }
            }
        }

        /* Pass 0. */
        if (refName) {
            addDataDesc(run, refName, refType, -1);
            for (i = 0; i < numsaves; i++)
                if (!savesused[i] && name_eq(saves[i].name, refName)) {
                    savesused[i] = TRUE;
                    saves[i].used = 1;
                }
        } else {
            run->refIndex = -1;
        }


        /* Pass 1. */
        if (numsaves && !saveall) {
            for (i = 0; i < numsaves; i++)
                if (!savesused[i])
                    for (j = 0; j < numNames; j++)
                        if (name_eq(saves[i].name, dataNames[j])) {
                            addDataDesc(run, dataNames[j], dataType, j);
                            savesused[i] = TRUE;
                            saves[i].used = 1;
                            break;
                        }
        } else {
            for (i = 0; i < numNames; i++)
                if (!refName || !name_eq(dataNames[i], refName))
                    /*  Save the node as long as it's an internal device node  */
                    if (!strstr(dataNames[i], "#internal") &&
                        !strstr(dataNames[i], "#source") &&
                        !strstr(dataNames[i], "#drain") &&
                        !strstr(dataNames[i], "#collector") &&
                        !strstr(dataNames[i], "#emitter") &&
                        !strstr(dataNames[i], "#base"))
                    {
                        addDataDesc(run, dataNames[i], dataType, i);
                    }
        }

        /* Pass 1 and a bit.
           This is a new pass which searches for all the internal device
           nodes, and saves the terminal currents instead  */

        if (savealli) {
            depind = 0;
            for (i = 0; i < numNames; i++) {
                if (strstr(dataNames[i], "#internal") ||
                    strstr(dataNames[i], "#source") ||
                    strstr(dataNames[i], "#drain") ||
                    strstr(dataNames[i], "#collector") ||
                    strstr(dataNames[i], "#emitter") ||
                    strstr(dataNames[i], "#base"))
                {
                    tmpname[0] = '@';
                    tmpname[1] = '\0';
                    strncat(tmpname, dataNames[i], BSIZE_SP-1);
                    ch = strchr(tmpname, '#');

                    if (strstr(ch, "#collector")) {
                        strcpy(ch, "[ic]");
                    } else if (strstr(ch, "#base")) {
                        strcpy(ch, "[ib]");
                    } else if (strstr(ch, "#emitter")) {
                        strcpy(ch, "[ie]");
                        if (parseSpecial(tmpname, namebuf, parambuf, depbuf))
                            addSpecialDesc(run, tmpname, namebuf, parambuf, depind);
                        strcpy(ch, "[is]");
                    } else if (strstr(ch, "#drain")) {
                        strcpy(ch, "[id]");
                        if (parseSpecial(tmpname, namebuf, parambuf, depbuf))
                            addSpecialDesc(run, tmpname, namebuf, parambuf, depind);
                        strcpy(ch, "[ig]");
                    } else if (strstr(ch, "#source")) {
                        strcpy(ch, "[is]");
                        if (parseSpecial(tmpname, namebuf, parambuf, depbuf))
                            addSpecialDesc(run, tmpname, namebuf, parambuf, depind);
                        strcpy(ch, "[ib]");
                    } else if (strstr(ch, "#internal") && (tmpname[1] == 'd')) {
                        strcpy(ch, "[id]");
                    } else {
                        fprintf(cp_err,
                                "Debug: could output current for %s\n", tmpname);
                        continue;
                    }
                    if (parseSpecial(tmpname, namebuf, parambuf, depbuf)) {
                        if (*depbuf) {
                            fprintf(stderr,
                                    "Warning : unexpected dependent variable on %s\n", tmpname);
                        } else {
                            addSpecialDesc(run, tmpname, namebuf, parambuf, depind);
                        }
                    }
                }
            }
        }


        /* Pass 2. */
        for (i = 0; i < numsaves; i++) {

            if (savesused[i])
                continue;

            if (!parseSpecial(saves[i].name, namebuf, parambuf, depbuf)) {
                if (saves[i].analysis)
                    fprintf(cp_err, "Warning: can't parse '%s': ignored\n",
                            saves[i].name);
                continue;
            }

            /* Now, if there's a dep variable, do we already have it? */
            if (*depbuf) {
                for (j = 0; j < run->numData; j++)
                    if (name_eq(depbuf, run->data[j].name))
                        break;
                if (j == run->numData) {
                    /* Better add it. */
                    for (j = 0; j < numNames; j++)
                        if (name_eq(depbuf, dataNames[j]))
                            break;
                    if (j == numNames) {
                        fprintf(cp_err,
                                "Warning: can't find '%s': value '%s' ignored\n",
                                depbuf, saves[i].name);
                        continue;
                    }
                    addDataDesc(run, dataNames[j], dataType, j);
                    savesused[i] = TRUE;
                    saves[i].used = 1;
                    depind = j;
                } else {
                    depind = run->data[j].outIndex;
                }
            }

            addSpecialDesc(run, saves[i].name, namebuf, parambuf, depind);
        }

        if (numsaves) {
            for (i = 0; i < numsaves; i++) {
                tfree(saves[i].analysis);
                tfree(saves[i].name);
            }
            tfree(savesused);
        }

        if (numNames &&
            ((run->numData == 1 && run->refIndex != -1) ||
             (run->numData == 0 && run->refIndex == -1)))
        {
            fprintf(cp_err, "Error: no data saved for %s; analysis not run\n",
                    spice_analysis_get_description(analysisPtr->JOBtype));
            return E_NOTFOUND;
        }

        /* Now that we have our own data structures built up, let's see what
         * nutmeg wants us to do.
         */
        run->writeOut = ft_getOutReq(&run->fp, &run->runPlot, &run->binary,
                                     run->type, run->name);

        if (run->writeOut) {
            fileInit(run);
        } else {
            plotInit(run);
            if (refName)
                run->runPlot->pl_ndims = 1;
        }
    }

    /*Start BLT, initilises the blt vectors saj*/
#ifdef TCL_MODULE
    blt_init(run);
#endif

    return (OK);
}


static int
addDataDesc(runDesc *run, char *name, int type, int ind)
{
    dataDesc *data;

    if (!run->numData)
        run->data = TMALLOC(dataDesc, 1);
    else
        run->data = TREALLOC(dataDesc, run->data, run->numData + 1);

    data = &run->data[run->numData];
    /* so freeRun will get nice NULL pointers for the fields we don't set */
    bzero(data, sizeof(dataDesc));

    data->name = copy(name);
    data->type = type;
    data->gtype = GRID_LIN;
    data->regular = TRUE;
    data->outIndex = ind;

    /* It's the reference vector. */
    if (ind == -1)
        run->refIndex = run->numData;

    run->numData++;

    return (OK);
}


static int
addSpecialDesc(runDesc *run, char *name, char *devname, char *param, int depind)
{
    dataDesc *data;
    char *unique;       /* unique char * from back-end */

    if (!run->numData)
        run->data = TMALLOC(dataDesc, 1);
    else
        run->data = TREALLOC(dataDesc, run->data, run->numData + 1);

    data = &run->data[run->numData];
    /* so freeRun will get nice NULL pointers for the fields we don't set */
    bzero(data, sizeof(dataDesc));

    data->name = copy(name);

    unique = copy(devname);

    /* MW. My "special" routine here */
    INPinsertNofree(&unique, ft_curckt->ci_symtab);
    data->specName = unique;

    data->specParamName = copy(param);

    data->specIndex = depind;
    data->specType = -1;
    data->specFast = NULL;
    data->regular = FALSE;

    run->numData++;

    return (OK);
}


int
OUTpData(runDesc *plotPtr, IFvalue *refValue, IFvalue *valuePtr)
{
    runDesc *run = plotPtr;  // FIXME
    int i;

#ifdef PARALLEL_ARCH
    if (ARCHme != 0)
        return (OK);
#endif

    run->pointCount++;

#ifdef TCL_MODULE
    steps_completed = run->pointCount;
#endif

    if (run->writeOut) {

        if (run->pointCount == 1)
            fileInit_pass2(run);

        fileStartPoint(run->fp, run->binary, run->pointCount);

        if (run->refIndex != -1) {
            if (run->isComplex) {
                fileAddComplexValue(run->fp, run->binary, refValue->cValue);

                /*  While we're looking at the reference value, print it to the screen
                    every quarter of a second, to give some feedback without using
                    too much CPU time  */
#ifndef HAS_WINDOWS
                currclock = clock();
                if ((currclock-lastclock) > (0.25*CLOCKS_PER_SEC)) {
                    fprintf(stderr, " Reference value : % 12.5e\r",
                            refValue->cValue.real);
                    lastclock = currclock;
                }
#endif
            } else {

                /*  And the same for a non-complex value  */

                fileAddRealValue(run->fp, run->binary, refValue->rValue);
#ifndef HAS_WINDOWS
                currclock = clock();
                if ((currclock-lastclock) > (0.25*CLOCKS_PER_SEC)) {
                    fprintf(stderr, " Reference value : % 12.5e\r",
                            refValue->rValue);
                    lastclock = currclock;
                }
#endif
            }
        }

        for (i = 0; i < run->numData; i++) {
            /* we've already printed reference vec first */
            if (run->data[i].outIndex == -1)
                continue;

#ifdef TCL_MODULE
            blt_add(i, refValue ? refValue->rValue : NAN);
#endif

            if (run->data[i].regular) {
                if (run->data[i].type == IF_REAL)
                    fileAddRealValue(run->fp, run->binary,
                                     valuePtr->v.vec.rVec [run->data[i].outIndex]);
                else if (run->data[i].type == IF_COMPLEX)
                    fileAddComplexValue(run->fp, run->binary,
                                        valuePtr->v.vec.cVec [run->data[i].outIndex]);
                else
                    fprintf(stderr, "OUTpData: unsupported data type\n");
            } else {
                IFvalue val;
                /* should pre-check instance */
                if (!getSpecial(&run->data[i], run, &val)) {

                    /*  If this is the first data point, print a warning for any unrecognized
                        variables, since this has not already been checked  */

                    if (run->pointCount == 1)
                        fprintf(stderr, "Warning: unrecognized variable - %s\n",
                                run->data[i].name);

                    if (run->isComplex) {
                        val.cValue.real = 0;
                        val.cValue.imag = 0;
                        fileAddComplexValue(run->fp, run->binary, val.cValue);
                    } else {
                        val.rValue = 0;
                        fileAddRealValue(run->fp, run->binary, val.rValue);
                    }

                    continue;
                }

                if (run->data[i].type == IF_REAL)
                    fileAddRealValue(run->fp, run->binary, val.rValue);
                else if (run->data[i].type == IF_COMPLEX)
                    fileAddComplexValue(run->fp, run->binary, val.cValue);
                else
                    fprintf(stderr, "OUTpData: unsupported data type\n");
            }

#ifdef TCL_MODULE
            blt_add(i, valuePtr->v.vec.rVec [run->data[i].outIndex]);
#endif

        }

        fileEndPoint(run->fp, run->binary);

        /*  Check that the write to disk completed successfully, otherwise abort  */

        if (ferror(run->fp)) {
            fprintf(stderr, "Warning: rawfile write error !!\n");
            shouldstop = TRUE;
        }

    } else {

        /*  This is interactive mode. Update the screen with the reference
            variable just the same  */

        currclock = clock();

#ifndef HAS_WINDOWS
        if ((currclock-lastclock) > (0.25*CLOCKS_PER_SEC)) {
            if (run->isComplex) {
                fprintf(stderr, " Reference value : % 12.5e\r",
                        refValue ? refValue->cValue.real : NAN);
            } else {
                fprintf(stderr, " Reference value : % 12.5e\r",
                        refValue ? refValue->rValue : NAN);
            }
            lastclock = currclock;
        }
#endif

        for (i = 0; i < run->numData; i++) {

#ifdef TCL_MODULE
            /*Locks the blt vector to stop access*/
            blt_lockvec(i);
#endif

            if (run->data[i].outIndex == -1) {
                if (run->data[i].type == IF_REAL)
                    plotAddRealValue(&run->data[i], refValue->rValue);
                else if (run->data[i].type == IF_COMPLEX)
                    plotAddComplexValue(&run->data[i], refValue->cValue);
            } else if (run->data[i].regular) {
                if (run->data[i].type == IF_REAL)
                    plotAddRealValue(&run->data[i],
                                     valuePtr->v.vec.rVec[run->data[i].outIndex]);
                else if (run->data[i].type == IF_COMPLEX)
                    plotAddComplexValue(&run->data[i],
                                        valuePtr->v.vec.cVec[run->data[i].outIndex]);
            } else {
                IFvalue val;
                /* should pre-check instance */
                if (!getSpecial(&run->data[i], run, &val))
                    continue;
                if (run->data[i].type == IF_REAL)
                    plotAddRealValue(&run->data[i], val.rValue);
                else if (run->data[i].type == IF_COMPLEX)
                    plotAddComplexValue(&run->data[i], val.cValue);
                else
                    fprintf(stderr, "OUTpData: unsupported data type\n");
            }

#ifdef TCL_MODULE
            /*relinks and unlocks vector*/
            blt_relink(i, (run->data[i]).vec);
#endif

        }

        gr_iplot(run->runPlot);
    }

    if (ft_bpcheck(run->runPlot, run->pointCount) == FALSE)
        shouldstop = TRUE;

#ifdef TCL_MODULE
    Tcl_ExecutePerLoop();
#endif

    return (OK);
}


/* ARGSUSED */ /* until some code gets written */
int
OUTwReference(void *plotPtr, IFvalue *valuePtr, void **refPtr)
{
    NG_IGNORE(refPtr);
    NG_IGNORE(valuePtr);
    NG_IGNORE(plotPtr);

    return (OK);
}


/* ARGSUSED */ /* until some code gets written */
int
OUTwData(runDesc *plotPtr, int dataIndex, IFvalue *valuePtr, void *refPtr)
{
    NG_IGNORE(refPtr);
    NG_IGNORE(valuePtr);
    NG_IGNORE(dataIndex);
    NG_IGNORE(plotPtr);

    return (OK);
}


/* ARGSUSED */ /* until some code gets written */
int
OUTwEnd(runDesc *plotPtr)
{
    NG_IGNORE(plotPtr);

    return (OK);
}


int
OUTendPlot(runDesc *plotPtr)
{
    runDesc *run = plotPtr;  // FIXME

#ifdef PARALLEL_ARCH
    if (ARCHme != 0)
        return (OK);
#endif

    if (run->writeOut) {
        fileEnd(run);
    } else {
        gr_end_iplot();
        plotEnd(run);
    }

    freeRun(run);

    return (OK);
}


/* ARGSUSED */ /* until some code gets written */
int
OUTbeginDomain(runDesc *plotPtr, IFuid refName, int refType, IFvalue *outerRefValue)
{
    NG_IGNORE(outerRefValue);
    NG_IGNORE(refType);
    NG_IGNORE(refName);
    NG_IGNORE(plotPtr);

    return (OK);
}


/* ARGSUSED */ /* until some code gets written */
int
OUTendDomain(runDesc *plotPtr)
{
    NG_IGNORE(plotPtr);

    return (OK);
}


/* ARGSUSED */ /* until some code gets written */
int
OUTattributes(runDesc *plotPtr, IFuid varName, int param, IFvalue *value)
{
    runDesc *run = plotPtr;  // FIXME
    GRIDTYPE type;

    NG_IGNORE(value);

    if (param == OUT_SCALE_LIN)
        type = GRID_LIN;
    else if (param == OUT_SCALE_LOG)
        type = GRID_XLOG;
    else
        return E_UNSUPP;

    if (run->writeOut) {
        if (varName) {
            int i;
            for (i = 0; i < run->numData; i++)
                if (!strcmp(varName, run->data[i].name))
                    run->data[i].gtype = type;
        } else {
            run->data[run->refIndex].gtype = type;
        }
    } else {
        if (varName) {
            struct dvec *d;
            for (d = run->runPlot->pl_dvecs; d; d = d->v_next)
                if (!strcmp(varName, d->v_name))
                    d->v_gridtype = type;
        } else {
            run->runPlot->pl_scale->v_gridtype = type;
        }
    }

    return (OK);
}


/* The file writing routines. */

static void
fileInit(runDesc *run)
{
    char buf[513];
    int i;
    size_t n;

    lastclock = clock();

    /* This is a hack. */
    run->isComplex = FALSE;
    for (i = 0; i < run->numData; i++)
        if (run->data[i].type == IF_COMPLEX)
            run->isComplex = TRUE;

    n = 0;
    sprintf(buf, "Title: %s\n", run->name);
    n += strlen(buf);
    fputs(buf, run->fp);
    sprintf(buf, "Date: %s\n", datestring());
    n += strlen(buf);
    fputs(buf, run->fp);
    sprintf(buf, "Plotname: %s\n", run->type);
    n += strlen(buf);
    fputs(buf, run->fp);
    sprintf(buf, "Flags: %s\n", run->isComplex ? "complex" : "real");
    n += strlen(buf);
    fputs(buf, run->fp);
    sprintf(buf, "No. Variables: %d\n", run->numData);
    n += strlen(buf);
    fputs(buf, run->fp);
    sprintf(buf, "No. Points: ");
    n += strlen(buf);
    fputs(buf, run->fp);

    fflush(run->fp);        /* Gotta do this for LATTICE. */
    if (run->fp == stdout || (run->pointPos = ftell(run->fp)) <= 0)
        run->pointPos = (long) n;
    fprintf(run->fp, "0       \n"); /* Save 8 spaces here. */

    /*fprintf(run->fp, "Command: version %s\n", ft_sim->version);*/
    fprintf(run->fp, "Variables:\n");

    printf("No. of Data Columns : %d  \n", run->numData);
}


static void
fileInit_pass2(runDesc *run)
{
    int i, type;

    for (i = 0; i < run->numData; i++) {

        char *name = run->data[i].name;

        if (substring("#branch", name))
            type = SV_CURRENT;
        else if (cieq(name, "time"))
            type = SV_TIME;
        else if (cieq(name, "frequency"))
            type = SV_FREQUENCY;
        else if (cieq(name, "temp-sweep"))
            type = SV_TEMP;
        else if (cieq(name, "res-sweep"))
            type = SV_RES;
        else if ((*name == '@') && (substring("[g", name)))
            type = SV_ADMITTANCE;
        else if ((*name == '@') && (substring("[c", name)))
            type = SV_CAPACITANCE;
        else if ((*name == '@') && (substring("[i", name)))
            type = SV_CURRENT;
        else if ((*name == '@') && (substring("[q", name)))
            type = SV_CHARGE;
        else
            type = SV_VOLTAGE;

        if (type == SV_CURRENT) {
            char *branch = strstr(name, "#branch");
            if (branch)
                *branch = '\0';
            fprintf(run->fp, "\t%d\ti(%s)\t%s", i, name, ft_typenames(type));
            if (branch)
                *branch = '#';
        } else if (type == SV_VOLTAGE) {
            fprintf(run->fp, "\t%d\tv(%s)\t%s", i, name, ft_typenames(type));
        } else {
            fprintf(run->fp, "\t%d\t%s\t%s", i, name, ft_typenames(type));
        }

        if (run->data[i].gtype == GRID_XLOG)
            fprintf(run->fp, "\tgrid=3");

        fprintf(run->fp, "\n");
    }

    fprintf(run->fp, "%s:\n", run->binary ? "Binary" : "Values");
    fflush(run->fp);

    /*  Allocate Row buffer  */

    if (run->binary) {
        rowbuflen = (size_t) (run->numData);
        if (run->isComplex)
            rowbuflen *= 2;
        rowbuf = TMALLOC(double, rowbuflen);
    } else {
        // fIXME rowbuflen = 0;
        rowbuf = NULL;
    }
}


static void
fileStartPoint(FILE *fp, bool bin, int num)
{
    if (!bin)
        fprintf(fp, "%d\t", num - 1);

    /*  reset buffer pointer to zero  */

    column = 0;
}


static void
fileAddRealValue(FILE *fp, bool bin, double value)
{
    if (bin)
        rowbuf[column++] = value;
    else
        fprintf(fp, "\t%.*e\n", DOUBLE_PRECISION, value);
}


static void
fileAddComplexValue(FILE *fp, bool bin, IFcomplex value)
{
    if (bin) {
        rowbuf[column++] = value.real;
        rowbuf[column++] = value.imag;
    } else {
        fprintf(fp, "\t%.*e,%.*e\n", DOUBLE_PRECISION, value.real,
                DOUBLE_PRECISION, value.imag);
    }
}


/* ARGSUSED */ /* until some code gets written */
static void
fileEndPoint(FILE *fp, bool bin)
{
    /*  write row buffer to file  */
    /* otherwise the data has already been written */

    if (bin)
        fwrite(rowbuf, sizeof(double), rowbuflen, fp);
}


/* Here's the hack...  Run back and fill in the number of points. */

static void
fileEnd(runDesc *run)
{
    if (run->fp != stdout) {
        long place = ftell(run->fp);
        fseek(run->fp, run->pointPos, SEEK_SET);
        fprintf(run->fp, "%d", run->pointCount);
        fprintf(stdout, "\nNo. of Data Rows : %d\n", run->pointCount);
        fseek(run->fp, place, SEEK_SET);
    } else {
        /* Yet another hack-around */
        fprintf(stderr, "@@@ %ld %d\n", run->pointPos, run->pointCount);
    }

    fflush(run->fp);

    if (run->binary) {
        /* deallocate row buffer */
        tfree(rowbuf);
    }
}


/* The plot maintenance routines. */

static void
plotInit(runDesc *run)
{
    struct plot *pl = plot_alloc(run->type);
    char buf[100];
    struct dvec *v;
    dataDesc *dd;
    int i;

    pl->pl_title = copy(run->name);
    pl->pl_name = copy(run->type);
    pl->pl_date = copy(datestring());
    pl->pl_ndims = 0;
    plot_new(pl);
    plot_setcur(pl->pl_typename);
    run->runPlot = pl;

    /* This is a hack. */
    /* if any of them complex, make them all complex */
    run->isComplex = FALSE;
    for (i = 0; i < run->numData; i++)
        if (run->data[i].type == IF_COMPLEX)
            run->isComplex = TRUE;

    for (i = 0; i < run->numData; i++) {
        dd = &run->data[i];
        v = alloc(struct dvec);
        if (isdigit(*dd->name)) {
            (void) sprintf(buf, "V(%s)", dd->name);
            v->v_name = copy(buf);
        } else {
            v->v_name = copy(dd->name);
        }
        if (substring("#branch", v->v_name))
            v->v_type = SV_CURRENT;
        else if (cieq(v->v_name, "time"))
            v->v_type = SV_TIME;
        else if (cieq(v->v_name, "frequency"))
            v->v_type = SV_FREQUENCY;
        else if (cieq(v->v_name, "onoise_spectrum"))
            v->v_type = SV_OUTPUT_N_DENS;
        else if (cieq(v->v_name, "onoise_integrated"))
            v->v_type = SV_OUTPUT_NOISE;
        else if (cieq(v->v_name, "inoise_spectrum"))
            v->v_type = SV_INPUT_N_DENS;
        else if (cieq(v->v_name, "inoise_integrated"))
            v->v_type = SV_INPUT_NOISE;
        else if (cieq(v->v_name, "temp-sweep"))
            v->v_type = SV_TEMP;
        else if (cieq(v->v_name, "res-sweep"))
            v->v_type = SV_RES;
        else if ((*(v->v_name) == '@') && (substring("[g", v->v_name)))
            v->v_type = SV_ADMITTANCE;
        else if ((*(v->v_name) == '@') && (substring("[c", v->v_name)))
            v->v_type = SV_CAPACITANCE;
        else if ((*(v->v_name) == '@') && (substring("[i", v->v_name)))
            v->v_type = SV_CURRENT;
        else if ((*(v->v_name) == '@') && (substring("[q", v->v_name)))
            v->v_type = SV_CHARGE;
        else
            v->v_type = SV_VOLTAGE;
        v->v_length = 0;
        v->v_scale = NULL;
        if (!run->isComplex) {
            v->v_flags = VF_REAL;
            v->v_realdata = NULL;
        } else {
            v->v_flags = VF_COMPLEX;
            v->v_compdata = NULL;
        }

        v->v_flags |= VF_PERMANENT;

        vec_new(v);
        dd->vec = v;
    }
}


static void
plotAddRealValue(dataDesc *desc, double value)
{
    struct dvec *v = desc->vec;

    if (isreal(v)) {
        v->v_realdata = TREALLOC(double, v->v_realdata, v->v_length + 1);
        v->v_realdata[v->v_length] = value;
    } else {
        /* a real parading as a VF_COMPLEX */
        v->v_compdata = TREALLOC(ngcomplex_t, v->v_compdata, v->v_length + 1);
        v->v_compdata[v->v_length].cx_real = value;
        v->v_compdata[v->v_length].cx_imag = 0.0;
    }

    v->v_length++;
    v->v_dims[0] = v->v_length; /* va, must be updated */
}


static void
plotAddComplexValue(dataDesc *desc, IFcomplex value)
{
    struct dvec *v = desc->vec;

    v->v_compdata = TREALLOC(ngcomplex_t, v->v_compdata, v->v_length + 1);
    v->v_compdata[v->v_length].cx_real = value.real;
    v->v_compdata[v->v_length].cx_imag = value.imag;

    v->v_length++;
    v->v_dims[0] = v->v_length; /* va, must be updated */
}


/* ARGSUSED */ /* until some code gets written */
static void
plotEnd(runDesc *run)
{
    fprintf(stderr, "\n");
    fprintf(stdout, "\nNo. of Data Rows : %d\n", run->pointCount);
}


/* ParseSpecial takes something of the form "@name[param,index]" and rips
 * out name, param, andstrchr.
 */

static bool
parseSpecial(char *name, char *dev, char *param, char *ind)
{
    char *s;

    *dev = *param = *ind = '\0';

    if (*name != '@')
        return FALSE;
    name++;

    s = dev;
    while (*name && (*name != '['))
        *s++ = *name++;
    *s = '\0';

    if (!*name)
        return TRUE;
    name++;

    s = param;
    while (*name && (*name != ',') && (*name != ']'))
        *s++ = *name++;
    *s = '\0';

    if (*name == ']')
        return (!name[1] ? TRUE : FALSE);
    else if (!*name)
        return FALSE;
    name++;

    s = ind;
    while (*name && (*name != ']'))
        *s++ = *name++;
    *s = '\0';

    if (*name && !name[1])
        return TRUE;
    else
        return FALSE;
}


/* This routine must match two names with or without a V() around them. */

static bool
name_eq(char *n1, char *n2)
{
    char buf1[BSIZE_SP], buf2[BSIZE_SP], *s;

    if ((s = strchr(n1, '(')) != NULL) {
        strcpy(buf1, s);
        if ((s = strchr(buf1, ')')) == NULL)
            return FALSE;
        *s = '\0';
        n1 = buf1;
    }

    if ((s = strchr(n2, '(')) != NULL) {
        strcpy(buf2, s);
        if ((s = strchr(buf2, ')')) == NULL)
            return FALSE;
        *s = '\0';
        n2 = buf2;
    }

    return (strcmp(n1, n2) ? FALSE : TRUE);
}


static bool
getSpecial(dataDesc *desc, runDesc *run, IFvalue *val)
{
    IFvalue selector;
    struct variable *vv;

    selector.iValue = desc->specIndex;
    if (INPaName(desc->specParamName, val, run->circuit, &desc->specType,
                 desc->specName, &desc->specFast, ft_sim, &desc->type,
                 &selector) == OK) {
        desc->type &= (IF_REAL | IF_COMPLEX);   /* mask out other bits */
        return TRUE;
    }

    if ((vv = if_getstat(run->circuit, &desc->name[1])) != NULL) {
        /* skip @ sign */
        desc->type = IF_REAL;
        if (vv->va_type == CP_REAL)
            val->rValue = vv->va_real;
        else if (vv->va_type == CP_NUM)
            val->rValue = vv->va_num;
        else if (vv->va_type == CP_BOOL)
            val->rValue = (vv->va_bool ? 1.0 : 0.0);
        else
            return FALSE; /* not a real */
        tfree(vv);
        return TRUE;
    }

    return FALSE;
}


static void
freeRun(runDesc *run)
{
    int i;

    for (i = 0; i < run->numData; i++) {
        tfree(run->data[i].name);
        tfree(run->data[i].specParamName);
    }

    tfree(run->data);
    tfree(run->type);
    tfree(run->name);

    tfree(run);
}


int
OUTstopnow(void)
{
    if (ft_intrpt || shouldstop) {
        ft_intrpt = shouldstop = FALSE;
        return (1);
    }

    return (0);
}


/* Print out error messages. */

static struct mesg {
    char *string;
    long flag;
} msgs[] = {
    { "Warning", ERR_WARNING } ,
    { "Fatal error", ERR_FATAL } ,
    { "Panic", ERR_PANIC } ,
    { "Note", ERR_INFO } ,
    { NULL, 0 }
};


void
OUTerror(int flags, char *format, IFuid *names)
{
    struct mesg *m;
    char buf[BSIZE_SP], *s, *bptr;
    int nindex = 0;

    if ((flags == ERR_INFO) && cp_getvar("printinfo", CP_BOOL, NULL))
        return;

    for (m = msgs; m->flag; m++)
        if (flags & m->flag)
            fprintf(cp_err, "%s: ", m->string);

    for (s = format, bptr = buf; *s; s++) {
        if (*s == '%' && (s == format || *(s-1) != '%') && *(s+1) == 's') {
            if (names[nindex])
                strcpy(bptr, names[nindex]);
            else
                strcpy(bptr, "(null)");
            bptr += strlen(bptr);
            s++;
            nindex++;
        } else {
            *bptr++ = *s;
        }
    }

    *bptr = '\0';
    fprintf(cp_err, "%s\n", buf);
    fflush(cp_err);
}
