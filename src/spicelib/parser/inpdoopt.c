/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

    /* INPdoOpts(ckt,option card)
     *  parse the options off of the given option card and add them to
     *  the given circuit 
     */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/cpdefs.h"
#include "ngspice/fteext.h"


void
INPdoOpts(
    CKTcircuit *ckt,
    JOB *anal,
    struct card *optCard,
    INPtables *tab)
{
    char *line;
    char *token;
    char *errmsg;
    IFvalue *val;
    int error;
    int which;

    which = ft_find_analysis("options");

    if(which == -1) {
        optCard->error = INPerrCat(optCard->error,INPmkTemp(
                                       "error:  analysis options table not found\n"));
        return;
    }

    line = optCard->line;

    INPgetTok(&line,&token,1);    /* throw away '.option' */

    while (*line) {

        IFparm *if_parm;

        INPgetTok(&line,&token,1);

        if_parm = ft_find_analysis_parm(which, token);

        if(if_parm && !(if_parm->dataType & IF_UNIMP_MASK)) {
            errmsg = tprintf(" Warning: %s not yet implemented - ignored \n", token);
            optCard->error = INPerrCat(optCard->error,errmsg);
            val = INPgetValue(ckt,&line, if_parm->dataType, tab);
            continue;
        }

        if(if_parm && (if_parm->dataType & IF_SET)) {
            val = INPgetValue(ckt,&line, if_parm->dataType&IF_VARTYPES, tab);
            error = ft_sim->setAnalysisParm (ckt, anal, if_parm->id, val, NULL);
            if(error) {
                errmsg = tprintf("Warning:  can't set option %s\n", token);
                optCard->error = INPerrCat(optCard->error, errmsg);
            }
            continue;
        }

        errmsg = TMALLOC(char, 100);
        (void) strcpy(errmsg," Error: unknown option - ignored\n");
        optCard->error = INPerrCat(optCard->error,errmsg);
        fprintf(stderr, "%s\n", optCard->error);
    }
}
