/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

    /* INPdoOpts(ckt,option card)
     *  parse the options off of the given option card and add them to
     *  the given circuit 
     */

#include "ngspice.h"
#include <stdio.h>
#include "inpdefs.h"
#include "ifsim.h"
#include "cpdefs.h"
#include "fteext.h"


void
INPdoOpts(ckt,anal,optCard,tab)
    void *ckt;
    void *anal;
    card *optCard;
    INPtables *tab;
{
    char *line;
    char *token;
    char *errmsg;
    IFvalue *val;
    int error;
    int i;
    int which;
    IFanalysis *prm = NULL;

    which = -1;
    i=0;
    for(i=0;i<ft_sim->numAnalyses;i++) {
        prm = ft_sim->analyses[i];
        if(strcmp(prm->name,"options")==0) {
            which = i;
            break;
        }
        i++;
    }
    if(which == -1) {
        optCard->error = INPerrCat(optCard->error,INPmkTemp(
                "error:  analysis options table not found\n"));
        return;
    }
    line = optCard->line;
    INPgetTok(&line,&token,1);    /* throw away '.option' */
    while (*line) {
        INPgetTok(&line,&token,1);
        for(i=0;i<prm->numParms;i++) {
            if(strcmp(token,prm->analysisParms[i].keyword) == 0) {
                if(!(prm->analysisParms[i].dataType & IF_UNIMP_MASK)) {
                    errmsg = (char *)MALLOC((45+strlen(token)) * sizeof(char));
                    (void) sprintf(errmsg,
                        " Warning: %s not yet implemented - ignored \n",token);
                    optCard->error = INPerrCat(optCard->error,errmsg);
                    val = INPgetValue(ckt,&line,
                            prm->analysisParms[i].dataType, tab);
                    break;
                }
                if(prm->analysisParms[i].dataType & IF_SET) {
                    val = INPgetValue(ckt,&line,
                            prm->analysisParms[i].dataType&IF_VARTYPES, tab);
                    error = (*(ft_sim->setAnalysisParm))(ckt,anal,
                            prm->analysisParms[i].id,val,(IFvalue*)NULL);
                    if(error) {
                        errmsg =(char *)MALLOC((35+strlen(token))*sizeof(char));
                        (void) sprintf(errmsg,
                                "Warning:  can't set option %s\n", token);
                        optCard->error = INPerrCat(optCard->error, errmsg);
                    }
                    break;
                }
            }
        }
        if(i == prm->numParms) {
            errmsg = (char *)MALLOC(100 * sizeof(char));
            (void) strcpy(errmsg," Error: unknown option - ignored\n");
            optCard->error = INPerrCat(optCard->error,errmsg);
            fprintf(stderr, "%s\n", optCard->error);
        }
    }
}
