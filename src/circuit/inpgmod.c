/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include <stdio.h>
#include "inpdefs.h"
#include "ifsim.h"
#include "cpstd.h"
#include "fteext.h"
#include "inp.h"

extern INPmodel *modtab;

char *
INPgetMod(void *ckt, char *name, INPmodel **model, INPtables *tab)
{
    INPmodel *modtmp;
    IFvalue * val;
    register int j;
    char * line;
    char *parm;
    char *err = NULL;
    char *temp;
    int error;

    for (modtmp = modtab;modtmp != (INPmodel *)NULL;modtmp =
            ((modtmp)->INPnextModel)) {
        if (strcmp((modtmp)->INPmodName,name) == 0) {
            /* found the model in question - now instantiate if necessary */
            /* and return an appropriate pointer to it */
            if(modtmp->INPmodType<0) {
                /* illegal device type, so can't handle */
                *model = (INPmodel *)NULL;
                err = (char *)MALLOC((35+strlen(name)) * sizeof(char));
                (void) sprintf(err,
                        "Unknown device type for model %s \n",name);
                return(err);
            }
            if(! ((modtmp)->INPmodUsed )) {
                /* not already defined, so create & give parameters */
                error = (*(ft_sim->newModel))( ckt,(modtmp)->INPmodType, 
                        &((modtmp)->INPmodfast), (modtmp)->INPmodName);
                if(error) return(INPerror(error));
                    /* parameter isolation, identification, binding */
                line = ((modtmp)->INPmodLine)->line;
                INPgetTok(&line,&parm,1);     /* throw away '.model' */
                INPgetTok(&line,&parm,1);     /* throw away 'modname' */
                while(*line != 0) {
                    INPgetTok(&line,&parm,1);
		    if (!*parm)
			continue;
                    for(j=0;j<(*(*(ft_sim->devices)[(modtmp)->INPmodType]).
                            numModelParms); j++) {
                        if (strcmp(parm,((*(ft_sim->devices) [ (modtmp)->
                                INPmodType ]).modelParms[j].keyword)) == 0) {
                            val = INPgetValue(ckt,&line,
                                    ((*(ft_sim->devices)[(modtmp)->
                                    INPmodType ]).modelParms[j].
                                    dataType),tab);
                            error = (*(ft_sim->setModelParm))(ckt, 
                                    ((modtmp)->INPmodfast),
                                    (*(ft_sim->devices)[(modtmp)->INPmodType ]).
                                    modelParms[j].id,val,(IFvalue*)NULL);
                            if(error) return(INPerror(error));
                            break;
                        } 
                    }
                    if (strcmp(parm,"level")==0) {
                        /* just grab the level number and throw away */
                        /* since we already have that info from pass1 */
                        val = INPgetValue(ckt,&line,IF_REAL,tab);
                    } else if(j >= 
                            (*(*(ft_sim->devices)[(modtmp)->INPmodType]).
                                    numModelParms)) {
                        temp = (char *)MALLOC((40+strlen(parm)) * sizeof(char));
                        (void)sprintf(temp,
                            "unrecognized parameter (%s) - ignored\n", parm);
                        err = INPerrCat(err,temp);
                    }
                    FREE(parm);
                }
                (modtmp)->INPmodUsed=1;
                (modtmp)->INPmodLine->error = err;
            }
            *model = modtmp;
            return((char *)NULL);
        }
    }
    /* didn't find model - ERROR  - return model */
    *model = (INPmodel *)NULL;
    err = (char *)MALLOC((60+strlen(name)) * sizeof(char));
    (void) sprintf(err,
            " unable to find definition of model %s - default assumed \n",name);
    return(err);
}

