/*============================================================================
FILE

MEMBER OF process XSPICE

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

 *
 * Copyright (c) 1985 Thomas L. Quarles
 *
 * NOTE:  Portions of this code are Copyright Thomas L. Quarles and University of
 *        California at Berkeley.  Other portions are modified and added by
 *        the Georgia Tech Research Institute.
 *

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the routine that allocates a new model structure and
    parses the .model card parameters.

INTERFACES

    MIFgetMod()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"

#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/cpstd.h"
#include "ngspice/fteext.h"

#include "ngspice/mifproto.h"
#include "ngspice/mifdefs.h"
#include "ngspice/mifcmdat.h"

#include "ngspice/suffix.h"

/*  This is the table of all models known to the program.  
    It is now defined in inpmkmod.c.      */
extern INPmodel *modtab;

/*
MIFgetMod

This function is a modified version of SPICE 3C1 INPgetMod().
MIFgetMod looks in the table of model information created on the
first pass of the parser to find the text of the .model card.  It
then checks to see if the .model card has already been processed
by a previous element card reference.  If so, it returns a
pointer to the previously created model structure.  If not, it
allocates a new model structure and processes the parameters on
the .model card.  Parameter values for parameters not found on
the .model card are not filled in by this function.  They are
defaulted later by MIFsetup().  The function returns NULL when
successful, and an error string on failure.
*/

char *MIFgetMod( 
    CKTcircuit *ckt,    /* The circuit structure */
    char      *name,    /* The name of the model to look for */
    INPmodel  **model,  /* The model found/created */
    INPtables *tab      /* Table of model info from first pass */
    )
{
    INPmodel *modtmp;
    IFvalue * val;
    register int j;
    char * line;
    char *parm;
    char *err = NULL;
    char *temp;
    int error;

    int               i;

    char              *err1;
    char              *err2;

    MIFmodel          *mdfast;
    /* Mif_Param_Info_t  *param_info;*/


    /* ===========  First locate the named model in the modtab list ================= */

#ifdef TRACE
    /* SDB debug statement */
    printf("In MIFgetMod, looking for model name = %s . . .\n", name);
#endif

    /* maschmann : remove : from name
     *    char *pos;
     * if((pos=strstr(name,":"))!=NULL) *pos=0;   
     */

    /*------------------------------------
   for (i = &modtab; *i != NULL; i = &((*i)->INPnextModel)) {
        if (strcmp((*i)->INPmodName, token) == 0) {
            return (OK);
        }
    }
    --------------------------*/

    /* loop through modtable looking for this model (*name) */
    for (modtmp = modtab; modtmp != NULL; modtmp = modtmp->INPnextModel) {

#ifdef TRACE
      /* SDB debug statement */
      printf("In MIFgetMod, checking model against stored model = %s . . .\n", modtmp->INPmodName);
#endif

        if (strcmp(modtmp->INPmodName, name) == 0) {

#ifdef TRACE
	/* SDB debug statement */
	printf("In MIFgetMod, found model!!!\n");
#endif

	/* ========= found the model in question - now instantiate if necessary ========== */
	/* ==============    and return an appropriate pointer to it ===================== */

            /* make sure the type is valid before proceeding */
	if(modtmp->INPmodType < 0) {
                /* illegal device type, so can't handle */
                *model = NULL;

		/* fixed by SDB -- magic number is 39, not 35.  
		 * Also needed parens to correctly compute # of bytes to malloc
		 */
                err = TMALLOC(char, 39 + strlen(name)); 

                sprintf(err, "MIF: Unknown device type for model %s \n",name);
                return(err);
            }

            /* check to see if this model's parameters have been processed */
            if(! modtmp->INPmodUsed) {

                /* not already processed, so create data struct */
                error = ft_sim->newModel ( ckt, modtmp->INPmodType,
                        &(modtmp->INPmodfast), modtmp->INPmodName);
                if(error)
                    return(INPerror(error));

                /* gtri modification: allocate and initialize MIF specific model struct items */
                mdfast = (MIFmodel*) modtmp->INPmodfast;
                mdfast->num_param = DEVices[modtmp->INPmodType]->DEVpublic.num_param;
                mdfast->param = TMALLOC(Mif_Param_Data_t *, mdfast->num_param);
                for(i = 0; i < mdfast->num_param; i++) {
                    mdfast->param[i] = TMALLOC(Mif_Param_Data_t, 1);
                    mdfast->param[i]->is_null = MIF_TRUE;
                    mdfast->param[i]->size = 0;
                    mdfast->param[i]->element = NULL;
                }
                /* remaining initializations will be done by MIFmParam() and MIFsetup() */

                /* parameter isolation, identification, binding */
                line = modtmp->INPmodLine->line;
                INPgetTok(&line,&parm,1);     /* throw away '.model' */
                tfree(parm);
                INPgetTok(&line,&parm,1);     /* throw away 'modname' */
                tfree(parm);

                /* throw away the modtype - we don't treat it as a parameter */
                /* like SPICE does                                           */
                INPgetTok(&line,&parm,1);     /* throw away 'modtype' */
                tfree(parm);

                while(*line != 0) {
                    INPgetTok(&line,&parm,1);
                    for(j=0 ; j < *(ft_sim->devices[modtmp->INPmodType]->numModelParms); j++) {
                        if (strcmp(parm, ft_sim->devices[modtmp->INPmodType]->modelParms[j].keyword) == 0) {
                           /* gtri modification: call MIFgetValue instead of INPgetValue */
                            err1 = NULL;
                            val = MIFgetValue(ckt,&line,
                                    ft_sim->devices[modtmp->INPmodType]->modelParms[j].dataType,
                                    tab, &err1);
                            if(err1) {
                                err2 = TMALLOC(char, 25 + strlen(name) + strlen(err1));
                                sprintf(err2, "MIF-ERROR - model: %s - %s\n", name, err1);
                                return(err2);
                            }
                            error = ft_sim->setModelParm (ckt,
                                    modtmp->INPmodfast,
                                    ft_sim->devices[modtmp->INPmodType]->modelParms[j].id,
                                    val, NULL);
                            if(error)
                                return(INPerror(error));
                            break;
                        }
                    }
                    /* gtri modification: processing of special parameter "level" removed */
                    if(j >= *(ft_sim->devices[modtmp->INPmodType]->numModelParms))
					{
						//err has not been allocated, but free() in INPerrCat()

						// This did not allocate enough memory you wanker, K.A. replaced 5 March 2000
		// temp = TMALLOC(char, 40 + strlen(parm));
						temp = TMALLOC(char, 42 + strlen(parm));// K.A. replaced 5 March 2000

						sprintf(temp, "MIF: unrecognized parameter (%s) - ignored\n", parm);

						fprintf(stdout,temp);
						err = TMALLOC(char, 2 * strlen(temp) + 2);// K.A. added 5 March 2000
                  
						*err = '\0';// K.A. added 5 March 2000

                        err = INPerrCat(err,temp);
                    }
                    FREE(parm);

                }  /* end while end of line not reached */

                modtmp->INPmodUsed = 1;
                modtmp->INPmodLine->error = err;

            }  /* end if model parameters not processed yet */

            *model = modtmp;
            return(NULL);

        } /* end if name matches */

    } /* end for all models in modtab linked list */


    /* didn't find model - ERROR  - return NULL model */
    *model = NULL;
    err = TMALLOC(char, 60 + strlen(name));
    sprintf(err, " MIF-ERROR - unable to find definition of model %s\n",name);

    return(err);
}

