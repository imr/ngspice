/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/ifsim.h"
#include "ngspice/inpmacs.h"
#include "ngspice/cktdefs.h"

#include "inppas3.h"

extern IFsimulator *ft_sim;

#ifdef RELAN
static void
dot_appendmodel (CKTcircuit *ckt, INPtables *tab, card *current)
{
    /* .appendmodel from_model to_model */

    int type;                  /* the type the model says it is */
    char *line ;               /* the part of the current line left to parse */
    char *relmodel_name ;      /* the relmodel name */
    char *appendmodel_name ;   /* the appendmodel name */
    char *save ;               /* saj - used to save the posn of the start of
                                  the parameters if the model is a mosfet */
    int error ;                /* error code temporary */
    char *model ;              /* the name of the model */
    INPmodel *thismodel ;      /* pointer to model description for user's model */
    GENmodel *mdfast ;         /* pointer to the actual model */
    IFuid uid ;                /* uid for default model */
    char* err_msg ;

#ifdef TRACE
    printf("INP2M: Parsing '%s'\n",current->line);
#endif

    line = current->line ;

    INPgetTok (&line, &relmodel_name, 1) ; /* throw away .appendmodel */
    tfree (relmodel_name) ;

    INPgetTok (&line, &relmodel_name, 1) ; /* get relmodel name */

    save = line ; /* saj - save the posn for later if the default mosfet model is used */
    thismodel = NULL ;

    err_msg = INPgetMod (ckt, relmodel_name, &thismodel, tab) ;
    tfree (err_msg) ;

    /* check if using model binning -- pass in line since need 'l' and 'w' */
    if (thismodel == NULL)
    {
        INPgetModBin (ckt, relmodel_name, &thismodel, tab, line) ;
    }

    model = relmodel_name ; /* mosfet */
    line = save ;        /* reset the posn to what it sould be */

    INPinsert (&model, tab) ;
    thismodel = NULL ;

#ifdef TRACE
    printf("INP2M: Looking up model\n");
#endif

    err_msg = INPgetMod (ckt, model, &thismodel, tab) ;
    if (thismodel == NULL)
    {
        INPgetModBin (ckt, model, &thismodel, tab, save) ;
        if (thismodel == NULL)
        {
            current->error = err_msg ;
        } else {
            tfree (err_msg) ;
        }
    } else {
        tfree (err_msg) ;
    }

    if (thismodel != NULL)
    {
        if (thismodel->INPmodType != INPtypelook ("Mos1")
                && thismodel->INPmodType != INPtypelook ("Mos2")
                && thismodel->INPmodType != INPtypelook ("Mos3")
                && thismodel->INPmodType != INPtypelook ("Mos5")
                && thismodel->INPmodType != INPtypelook ("Mos6")
                && thismodel->INPmodType != INPtypelook ("Mos8")
                && thismodel->INPmodType != INPtypelook ("Mos9")
                && thismodel->INPmodType != INPtypelook ("BSIM1")
                && thismodel->INPmodType != INPtypelook ("BSIM2")
                && thismodel->INPmodType != INPtypelook ("BSIM3")
                && thismodel->INPmodType != INPtypelook ("BSIM3v32")
                && thismodel->INPmodType != INPtypelook ("B4SOI")
                && thismodel->INPmodType != INPtypelook ("B3SOIPD")
                && thismodel->INPmodType != INPtypelook ("B3SOIFD")
                && thismodel->INPmodType != INPtypelook ("B3SOIDD")
                && thismodel->INPmodType != INPtypelook ("BSIM4")
                && thismodel->INPmodType != INPtypelook ("BSIM4v5")
                && thismodel->INPmodType != INPtypelook ("BSIM4v6")
                && thismodel->INPmodType != INPtypelook ("BSIM4v7")
                && thismodel->INPmodType != INPtypelook ("BSIM3v0")
                && thismodel->INPmodType != INPtypelook ("BSIM3v1")
                && thismodel->INPmodType != INPtypelook ("SOI3")
                && thismodel->INPmodType != INPtypelook ("HiSIM2")
                && thismodel->INPmodType != INPtypelook ("HiSIMHV")
                && thismodel->INPmodType != INPtypelook ("RELMODEL")
           )
        {
            LITERR ("incorrect model type") ;
            return ;
        }
        type = thismodel->INPmodType ;
        mdfast = (thismodel->INPmodfast) ;
    } else {
        type = INPtypelook ("Mos1") ;
        if (type < 0)
        {
            LITERR ("Device type MOS1 not supported by this binary\n") ;
            return ;
        }
        if (!tab->defMmod)
        {
            /* create default M model */
            IFnewUid (ckt, &uid, NULL, "M", UID_MODEL, NULL) ;
            IFC (newModel, (ckt, type, &(tab->defMmod), uid)) ;
        }
        mdfast = tab->defMmod ;
    }

    INPgetTok (&line, &appendmodel_name, 1) ; /* get appendmodel name */
    thismodel = NULL ;
    err_msg = INPgetMod (ckt, appendmodel_name, &thismodel, tab) ;
    tfree (err_msg) ;
    if (thismodel == NULL)
    {
        INPgetModBin (ckt, relmodel_name, &thismodel, tab, line) ;
    }
    thismodel->INPmodfast->GENrelmodelModel = mdfast ; /* Attach 'relmodel' model to 'appendmodel' model */

    return ;
}
#endif

/* pass 3 - Read all nodeset and IC lines. All circuit nodes will have
 * been created by now, (except for internal device nodes), so any
 * nodeset or IC nodes which have to be created are flagged with a
 * warning.  */

void
INPpas3(CKTcircuit *ckt, card *data, INPtables *tab, TSKtask *task,
        IFparm *nodeParms, int numNodeParms)
{

    card *current;
    int error;			/* used by the macros defined above */
    char *line;			/* the part of the current line left
                                   to parse */
    char *name;			/* the node's name */
    char *token=NULL;		/* a token from the line */
    IFparm *prm;		/* pointer to parameter to search
                                   through array */
    IFvalue ptemp;		/* a value structure to package
                                   resistance into */
    int which;			/* which analysis we are performing */
    CKTnode *node1;		/* the first node's node pointer */

    NG_IGNORE(task);

#ifdef TRACE
    /* SDB debug statement */
    printf("In INPpas3 . . . \n");
#endif

    for(current = data; current != NULL; current = current->nextcard) {
        line = current->line;
        FREE(token);
        INPgetTok(&line,&token,1);

        if (strcmp(token,".nodeset")==0) {
            which = -1;

            for(prm = nodeParms; prm < nodeParms + numNodeParms; prm++) {
                if(strcmp(prm->keyword,"nodeset")==0) {
                    which = prm->id;
                    break;
                }
            }

            if(which == -1) {
                LITERR("nodeset unknown to simulator. \n");
                goto quit;
            }

            for(;;) {
                /* loop until we run out of data */
                INPgetTok(&line,&name,1);
                if( *name == 0) break; /* end of line */

                /* If we have 'all = value' , then set all voltage nodes to 'value',
                   except for ground node at node->number 0 */
                if ( cieq(name, "all")) {
                    ptemp.rValue = INPevaluate(&line,&error,1);
                    for (node1 = ckt->CKTnodes; node1 != NULL; node1 = node1->next) {
                        if ((node1->type == SP_VOLTAGE) && (node1->number > 0))
                            IFC(setNodeParm, (ckt, node1, which, &ptemp, NULL));
                    }
                    break;
                }
                /* check to see if in the form V(xxx) and grab the xxx */
                if( (*name == 'V' || *name == 'v') && !name[1] ) {
                    /* looks like V - must be V(xx) - get xx now*/
                    INPgetTok(&line,&name,1);
                    if (INPtermInsert(ckt,&name,tab,&node1)!=E_EXISTS)
                        fprintf(stderr,
                                "Warning : Nodeset on non-existant node - %s\n", name);
                    ptemp.rValue = INPevaluate(&line,&error,1);
                    IFC(setNodeParm, (ckt, node1, which, &ptemp, NULL));
                    continue;
                }
                LITERR(" Error: .nodeset syntax error.\n");
                break;
            }
        } else if ((strcmp(token,".ic") == 0)) {
            /* .ic */
            which = -1;
            for(prm = nodeParms; prm < nodeParms + numNodeParms; prm++) {
                if(strcmp(prm->keyword,"ic")==0) {
                    which = prm->id;
                    break;
                }
            }

            if(which==-1) {
                LITERR("ic unknown to simulator. \n");
                goto quit;
            }

            for(;;) {
                /* loop until we run out of data */
                INPgetTok(&line,&name,1);
                /* check to see if in the form V(xxx) and grab the xxx */
                if( *name == 0) {
                    FREE(name);
                    break; /* end of line */
                }
                if( (*name == 'V' || *name == 'v') && !name[1] ) {
                    /* looks like V - must be V(xx) - get xx now*/
                    INPgetTok(&line,&name,1);
                    if (INPtermInsert(ckt,&name,tab,&node1)!=E_EXISTS)
                        fprintf(stderr,
                                "Warning : IC on non-existant node - %s\n", name);
                    ptemp.rValue = INPevaluate(&line,&error,1);
                    IFC(setNodeParm, (ckt, node1, which, &ptemp, NULL));
                    continue;
                }
                LITERR(" Error: .ic syntax error.\n");
                break;
            }
        }

#ifdef RELAN
        else if (strcmp (token, ".appendmodel") == 0) {
                dot_appendmodel (ckt, tab, current) ;
                goto quit ;
        }
#endif

    }
quit:
    FREE(token);
    return;
}

