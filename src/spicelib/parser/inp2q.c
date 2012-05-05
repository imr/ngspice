/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: 2001 Paolo Nenzi (Cider Integration)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inp.h"

void INP2Q(CKTcircuit *ckt, INPtables * tab, card * current, CKTnode *gnode)
{

    /* Qname <node> <node> <node> [<node>] <model> [<val>] [OFF]
     *       [IC=<val>,<val>] */

    int type;                   /* the type the model says it is */
    char *line;                 /* the part of the current line left to parse */
    char *name;                 /* the resistor's name */
    char *nname1;               /* the first node's name */
    char *nname2;               /* the second node's name */
    char *nname3;               /* the third node's name */
    char *nname4;               /* the fourth node's name */
#ifdef ADMS
    char *nname5;               /* the fifth node's name */
#endif
    CKTnode *node1;             /* the first node's node pointer */
    CKTnode *node2;             /* the second node's node pointer */
    CKTnode *node3;             /* the third node's node pointer */
    CKTnode *node4;             /* the fourth node's node pointer */
#ifdef ADMS
    CKTnode *node5;             /* the fifth node's node pointer */
#endif
    int error;                  /* error code temporary */
    int nodeflag;               /* flag indicating 4 or 5 nodes */
    GENinstance *fast;          /* pointer to the actual instance */
    IFvalue ptemp;              /* a value structure to package resistance into */
    int waslead;                /* flag to indicate that funny unlabeled number was found */
    double leadval;             /* actual value of unlabeled number */
    char *model;                /* the name of the model */
    INPmodel *thismodel;        /* pointer to model description for user's model */
    GENmodel *mdfast;           /* pointer to the actual model */
    IFuid uid;                  /* uid of default model */

#ifdef TRACE
    printf("INP2Q: Parsing '%s'\n", current->line);
#endif

    nodeflag = 0;               /*  initially specify a 4 terminal device  */
    line = current->line;
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);
    INPgetNetTok(&line, &nname1, 1);
    INPtermInsert(ckt, &nname1, tab, &node1);
    INPgetNetTok(&line, &nname2, 1);
    INPtermInsert(ckt, &nname2, tab, &node2);
    INPgetNetTok(&line, &nname3, 1);
    INPtermInsert(ckt, &nname3, tab, &node3);
    INPgetTok(&line, &model, 1);

    thismodel = NULL;
    /*  See if 4th token after device specification is a model name  */
    if (INPlookMod(model)) {
        /* 3-terminal device - substrate to ground */
        node4 = gnode;
        INPinsert(&model, tab);
#ifdef TRACE
        printf("INP2Q: 3-terminal device - substrate to ground\n");
#endif
        current->error = INPgetMod(ckt, model, &thismodel, tab);
    } else {
        nname4 = model;
        INPtermInsert(ckt, &nname4, tab, &node4);
        INPgetTok(&line, &model, 1);
        /*  See if 5th token after device specification is a model name  */
#ifdef TRACE
        printf("INP2Q: checking for 4 node device\n");
#endif
        if (INPlookMod(model)) {
           /* 4-terminal device - special case with tnodeout flag not handled */
           INPinsert(&model, tab);
           current->error = INPgetMod(ckt, model, &thismodel, tab);
#ifdef ADMS
           if (thismodel == NULL) {
        	   fprintf(stderr, "%s\nPlease check model, level or number of terminals!\n", current->error);
        	   controlled_exit(EXIT_BAD);
           }
           else if ((thismodel->INPmodType == INPtypelook("hicum0"))
            || (thismodel->INPmodType == INPtypelook("hicum2"))
            || (thismodel->INPmodType == INPtypelook("bjt504t")))
           {
               node5 = gnode; /* 4-terminal adms device - thermal node to ground */
               nname5 = copy("0");
               INPtermInsert(ckt, &nname5, tab, &node5);
               nodeflag = 1;  /* now specify a 5 node device  */
           }
        } else {
           /* 5-terminal device */
#ifdef TRACE
           printf("INP2Q: checking for 5 node device\n");
#endif
           nodeflag = 1;                /*  now specify a 5 node device  */
           nname5 = model;
           INPtermInsert(ckt, &nname5, tab, &node5);
           INPgetTok(&line, &model, 1);
           INPinsert(&model, tab);
           current->error = INPgetMod(ckt, model, &thismodel, tab);
#endif
        }
    }

#ifdef TRACE
    printf("INP2Q: Looking up model\n");
#endif

    if (thismodel != NULL) {
        if ((thismodel->INPmodType != INPtypelook("BJT"))
#ifdef CIDER
         && (thismodel->INPmodType != INPtypelook("NBJT"))
         && (thismodel->INPmodType != INPtypelook("NBJT2"))
#endif
#ifdef ADMS
         && (thismodel->INPmodType != INPtypelook("hicum0"))
         && (thismodel->INPmodType != INPtypelook("hicum2"))
         && (thismodel->INPmodType != INPtypelook("bjt504t"))
#endif
         && (thismodel->INPmodType != INPtypelook("VBIC")))
        {
            LITERR("incorrect model type")
            return;
        }
#ifdef ADMS
        if ((nodeflag && (thismodel->INPmodType != INPtypelook("hicum0")))
         && (nodeflag && (thismodel->INPmodType != INPtypelook("hicum2")))
         && (nodeflag && (thismodel->INPmodType != INPtypelook("bjt504t"))))
        {
            LITERR("Too much nodes for this model type")
            return;
        }
#endif
        type = (thismodel->INPmodType);
        mdfast = (thismodel->INPmodfast);
    } else {
       /* no model found */
       type = INPtypelook("BJT");
        if (type < 0) {
            LITERR("Device type BJT not supported by this binary\n");
            return;
        }
        if (!tab->defQmod) {
            /* create default Q model */
            char *err;
            IFnewUid(ckt, &uid, NULL, "Q", UID_MODEL, NULL);
            IFC(newModel, (ckt, type, &(tab->defQmod), uid));
            err = TMALLOC(char, 70 + strlen(model));
            (void) sprintf(err, "Unable to find definition of model %s - default BJT assumed \n", model);
            LITERR(err);
            tfree(err);
        }
        mdfast = tab->defQmod;
    }

#ifdef TRACE
    printf("INP2Q: Type: %d nodeflag: %d instancename: %s\n", type, nodeflag, name);
#endif
    IFC(newInstance, (ckt, mdfast, &fast, name));
    IFC(bindNode, (ckt, fast, 1, node1));
    IFC(bindNode, (ckt, fast, 2, node2));
    IFC(bindNode, (ckt, fast, 3, node3));
    IFC(bindNode, (ckt, fast, 4, node4));

#ifdef ADMS
    if (nodeflag) { /* 5-node device */
        IFC(bindNode, (ckt, fast, 5, node5));
    }
#endif

    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
#ifdef CIDER
        if( type == INPtypelook("NBJT2") ) {
            LITERR(" error: no unlabeled parameter permitted on NBJT2\n")
        } else {
#endif
            ptemp.rValue = leadval;
            GCA(INPpName, ("area", &ptemp, ckt, type, fast));
        }
#ifdef CIDER
   }
#endif
}
