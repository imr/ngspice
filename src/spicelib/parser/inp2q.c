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
#include "inpxx.h"


static int
model_numnodes(int type)
{
#ifdef ADMS
    if (type == INPtypelook("hicum0") ||
        type == INPtypelook("hicum2") ||
        type == INPtypelook("bjt504t"))
        return 5;
#else
    NG_IGNORE(type);
#endif

    return 4;
}


void INP2Q(CKTcircuit *ckt, INPtables * tab, card * current, CKTnode *gnode)
{

    /* Qname <node> <node> <node> [<node>] <model> [<val>] [OFF]
     *       [IC=<val>,<val>] */

    int type;                   /* the type the model says it is */
    char *line;                 /* the part of the current line left to parse */
    char *name;                 /* the resistor's name */
#ifdef ADMS
    const int max_i = 4;
    char *nname[6];
    CKTnode *node[5];
#else
    const int max_i = 5;
    char *nname[5];
    CKTnode *node[4];
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
    int i;

#ifdef TRACE
    printf("INP2Q: Parsing '%s'\n", current->line);
#endif

    nodeflag = 4;               /*  initially specify a 4 terminal device  */
    line = current->line;
    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);

    model = NULL;
    thismodel = NULL;

    for (i = 0; ; i++) {
        INPgetTok(&line, &nname[i], 1);
        if (i >= 3 && INPlookMod(nname[i])) {
            model = nname[i];
            INPinsert(&model, tab);
            current->error = INPgetMod(ckt, model, &thismodel, tab);
            break;
        }
        if (i >= max_i)
            break;
        INPtermInsert(ckt, &nname[i], tab, &node[i]);
    }

    if (!model) {
        LITERR ("could not find a valid modelname");
        return;
    }

    if (i == 3) {
        /* 3-terminal device - substrate to ground */
        node[3] = gnode;
        nodeflag = 4;
    }

    if (i == 4) {
        nodeflag = 4;
#ifdef ADMS
        /* 4-terminal device - special case with tnodeout flag not handled */
        if (thismodel == NULL) {
            fprintf(stderr, "%s\nPlease check model, level or number of terminals!\n", current->error);
            controlled_exit(EXIT_BAD);
        }
        if (5 == model_numnodes(thismodel->INPmodType)) {
            node[4] = gnode; /* 4-terminal adms device - thermal node to ground */
            nname[4] = copy("0");
            INPtermInsert(ckt, &nname[4], tab, &node[4]);
            nodeflag = 5;  /* now specify a 5 node device  */
        }
#endif
    }

    if (i == 5)
        nodeflag = 5;

#ifdef TRACE
    printf("INP2Q: Looking up model\n");
#endif

    if (thismodel != NULL) {
        if (thismodel->INPmodType != INPtypelook("BJT") &&
#ifdef CIDER
         thismodel->INPmodType != INPtypelook("NBJT") &&
         thismodel->INPmodType != INPtypelook("NBJT2") &&
#endif
#ifdef ADMS
         thismodel->INPmodType != INPtypelook("hicum0") &&
         thismodel->INPmodType != INPtypelook("hicum2") &&
         thismodel->INPmodType != INPtypelook("bjt504t") &&
#endif
         thismodel->INPmodType != INPtypelook("VBIC"))
        {
            LITERR("incorrect model type");
            return;
        }
        if (nodeflag > model_numnodes(thismodel->INPmodType))
        {
            LITERR("Too much nodes for this model type");
            return;
        }
        type = thismodel->INPmodType;
        mdfast = thismodel->INPmodfast;
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
            err = tprintf("Unable to find definition of model %s\n", model);
            LITERR(err);
            tfree(err);
        }
        mdfast = tab->defQmod;
    }

#ifdef TRACE
    printf("INP2Q: Type: %d nodeflag: %d instancename: %s\n", type, nodeflag, name);
#endif
    IFC(newInstance, (ckt, mdfast, &fast, name));
    for (i = 0; i < nodeflag; i++)
        IFC(bindNode, (ckt, fast, i + 1, node[i]));

    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead) {
#ifdef CIDER
        if( type == INPtypelook("NBJT2") ) {
            LITERR(" error: no unlabeled parameter permitted on NBJT2\n");
            return;
        }
#endif
            ptemp.rValue = leadval;
            GCA(INPpName, ("area", &ptemp, ckt, type, fast));
   }
}
