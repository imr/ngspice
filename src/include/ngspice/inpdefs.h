/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#ifndef ngspice_INPDEFS_H
#define ngspice_INPDEFS_H

    /* structure declarations used by either/both input package */

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/inpptree.h"

typedef struct INPtables INPtables;
typedef struct card card;
typedef struct INPmodel INPmodel;

#define NESTINGDEPTH 10

struct INPtab {
    char *t_ent;
    struct INPtab *t_next;
};

struct INPnTab {
    char *t_ent;
    CKTnode *t_node;
    struct INPnTab *t_next;
};

struct INPtables{
    struct INPtab **INPsymtab;
    struct INPnTab **INPtermsymtab;
    int INPsize;
    int INPtermsize;
    GENmodel *defAmod;
    GENmodel *defBmod;
    GENmodel *defCmod;
    GENmodel *defDmod;
    GENmodel *defEmod;
    GENmodel *defFmod;
    GENmodel *defGmod;
    GENmodel *defHmod;
    GENmodel *defImod;
    GENmodel *defJmod;
    GENmodel *defKmod;
    GENmodel *defLmod;
    GENmodel *defMmod;
    GENmodel *defNmod;
    GENmodel *defOmod;
    GENmodel *defPmod;
    GENmodel *defQmod;
    GENmodel *defRmod;
    GENmodel *defSmod;
    GENmodel *defTmod;
    GENmodel *defUmod;
    GENmodel *defVmod;
    GENmodel *defWmod;
    GENmodel *defYmod;
    GENmodel *defZmod;
};

struct modellist;

struct nscope {
    struct nscope *next;
    struct line_assoc *subckts;
    struct modellist *models;
};

struct line_assoc {
    const char *name;
    struct line *line;
    struct line_assoc *next;
};

struct card{
    int linenum;
    int linenum_orig;
    char *line;
    char *error;
    card *nextcard;
    card *actualLine;
    struct nscope *level;
};

/* structure used to save models in after they are read during pass 1 */
struct INPmodel{
    IFuid INPmodName;   /* uid of model */
    int INPmodType;     /* type index of device type */
    INPmodel *INPnextModel;  /* link to next model */
    card *INPmodLine;   /* pointer to line describing model */
    GENmodel *INPmodfast;   /* high speed pointer to model for access */
};



/* listing types - used for debug listings */
#define LOGICAL 1
#define PHYSICAL 2

int IFnewUid(CKTcircuit *, IFuid *, IFuid, char *, int, CKTnode **);
int IFdelUid(CKTcircuit *, IFuid, int);
int INPaName(char *, IFvalue *, CKTcircuit *, int *, char *, GENinstance **, IFsimulator *, int *,
        IFvalue *);
int INPapName(CKTcircuit *, int, JOB *, char *, IFvalue *);
void INPcaseFix(char *);
char *INPdevParse(char **, CKTcircuit *, int, GENinstance *, double *, int *, INPtables *);
char *INPdomodel(CKTcircuit *, card *, INPtables *);
void INPdoOpts(CKTcircuit *, JOB *, card *, INPtables *);
char *INPerrCat(char *, char *);
char *INPstrCat(char *, char *, char *);
char *INPerror(int);
double INPevaluate(char **, int *, int);
char *INPfindLev(char *, int *);
char *INPgetMod(CKTcircuit *, char *, INPmodel **, INPtables *);
char *INPgetModBin(CKTcircuit *, char *, INPmodel **, INPtables *, char *);
int INPgetTok(char **, char **, int);
int INPgetNetTok(char **, char **, int);
void INPgetTree(char **, INPparseTree **, CKTcircuit *, INPtables *);
void INPfreeTree(IFparseTree *);
IFvalue *INPgetValue(CKTcircuit *, char **, int, INPtables *);
int INPgndInsert(CKTcircuit *, char **, INPtables *, CKTnode **);
int INPinsertNofree(char **token, INPtables *tab);
int INPinsert(char **, INPtables *);
int INPretrieve(char **, INPtables *);
int INPremove(char *, INPtables *);
INPmodel *INPlookMod(const char *);
int INPmakeMod(char *, int, card *);
char *INPmkTemp(char *);
void INPpas1(CKTcircuit *, card *, INPtables *);
void INPpas2(CKTcircuit *, card *, INPtables *, TSKtask *);
void INPpas3(CKTcircuit *, card *, INPtables *, TSKtask *, IFparm *, int);
int INPpName(char *, IFvalue *, CKTcircuit *, int, GENinstance *);
int INPtermInsert(CKTcircuit *, char **, INPtables *, CKTnode **);
int INPmkTerm(CKTcircuit *, char **, INPtables *, CKTnode **);
int INPtypelook(char *);
int INP2dot(CKTcircuit *, INPtables *, card *, TSKtask *, CKTnode *);
INPtables *INPtabInit(int);
void INPkillMods(void);
void INPtabEnd(INPtables *);
char * INPfindVer(char *line, char *version);
int INPgetStr(char **line, char **token, int gobble);
int INPgetTitle(CKTcircuit **ckt, card **data);
int INPgetUTok(char **line, char **token, int gobble);
int INPgetU2Tok(char **line, char **token, int gobble);
int INPremTerm(char *token, INPtables *tab);



#endif
