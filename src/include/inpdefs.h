/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#ifndef INP
#define INP

    /* structure declarations used by either/both input package */

#include "ifsim.h"
#include "gendefs.h"
#include "inpptree.h"

typedef struct INPtables INPtables;
typedef struct card card;
typedef struct INPmodel INPmodel;


struct INPtab {
    char *t_ent;
    struct INPtab *t_next;
};

struct INPnTab {
    char *t_ent;
    void *t_node;
    struct INPnTab *t_next;
};

struct INPtables{
    struct INPtab **INPsymtab;
    struct INPnTab **INPtermsymtab;
    int INPsize;
    int INPtermsize;
    void *defAmod;
    void *defBmod;
    void *defCmod;
    void *defDmod;
    void *defEmod;
    void *defFmod;
    void *defGmod;
    void *defHmod;
    void *defImod;
    void *defJmod;
    void *defKmod;
    void *defLmod;
    void *defMmod;
    void *defNmod;
    void *defOmod;
    void *defPmod;
    void *defQmod;
    void *defRmod;
    void *defSmod;
    void *defTmod;
    void *defUmod;
    void *defVmod;
    void *defWmod;
    void *defYmod;
    void *defZmod;
};

struct card{
    int linenum;
    int linenum_orig;
    char *line;
    char *error;
    card *nextcard;
    card *actualLine;
};

/* structure used to save models in after they are read during pass 1 */
struct INPmodel{
    IFuid INPmodName;   /* uid of model */
    int INPmodType;     /* type index of device type */
    INPmodel *INPnextModel;  /* link to next model */
    int INPmodUsed;     /* flag to indicate it has already been used */
    card *INPmodLine;   /* pointer to line describing model */
    void *INPmodfast;   /* high speed pointer to model for access */
};



/* listing types - used for debug listings */
#define LOGICAL 1
#define PHYSICAL 2

int IFnewUid(CKTcircuit *, IFuid *, IFuid, char *, int, void **);
int IFdelUid(CKTcircuit *, IFuid, int);
int INPaName(char *, IFvalue *, CKTcircuit *, int *, char *, void **, IFsimulator *, int *,
        IFvalue *);
int INPapName(CKTcircuit *, int, void *, char *, IFvalue *);
void INPcaseFix(char *);
char *INPdevParse(char **, CKTcircuit *, int, void *, double *, int *, INPtables *);
char *INPdomodel(CKTcircuit *, card *, INPtables *);
void INPdoOpts(CKTcircuit *, void *, card *, INPtables *);
char *INPerrCat(char *, char *);
char *INPerror(int);
double INPevaluate(char **, int *, int);
char *INPfindLev(char *, int *);
char *INPgetMod(CKTcircuit *, char *, INPmodel **, INPtables *);
char *INPgetModBin(CKTcircuit *, char *, INPmodel **, INPtables *, char *);
int INPgetTok(char **, char **, int);
int INPgetNetTok(char **, char **, int);
void INPgetTree(char **, INPparseTree **, CKTcircuit *, INPtables *);
IFvalue *INPgetValue(CKTcircuit *, char **, int, INPtables *);
int INPgndInsert(CKTcircuit *, char **, INPtables *, void **);
int INPinsertNofree(char **token, INPtables *tab);
int INPinsert(char **, INPtables *);
int INPretrieve(char **, INPtables *);
int INPremove(char *, INPtables *);
int INPlookMod(char *);
int INPmakeMod(char *, int, card *);
char *INPmkTemp(char *);
void INPpas1(CKTcircuit *, card *, INPtables *);
void INPpas2(CKTcircuit *, card *, INPtables *, void *);
void INPpas3(CKTcircuit *, card *, INPtables *, void *, IFparm *, int);
int INPpName(char *, IFvalue *, CKTcircuit *, int, void *);
int INPtermInsert(CKTcircuit *, char **, INPtables *, void **);
int INPmkTerm(CKTcircuit *, char **, INPtables *, void **);
int INPtypelook(char *);
void INP2B(CKTcircuit *, INPtables *, card *);
void INP2C(CKTcircuit *, INPtables *, card *);
void INP2D(CKTcircuit *, INPtables *, card *);
void INP2E(CKTcircuit *, INPtables *, card *);
void INP2F(CKTcircuit *, INPtables *, card *);
void INP2G(CKTcircuit *, INPtables *, card *);
void INP2H(CKTcircuit *, INPtables *, card *);
void INP2I(CKTcircuit *, INPtables *, card *);
void INP2J(CKTcircuit *, INPtables *, card *);
void INP2K(CKTcircuit *, INPtables *, card *);
void INP2L(CKTcircuit *, INPtables *, card *);
void INP2M(CKTcircuit *, INPtables *, card *);
void INP2O(CKTcircuit *, INPtables *, card *);
void INP2P(CKTcircuit *, INPtables *, card *);
void INP2Q(CKTcircuit *, INPtables *, card *, void *);
void INP2R(CKTcircuit *, INPtables *, card *);
void INP2S(CKTcircuit *, INPtables *, card *);
void INP2T(CKTcircuit *, INPtables *, card *);
void INP2U(CKTcircuit *, INPtables *, card *);
void INP2V(CKTcircuit *, INPtables *, card *);
void INP2W(CKTcircuit *, INPtables *, card *);
void INP2Y(CKTcircuit *, INPtables *, card *);
void INP2Z(CKTcircuit *, INPtables *, card *);
int INP2dot(CKTcircuit *, INPtables *, card *, void *, void *);
INPtables *INPtabInit(int);
void INPkillMods(void);
void INPtabEnd(INPtables *);

#endif /*INP*/
