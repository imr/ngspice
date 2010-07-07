/*************
 * Header file for inpxx.c
 * 1999 E. Rouat
 ************/

#ifndef INP_H_INCLUDED
#define INP_H_INCLUDED

/* ifeval.c */

int IFeval(IFparseTree *tree, double gmin, double *result, double *vals, 
	   double *derivs);

/* ifnewuid.c */

int IFnewUid(CKTcircuit *ckt, IFuid *newuid, IFuid olduid, char *suffix, int type, 
	     void **nodedata);
int IFdelUid(CKTcircuit *ckt, IFuid uid, int type);

/* inp2xx.c */

void INP2B(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2C(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2D(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2E(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2F(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2G(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2H(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2I(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2J(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2K(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2L(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2M(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2O(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2P(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2Q(CKTcircuit *ckt, INPtables *tab, card *current, void *gnode);
void INP2R(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2S(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2T(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2U(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2V(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2W(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2Y(CKTcircuit *ckt, INPtables *tab, card *current);
void INP2Z(CKTcircuit *ckt, INPtables *tab, card *current);
int INP2dot(CKTcircuit *ckt, INPtables *tab, card *current, void *task, void *gnode);

/* inpxxxx.c */

int INPaName(char *parm, IFvalue *val, CKTcircuit *ckt, int *dev, char *devnam, 
	     void **fast, IFsimulator *sim, int *dataType, IFvalue *selector);
int INPapName(CKTcircuit *ckt, int type, void *analPtr, char *parmname, IFvalue *value);
void INPcaseFix(register char *string);
char * INPdomodel(CKTcircuit *ckt, card *image, INPtables *tab);
void INPdoOpts(CKTcircuit *ckt, void *anal, card *optCard, INPtables *tab);
char * INPdevParse(char **line, CKTcircuit *ckt, int dev, void *fast, double *leading, 
		   int *waslead, INPtables *tab);
char * INPerrCat(char *a, char *b);
char * INPerror(int type);
double INPevaluate(char **line, int *error, int gobble);
char * INPfindLev(char *line, int *level);
char * INPfindVer(char *line, char *version);
char * INPgetMod(CKTcircuit *ckt, char *name, INPmodel **model, INPtables *tab);
int INPgetStr(char **line, char **token, int gobble);
int INPgetTitle(CKTcircuit **ckt, card **data);
int INPgetTok(char **line, char **token, int gobble);
int INPgetUTok(char **line, char **token, int gobble);
IFvalue * INPgetValue(CKTcircuit *ckt, char **line, int type, INPtables *tab);
void INPkillMods(void);
void INPlist(FILE *file, card *deck, int type);
int INPlookMod(char *name);
int INPmakeMod(char *token, int type, card *line);
char * INPmkTemp(char *string);
void INPpas1(CKTcircuit *ckt, card *deck, INPtables *tab);
void INPpas2(CKTcircuit *ckt, card *data, INPtables *tab, void *task);
int INPpName(char *parm, IFvalue *val, CKTcircuit *ckt, int dev, void *fast);

/* inpptree.c */

void INPgetTree(char **line, INPparseTree **pt, CKTcircuit *ckt, INPtables *tab);


/* inpsymt.c */

INPtables * INPtabInit(int numlines);
int INPtermInsert(CKTcircuit *ckt, char **token, INPtables *tab, void **node);
int INPmkTerm(CKTcircuit *ckt, char **token, INPtables *tab, void **node);
int INPgndInsert(CKTcircuit *ckt, char **token, INPtables *tab, void **node);
int INPretrieve(char **token, INPtables *tab);
int INPinsert(char **token, INPtables *tab);
int INPinsertNofree(char **token, INPtables *tab);
int INPremove(char *token, INPtables *tab);
int INPremTerm(char *token, INPtables *tab);
void INPtabEnd(INPtables *tab);

int INPtypelook(char *type);

/* ptfuncs.c */

double PTabs(double arg);
double PTsgn(double arg);
double PTplus(double arg1, double arg2);
double PTminus(double arg1, double arg2);
double PTtimes(double arg1, double arg2);
double PTtimes(double arg1, double arg2);
double PTdivide(double arg1, double arg2);
double PTpower(double arg1, double arg2);
double PTacos(double arg);
double PTacosh(double arg);
double PTasin(double arg);
double PTasinh(double arg);
double PTatan(double arg);
double PTatanh(double arg);
double PTustep(double arg);
double PTuramp(double arg);
double PTcos(double arg);
double PTcosh(double arg);
double PTexp(double arg);
double PTln(double arg);
double PTlog(double arg);
double PTsin(double arg);
double PTsinh(double arg);
double PTsqrt(double arg);
double PTtan(double arg);
double PTuminus(double arg);

/* sperror.c */

#endif
