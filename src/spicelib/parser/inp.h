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

int IFnewUid(void *ckt, IFuid *newuid, IFuid olduid, char *suffix, int type, 
	     void **nodedata);
int IFdelUid(void *ckt, IFuid uid, int type);

/* inp2xx.c */

void INP2B(void *ckt, INPtables *tab, card *current);
void INP2C(void *ckt, INPtables *tab, card *current);
void INP2D(void *ckt, INPtables *tab, card *current);
void INP2E(void *ckt, INPtables *tab, card *current);
void INP2F(void *ckt, INPtables *tab, card *current);
void INP2G(void *ckt, INPtables *tab, card *current);
void INP2H(void *ckt, INPtables *tab, card *current);
void INP2I(void *ckt, INPtables *tab, card *current);
void INP2J(void *ckt, INPtables *tab, card *current);
void INP2K(void *ckt, INPtables *tab, card *current);
void INP2L(void *ckt, INPtables *tab, card *current);
void INP2M(void *ckt, INPtables *tab, card *current);
void INP2O(void *ckt, INPtables *tab, card *current);
void INP2P(void *ckt, INPtables *tab, card *current);
void INP2Q(void *ckt, INPtables *tab, card *current, void *gnode);
void INP2R(void *ckt, INPtables *tab, card *current);
void INP2S(void *ckt, INPtables *tab, card *current);
void INP2T(void *ckt, INPtables *tab, card *current);
void INP2U(void *ckt, INPtables *tab, card *current);
void INP2V(void *ckt, INPtables *tab, card *current);
void INP2W(void *ckt, INPtables *tab, card *current);
void INP2Y(void *ckt, INPtables *tab, card *current);
void INP2Z(void *ckt, INPtables *tab, card *current);
int INP2dot(void *ckt, INPtables *tab, card *current, void *task, void *gnode);

/* inpxxxx.c */

int INPaName(char *parm, IFvalue *val, void *ckt, int *dev, char *devnam, 
	     void **fast, IFsimulator *sim, int *dataType, IFvalue *selector);
int INPapName(void *ckt, int type, void *analPtr, char *parmname, IFvalue *value);
void INPcaseFix(register char *string);
char * INPdomodel(void *ckt, card *image, INPtables *tab);
void INPdoOpts(void *ckt, void *anal, card *optCard, INPtables *tab);
char * INPdevParse(char **line, void *ckt, int dev, void *fast, double *leading, 
		   int *waslead, INPtables *tab);
char * INPerrCat(char *a, char *b);
char * INPerror(int type);
double INPevaluate(char **line, int *error, int gobble);
char * INPfindLev(char *line, int *level);
char * INPgetMod(void *ckt, char *name, INPmodel **model, INPtables *tab);
int INPgetStr(char **line, char **token, int gobble);
int INPgetTitle(void **ckt, card **data);
int INPgetTok(char **line, char **token, int gobble);
int INPgetUTok(char **line, char **token, int gobble);
IFvalue * INPgetValue(void *ckt, char **line, int type, INPtables *tab);
void INPkillMods(void);
void INPlist(FILE *file, card *deck, int type);
int INPlookMod(char *name);
int INPmakeMod(char *token, int type, card *line);
char * INPmkTemp(char *string);
void INPpas1(void *ckt, card *deck, INPtables *tab);
void INPpas2(void *ckt, card *data, INPtables *tab, void *task);
int INPpName(char *parm, IFvalue *val, void *ckt, int dev, void *fast);

/* inpptree.c */

void INPgetTree(char **line, INPparseTree **pt, void *ckt, INPtables *tab);


/* inpsymt.c */

INPtables * INPtabInit(int numlines);
int INPtermInsert(void *ckt, char **token, INPtables *tab, void **node);
int INPmkTerm(void *ckt, char **token, INPtables *tab, void **node);
int INPgndInsert(void *ckt, char **token, INPtables *tab, void **node);
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
