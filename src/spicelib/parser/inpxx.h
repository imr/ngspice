/*************
 * Header file for inpxx.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_INPXX_H
#define ngspice_INPXX_H

/* inp2xx.c */

void INP2B(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2C(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2D(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2E(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2F(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2G(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2H(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2I(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2J(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2K(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2L(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2M(CKTcircuit *ckt, INPtables *tab, struct card *current);
#ifdef OSDI
void INP2N(CKTcircuit *ckt, INPtables *tab, struct card *current);
#endif
void INP2O(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2P(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2Q(CKTcircuit *ckt, INPtables *tab, struct card *current, CKTnode *gnode);
void INP2R(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2S(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2T(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2U(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2V(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2W(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2Y(CKTcircuit *ckt, INPtables *tab, struct card *current);
void INP2Z(CKTcircuit *ckt, INPtables *tab, struct card *current);

#ifdef OSDI
/* inpdomod.c: model level bound to an OSDI module by command osdi_bind */
struct osdi_bind {
    char *kind;         /* "mos" or "dio" */
    int level;
    char *module;       /* OSDI module name */
    char *has;          /* selected if the card has this parameter, or NULL */
    char *version;      /* selected if the card's version starts so, or NULL */
    struct wordlist *alias;    /* instance parameter names: from, to, from, to, ... */
    char *polarity;     /* set by nmos/npn to nval, by pmos/pnp to pval, or NULL */
    char *nval, *pval;  /* default "1" and "-1" */
    struct wordlist *ignore;   /* card parameters to skip */
    struct osdi_bind *next;
};

const struct osdi_bind *INPosdiBindForCard(const char *kind, const char *card);
bool INPosdiCardFits(char letter, const char *card);
const struct osdi_bind *INPosdiBindOfCard(const char *card);
const char *INPosdiAlias(const struct osdi_bind *b, const char *name);
char *INPosdiAliasLine(const struct osdi_bind *b, const char *line);
#endif


/* ptfuncs.c */

double PTabs(double arg);
double PTsgn(double arg);
double PTplus(double arg1, double arg2);
double PTminus(double arg1, double arg2);
double PTtimes(double arg1, double arg2);
double PTdivide(double arg1, double arg2);
double PTpower(double arg1, double arg2);
double PTpowerH(double arg1, double arg2);
double PTpwr(double arg1, double arg2);
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
double PTlog(double arg);
double PTlog10(double arg);
double PTsin(double arg);
double PTsinh(double arg);
double PTsqrt(double arg);
double PTtan(double arg);
double PTtanh(double arg);
double PTuminus(double arg);
double PTustep2(double arg);
double PTpwl(double arg, void *data);
double PTpwl_derivative(double arg, void *data);
double PTmin(double arg1, double arg2);
double PTmax(double arg1, double arg2);
double PTeq0(double arg);
double PTne0(double arg);
double PTgt0(double arg);
double PTlt0(double arg);
double PTge0(double arg);
double PTle0(double arg);
double PTceil(double arg);
double PTfloor(double arg);
double PTnint(double arg);
double PTddt(double arg, void* data);

#endif
