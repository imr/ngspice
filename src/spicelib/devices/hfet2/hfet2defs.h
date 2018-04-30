#ifndef HFET2
#define HFET2

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"


typedef struct sHFET2instance {

    struct GENinstance gen;

#define HFET2modPtr(inst) ((struct sHFET2model *)((inst)->gen.GENmodPtr))
#define HFET2nextInstance(inst) ((struct sHFET2instance *)((inst)->gen.GENnextInstance))
#define HFET2name gen.GENname
#define HFET2state gen.GENstate

    int HFET2drainNode;
    int HFET2gateNode;
    int HFET2sourceNode;
    int HFET2drainPrimeNode;
    int HFET2sourcePrimeNode;
    double HFET2length;
    double HFET2width;
    double HFET2m;
    double HFET2temp;
    double HFET2dtemp;
    double HFET2tLambda;
    double HFET2tMu;
    double HFET2tNmax;
    double HFET2tVto;
    double HFET2icVDS;
    double HFET2icVGS;
    double *HFET2drainDrainPrimePtr;
    double *HFET2gateDrainPrimePtr;
    double *HFET2gateSourcePrimePtr;
    double *HFET2sourceSourcePrimePtr;
    double *HFET2drainPrimeDrainPtr;
    double *HFET2drainPrimeGatePtr;
    double *HFET2drainPriHFET2ourcePrimePtr;
    double *HFET2sourcePrimeGatePtr;
    double *HFET2sourcePriHFET2ourcePtr;
    double *HFET2sourcePrimeDrainPrimePtr;
    double *HFET2drainDrainPtr;
    double *HFET2gateGatePtr;  
    double *HFET2sourceSourcePtr;
    double *HFET2drainPrimeDrainPrimePtr;
    double *HFET2sourcePriHFET2ourcePrimePtr;

   
#define HFET2vgs HFET2state 
#define HFET2vgd HFET2state+1 
#define HFET2cg HFET2state+2 
#define HFET2cd HFET2state+3 
#define HFET2cgd HFET2state+4 
#define HFET2gm HFET2state+5 
#define HFET2gds HFET2state+6 
#define HFET2ggs HFET2state+7 
#define HFET2ggd HFET2state+8 
#define HFET2qgs HFET2state+9 
#define HFET2cqgs HFET2state+10 
#define HFET2qgd HFET2state+11 
#define HFET2cqgd HFET2state+12 

#define HFET2numStates 13

    int HFET2mode;
    int HFET2off;
    
    unsigned HFET2icVDSGiven  : 1;
    unsigned HFET2icVGSGiven  : 1;
    unsigned HFET2lengthGiven : 1;
    unsigned HFET2widthGiven  : 1;
    unsigned HFET2mGiven      : 1;
    unsigned HFET2tempGiven   : 1;
    unsigned HFET2dtempGiven  : 1;
        
    double HFET2n0;
    double HFET2n01;
    double HFET2n02;
    double HFET2gchi0;
    double HFET2imax;
    double HFET2vcrit;
    double HFET2ggrlw;
    double HFET2jslw;
    
} HFET2instance ;



typedef struct sHFET2model {

    struct GENmodel gen;

#define HFET2modType gen.GENmodType
#define HFET2nextModel(inst) ((struct sHFET2model *)((inst)->gen.GENnextModel))
#define HFET2instances(inst) ((HFET2instance *)((inst)->gen.GENinstances))
#define HFET2modName gen.GENmodName

    int HFET2type;
    
    double HFET2cf;
    double HFET2d1;
    double HFET2d2;
    double HFET2del;
    double HFET2delta;
    double HFET2deltad;
    double HFET2di;
    double HFET2epsi;
    double HFET2eta;
    double HFET2eta1;
    double HFET2eta2;
    double HFET2gamma;
    double HFET2ggr;
    double HFET2js;
    double HFET2klambda;
    double HFET2kmu;
    double HFET2knmax;
    double HFET2kvto;
    double HFET2lambda;
    double HFET2m;
    double HFET2mc;
    double HFET2mu;
    double HFET2n;
    double HFET2nmax;
    double HFET2p;
    double HFET2rd;
    double HFET2rdi;    
    double HFET2rs;
    double HFET2rsi;
    double HFET2sigma0;
    double HFET2vs;    
    double HFET2vsigma;    
    double HFET2vsigmat;
    double HFET2vt1;
    double HFET2vt2;
    double HFET2vto;
    
    double HFET2drainConduct;
    double HFET2sourceConduct;
    double HFET2deltaSqr;

    unsigned HFET2cfGiven : 1;
    unsigned HFET2d1Given : 1;
    unsigned HFET2d2Given : 1;
    unsigned HFET2delGiven : 1;
    unsigned HFET2deltaGiven : 1;
    unsigned HFET2deltadGiven : 1;
    unsigned HFET2diGiven : 1;
    unsigned HFET2epsiGiven : 1;
    unsigned HFET2etaGiven : 1;
    unsigned HFET2eta1Given : 1;
    unsigned HFET2eta2Given : 1;
    unsigned HFET2gammaGiven : 1;
    unsigned HFET2ggrGiven : 1;
    unsigned HFET2jsGiven : 1;
    unsigned HFET2klambdaGiven : 1;
    unsigned HFET2kmuGiven : 1;
    unsigned HFET2knmaxGiven : 1;
    unsigned HFET2kvtoGiven : 1;
    unsigned HFET2lambdaGiven : 1;
    unsigned HFET2mGiven : 1;
    unsigned HFET2mcGiven : 1;
    unsigned HFET2muGiven : 1;
    unsigned HFET2nGiven : 1;
    unsigned HFET2nmaxGiven : 1;
    unsigned HFET2pGiven : 1;
    unsigned HFET2rdGiven : 1;
    unsigned HFET2rdiGiven : 1;
    unsigned HFET2rsGiven : 1;
    unsigned HFET2rsiGiven : 1;
    unsigned HFET2sigma0Given : 1;
    unsigned HFET2vsGiven : 1;
    unsigned HFET2vsigmaGiven : 1;
    unsigned HFET2vsigmatGiven : 1;
    unsigned HFET2vt1Given : 1;
    unsigned HFET2vt2Given : 1;
    unsigned HFET2vtoGiven : 1;

} HFET2model;


#ifndef NHFET
#define NHFET 1
#define PHFET -1
#endif /*NMF*/

/* device parameters */
enum {
    HFET2_LENGTH = 1,
    HFET2_WIDTH,
    HFET2_IC_VDS,
    HFET2_IC_VGS,
    HFET2_IC,
    HFET2_OFF,
    HFET2_CS,
    HFET2_POWER,
    HFET2_TEMP,
    HFET2_DTEMP,
    HFET2_M,
};

/* model parameters */
enum {
    HFET2_MOD_NHFET = 101,
    HFET2_MOD_PHFET,
    HFET2_MOD_CF,
    HFET2_MOD_D1,
    HFET2_MOD_D2,
    HFET2_MOD_DEL,
    HFET2_MOD_DELTA,
    HFET2_MOD_DELTAD,
    HFET2_MOD_DI,
    HFET2_MOD_EPSI,
    HFET2_MOD_ETA,
    HFET2_MOD_ETA1,
    HFET2_MOD_ETA2,
    HFET2_MOD_GAMMA,
    HFET2_MOD_GGR,
    HFET2_MOD_JS,
    HFET2_MOD_KLAMBDA,
    HFET2_MOD_KMU,
    HFET2_MOD_KNMAX,
    HFET2_MOD_KVTO,
    HFET2_MOD_LAMBDA,
    HFET2_MOD_M,
    HFET2_MOD_MC,
    HFET2_MOD_MU,
    HFET2_MOD_N,
    HFET2_MOD_NMAX,
    HFET2_MOD_P,
    HFET2_MOD_RD,
    HFET2_MOD_RDI,
    HFET2_MOD_RS,
    HFET2_MOD_RSI,
    HFET2_MOD_SIGMA0,
    HFET2_MOD_VS,
    HFET2_MOD_VSIGMA,
    HFET2_MOD_VSIGMAT,
    HFET2_MOD_VT1,
    HFET2_MOD_VT2,
    HFET2_MOD_VTO,
    HFET2_MOD_TYPE,
};

/* device questions */

enum {
    HFET2_DRAINNODE = 201,
    HFET2_GATENODE,
    HFET2_SOURCENODE,
    HFET2_DRAINPRIMENODE,
    HFET2_SOURCEPRIMENODE,
    HFET2_VGS,
    HFET2_VGD,
    HFET2_CG,
    HFET2_CD,
    HFET2_CGD,
    HFET2_GM,
    HFET2_GDS,
    HFET2_GGS,
    HFET2_GGD,
    HFET2_QGS,
    HFET2_CQGS,
    HFET2_QGD,
    HFET2_CQGD,
};

/* model questions */

enum {
    HFET2_MOD_DRAINCONDUCT = 301,
    HFET2_MOD_SOURCECONDUCT,
    HFET2_MOD_DEPLETIONCAP,
    HFET2_MOD_VCRIT,
};

#define CF      (model->HFET2cf)
#define D1      (model->HFET2d1)
#define D2      (model->HFET2d2)
#define DEL     (model->HFET2del)
#define DELTA   (model->HFET2delta)
#define DELTAD  (model->HFET2deltad)
#define DI      (model->HFET2di)
#define EPSI    (model->HFET2epsi)
#define ETA     (model->HFET2eta)
#define ETA1    (model->HFET2eta1)
#define ETA2    (model->HFET2eta2)
#define GAMMA   (model->HFET2gamma)
#define GGR     (model->HFET2ggr)
#define JS      (model->HFET2js)
#define KLAMBDA (model->HFET2klambda)
#define KMU     (model->HFET2kmu)
#define KNMAX   (model->HFET2knmax)
#define KVTO    (model->HFET2kvto)
#define LAMBDA  (model->HFET2lambda)
#define M       (model->HFET2m)
#define MC      (model->HFET2mc)
#define MU      (model->HFET2mu)
#define N       (model->HFET2n)
#define NMAX    (model->HFET2nmax)
#define PP      (model->HFET2p)
#define RD      (model->HFET2rd)
#define RDI     (model->HFET2rdi)
#define RS      (model->HFET2rs)
#define RSI     (model->HFET2rsi)
#define SIGMA0  (model->HFET2sigma0)
#define TYPE    (model->HFET2type)
#define VS      (model->HFET2vs)
#define VSIGMA  (model->HFET2vsigma)
#define VSIGMAT (model->HFET2vsigmat)
#define HFET2_VT1 (model->HFET2vt1)  /* Fix a redefinition in include files */
#define VT2     (model->HFET2vt2)
#define VTO     (model->HFET2vto)

#define DELTA2  (model->HFET2deltaSqr)

#define GCHI0   (here->HFET2gchi0)
#define GGRLW   (here->HFET2ggrlw)
#define JSLW    (here->HFET2jslw)
#define IMAX    (here->HFET2imax)
#define L       (here->HFET2length)
#define N0      (here->HFET2n0)
#define N01     (here->HFET2n01)
#define N02     (here->HFET2n02)
#define TEMP    (here->HFET2temp)
#define TLAMBDA (here->HFET2tLambda)
#define TMU     (here->HFET2tMu)
#define TNMAX   (here->HFET2tNmax)
#define TVTO    (here->HFET2tVto)
#define VCRIT   (here->HFET2vcrit)
#define W       (here->HFET2width)

#include "hfet2ext.h"

#endif /*HFET2*/
