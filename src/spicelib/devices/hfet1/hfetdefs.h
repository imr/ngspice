/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#ifndef HFETA
#define HFETA

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"

#define HFETAnumStates 24

typedef struct sHFETAinstance {
    struct sHFETAmodel *HFETAmodPtr;
    struct sHFETAinstance *HFETAnextInstance;
    IFuid HFETAname;
    int HFETAowner;	  /* number of owner process */
    int HFETAstate;   /* index into state table for this device */
    
    int HFETAdrainNode;
    int HFETAgateNode;
    int HFETAsourceNode;
    int HFETAdrainPrimeNode;
    int HFETAgatePrimeNode;
    int HFETAsourcePrimeNode;
    int HFETAdrainPrmPrmNode;
    int HFETAsourcePrmPrmNode;
    double HFETAlength;
    double HFETAwidth;
    double HFETAm;
    double HFETAicVDS;
    double HFETAicVGS;
    double HFETAtemp;
    double HFETAdtemp;
    double HFETAtVto;
    double HFETAtMu;
    double HFETAtLambda;
    double HFETAtLambdahf;
    double *HFETAdrainDrainPrimePtr;
    double *HFETAgatePrimeDrainPrimePtr;
    double *HFETAgatePrimeSourcePrimePtr;
    double *HFETAsourceSourcePrimePtr;
    double *HFETAdrainPrimeDrainPtr;
    double *HFETAdrainPrimeGatePrimePtr;
    double *HFETAdrainPrimeSourcePrimePtr;
    double *HFETAsourcePrimeGatePrimePtr;
    double *HFETAsourcePrimeSourcePtr;
    double *HFETAsourcePrimeDrainPrimePtr;
    double *HFETAdrainDrainPtr;
    double *HFETAgatePrimeGatePrimePtr;
    double *HFETAsourceSourcePtr;
    double *HFETAdrainPrimeDrainPrimePtr;
    double *HFETAsourcePrimeSourcePrimePtr;
    double *HFETAdrainPrmPrmDrainPrmPrmPtr;
    double *HFETAdrainPrmPrmDrainPrimePtr;
    double *HFETAdrainPrimeDrainPrmPrmPtr;
    double *HFETAdrainPrmPrmGatePrimePtr;
    double *HFETAgatePrimeDrainPrmPrmPtr;
    double *HFETAsourcePrmPrmSourcePrmPrmPtr;
    double *HFETAsourcePrmPrmSourcePrimePtr;
    double *HFETAsourcePrimeSourcePrmPrmPtr;
    double *HFETAsourcePrmPrmGatePrimePtr;
    double *HFETAgatePrimeSourcePrmPrmPtr;
    double *HFETAgateGatePtr;
    double *HFETAgateGatePrimePtr;
    double *HFETAgatePrimeGatePtr;
    
    
#define HFETAvgs   HFETAstate 
#define HFETAvgd   HFETAstate+1 
#define HFETAcg    HFETAstate+2 
#define HFETAcd    HFETAstate+3 
#define HFETAcgd   HFETAstate+4 
#define HFETAcgs   HFETAstate+5 
#define HFETAgm    HFETAstate+6 
#define HFETAgds   HFETAstate+7 
#define HFETAggs   HFETAstate+8 
#define HFETAggd   HFETAstate+9 
#define HFETAqgs   HFETAstate+10
#define HFETAcqgs  HFETAstate+11 
#define HFETAqgd   HFETAstate+12 
#define HFETAcqgd  HFETAstate+13
#define HFETAvgspp HFETAstate+14
#define HFETAggspp HFETAstate+15
#define HFETAcgspp HFETAstate+16
#define HFETAvgdpp HFETAstate+17
#define HFETAggdpp HFETAstate+18
#define HFETAcgdpp HFETAstate+19
#define HFETAqds   HFETAstate+20
#define HFETAcqds  HFETAstate+21
#define HFETAgmg   HFETAstate+22
#define HFETAgmd   HFETAstate+23



    int HFETAoff;
    unsigned HFETAlengthGiven : 1;
    unsigned HFETAwidthGiven  : 1;
    unsigned HFETAmGiven      : 1;
    unsigned HFETAicVDSGiven  : 1;
    unsigned HFETAicVGSGiven  : 1;
    unsigned HFETAtempGiven   : 1;
    unsigned HFETAdtempGiven  : 1;
    int HFETAmode;

    double HFETAn0;
    double HFETAn01;
    double HFETAn02;
    double HFETAgchi0;
    double HFETAcf;
    double HFETAis1d;
    double HFETAis2d;
    double HFETAis1s;
    double HFETAis2s;
    double HFETAiso;
    double HFETAimax;
    double HFETAvcrit;
    double HFETAdelf;
    double HFETAfgds;
    double HFETAggrwl;

} HFETAinstance ;


/* per model data */

typedef struct sHFETAmodel {
    int HFETAmodType;
    struct sHFETAmodel *HFETAnextModel; 
    HFETAinstance *HFETAinstances;
    IFuid HFETAmodName;
    int HFETAtype;
    int HFETAgatemod;
    
    double HFETAthreshold;
    double HFETAlambda;
    double HFETAeta;
    double HFETAm;
    double HFETAmc;
    double HFETAgamma;
    double HFETAsigma0;
    double HFETAvsigmat;
    double HFETAvsigma;
    double HFETAmu;
    double HFETAdi;
    double HFETAdelta;
    double HFETAvs;
    double HFETAnmax;
    double HFETAdeltad;
    double HFETAjs1d;
    double HFETAjs2d;
    double HFETAjs1s;
    double HFETAjs2s;
    double HFETAm1d;
    double HFETAm2d;
    double HFETAm1s;
    double HFETAm2s;
    double HFETArd;
    double HFETArs;
    double HFETArg;
    double HFETArdi;
    double HFETArsi;
    double HFETArgs;
    double HFETArgd;
    double HFETAri;
    double HFETArf;
    double HFETAepsi;
    double HFETAa1;
    double HFETAa2;
    double HFETAmv1;
    double HFETAp;
    double HFETAkappa;
    double HFETAdelf;
    double HFETAfgds;
    double HFETAtf;
    double HFETAcds;
    double HFETAphib;
    double HFETAtalpha;
    double HFETAmt1;
    double HFETAmt2;
    double HFETAck1;
    double HFETAck2;
    double HFETAcm1;
    double HFETAcm2;
    double HFETAcm3;
    double HFETAastar;
    double HFETAeta1;
    double HFETAd1;
    double HFETAvt1;
    double HFETAeta2;
    double HFETAd2;
    double HFETAvt2;
    double HFETAggr;
    double HFETAdel;
    double HFETAklambda;
    double HFETAkmu;
    double HFETAkvto;
    
    double HFETAdrainConduct;
    double HFETAsourceConduct;
    double HFETAgateConduct;
    double HFETAgi;
    double HFETAgf;
    double HFETAdeltaSqr;

    unsigned HFETAgatemodGiven:1;
    unsigned HFETAthresholdGiven:1;
    unsigned HFETAlambdaGiven:1;
    unsigned HFETAetaGiven:1;
    unsigned HFETAmGiven:1;
    unsigned HFETAmcGiven:1;
    unsigned HFETAgammaGiven:1;
    unsigned HFETAsigma0Given:1;
    unsigned HFETAvsigmatGiven:1;
    unsigned HFETAvsigmaGiven:1;
    unsigned HFETAmuGiven:1;
    unsigned HFETAdiGiven:1;
    unsigned HFETAdeltaGiven:1;
    unsigned HFETAvsGiven:1;
    unsigned HFETAnmaxGiven:1;
    unsigned HFETAdeltadGiven:1;
    unsigned HFETAjs1dGiven:1;
    unsigned HFETAjs2dGiven:1;
    unsigned HFETAjs1sGiven:1;
    unsigned HFETAjs2sGiven:1;
    unsigned HFETAm1dGiven:1;
    unsigned HFETAm2dGiven:1;
    unsigned HFETAm1sGiven:1;
    unsigned HFETAm2sGiven:1;
    unsigned HFETArdGiven:1;
    unsigned HFETArsGiven:1;
    unsigned HFETArgGiven:1;
    unsigned HFETArdiGiven:1;
    unsigned HFETArsiGiven:1;
    unsigned HFETArgsGiven:1;
    unsigned HFETArgdGiven:1;
    unsigned HFETAriGiven:1;
    unsigned HFETArfGiven:1;
    unsigned HFETAepsiGiven:1;
    unsigned HFETAa1Given:1;
    unsigned HFETAa2Given:1;
    unsigned HFETAmv1Given:1;
    unsigned HFETApGiven:1;
    unsigned HFETAkappaGiven:1;
    unsigned HFETAdelfGiven:1;
    unsigned HFETAfgdsGiven:1;
    unsigned HFETAtfGiven:1;
    unsigned HFETAcdsGiven:1;
    unsigned HFETAphibGiven:1;
    unsigned HFETAtalphaGiven:1;
    unsigned HFETAmt1Given:1;
    unsigned HFETAmt2Given:1;
    unsigned HFETAck1Given:1;
    unsigned HFETAck2Given:1;
    unsigned HFETAcm1Given:1;
    unsigned HFETAcm2Given:1;
    unsigned HFETAcm3Given:1;
    unsigned HFETAastarGiven:1;
    unsigned HFETAeta1Given:1;
    unsigned HFETAd1Given:1;
    unsigned HFETAvt1Given:1;
    unsigned HFETAeta2Given:1;
    unsigned HFETAd2Given:1;
    unsigned HFETAvt2Given:1;
    unsigned HFETAggrGiven:1;
    unsigned HFETAdelGiven:1;
    unsigned HFETAklambdaGiven:1;
    unsigned HFETAkmuGiven:1;
    unsigned HFETAkvtoGiven:1;
    
} HFETAmodel;

#ifndef NHFET
#define NHFET  1
#define PHFET -1
#endif

/* device parameters */
#define HFETA_LENGTH   1
#define HFETA_WIDTH    2
#define HFETA_IC_VDS   3
#define HFETA_IC_VGS   4
#define HFETA_TEMP     5
#define HFETA_IC       6
#define HFETA_OFF      7
#define HFETA_CS       8
#define HFETA_POWER    9
#define HFETA_DTEMP   10
#define HFETA_M       11

/* model parameters */
#define HFETA_MOD_VTO      101
#define HFETA_MOD_LAMBDA   102
#define HFETA_MOD_RD       103
#define HFETA_MOD_RS       104
#define HFETA_MOD_RG       105
#define HFETA_MOD_RGS      106
#define HFETA_MOD_RGD      107
#define HFETA_MOD_RI       108
#define HFETA_MOD_RF       109
#define HFETA_MOD_ETA      110
#define HFETA_MOD_M        111
#define HFETA_MOD_MC       112
#define HFETA_MOD_GAMMA    113
#define HFETA_MOD_SIGMA0   114
#define HFETA_MOD_VSIGMAT  115
#define HFETA_MOD_VSIGMA   116
#define HFETA_MOD_MU       117
#define HFETA_MOD_DI       118
#define HFETA_MOD_DELTA    119
#define HFETA_MOD_VS       120
#define HFETA_MOD_NMAX     121
#define HFETA_MOD_DELTAD   122
#define HFETA_MOD_JS1D     123
#define HFETA_MOD_JS2D     124
#define HFETA_MOD_JS1S     125
#define HFETA_MOD_JS2S     126
#define HFETA_MOD_M1D      127
#define HFETA_MOD_M2D      128
#define HFETA_MOD_M1S      129
#define HFETA_MOD_M2S      130
#define HFETA_MOD_EPSI     132
#define HFETA_MOD_RDI      133
#define HFETA_MOD_RSI      134
#define HFETA_MOD_A1       135
#define HFETA_MOD_A2       136
#define HFETA_MOD_MV1      137
#define HFETA_MOD_P        138
#define HFETA_MOD_KAPPA    139
#define HFETA_MOD_DELF     140
#define HFETA_MOD_FGDS     141
#define HFETA_MOD_TF       142
#define HFETA_MOD_CDS      143
#define HFETA_MOD_PHIB     144
#define HFETA_MOD_TALPHA   145
#define HFETA_MOD_MT1      146
#define HFETA_MOD_MT2      147
#define HFETA_MOD_CK1      148
#define HFETA_MOD_CK2      149
#define HFETA_MOD_CM1      150
#define HFETA_MOD_CM2      151
#define HFETA_MOD_CM3      152
#define HFETA_MOD_ASTAR    153
#define HFETA_MOD_ETA1     154
#define HFETA_MOD_D1       155
#define HFETA_MOD_VT1      156
#define HFETA_MOD_ETA2     157
#define HFETA_MOD_D2       158
#define HFETA_MOD_VT2      159
#define HFETA_MOD_GGR      160
#define HFETA_MOD_DEL      161
#define HFETA_MOD_GATEMOD  162
#define HFETA_MOD_KLAMBDA  163
#define HFETA_MOD_KMU      164
#define HFETA_MOD_KVTO     165
#define HFETA_MOD_NHFET    166
#define HFETA_MOD_PHFET    167
#define HFETA_MOD_TYPE	  168

/* device questions */

#define HFETA_DRAINNODE       201
#define HFETA_GATENODE        202
#define HFETA_SOURCENODE      203
#define HFETA_DRAINPRIMENODE  204
#define HFETA_SOURCEPRIMENODE 205

#define HFETA_VGS         206
#define HFETA_VGD         207
#define HFETA_CG          208
#define HFETA_CD          209
#define HFETA_CGD         210
#define HFETA_GM          211
#define HFETA_GDS         212
#define HFETA_GGS         213
#define HFETA_GGD         214
#define HFETA_QGS         215
#define HFETA_CQGS        216
#define HFETA_QGD         217
#define HFETA_CQGD        218

/* model questions */

#define HFETA_MOD_DRAINCONDUCT    301
#define HFETA_MOD_SOURCECONDUCT   302 
#define HFETA_MOD_DEPLETIONCAP    303
#define HFETA_MOD_VCRIT           304


#define L        (here->HFETAlength)
#define W        (here->HFETAwidth)
#define VTO      (model->HFETAthreshold)
#define LAMBDA   (model->HFETAlambda)
#define RDI      (model->HFETArdi)
#define RSI      (model->HFETArsi)
#define RD       (model->HFETArd)
#define RS       (model->HFETArs)
#define RG       (model->HFETArg)
#define RF       (model->HFETArf)
#define RI       (model->HFETAri)
#define RGS      (model->HFETArgs)
#define RGD      (model->HFETArgd)
#define ETA      (model->HFETAeta)
#define M        (model->HFETAm)
#define MC       (model->HFETAmc)
#define GAMMA    (model->HFETAgamma)
#define SIGMA0   (model->HFETAsigma0)
#define VSIGMAT  (model->HFETAvsigmat)
#define VSIGMA   (model->HFETAvsigma)
#define MU       (model->HFETAmu)
#define DI       (model->HFETAdi)
#define DELTAD   (model->HFETAdeltad)
#define DELTASQR (model->HFETAdeltaSqr)
#define VS       (model->HFETAvs)
#define NMAX     (model->HFETAnmax)
#define EPSI     (model->HFETAepsi)
#define JS1D     (model->HFETAjs1d)
#define JS2D     (model->HFETAjs2d)
#define JS1S     (model->HFETAjs1s)
#define JS2S     (model->HFETAjs2s)
#define M1D      (model->HFETAm1d)
#define M2D      (model->HFETAm2d)
#define M1S      (model->HFETAm1s)
#define M2S      (model->HFETAm2s)
#define ASTAR    (model->HFETAastar)
#define PHIB     (model->HFETAphib)
#define TALPHA   (model->HFETAtalpha)
#define MT1      (model->HFETAmt1)
#define MT2      (model->HFETAmt2)
#define CK1      (model->HFETAck1)
#define CK2      (model->HFETAck2)
#define CM1      (model->HFETAcm1)
#define CM2      (model->HFETAcm2)
#define CM3      (model->HFETAcm3)
#define A1       (model->HFETAa1)
#define A2       (model->HFETAa2)
#define MV1      (model->HFETAmv1)
#define PM       (model->HFETAp)
#define CDS      (model->HFETAcds)
#define ETA1     (model->HFETAeta1)
#define D1       (model->HFETAd1)
#define IN_VT1   (model->HFETAvt1)  /* VT1 was defined in termios.h */
#define ETA2     (model->HFETAeta2)
#define D2       (model->HFETAd2)
#define VT2      (model->HFETAvt2)
#define GGR      (model->HFETAggr)
#define DEL      (model->HFETAdel)
#define KLAMBDA  (model->HFETAklambda)
#define KMU      (model->HFETAkmu)
#define KVTO     (model->HFETAkvto)

#define GCHI0    (here->HFETAgchi0)
#define N0       (here->HFETAn0)
#define N01      (here->HFETAn01)
#define N02      (here->HFETAn02)
#define CF       (here->HFETAcf)
#define IMAX     (here->HFETAimax)
#define ISO      (here->HFETAiso)
#define TEMP     (here->HFETAtemp)
#define IS1D     (here->HFETAis1d)
#define IS2D     (here->HFETAis2d)
#define IS1S     (here->HFETAis1s)
#define IS2S     (here->HFETAis2s)
#define FGDS     (here->HFETAfgds)
#define DELF     (here->HFETAdelf)
#define GGRWL    (here->HFETAggrwl)
#define TLAMBDA  (here->HFETAtLambda)
#define TMU      (here->HFETAtMu)
#define TVTO     (here->HFETAtVto)

#include "hfetext.h"

#endif /*HFETA*/
