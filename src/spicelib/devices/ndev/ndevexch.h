#ifndef _netexchange_h_
#define _netexchange_h_

#define NG_QUERY   "This is ngspice. Are you ready?"
#define NDEV_REPLY "Waiting orders!"
#define NG_STOP    "Ngspice finished, goodbye."

#define NDEV_LOAD               0x0001
#define NDEV_ACCEPT             0x0002
#define NDEV_CONVERGINCE_TEST   0x0004
#define NDEV_TRUNCATION_ERROR   0x0008

#define NDEV_TEMPERATURE        0x1000
#define NDEV_AC_LOAD            0x0010
#define NDEV_PZ_LOAD            0x0020

#ifndef ngspice_CKTDEFS_H
/* defines for CKTmode */
/* this should be the same as cktdefs.h */
/* old 'mode' parameters */
#define MODE 0x3
#define MODETRAN 0x1
#define MODEAC 0x2

/* old 'modedc' parameters */
#define MODEDC 0x70
#define MODEDCOP 0x10
#define MODETRANOP 0x20
#define MODEDCTRANCURVE 0x40

/* old 'initf' parameters */
#define INITF 0x3f00
#define MODEINITFLOAT 0x100
#define MODEINITJCT 0x200
#define MODEINITFIX 0x400
#define MODEINITSMSIG 0x800
#define MODEINITTRAN 0x1000
#define MODEINITPRED 0x2000

/* old 'nosolv' paramater */
#define MODEUIC 0x10000l
#endif

typedef struct {
char NDEVname[32];
int  term;
}sDeviceinfo;

 
typedef struct {
int  pin;
char name[32];
double V,I;
double V_old;
double dI_dV[7];
} sPINinfo;

typedef struct {
long   DEV_CALL;
long   CKTmode;
double time;
double dt;
double dt_old;
double omega;
int    accept_flag;
int    convergence_flag;
}sCKTinfo;

#endif
