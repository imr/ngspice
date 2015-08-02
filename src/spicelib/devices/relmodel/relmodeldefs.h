/**********
Author: Francesco Lannutti - July 2015
**********/

#ifndef RELMODEL
#define RELMODEL

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

typedef struct sRELMODELrelList {
    double time ;
    double deltaVth ;
    struct sRELMODELrelList *next ;
} RELMODELrelList ;

typedef struct sRELMODELrelStruct {
    double time ;
    double offsetTime ;
    double deltaVth ;
    double deltaVthMax ;
    double t_star ;
    int IsON ;
    RELMODELrelList *deltaVthList ;
    unsigned int semiPeriods ;
} RELMODELrelStruct ;

typedef struct sRELMODELmodel
{
    int RELMODELmodType ;
    struct sRELMODELmodel *RELMODELnextModel ;
    IFuid RELMODELmodName ;

    double RELMODELk_b ;
    double RELMODELh_cut ;
    double RELMODELnts ;
    double RELMODELeps_hk ;
    double RELMODELeps_SiO2 ;
    double RELMODELm_star ;
    double RELMODELw ;
    double RELMODELtau_0 ;
    double RELMODELbeta ;
    double RELMODELtau_e ;
    double RELMODELbeta1 ;
    int RELMODELrelmodel ;

    unsigned RELMODELk_bGiven : 1 ;
    unsigned RELMODELh_cutGiven : 1 ;
    unsigned RELMODELntsGiven : 1 ;
    unsigned RELMODELeps_hkGiven : 1 ;
    unsigned RELMODELeps_SiO2Given : 1 ;
    unsigned RELMODELm_starGiven : 1 ;
    unsigned RELMODELwGiven : 1 ;
    unsigned RELMODELtau_0Given : 1 ;
    unsigned RELMODELbetaGiven : 1 ;
    unsigned RELMODELtau_eGiven : 1 ;
    unsigned RELMODELbeta1Given : 1 ;
    unsigned RELMODELrelmodelGiven : 1 ;
} RELMODELmodel ;


/* Global parameters */
#define RELMODEL_MOD_KB       1
#define RELMODEL_MOD_HCUT     2
#define RELMODEL_MOD_NTS      3
#define RELMODEL_MOD_EPSHK    4
#define RELMODEL_MOD_EPSSIO2  5
#define RELMODEL_MOD_MSTAR    6
#define RELMODEL_MOD_W        7
#define RELMODEL_MOD_TAU0     8
#define RELMODEL_MOD_BETA     9
#define RELMODEL_MOD_TAUE     10
#define RELMODEL_MOD_BETA1    11
#define RELMODEL_MOD_RELMODEL 12


#include "relmodelext.h"
#endif /* RELMODEL */
