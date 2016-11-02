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
    double Vstress ;
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

    int RELMODELtype ;

    double RELMODELalpha_new ;
    double RELMODELb_new ;
    double RELMODELbeta_new ;
    double RELMODELe0_new ;
    double RELMODELk_new ;
    double RELMODELkb_new ;
    double RELMODELtau_c_fast_new ;
    double RELMODELtau_c_slow_new ;
    double RELMODELtau_e_fast_new ;
    double RELMODELtau_e_slow_new ;
    double RELMODELt_r_new ;
    double RELMODELt_s_new ;

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

    unsigned RELMODELtypeGiven : 1 ;

    unsigned RELMODELalpha_newGiven : 1 ;
    unsigned RELMODELb_newGiven : 1 ;
    unsigned RELMODELbeta_newGiven : 1 ;
    unsigned RELMODELe0_newGiven : 1 ;
    unsigned RELMODELk_newGiven : 1 ;
    unsigned RELMODELkb_newGiven : 1 ;
    unsigned RELMODELtau_c_fast_newGiven : 1 ;
    unsigned RELMODELtau_c_slow_newGiven : 1 ;
    unsigned RELMODELtau_e_fast_newGiven : 1 ;
    unsigned RELMODELtau_e_slow_newGiven : 1 ;
    unsigned RELMODELt_r_newGiven : 1 ;
    unsigned RELMODELt_s_newGiven : 1 ;
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

#define RELMODEL_MOD_TYPE 13

#define RELMODEL_MOD_ALPHA_NEW      14
#define RELMODEL_MOD_B_NEW          15
#define RELMODEL_MOD_BETA_NEW       16
#define RELMODEL_MOD_E0_NEW         17
#define RELMODEL_MOD_K_NEW          18
#define RELMODEL_MOD_KB_NEW         19
#define RELMODEL_MOD_TAU_C_FAST_NEW 20
#define RELMODEL_MOD_TAU_C_SLOW_NEW 21
#define RELMODEL_MOD_TAU_E_FAST_NEW 22
#define RELMODEL_MOD_TAU_E_SLOW_NEW 23
#define RELMODEL_MOD_T_R_NEW        24
#define RELMODEL_MOD_T_S_NEW        25


#include "relmodelext.h"
#endif /* RELMODEL */
