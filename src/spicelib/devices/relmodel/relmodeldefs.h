/**********
Author: Francesco Lannutti - August 2014
**********/

#ifndef RELMODEL
#define RELMODEL

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

typedef struct sRELMODELmodel
{
    int RELMODELmodType ;
    struct sRELMODELmodel *RELMODELnextModel ;
    IFuid RELMODELmodName ;

    double RELMODELt0 ;
    double RELMODELea ;
    double RELMODELk1_2 ;
    double RELMODELe_01 ;
    double RELMODELx1 ;
    double RELMODELx2 ;
    double RELMODELt_clk ;
    double RELMODELalfa ;
    int RELMODELrelmodel ;

    unsigned RELMODELt0Given : 1 ;
    unsigned RELMODELeaGiven : 1 ;
    unsigned RELMODELk1_2Given : 1 ;
    unsigned RELMODELe_01Given : 1 ;
    unsigned RELMODELx1Given : 1 ;
    unsigned RELMODELx2Given : 1 ;
    unsigned RELMODELt_clkGiven : 1 ;
    unsigned RELMODELalfaGiven : 1 ;
    unsigned RELMODELrelmodelGiven : 1 ;
} RELMODELmodel ;


/* Global parameters */
#define RELMODEL_MOD_T0    1
#define RELMODEL_MOD_EA    2
#define RELMODEL_MOD_K1_2  3
#define RELMODEL_MOD_E_01  4
#define RELMODEL_MOD_X1    5
#define RELMODEL_MOD_X2    6
#define RELMODEL_MOD_T_CLK 7
#define RELMODEL_MOD_ALFA  8
#define RELMODEL_MOD_RELMODEL 9


#include "relmodelext.h"
#endif /* RELMODEL */
