/**********
Author: Francesco Lannutti - August 2014
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "relmodeldefs.h"
#include "ngspice/suffix.h"

IFparm RELMODELpTable [] = { /* parameters */
} ;

IFparm RELMODELmPTable [] = { /* model parameters */
IOP ("t0", RELMODEL_MOD_T0, IF_REAL, "T0"),
IOP ("ea", RELMODEL_MOD_EA, IF_REAL, "Activation Energy [eV]"),
IOP ("k1_2", RELMODEL_MOD_K1_2, IF_REAL, "K1^2"),
IOP ("e_01", RELMODEL_MOD_E_01, IF_REAL, "E_01"),
IOP ("x1", RELMODEL_MOD_X1, IF_REAL, "X1"),
IOP ("x2", RELMODEL_MOD_X2, IF_REAL, "X2"),
IOP ("t_clk", RELMODEL_MOD_T_CLK, IF_REAL, "Period"),
IOP ("alfa", RELMODEL_MOD_ALFA, IF_REAL, "Switching Activity"),
IP  ("relmodel", RELMODEL_MOD_RELMODEL,  IF_FLAG, "Flag to indicate RELMODEL"),
} ;

char *RELMODELnames [] = {
} ;

int RELMODELnSize = NUMELEMS(RELMODELnames) ;
int RELMODELpTSize = NUMELEMS(RELMODELpTable) ;
int RELMODELmPTSize = NUMELEMS(RELMODELmPTable) ;
int RELMODELiSize = 0 ;
int RELMODELmSize = sizeof(RELMODELmodel) ;
