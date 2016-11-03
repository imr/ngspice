/**********
Author: Francesco Lannutti - July 2015
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "relmodeldefs.h"
#include "ngspice/suffix.h"

IFparm RELMODELpTable [] = { /* parameters */
} ;

IFparm RELMODELmPTable [] = { /* model parameters */
IOP ("type", RELMODEL_MOD_TYPE, IF_INTEGER, "Aging Model Type"),

IOP ("k_b", RELMODEL_MOD_KB, IF_REAL, "Boltzmann Constant"),
IOP ("h_cut", RELMODEL_MOD_HCUT, IF_REAL, "h_cut"),
IOP ("nts", RELMODEL_MOD_NTS, IF_REAL, "Nts"),
IOP ("eps_hk", RELMODEL_MOD_EPSHK, IF_REAL, "Dielectric Constant High-K"),
IOP ("eps_SiO2", RELMODEL_MOD_EPSSIO2, IF_REAL, "Dielectric Constant SiO2"),
IOP ("m_star", RELMODEL_MOD_MSTAR, IF_REAL, "m_star"),
IOP ("w", RELMODEL_MOD_W, IF_REAL, "W"),
IOP ("tau_0", RELMODEL_MOD_TAU0, IF_REAL, "tau_0"),
IOP ("beta", RELMODEL_MOD_BETA, IF_REAL, "beta"),
IOP ("tau_e", RELMODEL_MOD_TAUE, IF_REAL, "tau_e"),
IOP ("beta1", RELMODEL_MOD_BETA1, IF_REAL, "beta1"),

IOP ("alpha_new", RELMODEL_MOD_ALPHA_NEW, IF_REAL, "alpha"),
IOP ("b_new", RELMODEL_MOD_B_NEW, IF_REAL, "b"),
IOP ("beta_new", RELMODEL_MOD_BETA_NEW, IF_REAL, "beta"),
IOP ("e0_new", RELMODEL_MOD_E0_NEW, IF_REAL, "e0"),
IOP ("k_new", RELMODEL_MOD_K_NEW, IF_REAL, "k"),
IOP ("kb_new", RELMODEL_MOD_KB_NEW, IF_REAL, "Boltzmann Constant"),
IOP ("tau_c_fast_new", RELMODEL_MOD_TAU_C_FAST_NEW, IF_REAL, "tau_c_fast"),
IOP ("tau_c_slow_new", RELMODEL_MOD_TAU_C_SLOW_NEW, IF_REAL, "tau_c_slow"),
IOP ("tau_e_fast_new", RELMODEL_MOD_TAU_E_FAST_NEW, IF_REAL, "tau_e_fast"),
IOP ("tau_e_slow_new", RELMODEL_MOD_TAU_E_SLOW_NEW, IF_REAL, "tau_e_slow"),
IOP ("t_r_new", RELMODEL_MOD_T_R_NEW, IF_REAL, "t_r"),
IOP ("t_s_new", RELMODEL_MOD_T_S_NEW, IF_REAL, "t_s"),

IP  ("relmodel", RELMODEL_MOD_RELMODEL,  IF_FLAG, "Flag to indicate RELMODEL")
} ;

char *RELMODELnames [] = {
} ;

int RELMODELnSize = NUMELEMS(RELMODELnames) ;
int RELMODELpTSize = NUMELEMS(RELMODELpTable) ;
int RELMODELmPTSize = NUMELEMS(RELMODELmPTable) ;
int RELMODELiSize = 0 ;
int RELMODELmSize = sizeof(RELMODELmodel) ;
