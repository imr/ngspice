/*============================================================================
FILE    MIF.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file allocates globals used by various packages, including MIF.

INTERFACES

    g_mif_info

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/
 

#include "ngspice/mif.h"

int MIFiSize = sizeof(MIFinstance);  
int MIFmSize = sizeof(MIFmodel); 


/* Allocate global used to pass info on analysis type, etc. from */
/* SPICE to the MIF load routine                                 */


/* This must be initialized so that EVTfindvec can check for */
/* NULL pointer in g_mif_info.ckt */

Mif_Info_t  g_mif_info = {
  { MIF_FALSE, MIF_FALSE, MIF_DC, MIF_ANALOG, 0.0,},
  NULL,
  NULL,
  NULL,
  { 0.0, 0.0,},
  { MIF_FALSE, MIF_FALSE,},
};
