/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hisimhv.h

 DATE : 2013.04.30

 released by
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "hsmhvdef.h"
#include "ngspice/cktdefs.h"

#ifndef _HiSIMHV_H
#define _HiSIMHV_H

/* return value */
#ifndef OK
#define HiSIM_OK        0
#define HiSIM_ERROR     1
#else
#define HiSIM_OK        OK
#define HiSIM_ERROR     E_PANIC
#endif

/* MOS type */
#ifndef NMOS
#define NMOS     1
#define PMOS    -1
#endif

/* device working mode */
#ifndef CMI_NORMAL_MODE
#define HiSIM_NORMAL_MODE    1
#define HiSIM_REVERSE_MODE  -1
#else
#define HiSIM_NORMAL_MODE  CMI_NORMAL_MODE
#define HiSIM_REVERSE_MODE CMI_REVERSE_MODE
#endif

/* others */
#ifndef NULL
#define NULL            0
#endif

#define HiSIM_FALSE     0
#define HiSIM_TRUE      1

#ifndef return_if_error
#define return_if_error(s) { int error = s; if(error) return(error); }
#endif

extern int HSMHVevaluate
(
 double ivds,
 double ivgs,
 double ivbs,
 double ivdsi,
 double ivgsi,
 double ivbsi,
 double vbs_jct,
 double vbd_jct,
 double vsubs,
 double deltemp,
 HSMHVinstance *here,
 HSMHVmodel    *model,
 CKTcircuit   *ckt
 ) ;

#endif /* _HiSIMHV_H */
