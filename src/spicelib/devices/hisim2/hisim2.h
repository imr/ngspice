/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM
 ( VERSION : 2  SUBVERSION : 8  REVISION : 0 )
 
 FILE : hisim2.h

 Date : 2014.6.5

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "hsm2def.h"
#include "ngspice/cktdefs.h"

#ifndef _HiSIM2_H
#define _HiSIM2_H

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

extern int HSM2evaluate
(
 double ivds,
 double ivgs,
 double ivbs,
 double vbs_jct,
 double vbd_jct,
 HSM2instance *here,
 HSM2model    *model,
 CKTcircuit   *ckt
 ) ;

#endif /* _HiSIM2_H */
