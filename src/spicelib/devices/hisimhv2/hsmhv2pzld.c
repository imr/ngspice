/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvpzld.c

 DATE : 2014.6.11

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

/**********************************************************************

The following source code, and all copyrights, trade secrets or other
intellectual property rights in and to the source code in its entirety,
is owned by the Hiroshima University and the STARC organization.

All users need to follow the "HISIM_HV Distribution Statement and
Copyright Notice" attached to HiSIM_HV model.

-----HISIM_HV Distribution Statement and Copyright Notice--------------

Software is distributed as is, completely without warranty or service
support. Hiroshima University or STARC and its employees are not liable
for the condition or performance of the software.

Hiroshima University and STARC own the copyright and grant users a perpetual,
irrevocable, worldwide, non-exclusive, royalty-free license with respect 
to the software as set forth below.   

Hiroshima University and STARC hereby disclaims all implied warranties.

Hiroshima University and STARC grant the users the right to modify, copy,
and redistribute the software and documentation, both within the user's
organization and externally, subject to the following restrictions

1. The users agree not to charge for Hiroshima University and STARC code
itself but may charge for additions, extensions, or support.

2. In any product based on the software, the users agree to acknowledge
Hiroshima University and STARC that developed the software. This
acknowledgment shall appear in the product documentation.

3. The users agree to reproduce any copyright notice which appears on
the software on any copy or modification of such made available
to others."

Toshimasa Asahara, President, Hiroshima University
Mitiko Miura-Mattausch, Professor, Hiroshima University
Katsuhiro Shimohigashi, President&CEO, STARC
June 2008 (revised October 2011) 
*************************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "hsmhv2def.h"
#include "ngspice/suffix.h"

int HSMHV2pzLoad(
     GENmodel *inModel,
     CKTcircuit *ckt,
     SPcomplex *s)
{
  HSMHV2model *model = (HSMHV2model*)inModel;
  HSMHV2instance *here;
  int flg_nqs =0 ;

  NG_IGNORE(ckt);

#define dNode        0
#define dNodePrime   1
#define gNode        2
#define gNodePrime   3
#define sNode        4
#define sNodePrime   5
#define bNodePrime   6
#define bNode        7 
#define dbNode       8
#define sbNode       9
#define tempNode    10
#define qiNode      11
#define qbNode      12

  
  for ( ;model != NULL ;model = HSMHV2nextModel(model)) {
    for ( here = HSMHV2instances(model);here!= NULL ;
	  here = HSMHV2nextInstance(here)) {

      flg_nqs = model->HSMHV2_conqs ;

      /* stamp matrix */
     
      /*drain*/
      *(here->HSMHV2DdPtr) += here->HSMHV2_ydc_d[dNode] + here->HSMHV2_ydyn_d[dNode] * s->real;
      *(here->HSMHV2DdPtr +1) += here->HSMHV2_ydyn_d[dNode] * s->imag;
      *(here->HSMHV2DdpPtr) += here->HSMHV2_ydc_d[dNodePrime] + here->HSMHV2_ydyn_d[dNodePrime] * s->real;
      *(here->HSMHV2DdpPtr +1) += here->HSMHV2_ydyn_d[dNodePrime] * s->imag;
      *(here->HSMHV2DgpPtr) += here->HSMHV2_ydc_d[gNodePrime] + here->HSMHV2_ydyn_d[gNodePrime] * s->real;
      *(here->HSMHV2DgpPtr +1) += here->HSMHV2_ydyn_d[gNodePrime] * s->imag;
      *(here->HSMHV2DsPtr) += here->HSMHV2_ydc_d[sNode] + here->HSMHV2_ydyn_d[sNode] * s->real;
      *(here->HSMHV2DsPtr +1) += here->HSMHV2_ydyn_d[sNode] * s->imag;
      *(here->HSMHV2DbpPtr) += here->HSMHV2_ydc_d[bNodePrime] + here->HSMHV2_ydyn_d[bNodePrime] * s->real;
      *(here->HSMHV2DbpPtr +1) += here->HSMHV2_ydyn_d[bNodePrime] * s->imag;
      *(here->HSMHV2DdbPtr) += here->HSMHV2_ydc_d[dbNode] + here->HSMHV2_ydyn_d[dbNode] * s->real;
      *(here->HSMHV2DdbPtr +1) += here->HSMHV2_ydyn_d[dbNode] * s->imag;
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2DtempPtr) += model->HSMHV2_type * (here->HSMHV2_ydc_d[tempNode] + here->HSMHV2_ydyn_d[tempNode] * s->real);
        *(here->HSMHV2DtempPtr +1) += model->HSMHV2_type * here->HSMHV2_ydyn_d[tempNode] * s->imag;
      }

      /*drain prime*/
      *(here->HSMHV2DPdPtr) +=  here->HSMHV2_ydc_dP[dNode] + here->HSMHV2_ydyn_dP[dNode] * s->real;
      *(here->HSMHV2DPdPtr +1) +=  here->HSMHV2_ydyn_dP[dNode] * s->imag;
      *(here->HSMHV2DPdpPtr) +=  here->HSMHV2_ydc_dP[dNodePrime] + here->HSMHV2_ydyn_dP[dNodePrime] * s->real;
      *(here->HSMHV2DPdpPtr +1) +=  here->HSMHV2_ydyn_dP[dNodePrime] * s->imag;
      *(here->HSMHV2DPgpPtr) +=  here->HSMHV2_ydc_dP[gNodePrime] + here->HSMHV2_ydyn_dP[gNodePrime] * s->real;
      *(here->HSMHV2DPgpPtr +1) +=  here->HSMHV2_ydyn_dP[gNodePrime] * s->imag;
      *(here->HSMHV2DPsPtr) +=  here->HSMHV2_ydc_dP[sNode] + here->HSMHV2_ydyn_dP[sNode] * s->real;
      *(here->HSMHV2DPsPtr +1) += here->HSMHV2_ydyn_dP[sNode] * s->imag;
      *(here->HSMHV2DPspPtr) +=  here->HSMHV2_ydc_dP[sNodePrime] + here->HSMHV2_ydyn_dP[sNodePrime] * s->real;
      *(here->HSMHV2DPspPtr +1) +=  here->HSMHV2_ydyn_dP[sNodePrime] * s->imag;
      *(here->HSMHV2DPbpPtr) +=  here->HSMHV2_ydc_dP[bNodePrime] + here->HSMHV2_ydyn_dP[bNodePrime] * s->real;
      *(here->HSMHV2DPbpPtr +1) +=  here->HSMHV2_ydyn_dP[bNodePrime] * s->imag;
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2DPtempPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_dP[tempNode] + here->HSMHV2_ydyn_dP[tempNode] * s->real);
        *(here->HSMHV2DPtempPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_dP[tempNode] * s->imag;
      }
      if (flg_nqs) {
        *(here->HSMHV2DPqiPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_dP[qiNode] + here->HSMHV2_ydyn_dP[qiNode] * s->real);
        *(here->HSMHV2DPqiPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_dP[qiNode] * s->imag;
      }


      /*gate*/
      *(here->HSMHV2GgPtr) +=  here->HSMHV2_ydc_g[gNode] + here->HSMHV2_ydyn_g[gNode] * s->real;
      *(here->HSMHV2GgPtr +1) +=  here->HSMHV2_ydyn_g[gNode] * s->imag;
      *(here->HSMHV2GgpPtr) +=  here->HSMHV2_ydc_g[gNodePrime] + here->HSMHV2_ydyn_g[gNodePrime] * s->real;
      *(here->HSMHV2GgpPtr +1) +=  here->HSMHV2_ydyn_g[gNodePrime] * s->imag;
     
      /*gate prime*/
      *(here->HSMHV2GPdPtr) +=  here->HSMHV2_ydc_gP[dNode] + here->HSMHV2_ydyn_gP[dNode] * s->real;
      *(here->HSMHV2GPdPtr +1) +=  here->HSMHV2_ydyn_gP[dNode] * s->imag;
      *(here->HSMHV2GPdpPtr) +=  here->HSMHV2_ydc_gP[dNodePrime] + here->HSMHV2_ydyn_gP[dNodePrime] * s->real;
      *(here->HSMHV2GPdpPtr +1) +=  here->HSMHV2_ydyn_gP[dNodePrime] * s->imag;
      *(here->HSMHV2GPgPtr) +=  here->HSMHV2_ydc_gP[gNode] + here->HSMHV2_ydyn_gP[gNode] * s->real;
      *(here->HSMHV2GPgPtr +1) +=  here->HSMHV2_ydyn_gP[gNode] * s->imag;
      *(here->HSMHV2GPgpPtr) +=  here->HSMHV2_ydc_gP[gNodePrime] + here->HSMHV2_ydyn_gP[gNodePrime] * s->real;
      *(here->HSMHV2GPgpPtr +1) +=  here->HSMHV2_ydyn_gP[gNodePrime] * s->imag;
      *(here->HSMHV2GPsPtr) +=  here->HSMHV2_ydc_gP[sNode] + here->HSMHV2_ydyn_gP[sNode] * s->real;
      *(here->HSMHV2GPsPtr +1) +=  here->HSMHV2_ydyn_gP[sNode] * s->imag;
      *(here->HSMHV2GPspPtr) +=  here->HSMHV2_ydc_gP[sNodePrime] + here->HSMHV2_ydyn_gP[sNodePrime] * s->real;
      *(here->HSMHV2GPspPtr +1) +=  here->HSMHV2_ydyn_gP[sNodePrime] * s->imag;
      *(here->HSMHV2GPbpPtr) +=  here->HSMHV2_ydc_gP[bNodePrime] + here->HSMHV2_ydyn_gP[bNodePrime] * s->real;
      *(here->HSMHV2GPbpPtr +1) +=  here->HSMHV2_ydyn_gP[bNodePrime] * s->imag;
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2GPtempPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_gP[tempNode] + here->HSMHV2_ydyn_gP[tempNode] * s->real);
        *(here->HSMHV2GPtempPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_gP[tempNode] * s->imag;
      }
      if (flg_nqs) {
	*(here->HSMHV2GPqiPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_gP[qiNode] + here->HSMHV2_ydyn_gP[qiNode] * s->real);
	*(here->HSMHV2GPqiPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_gP[qiNode] * s->imag;
	*(here->HSMHV2GPqbPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_gP[qbNode] + here->HSMHV2_ydyn_gP[qbNode] * s->real);
	*(here->HSMHV2GPqbPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_gP[qbNode] * s->imag;
      }

      /*source*/
      *(here->HSMHV2SdPtr) += here->HSMHV2_ydc_s[dNode] + here->HSMHV2_ydyn_s[dNode] * s->real;
      *(here->HSMHV2SdPtr +1) += here->HSMHV2_ydyn_s[dNode] * s->imag;
      *(here->HSMHV2SgpPtr) += here->HSMHV2_ydc_s[gNodePrime] + here->HSMHV2_ydyn_s[gNodePrime] * s->real;
      *(here->HSMHV2SgpPtr +1) += here->HSMHV2_ydyn_s[gNodePrime] * s->imag;
      *(here->HSMHV2SsPtr) +=  here->HSMHV2_ydc_s[sNode] + here->HSMHV2_ydyn_s[sNode] * s->real;
      *(here->HSMHV2SsPtr +1) +=  here->HSMHV2_ydyn_s[sNode] * s->imag;
      *(here->HSMHV2SspPtr) +=  here->HSMHV2_ydc_s[sNodePrime] + here->HSMHV2_ydyn_s[sNodePrime] * s->real;
      *(here->HSMHV2SspPtr +1) +=  here->HSMHV2_ydyn_s[sNodePrime] * s->imag;
      *(here->HSMHV2SbpPtr) += here->HSMHV2_ydc_s[bNodePrime] + here->HSMHV2_ydyn_s[bNodePrime] * s->real;
      *(here->HSMHV2SbpPtr +1) += here->HSMHV2_ydyn_s[bNodePrime] * s->imag;
      *(here->HSMHV2SsbPtr) += here->HSMHV2_ydc_s[sbNode] + here->HSMHV2_ydyn_s[sbNode] * s->real;
      *(here->HSMHV2SsbPtr +1) += here->HSMHV2_ydyn_s[sbNode] * s->imag;
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2StempPtr) += model->HSMHV2_type * (here->HSMHV2_ydc_s[tempNode] + here->HSMHV2_ydyn_s[tempNode] * s->real);
        *(here->HSMHV2StempPtr +1) += model->HSMHV2_type * here->HSMHV2_ydyn_s[tempNode] * s->imag;
      }
 
      /*source prime*/
      *(here->HSMHV2SPdPtr) +=  here->HSMHV2_ydc_sP[dNode] + here->HSMHV2_ydyn_sP[dNode] * s->real;
      *(here->HSMHV2SPdPtr +1) +=  here->HSMHV2_ydyn_sP[dNode] * s->imag;
      *(here->HSMHV2SPdpPtr) +=  here->HSMHV2_ydc_sP[dNodePrime] + here->HSMHV2_ydyn_sP[dNodePrime] * s->real;
      *(here->HSMHV2SPdpPtr +1) +=  here->HSMHV2_ydyn_sP[dNodePrime] * s->imag;
      *(here->HSMHV2SPgpPtr) +=  here->HSMHV2_ydc_sP[gNodePrime] + here->HSMHV2_ydyn_sP[gNodePrime] * s->real;
      *(here->HSMHV2SPgpPtr +1) +=  here->HSMHV2_ydyn_sP[gNodePrime] * s->imag;
      *(here->HSMHV2SPsPtr) +=  here->HSMHV2_ydc_sP[sNode] + here->HSMHV2_ydyn_sP[sNode] * s->real;
      *(here->HSMHV2SPsPtr +1) +=  here->HSMHV2_ydyn_sP[sNode] * s->imag;
      *(here->HSMHV2SPspPtr) +=  here->HSMHV2_ydc_sP[sNodePrime] + here->HSMHV2_ydyn_sP[sNodePrime] * s->real;
      *(here->HSMHV2SPspPtr +1) +=  here->HSMHV2_ydyn_sP[sNodePrime] * s->imag;
      *(here->HSMHV2SPbpPtr) +=  here->HSMHV2_ydc_sP[bNodePrime] + here->HSMHV2_ydyn_sP[bNodePrime] * s->real;
      *(here->HSMHV2SPbpPtr +1) +=  here->HSMHV2_ydyn_sP[bNodePrime] * s->imag;
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2SPtempPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_sP[tempNode] + here->HSMHV2_ydyn_sP[tempNode] * s->real);
        *(here->HSMHV2SPtempPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_sP[tempNode] * s->imag;
      }
      if (flg_nqs) {
	*(here->HSMHV2SPqiPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_sP[qiNode] + here->HSMHV2_ydyn_sP[qiNode] * s->real);
	*(here->HSMHV2SPqiPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_sP[qiNode] * s->imag;
      }
     
      /*bulk prime*/
      *(here->HSMHV2BPdpPtr) +=  here->HSMHV2_ydc_bP[dNodePrime] + here->HSMHV2_ydyn_bP[dNodePrime] * s->real;
      *(here->HSMHV2BPdpPtr +1) +=  here->HSMHV2_ydyn_bP[dNodePrime] * s->imag;
      *(here->HSMHV2BPgpPtr) +=  here->HSMHV2_ydc_bP[gNodePrime] + here->HSMHV2_ydyn_bP[gNodePrime] * s->real;
      *(here->HSMHV2BPgpPtr +1) +=  here->HSMHV2_ydyn_bP[gNodePrime] * s->imag;
      *(here->HSMHV2BPspPtr) +=  here->HSMHV2_ydc_bP[sNodePrime] + here->HSMHV2_ydyn_bP[sNodePrime] * s->real;
      *(here->HSMHV2BPspPtr +1) +=  here->HSMHV2_ydyn_bP[sNodePrime] * s->imag;
      *(here->HSMHV2BPbpPtr) +=  here->HSMHV2_ydc_bP[bNodePrime] + here->HSMHV2_ydyn_bP[bNodePrime] * s->real;
      *(here->HSMHV2BPbpPtr +1) +=  here->HSMHV2_ydyn_bP[bNodePrime] * s->imag;
      *(here->HSMHV2BPbPtr) +=  here->HSMHV2_ydc_bP[bNode] + here->HSMHV2_ydyn_bP[bNode] * s->real;
      *(here->HSMHV2BPbPtr +1) +=  here->HSMHV2_ydyn_bP[bNode] * s->imag;
      *(here->HSMHV2BPdbPtr) +=  here->HSMHV2_ydc_bP[dbNode] + here->HSMHV2_ydyn_bP[dbNode] * s->real;
      *(here->HSMHV2BPdbPtr +1) +=  here->HSMHV2_ydyn_bP[dbNode] * s->imag;
      *(here->HSMHV2BPsbPtr) +=  here->HSMHV2_ydc_bP[sbNode] + here->HSMHV2_ydyn_bP[sbNode] * s->real;
      *(here->HSMHV2BPsbPtr +1) +=  here->HSMHV2_ydyn_bP[sbNode] * s->imag;
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2BPtempPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_bP[tempNode] + here->HSMHV2_ydyn_bP[tempNode] * s->real);
        *(here->HSMHV2BPtempPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_bP[tempNode] * s->imag;
      }
      if (flg_nqs) {
	*(here->HSMHV2BPqbPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_bP[qbNode] + here->HSMHV2_ydyn_bP[qbNode] * s->real);
	*(here->HSMHV2BPqbPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_bP[qbNode] * s->imag;
      }
     
      /*bulk*/
      *(here->HSMHV2BbpPtr) +=  here->HSMHV2_ydc_b[bNodePrime] + here->HSMHV2_ydyn_b[bNodePrime] * s->real;
      *(here->HSMHV2BbpPtr +1) +=  here->HSMHV2_ydyn_b[bNodePrime] * s->imag;
      *(here->HSMHV2BbPtr) +=  here->HSMHV2_ydc_b[bNode] + here->HSMHV2_ydyn_b[bNode] * s->real;
      *(here->HSMHV2BbPtr +1) +=  here->HSMHV2_ydyn_b[bNode] * s->imag;

      /*drain bulk*/
      *(here->HSMHV2DBdPtr)  +=  here->HSMHV2_ydc_db[dNode] + here->HSMHV2_ydyn_db[dNode] * s->real;
      *(here->HSMHV2DBdPtr +1)  +=  here->HSMHV2_ydyn_db[dNode] * s->imag;
      *(here->HSMHV2DBbpPtr) +=  here->HSMHV2_ydc_db[bNodePrime] + here->HSMHV2_ydyn_db[bNodePrime] * s->real;
      *(here->HSMHV2DBbpPtr +1) +=  here->HSMHV2_ydyn_db[bNodePrime] * s->imag;
      *(here->HSMHV2DBdbPtr) +=  here->HSMHV2_ydc_db[dbNode] + here->HSMHV2_ydyn_db[dbNode] * s->real;
      *(here->HSMHV2DBdbPtr +1) +=  here->HSMHV2_ydyn_db[dbNode] * s->imag;
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2DBtempPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_db[tempNode] + here->HSMHV2_ydyn_db[tempNode] * s->real);
        *(here->HSMHV2DBtempPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_db[tempNode] * s->imag;
      }
     
      /*source bulk*/
      *(here->HSMHV2SBsPtr)  +=  here->HSMHV2_ydc_sb[sNode] + here->HSMHV2_ydyn_sb[sNode] * s->real;
      *(here->HSMHV2SBsPtr +1)  +=  here->HSMHV2_ydyn_sb[sNode] * s->imag;
      *(here->HSMHV2SBbpPtr) +=  here->HSMHV2_ydc_sb[bNodePrime] + here->HSMHV2_ydyn_sb[bNodePrime] * s->real;
      *(here->HSMHV2SBbpPtr +1) +=  here->HSMHV2_ydyn_sb[bNodePrime] * s->imag;
      *(here->HSMHV2SBsbPtr) +=  here->HSMHV2_ydc_sb[sbNode] + here->HSMHV2_ydyn_sb[sbNode] * s->real;
      *(here->HSMHV2SBsbPtr +1) +=  here->HSMHV2_ydyn_sb[sbNode] * s->imag;
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2SBtempPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_sb[tempNode] + here->HSMHV2_ydyn_sb[tempNode] * s->real);
        *(here->HSMHV2SBtempPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_sb[tempNode] * s->imag;
      }
     
      /*temp*/
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2TempdPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_t[dNode] + here->HSMHV2_ydyn_t[dNode] * s->real);
        *(here->HSMHV2TempdPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_t[dNode] * s->imag;
        *(here->HSMHV2TempdpPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_t[dNodePrime] + here->HSMHV2_ydyn_t[dNodePrime] * s->real);
        *(here->HSMHV2TempdpPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_t[dNodePrime] * s->imag;
        *(here->HSMHV2TempgpPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_t[gNodePrime] + here->HSMHV2_ydyn_t[gNodePrime] * s->real);
        *(here->HSMHV2TempgpPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_t[gNodePrime] * s->imag;
        *(here->HSMHV2TempsPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_t[sNode] + here->HSMHV2_ydyn_t[sNode] * s->real);
        *(here->HSMHV2TempsPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_t[sNode] * s->imag;
        *(here->HSMHV2TempspPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_t[sNodePrime] + here->HSMHV2_ydyn_t[sNodePrime] * s->real);
        *(here->HSMHV2TempspPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_t[sNodePrime] * s->imag;
        *(here->HSMHV2TempbpPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_t[bNodePrime] + here->HSMHV2_ydyn_t[bNodePrime] * s->real);
        *(here->HSMHV2TempbpPtr +1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_t[bNodePrime] * s->imag;
        *(here->HSMHV2TemptempPtr) +=  here->HSMHV2_ydc_t[tempNode] + here->HSMHV2_ydyn_t[tempNode] * s->real;
        *(here->HSMHV2TemptempPtr +1) +=  here->HSMHV2_ydyn_t[tempNode] * s->imag;
      }
      /* additional entries for flat nqs handling */
      if ( flg_nqs ) {
        /*qi*/
	*(here->HSMHV2QIdpPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_qi[dNodePrime] + here->HSMHV2_ydyn_qi[dNodePrime] * s->real);
	*(here->HSMHV2QIdpPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_qi[dNodePrime] * s->imag;
	*(here->HSMHV2QIgpPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_qi[gNodePrime] + here->HSMHV2_ydyn_qi[gNodePrime] * s->real);
	*(here->HSMHV2QIgpPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_qi[gNodePrime] * s->imag;
	*(here->HSMHV2QIspPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_qi[sNodePrime] + here->HSMHV2_ydyn_qi[sNodePrime] * s->real);
	*(here->HSMHV2QIspPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_qi[sNodePrime] * s->imag;
	*(here->HSMHV2QIbpPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_qi[bNodePrime] + here->HSMHV2_ydyn_qi[bNodePrime] * s->real);
	*(here->HSMHV2QIbpPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_qi[bNodePrime] * s->imag;
	*(here->HSMHV2QIqiPtr) +=                     here->HSMHV2_ydc_qi[qiNode] + here->HSMHV2_ydyn_qi[qiNode] * s->real;
	*(here->HSMHV2QIqiPtr+1) +=                   here->HSMHV2_ydyn_qi[qiNode] * s->imag;
        if ( here->HSMHV2tempNode > 0 ) {
	  *(here->HSMHV2QItempPtr) +=                 here->HSMHV2_ydc_qi[tempNode] + here->HSMHV2_ydyn_qi[tempNode] * s->real;
	  *(here->HSMHV2QItempPtr+1) +=               here->HSMHV2_ydyn_qi[tempNode] * s->imag;
        }

        /*qb*/
	*(here->HSMHV2QBdpPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_qb[dNodePrime] + here->HSMHV2_ydyn_qb[dNodePrime] * s->real);
	*(here->HSMHV2QBdpPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_qb[dNodePrime] * s->imag;
	*(here->HSMHV2QBgpPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_qb[gNodePrime] + here->HSMHV2_ydyn_qb[gNodePrime] * s->real);
	*(here->HSMHV2QBgpPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_qb[gNodePrime] * s->imag;
	*(here->HSMHV2QBspPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_qb[sNodePrime] + here->HSMHV2_ydyn_qb[sNodePrime] * s->real);
	*(here->HSMHV2QBspPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_qb[sNodePrime] * s->imag;
	*(here->HSMHV2QBbpPtr) +=  model->HSMHV2_type * (here->HSMHV2_ydc_qb[bNodePrime] + here->HSMHV2_ydyn_qb[bNodePrime] * s->real);
	*(here->HSMHV2QBbpPtr+1) +=  model->HSMHV2_type * here->HSMHV2_ydyn_qb[bNodePrime] * s->imag;
	*(here->HSMHV2QBqbPtr) +=                     here->HSMHV2_ydc_qb[qbNode] + here->HSMHV2_ydyn_qb[qbNode] * s->real;
	*(here->HSMHV2QBqbPtr+1) +=                   here->HSMHV2_ydyn_qb[qbNode] * s->imag;
        if ( here->HSMHV2tempNode > 0 ) {
	  *(here->HSMHV2QBtempPtr) +=                 here->HSMHV2_ydc_qb[tempNode] + here->HSMHV2_ydyn_qb[tempNode] * s->real;
	  *(here->HSMHV2QBtempPtr+1) +=               here->HSMHV2_ydyn_qb[tempNode] * s->imag;
        }
      }
    }
  }
  return(OK);
}

