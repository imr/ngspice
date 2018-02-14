/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvacld.c

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
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "hsmhv2def.h"

int HSMHV2acLoad(
     GENmodel *inModel,
     CKTcircuit *ckt)
{
  HSMHV2model *model = (HSMHV2model*)inModel;
  HSMHV2instance *here;

  double omega=0.0 ;
  int flg_nqs =0 ;
  int flg_subNode = 0 ;

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
#define subNode     10
#define tempNode    11
#define qiNode      12
#define qbNode      13


  omega = ckt->CKTomega;
  for ( ; model != NULL; model = HSMHV2nextModel(model)) {
    for ( here = HSMHV2instances(model); here!= NULL; here = HSMHV2nextInstance(here)) {

      flg_nqs = model->HSMHV2_conqs ;
      flg_subNode = here->HSMHV2subNode ; /* if flg_subNode > 0, external(/internal) substrate node exists */

      /* stamp matrix */
     
      /*drain*/
      *(here->HSMHV2DdPtr) += here->HSMHV2_ydc_d[dNode] ;
      *(here->HSMHV2DdPtr +1) += omega*here->HSMHV2_ydyn_d[dNode] ;
      *(here->HSMHV2DdpPtr) += here->HSMHV2_ydc_d[dNodePrime] ;
      *(here->HSMHV2DdpPtr +1) += omega*here->HSMHV2_ydyn_d[dNodePrime];
      *(here->HSMHV2DgpPtr) += here->HSMHV2_ydc_d[gNodePrime];
      *(here->HSMHV2DgpPtr +1) += omega*here->HSMHV2_ydyn_d[gNodePrime];
      *(here->HSMHV2DsPtr) += here->HSMHV2_ydc_d[sNode];
      *(here->HSMHV2DsPtr +1) += omega*here->HSMHV2_ydyn_d[sNode];
      *(here->HSMHV2DbpPtr) += here->HSMHV2_ydc_d[bNodePrime];
      *(here->HSMHV2DbpPtr +1) += omega*here->HSMHV2_ydyn_d[bNodePrime];
      *(here->HSMHV2DdbPtr) += here->HSMHV2_ydc_d[dbNode];
      *(here->HSMHV2DdbPtr +1) += omega*here->HSMHV2_ydyn_d[dbNode];
      if (flg_subNode > 0) {
	*(here->HSMHV2DsubPtr) += here->HSMHV2_ydc_d[subNode];
      }
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2DtempPtr) += model->HSMHV2_type*here->HSMHV2_ydc_d[tempNode];
        *(here->HSMHV2DtempPtr +1) += model->HSMHV2_type*omega*here->HSMHV2_ydyn_d[tempNode];
      }

      /*drain prime*/
      *(here->HSMHV2DPdPtr) +=  here->HSMHV2_ydc_dP[dNode] ;
      *(here->HSMHV2DPdPtr +1) +=  omega*here->HSMHV2_ydyn_dP[dNode];
      *(here->HSMHV2DPdpPtr) +=  here->HSMHV2_ydc_dP[dNodePrime];
      *(here->HSMHV2DPdpPtr +1) +=  omega*here->HSMHV2_ydyn_dP[dNodePrime];
      *(here->HSMHV2DPgpPtr) +=  here->HSMHV2_ydc_dP[gNodePrime];
      *(here->HSMHV2DPgpPtr +1) +=  omega*here->HSMHV2_ydyn_dP[gNodePrime];
      *(here->HSMHV2DPsPtr) +=  here->HSMHV2_ydc_dP[sNode] ;
      *(here->HSMHV2DPsPtr +1) += omega*here->HSMHV2_ydyn_dP[sNode];
      *(here->HSMHV2DPspPtr) +=  here->HSMHV2_ydc_dP[sNodePrime] ;
      *(here->HSMHV2DPspPtr +1) +=  omega*here->HSMHV2_ydyn_dP[sNodePrime];
      *(here->HSMHV2DPbpPtr) +=  here->HSMHV2_ydc_dP[bNodePrime] ;
      *(here->HSMHV2DPbpPtr +1) +=  omega*here->HSMHV2_ydyn_dP[bNodePrime];
      if (flg_subNode > 0) {
	*(here->HSMHV2DPsubPtr) += here->HSMHV2_ydc_dP[subNode];
      }
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2DPtempPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_dP[tempNode];
        *(here->HSMHV2DPtempPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_dP[tempNode];
      }
      if (flg_nqs) {
        *(here->HSMHV2DPqiPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_dP[qiNode];
        *(here->HSMHV2DPqiPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_dP[qiNode];
      }


      /*gate*/
      *(here->HSMHV2GgPtr) +=  here->HSMHV2_ydc_g[gNode] ;
      *(here->HSMHV2GgPtr +1) +=  omega*here->HSMHV2_ydyn_g[gNode];
      *(here->HSMHV2GgpPtr) +=  here->HSMHV2_ydc_g[gNodePrime] ;
      *(here->HSMHV2GgpPtr +1) +=  omega*here->HSMHV2_ydyn_g[gNodePrime];
     
      /*gate prime*/
      *(here->HSMHV2GPdPtr) +=  here->HSMHV2_ydc_gP[dNode] ;
      *(here->HSMHV2GPdPtr +1) +=  omega*here->HSMHV2_ydyn_gP[dNode];
      *(here->HSMHV2GPdpPtr) +=  here->HSMHV2_ydc_gP[dNodePrime] ;
      *(here->HSMHV2GPdpPtr +1) +=  omega*here->HSMHV2_ydyn_gP[dNodePrime];
      *(here->HSMHV2GPgPtr) +=  here->HSMHV2_ydc_gP[gNode];
      *(here->HSMHV2GPgPtr +1) +=  omega*here->HSMHV2_ydyn_gP[gNode];
      *(here->HSMHV2GPgpPtr) +=  here->HSMHV2_ydc_gP[gNodePrime] ;
      *(here->HSMHV2GPgpPtr +1) +=  omega*here->HSMHV2_ydyn_gP[gNodePrime];
      *(here->HSMHV2GPsPtr) +=  here->HSMHV2_ydc_gP[sNode];
      *(here->HSMHV2GPsPtr +1) +=  omega*here->HSMHV2_ydyn_gP[sNode];
      *(here->HSMHV2GPspPtr) +=  here->HSMHV2_ydc_gP[sNodePrime] ;
      *(here->HSMHV2GPspPtr +1) +=  omega*here->HSMHV2_ydyn_gP[sNodePrime];
      *(here->HSMHV2GPbpPtr) +=  here->HSMHV2_ydc_gP[bNodePrime] ;
      *(here->HSMHV2GPbpPtr +1) +=  omega*here->HSMHV2_ydyn_gP[bNodePrime];
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2GPtempPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_gP[tempNode] ;
        *(here->HSMHV2GPtempPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_gP[tempNode];
      }
      if (flg_nqs) {
	*(here->HSMHV2GPqiPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_gP[qiNode];
	*(here->HSMHV2GPqiPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_gP[qiNode];
	*(here->HSMHV2GPqbPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_gP[qbNode];
	*(here->HSMHV2GPqbPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_gP[qbNode];
      }

      /*source*/
      *(here->HSMHV2SdPtr) += here->HSMHV2_ydc_s[dNode];
      *(here->HSMHV2SdPtr +1) += omega*here->HSMHV2_ydyn_s[dNode];
      *(here->HSMHV2SgpPtr) += here->HSMHV2_ydc_s[gNodePrime];
      *(here->HSMHV2SgpPtr +1) += omega*here->HSMHV2_ydyn_s[gNodePrime];
      *(here->HSMHV2SsPtr) +=  here->HSMHV2_ydc_s[sNode] ;
      *(here->HSMHV2SsPtr +1) +=  omega*here->HSMHV2_ydyn_s[sNode];
      *(here->HSMHV2SspPtr) +=  here->HSMHV2_ydc_s[sNodePrime] ;
      *(here->HSMHV2SspPtr +1) +=  omega*here->HSMHV2_ydyn_s[sNodePrime];
      *(here->HSMHV2SbpPtr) += here->HSMHV2_ydc_s[bNodePrime];
      *(here->HSMHV2SbpPtr +1) += omega*here->HSMHV2_ydyn_s[bNodePrime];
      *(here->HSMHV2SsbPtr) += here->HSMHV2_ydc_s[sbNode] ;
      *(here->HSMHV2SsbPtr +1) += omega*here->HSMHV2_ydyn_s[sbNode];
      if (flg_subNode > 0) {
	*(here->HSMHV2SsubPtr) += here->HSMHV2_ydc_s[subNode];
      }
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2StempPtr) += model->HSMHV2_type*here->HSMHV2_ydc_s[tempNode];
        *(here->HSMHV2StempPtr +1) += model->HSMHV2_type*omega*here->HSMHV2_ydyn_s[tempNode];
      }
 
      /*source prime*/
      *(here->HSMHV2SPdPtr) +=  here->HSMHV2_ydc_sP[dNode] ;
      *(here->HSMHV2SPdPtr +1) +=  omega*here->HSMHV2_ydyn_sP[dNode];
      *(here->HSMHV2SPdpPtr) +=  here->HSMHV2_ydc_sP[dNodePrime] ;
      *(here->HSMHV2SPdpPtr +1) +=  omega*here->HSMHV2_ydyn_sP[dNodePrime];
      *(here->HSMHV2SPgpPtr) +=  here->HSMHV2_ydc_sP[gNodePrime] ;
      *(here->HSMHV2SPgpPtr +1) +=  omega*here->HSMHV2_ydyn_sP[gNodePrime];
      *(here->HSMHV2SPsPtr) +=  here->HSMHV2_ydc_sP[sNode] ;
      *(here->HSMHV2SPsPtr +1) +=  omega*here->HSMHV2_ydyn_sP[sNode];
      *(here->HSMHV2SPspPtr) +=  here->HSMHV2_ydc_sP[sNodePrime] ;
      *(here->HSMHV2SPspPtr +1) +=  omega*here->HSMHV2_ydyn_sP[sNodePrime];
      *(here->HSMHV2SPbpPtr) +=  here->HSMHV2_ydc_sP[bNodePrime];
      *(here->HSMHV2SPbpPtr +1) +=  omega*here->HSMHV2_ydyn_sP[bNodePrime];
      if (flg_subNode > 0) {
	*(here->HSMHV2SPsubPtr) += here->HSMHV2_ydc_sP[subNode];
      }
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2SPtempPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_sP[tempNode] ;
        *(here->HSMHV2SPtempPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_sP[tempNode];
      }
      if (flg_nqs) {
	*(here->HSMHV2SPqiPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_sP[qiNode];
	*(here->HSMHV2SPqiPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_sP[qiNode];
      }
     
      /*bulk prime*/
      *(here->HSMHV2BPdPtr) +=  here->HSMHV2_ydc_bP[dNode];
      *(here->HSMHV2BPdPtr +1) +=  omega*here->HSMHV2_ydyn_bP[dNode];
      *(here->HSMHV2BPsPtr) +=  here->HSMHV2_ydc_bP[sNode];
      *(here->HSMHV2BPsPtr +1) +=  omega*here->HSMHV2_ydyn_bP[sNode];
      *(here->HSMHV2BPdpPtr) +=  here->HSMHV2_ydc_bP[dNodePrime];
      *(here->HSMHV2BPdpPtr +1) +=  omega*here->HSMHV2_ydyn_bP[dNodePrime];
      *(here->HSMHV2BPgpPtr) +=  here->HSMHV2_ydc_bP[gNodePrime] ;
      *(here->HSMHV2BPgpPtr +1) +=  omega*here->HSMHV2_ydyn_bP[gNodePrime];
      *(here->HSMHV2BPspPtr) +=  here->HSMHV2_ydc_bP[sNodePrime];
      *(here->HSMHV2BPspPtr +1) +=  omega*here->HSMHV2_ydyn_bP[sNodePrime];
      *(here->HSMHV2BPbpPtr) +=  here->HSMHV2_ydc_bP[bNodePrime];
      *(here->HSMHV2BPbpPtr +1) +=  omega*here->HSMHV2_ydyn_bP[bNodePrime];
      *(here->HSMHV2BPbPtr) +=  here->HSMHV2_ydc_bP[bNode];
      *(here->HSMHV2BPbPtr +1) +=  omega*here->HSMHV2_ydyn_bP[bNode];
      *(here->HSMHV2BPdbPtr) +=  here->HSMHV2_ydc_bP[dbNode] ;
      *(here->HSMHV2BPdbPtr +1) +=  omega*here->HSMHV2_ydyn_bP[dbNode];
      *(here->HSMHV2BPsbPtr) +=  here->HSMHV2_ydc_bP[sbNode] ;
      *(here->HSMHV2BPsbPtr +1) +=  omega*here->HSMHV2_ydyn_bP[sbNode];
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2BPtempPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_bP[tempNode] ;
        *(here->HSMHV2BPtempPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_bP[tempNode];
      }
      if (flg_nqs) {
	*(here->HSMHV2BPqbPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_bP[qbNode];
	*(here->HSMHV2BPqbPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_bP[qbNode];
      }
     
      /*bulk*/
      *(here->HSMHV2BbpPtr) +=  here->HSMHV2_ydc_b[bNodePrime] ;
      *(here->HSMHV2BbpPtr +1) +=  omega*here->HSMHV2_ydyn_b[bNodePrime];
      *(here->HSMHV2BbPtr) +=  here->HSMHV2_ydc_b[bNode] ;
      *(here->HSMHV2BbPtr +1) +=  omega*here->HSMHV2_ydyn_b[bNode];

      /*drain bulk*/
      *(here->HSMHV2DBdPtr)  +=  here->HSMHV2_ydc_db[dNode] ;
      *(here->HSMHV2DBdPtr +1)  +=  omega*here->HSMHV2_ydyn_db[dNode];
      *(here->HSMHV2DBbpPtr) +=  here->HSMHV2_ydc_db[bNodePrime] ;
      *(here->HSMHV2DBbpPtr +1) +=  omega*here->HSMHV2_ydyn_db[bNodePrime];
      *(here->HSMHV2DBdbPtr) +=  here->HSMHV2_ydc_db[dbNode] ;
      *(here->HSMHV2DBdbPtr +1) +=  omega*here->HSMHV2_ydyn_db[dbNode];
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2DBtempPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_db[tempNode] ;
        *(here->HSMHV2DBtempPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_db[tempNode];
      }
     
      /*source bulk*/
      *(here->HSMHV2SBsPtr)  +=  here->HSMHV2_ydc_sb[sNode] ;
      *(here->HSMHV2SBsPtr +1)  +=  omega*here->HSMHV2_ydyn_sb[sNode];
      *(here->HSMHV2SBbpPtr) +=  here->HSMHV2_ydc_sb[bNodePrime];
      *(here->HSMHV2SBbpPtr +1) +=  omega*here->HSMHV2_ydyn_sb[bNodePrime];
      *(here->HSMHV2SBsbPtr) +=  here->HSMHV2_ydc_sb[sbNode] ;
      *(here->HSMHV2SBsbPtr +1) +=  omega*here->HSMHV2_ydyn_sb[sbNode];
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2SBtempPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_sb[tempNode];
        *(here->HSMHV2SBtempPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_sb[tempNode];
      }
     
      /*temp*/
      if( here->HSMHV2tempNode > 0) { 
        *(here->HSMHV2TempdPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_t[dNode] ;
        *(here->HSMHV2TempdPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_t[dNode];
        *(here->HSMHV2TempdpPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_t[dNodePrime] ;
        *(here->HSMHV2TempdpPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_t[dNodePrime];
        *(here->HSMHV2TempgpPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_t[gNodePrime];
        *(here->HSMHV2TempgpPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_t[gNodePrime];
        *(here->HSMHV2TempsPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_t[sNode] ;
        *(here->HSMHV2TempsPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_t[sNode];
        *(here->HSMHV2TempspPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_t[sNodePrime] ;
        *(here->HSMHV2TempspPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_t[sNodePrime];
        *(here->HSMHV2TempbpPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_t[bNodePrime] ;
        *(here->HSMHV2TempbpPtr +1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_t[bNodePrime];
        *(here->HSMHV2TemptempPtr) +=  here->HSMHV2_ydc_t[tempNode] ;
        *(here->HSMHV2TemptempPtr +1) +=  omega*here->HSMHV2_ydyn_t[tempNode];
      }
      /* additional entries for flat nqs handling */
      if ( flg_nqs ) {
        /*qi*/
	*(here->HSMHV2QIdpPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_qi[dNodePrime];
	*(here->HSMHV2QIdpPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_qi[dNodePrime];
	*(here->HSMHV2QIgpPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_qi[gNodePrime];
	*(here->HSMHV2QIgpPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_qi[gNodePrime];
	*(here->HSMHV2QIspPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_qi[sNodePrime];
	*(here->HSMHV2QIspPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_qi[sNodePrime];
	*(here->HSMHV2QIbpPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_qi[bNodePrime];
	*(here->HSMHV2QIbpPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_qi[bNodePrime];
	*(here->HSMHV2QIqiPtr) +=                     here->HSMHV2_ydc_qi[qiNode];
	*(here->HSMHV2QIqiPtr+1) +=                   omega*here->HSMHV2_ydyn_qi[qiNode];
        if ( here->HSMHV2tempNode > 0 ) {
	  *(here->HSMHV2QItempPtr) +=                 here->HSMHV2_ydc_qi[tempNode];
	  *(here->HSMHV2QItempPtr+1) +=               omega*here->HSMHV2_ydyn_qi[tempNode];
        }

        /*qb*/
	*(here->HSMHV2QBdpPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_qb[dNodePrime];
	*(here->HSMHV2QBdpPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_qb[dNodePrime];
	*(here->HSMHV2QBgpPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_qb[gNodePrime];
	*(here->HSMHV2QBgpPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_qb[gNodePrime];
	*(here->HSMHV2QBspPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_qb[sNodePrime];
	*(here->HSMHV2QBspPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_qb[sNodePrime];
	*(here->HSMHV2QBbpPtr) +=  model->HSMHV2_type*here->HSMHV2_ydc_qb[bNodePrime];
	*(here->HSMHV2QBbpPtr+1) +=  model->HSMHV2_type*omega*here->HSMHV2_ydyn_qb[bNodePrime];
	*(here->HSMHV2QBqbPtr) +=                     here->HSMHV2_ydc_qb[qbNode];
	*(here->HSMHV2QBqbPtr+1) +=                   omega*here->HSMHV2_ydyn_qb[qbNode];
        if ( here->HSMHV2tempNode > 0 ) {
	  *(here->HSMHV2QBtempPtr) +=                 here->HSMHV2_ydc_qb[tempNode];
	  *(here->HSMHV2QBtempPtr+1) +=               omega*here->HSMHV2_ydyn_qb[tempNode];
        }
      }
    }
  }

  return(OK);
}
