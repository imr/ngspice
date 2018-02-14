/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvacld.c

 DATE : 2013.04.30

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "hsmhvdef.h"

int HSMHVacLoad(
     GENmodel *inModel,
     register CKTcircuit *ckt)
{
  register HSMHVmodel *model = (HSMHVmodel*)inModel;
  register HSMHVinstance *here;

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
  for ( ; model != NULL; model = HSMHVnextModel(model)) {
    for ( here = HSMHVinstances(model); here!= NULL; here = HSMHVnextInstance(here)) {

      flg_nqs = model->HSMHV_conqs ;
      flg_subNode = here->HSMHVsubNode ; /* if flg_subNode > 0, external(/internal) substrate node exists */

      /* stamp matrix */
     
      /*drain*/
      *(here->HSMHVDdPtr) += here->HSMHV_ydc_d[dNode] ;
      *(here->HSMHVDdPtr +1) += omega*here->HSMHV_ydyn_d[dNode] ;
      *(here->HSMHVDdpPtr) += here->HSMHV_ydc_d[dNodePrime] ;
      *(here->HSMHVDdpPtr +1) += omega*here->HSMHV_ydyn_d[dNodePrime];
      *(here->HSMHVDgpPtr) += here->HSMHV_ydc_d[gNodePrime];
      *(here->HSMHVDgpPtr +1) += omega*here->HSMHV_ydyn_d[gNodePrime];
      *(here->HSMHVDsPtr) += here->HSMHV_ydc_d[sNode];
      *(here->HSMHVDsPtr +1) += omega*here->HSMHV_ydyn_d[sNode];
      *(here->HSMHVDbpPtr) += here->HSMHV_ydc_d[bNodePrime];
      *(here->HSMHVDbpPtr +1) += omega*here->HSMHV_ydyn_d[bNodePrime];
      *(here->HSMHVDdbPtr) += here->HSMHV_ydc_d[dbNode];
      *(here->HSMHVDdbPtr +1) += omega*here->HSMHV_ydyn_d[dbNode];
      if (flg_subNode > 0) {
	*(here->HSMHVDsubPtr) += here->HSMHV_ydc_d[subNode];
      }
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVDtempPtr) += model->HSMHV_type*here->HSMHV_ydc_d[tempNode];
        *(here->HSMHVDtempPtr +1) += model->HSMHV_type*omega*here->HSMHV_ydyn_d[tempNode];
      }

      /*drain prime*/
      *(here->HSMHVDPdPtr) +=  here->HSMHV_ydc_dP[dNode] ;
      *(here->HSMHVDPdPtr +1) +=  omega*here->HSMHV_ydyn_dP[dNode];
      *(here->HSMHVDPdpPtr) +=  here->HSMHV_ydc_dP[dNodePrime];
      *(here->HSMHVDPdpPtr +1) +=  omega*here->HSMHV_ydyn_dP[dNodePrime];
      *(here->HSMHVDPgpPtr) +=  here->HSMHV_ydc_dP[gNodePrime];
      *(here->HSMHVDPgpPtr +1) +=  omega*here->HSMHV_ydyn_dP[gNodePrime];
      *(here->HSMHVDPsPtr) +=  here->HSMHV_ydc_dP[sNode] ;
      *(here->HSMHVDPsPtr +1) += omega*here->HSMHV_ydyn_dP[sNode];
      *(here->HSMHVDPspPtr) +=  here->HSMHV_ydc_dP[sNodePrime] ;
      *(here->HSMHVDPspPtr +1) +=  omega*here->HSMHV_ydyn_dP[sNodePrime];
      *(here->HSMHVDPbpPtr) +=  here->HSMHV_ydc_dP[bNodePrime] ;
      *(here->HSMHVDPbpPtr +1) +=  omega*here->HSMHV_ydyn_dP[bNodePrime];
      if (flg_subNode > 0) {
	*(here->HSMHVDPsubPtr) += here->HSMHV_ydc_dP[subNode];
      }
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVDPtempPtr) +=  model->HSMHV_type*here->HSMHV_ydc_dP[tempNode];
        *(here->HSMHVDPtempPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_dP[tempNode];
      }
      if (flg_nqs) {
        *(here->HSMHVDPqiPtr) +=  model->HSMHV_type*here->HSMHV_ydc_dP[qiNode];
        *(here->HSMHVDPqiPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_dP[qiNode];
      }


      /*gate*/
      *(here->HSMHVGgPtr) +=  here->HSMHV_ydc_g[gNode] ;
      *(here->HSMHVGgPtr +1) +=  omega*here->HSMHV_ydyn_g[gNode];
      *(here->HSMHVGgpPtr) +=  here->HSMHV_ydc_g[gNodePrime] ;
      *(here->HSMHVGgpPtr +1) +=  omega*here->HSMHV_ydyn_g[gNodePrime];
     
      /*gate prime*/
      *(here->HSMHVGPdPtr) +=  here->HSMHV_ydc_gP[dNode] ;
      *(here->HSMHVGPdPtr +1) +=  omega*here->HSMHV_ydyn_gP[dNode];
      *(here->HSMHVGPdpPtr) +=  here->HSMHV_ydc_gP[dNodePrime] ;
      *(here->HSMHVGPdpPtr +1) +=  omega*here->HSMHV_ydyn_gP[dNodePrime];
      *(here->HSMHVGPgPtr) +=  here->HSMHV_ydc_gP[gNode];
      *(here->HSMHVGPgPtr +1) +=  omega*here->HSMHV_ydyn_gP[gNode];
      *(here->HSMHVGPgpPtr) +=  here->HSMHV_ydc_gP[gNodePrime] ;
      *(here->HSMHVGPgpPtr +1) +=  omega*here->HSMHV_ydyn_gP[gNodePrime];
      *(here->HSMHVGPsPtr) +=  here->HSMHV_ydc_gP[sNode];
      *(here->HSMHVGPsPtr +1) +=  omega*here->HSMHV_ydyn_gP[sNode];
      *(here->HSMHVGPspPtr) +=  here->HSMHV_ydc_gP[sNodePrime] ;
      *(here->HSMHVGPspPtr +1) +=  omega*here->HSMHV_ydyn_gP[sNodePrime];
      *(here->HSMHVGPbpPtr) +=  here->HSMHV_ydc_gP[bNodePrime] ;
      *(here->HSMHVGPbpPtr +1) +=  omega*here->HSMHV_ydyn_gP[bNodePrime];
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVGPtempPtr) +=  model->HSMHV_type*here->HSMHV_ydc_gP[tempNode] ;
        *(here->HSMHVGPtempPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_gP[tempNode];
      }
      if (flg_nqs) {
	*(here->HSMHVGPqiPtr) +=  model->HSMHV_type*here->HSMHV_ydc_gP[qiNode];
	*(here->HSMHVGPqiPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_gP[qiNode];
	*(here->HSMHVGPqbPtr) +=  model->HSMHV_type*here->HSMHV_ydc_gP[qbNode];
	*(here->HSMHVGPqbPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_gP[qbNode];
      }

      /*source*/
      *(here->HSMHVSdPtr) += here->HSMHV_ydc_s[dNode];
      *(here->HSMHVSdPtr +1) += omega*here->HSMHV_ydyn_s[dNode];
      *(here->HSMHVSgpPtr) += here->HSMHV_ydc_s[gNodePrime];
      *(here->HSMHVSgpPtr +1) += omega*here->HSMHV_ydyn_s[gNodePrime];
      *(here->HSMHVSsPtr) +=  here->HSMHV_ydc_s[sNode] ;
      *(here->HSMHVSsPtr +1) +=  omega*here->HSMHV_ydyn_s[sNode];
      *(here->HSMHVSspPtr) +=  here->HSMHV_ydc_s[sNodePrime] ;
      *(here->HSMHVSspPtr +1) +=  omega*here->HSMHV_ydyn_s[sNodePrime];
      *(here->HSMHVSbpPtr) += here->HSMHV_ydc_s[bNodePrime];
      *(here->HSMHVSbpPtr +1) += omega*here->HSMHV_ydyn_s[bNodePrime];
      *(here->HSMHVSsbPtr) += here->HSMHV_ydc_s[sbNode] ;
      *(here->HSMHVSsbPtr +1) += omega*here->HSMHV_ydyn_s[sbNode];
      if (flg_subNode > 0) {
	*(here->HSMHVSsubPtr) += here->HSMHV_ydc_s[subNode];
      }
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVStempPtr) += model->HSMHV_type*here->HSMHV_ydc_s[tempNode];
        *(here->HSMHVStempPtr +1) += model->HSMHV_type*omega*here->HSMHV_ydyn_s[tempNode];
      }
 
      /*source prime*/
      *(here->HSMHVSPdPtr) +=  here->HSMHV_ydc_sP[dNode] ;
      *(here->HSMHVSPdPtr +1) +=  omega*here->HSMHV_ydyn_sP[dNode];
      *(here->HSMHVSPdpPtr) +=  here->HSMHV_ydc_sP[dNodePrime] ;
      *(here->HSMHVSPdpPtr +1) +=  omega*here->HSMHV_ydyn_sP[dNodePrime];
      *(here->HSMHVSPgpPtr) +=  here->HSMHV_ydc_sP[gNodePrime] ;
      *(here->HSMHVSPgpPtr +1) +=  omega*here->HSMHV_ydyn_sP[gNodePrime];
      *(here->HSMHVSPsPtr) +=  here->HSMHV_ydc_sP[sNode] ;
      *(here->HSMHVSPsPtr +1) +=  omega*here->HSMHV_ydyn_sP[sNode];
      *(here->HSMHVSPspPtr) +=  here->HSMHV_ydc_sP[sNodePrime] ;
      *(here->HSMHVSPspPtr +1) +=  omega*here->HSMHV_ydyn_sP[sNodePrime];
      *(here->HSMHVSPbpPtr) +=  here->HSMHV_ydc_sP[bNodePrime];
      *(here->HSMHVSPbpPtr +1) +=  omega*here->HSMHV_ydyn_sP[bNodePrime];
      if (flg_subNode > 0) {
	*(here->HSMHVSPsubPtr) += here->HSMHV_ydc_sP[subNode];
      }
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVSPtempPtr) +=  model->HSMHV_type*here->HSMHV_ydc_sP[tempNode] ;
        *(here->HSMHVSPtempPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_sP[tempNode];
      }
      if (flg_nqs) {
	*(here->HSMHVSPqiPtr) +=  model->HSMHV_type*here->HSMHV_ydc_sP[qiNode];
	*(here->HSMHVSPqiPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_sP[qiNode];
      }
     
      /*bulk prime*/
      *(here->HSMHVBPdPtr) +=  here->HSMHV_ydc_bP[dNode];
      *(here->HSMHVBPdPtr +1) +=  omega*here->HSMHV_ydyn_bP[dNode];
      *(here->HSMHVBPsPtr) +=  here->HSMHV_ydc_bP[sNode];
      *(here->HSMHVBPsPtr +1) +=  omega*here->HSMHV_ydyn_bP[sNode];
      *(here->HSMHVBPdpPtr) +=  here->HSMHV_ydc_bP[dNodePrime];
      *(here->HSMHVBPdpPtr +1) +=  omega*here->HSMHV_ydyn_bP[dNodePrime];
      *(here->HSMHVBPgpPtr) +=  here->HSMHV_ydc_bP[gNodePrime] ;
      *(here->HSMHVBPgpPtr +1) +=  omega*here->HSMHV_ydyn_bP[gNodePrime];
      *(here->HSMHVBPspPtr) +=  here->HSMHV_ydc_bP[sNodePrime];
      *(here->HSMHVBPspPtr +1) +=  omega*here->HSMHV_ydyn_bP[sNodePrime];
      *(here->HSMHVBPbpPtr) +=  here->HSMHV_ydc_bP[bNodePrime];
      *(here->HSMHVBPbpPtr +1) +=  omega*here->HSMHV_ydyn_bP[bNodePrime];
      *(here->HSMHVBPbPtr) +=  here->HSMHV_ydc_bP[bNode];
      *(here->HSMHVBPbPtr +1) +=  omega*here->HSMHV_ydyn_bP[bNode];
      *(here->HSMHVBPdbPtr) +=  here->HSMHV_ydc_bP[dbNode] ;
      *(here->HSMHVBPdbPtr +1) +=  omega*here->HSMHV_ydyn_bP[dbNode];
      *(here->HSMHVBPsbPtr) +=  here->HSMHV_ydc_bP[sbNode] ;
      *(here->HSMHVBPsbPtr +1) +=  omega*here->HSMHV_ydyn_bP[sbNode];
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVBPtempPtr) +=  model->HSMHV_type*here->HSMHV_ydc_bP[tempNode] ;
        *(here->HSMHVBPtempPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_bP[tempNode];
      }
      if (flg_nqs) {
	*(here->HSMHVBPqbPtr) +=  model->HSMHV_type*here->HSMHV_ydc_bP[qbNode];
	*(here->HSMHVBPqbPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_bP[qbNode];
      }
     
      /*bulk*/
      *(here->HSMHVBbpPtr) +=  here->HSMHV_ydc_b[bNodePrime] ;
      *(here->HSMHVBbpPtr +1) +=  omega*here->HSMHV_ydyn_b[bNodePrime];
      *(here->HSMHVBbPtr) +=  here->HSMHV_ydc_b[bNode] ;
      *(here->HSMHVBbPtr +1) +=  omega*here->HSMHV_ydyn_b[bNode];

      /*drain bulk*/
      *(here->HSMHVDBdPtr)  +=  here->HSMHV_ydc_db[dNode] ;
      *(here->HSMHVDBdPtr +1)  +=  omega*here->HSMHV_ydyn_db[dNode];
      *(here->HSMHVDBbpPtr) +=  here->HSMHV_ydc_db[bNodePrime] ;
      *(here->HSMHVDBbpPtr +1) +=  omega*here->HSMHV_ydyn_db[bNodePrime];
      *(here->HSMHVDBdbPtr) +=  here->HSMHV_ydc_db[dbNode] ;
      *(here->HSMHVDBdbPtr +1) +=  omega*here->HSMHV_ydyn_db[dbNode];
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVDBtempPtr) +=  model->HSMHV_type*here->HSMHV_ydc_db[tempNode] ;
        *(here->HSMHVDBtempPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_db[tempNode];
      }
     
      /*source bulk*/
      *(here->HSMHVSBsPtr)  +=  here->HSMHV_ydc_sb[sNode] ;
      *(here->HSMHVSBsPtr +1)  +=  omega*here->HSMHV_ydyn_sb[sNode];
      *(here->HSMHVSBbpPtr) +=  here->HSMHV_ydc_sb[bNodePrime];
      *(here->HSMHVSBbpPtr +1) +=  omega*here->HSMHV_ydyn_sb[bNodePrime];
      *(here->HSMHVSBsbPtr) +=  here->HSMHV_ydc_sb[sbNode] ;
      *(here->HSMHVSBsbPtr +1) +=  omega*here->HSMHV_ydyn_sb[sbNode];
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVSBtempPtr) +=  model->HSMHV_type*here->HSMHV_ydc_sb[tempNode];
        *(here->HSMHVSBtempPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_sb[tempNode];
      }
     
      /*temp*/
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVTempdPtr) +=  model->HSMHV_type*here->HSMHV_ydc_t[dNode] ;
        *(here->HSMHVTempdPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_t[dNode];
        *(here->HSMHVTempdpPtr) +=  model->HSMHV_type*here->HSMHV_ydc_t[dNodePrime] ;
        *(here->HSMHVTempdpPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_t[dNodePrime];
        *(here->HSMHVTempgpPtr) +=  model->HSMHV_type*here->HSMHV_ydc_t[gNodePrime];
        *(here->HSMHVTempgpPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_t[gNodePrime];
        *(here->HSMHVTempsPtr) +=  model->HSMHV_type*here->HSMHV_ydc_t[sNode] ;
        *(here->HSMHVTempsPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_t[sNode];
        *(here->HSMHVTempspPtr) +=  model->HSMHV_type*here->HSMHV_ydc_t[sNodePrime] ;
        *(here->HSMHVTempspPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_t[sNodePrime];
        *(here->HSMHVTempbpPtr) +=  model->HSMHV_type*here->HSMHV_ydc_t[bNodePrime] ;
        *(here->HSMHVTempbpPtr +1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_t[bNodePrime];
        *(here->HSMHVTemptempPtr) +=  here->HSMHV_ydc_t[tempNode] ;
        *(here->HSMHVTemptempPtr +1) +=  omega*here->HSMHV_ydyn_t[tempNode];
      }
      /* additional entries for flat nqs handling */
      if ( flg_nqs ) {
        /*qi*/
	*(here->HSMHVQIdpPtr) +=  model->HSMHV_type*here->HSMHV_ydc_qi[dNodePrime];
	*(here->HSMHVQIdpPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_qi[dNodePrime];
	*(here->HSMHVQIgpPtr) +=  model->HSMHV_type*here->HSMHV_ydc_qi[gNodePrime];
	*(here->HSMHVQIgpPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_qi[gNodePrime];
	*(here->HSMHVQIspPtr) +=  model->HSMHV_type*here->HSMHV_ydc_qi[sNodePrime];
	*(here->HSMHVQIspPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_qi[sNodePrime];
	*(here->HSMHVQIbpPtr) +=  model->HSMHV_type*here->HSMHV_ydc_qi[bNodePrime];
	*(here->HSMHVQIbpPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_qi[bNodePrime];
	*(here->HSMHVQIqiPtr) +=                     here->HSMHV_ydc_qi[qiNode];
	*(here->HSMHVQIqiPtr+1) +=                   omega*here->HSMHV_ydyn_qi[qiNode];
        if ( here->HSMHVtempNode > 0 ) {
	  *(here->HSMHVQItempPtr) +=                 here->HSMHV_ydc_qi[tempNode];
	  *(here->HSMHVQItempPtr+1) +=               omega*here->HSMHV_ydyn_qi[tempNode];
        }

        /*qb*/
	*(here->HSMHVQBdpPtr) +=  model->HSMHV_type*here->HSMHV_ydc_qb[dNodePrime];
	*(here->HSMHVQBdpPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_qb[dNodePrime];
	*(here->HSMHVQBgpPtr) +=  model->HSMHV_type*here->HSMHV_ydc_qb[gNodePrime];
	*(here->HSMHVQBgpPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_qb[gNodePrime];
	*(here->HSMHVQBspPtr) +=  model->HSMHV_type*here->HSMHV_ydc_qb[sNodePrime];
	*(here->HSMHVQBspPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_qb[sNodePrime];
	*(here->HSMHVQBbpPtr) +=  model->HSMHV_type*here->HSMHV_ydc_qb[bNodePrime];
	*(here->HSMHVQBbpPtr+1) +=  model->HSMHV_type*omega*here->HSMHV_ydyn_qb[bNodePrime];
	*(here->HSMHVQBqbPtr) +=                     here->HSMHV_ydc_qb[qbNode];
	*(here->HSMHVQBqbPtr+1) +=                   omega*here->HSMHV_ydyn_qb[qbNode];
        if ( here->HSMHVtempNode > 0 ) {
	  *(here->HSMHVQBtempPtr) +=                 here->HSMHV_ydc_qb[tempNode];
	  *(here->HSMHVQBtempPtr+1) +=               omega*here->HSMHV_ydyn_qb[tempNode];
        }
      }
    }
  }

  return(OK);
}
