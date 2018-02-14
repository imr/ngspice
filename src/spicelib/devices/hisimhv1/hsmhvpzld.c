/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvpzld.c

 DATE : 2013.04.30

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "hsmhvdef.h"
#include "ngspice/suffix.h"

int HSMHVpzLoad(
     GENmodel *inModel,
     register CKTcircuit *ckt,
     register SPcomplex *s)
{
  register HSMHVmodel *model = (HSMHVmodel*)inModel;
  register HSMHVinstance *here;
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

  
  for ( ;model != NULL ;model = HSMHVnextModel(model)) {
    for ( here = HSMHVinstances(model);here!= NULL ;
	  here = HSMHVnextInstance(here)) {

      flg_nqs = model->HSMHV_conqs ;

      /* stamp matrix */
     
      /*drain*/
      *(here->HSMHVDdPtr) += here->HSMHV_ydc_d[dNode] + here->HSMHV_ydyn_d[dNode] * s->real;
      *(here->HSMHVDdPtr +1) += here->HSMHV_ydyn_d[dNode] * s->imag;
      *(here->HSMHVDdpPtr) += here->HSMHV_ydc_d[dNodePrime] + here->HSMHV_ydyn_d[dNodePrime] * s->real;
      *(here->HSMHVDdpPtr +1) += here->HSMHV_ydyn_d[dNodePrime] * s->imag;
      *(here->HSMHVDgpPtr) += here->HSMHV_ydc_d[gNodePrime] + here->HSMHV_ydyn_d[gNodePrime] * s->real;
      *(here->HSMHVDgpPtr +1) += here->HSMHV_ydyn_d[gNodePrime] * s->imag;
      *(here->HSMHVDsPtr) += here->HSMHV_ydc_d[sNode] + here->HSMHV_ydyn_d[sNode] * s->real;
      *(here->HSMHVDsPtr +1) += here->HSMHV_ydyn_d[sNode] * s->imag;
      *(here->HSMHVDbpPtr) += here->HSMHV_ydc_d[bNodePrime] + here->HSMHV_ydyn_d[bNodePrime] * s->real;
      *(here->HSMHVDbpPtr +1) += here->HSMHV_ydyn_d[bNodePrime] * s->imag;
      *(here->HSMHVDdbPtr) += here->HSMHV_ydc_d[dbNode] + here->HSMHV_ydyn_d[dbNode] * s->real;
      *(here->HSMHVDdbPtr +1) += here->HSMHV_ydyn_d[dbNode] * s->imag;
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVDtempPtr) += model->HSMHV_type * (here->HSMHV_ydc_d[tempNode] + here->HSMHV_ydyn_d[tempNode] * s->real);
        *(here->HSMHVDtempPtr +1) += model->HSMHV_type * here->HSMHV_ydyn_d[tempNode] * s->imag;
      }

      /*drain prime*/
      *(here->HSMHVDPdPtr) +=  here->HSMHV_ydc_dP[dNode] + here->HSMHV_ydyn_dP[dNode] * s->real;
      *(here->HSMHVDPdPtr +1) +=  here->HSMHV_ydyn_dP[dNode] * s->imag;
      *(here->HSMHVDPdpPtr) +=  here->HSMHV_ydc_dP[dNodePrime] + here->HSMHV_ydyn_dP[dNodePrime] * s->real;
      *(here->HSMHVDPdpPtr +1) +=  here->HSMHV_ydyn_dP[dNodePrime] * s->imag;
      *(here->HSMHVDPgpPtr) +=  here->HSMHV_ydc_dP[gNodePrime] + here->HSMHV_ydyn_dP[gNodePrime] * s->real;
      *(here->HSMHVDPgpPtr +1) +=  here->HSMHV_ydyn_dP[gNodePrime] * s->imag;
      *(here->HSMHVDPsPtr) +=  here->HSMHV_ydc_dP[sNode] + here->HSMHV_ydyn_dP[sNode] * s->real;
      *(here->HSMHVDPsPtr +1) += here->HSMHV_ydyn_dP[sNode] * s->imag;
      *(here->HSMHVDPspPtr) +=  here->HSMHV_ydc_dP[sNodePrime] + here->HSMHV_ydyn_dP[sNodePrime] * s->real;
      *(here->HSMHVDPspPtr +1) +=  here->HSMHV_ydyn_dP[sNodePrime] * s->imag;
      *(here->HSMHVDPbpPtr) +=  here->HSMHV_ydc_dP[bNodePrime] + here->HSMHV_ydyn_dP[bNodePrime] * s->real;
      *(here->HSMHVDPbpPtr +1) +=  here->HSMHV_ydyn_dP[bNodePrime] * s->imag;
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVDPtempPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_dP[tempNode] + here->HSMHV_ydyn_dP[tempNode] * s->real);
        *(here->HSMHVDPtempPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_dP[tempNode] * s->imag;
      }
      if (flg_nqs) {
        *(here->HSMHVDPqiPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_dP[qiNode] + here->HSMHV_ydyn_dP[qiNode] * s->real);
        *(here->HSMHVDPqiPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_dP[qiNode] * s->imag;
      }


      /*gate*/
      *(here->HSMHVGgPtr) +=  here->HSMHV_ydc_g[gNode] + here->HSMHV_ydyn_g[gNode] * s->real;
      *(here->HSMHVGgPtr +1) +=  here->HSMHV_ydyn_g[gNode] * s->imag;
      *(here->HSMHVGgpPtr) +=  here->HSMHV_ydc_g[gNodePrime] + here->HSMHV_ydyn_g[gNodePrime] * s->real;
      *(here->HSMHVGgpPtr +1) +=  here->HSMHV_ydyn_g[gNodePrime] * s->imag;
     
      /*gate prime*/
      *(here->HSMHVGPdPtr) +=  here->HSMHV_ydc_gP[dNode] + here->HSMHV_ydyn_gP[dNode] * s->real;
      *(here->HSMHVGPdPtr +1) +=  here->HSMHV_ydyn_gP[dNode] * s->imag;
      *(here->HSMHVGPdpPtr) +=  here->HSMHV_ydc_gP[dNodePrime] + here->HSMHV_ydyn_gP[dNodePrime] * s->real;
      *(here->HSMHVGPdpPtr +1) +=  here->HSMHV_ydyn_gP[dNodePrime] * s->imag;
      *(here->HSMHVGPgPtr) +=  here->HSMHV_ydc_gP[gNode] + here->HSMHV_ydyn_gP[gNode] * s->real;
      *(here->HSMHVGPgPtr +1) +=  here->HSMHV_ydyn_gP[gNode] * s->imag;
      *(here->HSMHVGPgpPtr) +=  here->HSMHV_ydc_gP[gNodePrime] + here->HSMHV_ydyn_gP[gNodePrime] * s->real;
      *(here->HSMHVGPgpPtr +1) +=  here->HSMHV_ydyn_gP[gNodePrime] * s->imag;
      *(here->HSMHVGPsPtr) +=  here->HSMHV_ydc_gP[sNode] + here->HSMHV_ydyn_gP[sNode] * s->real;
      *(here->HSMHVGPsPtr +1) +=  here->HSMHV_ydyn_gP[sNode] * s->imag;
      *(here->HSMHVGPspPtr) +=  here->HSMHV_ydc_gP[sNodePrime] + here->HSMHV_ydyn_gP[sNodePrime] * s->real;
      *(here->HSMHVGPspPtr +1) +=  here->HSMHV_ydyn_gP[sNodePrime] * s->imag;
      *(here->HSMHVGPbpPtr) +=  here->HSMHV_ydc_gP[bNodePrime] + here->HSMHV_ydyn_gP[bNodePrime] * s->real;
      *(here->HSMHVGPbpPtr +1) +=  here->HSMHV_ydyn_gP[bNodePrime] * s->imag;
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVGPtempPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_gP[tempNode] + here->HSMHV_ydyn_gP[tempNode] * s->real);
        *(here->HSMHVGPtempPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_gP[tempNode] * s->imag;
      }
      if (flg_nqs) {
	*(here->HSMHVGPqiPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_gP[qiNode] + here->HSMHV_ydyn_gP[qiNode] * s->real);
	*(here->HSMHVGPqiPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_gP[qiNode] * s->imag;
	*(here->HSMHVGPqbPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_gP[qbNode] + here->HSMHV_ydyn_gP[qbNode] * s->real);
	*(here->HSMHVGPqbPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_gP[qbNode] * s->imag;
      }

      /*source*/
      *(here->HSMHVSdPtr) += here->HSMHV_ydc_s[dNode] + here->HSMHV_ydyn_s[dNode] * s->real;
      *(here->HSMHVSdPtr +1) += here->HSMHV_ydyn_s[dNode] * s->imag;
      *(here->HSMHVSgpPtr) += here->HSMHV_ydc_s[gNodePrime] + here->HSMHV_ydyn_s[gNodePrime] * s->real;
      *(here->HSMHVSgpPtr +1) += here->HSMHV_ydyn_s[gNodePrime] * s->imag;
      *(here->HSMHVSsPtr) +=  here->HSMHV_ydc_s[sNode] + here->HSMHV_ydyn_s[sNode] * s->real;
      *(here->HSMHVSsPtr +1) +=  here->HSMHV_ydyn_s[sNode] * s->imag;
      *(here->HSMHVSspPtr) +=  here->HSMHV_ydc_s[sNodePrime] + here->HSMHV_ydyn_s[sNodePrime] * s->real;
      *(here->HSMHVSspPtr +1) +=  here->HSMHV_ydyn_s[sNodePrime] * s->imag;
      *(here->HSMHVSbpPtr) += here->HSMHV_ydc_s[bNodePrime] + here->HSMHV_ydyn_s[bNodePrime] * s->real;
      *(here->HSMHVSbpPtr +1) += here->HSMHV_ydyn_s[bNodePrime] * s->imag;
      *(here->HSMHVSsbPtr) += here->HSMHV_ydc_s[sbNode] + here->HSMHV_ydyn_s[sbNode] * s->real;
      *(here->HSMHVSsbPtr +1) += here->HSMHV_ydyn_s[sbNode] * s->imag;
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVStempPtr) += model->HSMHV_type * (here->HSMHV_ydc_s[tempNode] + here->HSMHV_ydyn_s[tempNode] * s->real);
        *(here->HSMHVStempPtr +1) += model->HSMHV_type * here->HSMHV_ydyn_s[tempNode] * s->imag;
      }
 
      /*source prime*/
      *(here->HSMHVSPdPtr) +=  here->HSMHV_ydc_sP[dNode] + here->HSMHV_ydyn_sP[dNode] * s->real;
      *(here->HSMHVSPdPtr +1) +=  here->HSMHV_ydyn_sP[dNode] * s->imag;
      *(here->HSMHVSPdpPtr) +=  here->HSMHV_ydc_sP[dNodePrime] + here->HSMHV_ydyn_sP[dNodePrime] * s->real;
      *(here->HSMHVSPdpPtr +1) +=  here->HSMHV_ydyn_sP[dNodePrime] * s->imag;
      *(here->HSMHVSPgpPtr) +=  here->HSMHV_ydc_sP[gNodePrime] + here->HSMHV_ydyn_sP[gNodePrime] * s->real;
      *(here->HSMHVSPgpPtr +1) +=  here->HSMHV_ydyn_sP[gNodePrime] * s->imag;
      *(here->HSMHVSPsPtr) +=  here->HSMHV_ydc_sP[sNode] + here->HSMHV_ydyn_sP[sNode] * s->real;
      *(here->HSMHVSPsPtr +1) +=  here->HSMHV_ydyn_sP[sNode] * s->imag;
      *(here->HSMHVSPspPtr) +=  here->HSMHV_ydc_sP[sNodePrime] + here->HSMHV_ydyn_sP[sNodePrime] * s->real;
      *(here->HSMHVSPspPtr +1) +=  here->HSMHV_ydyn_sP[sNodePrime] * s->imag;
      *(here->HSMHVSPbpPtr) +=  here->HSMHV_ydc_sP[bNodePrime] + here->HSMHV_ydyn_sP[bNodePrime] * s->real;
      *(here->HSMHVSPbpPtr +1) +=  here->HSMHV_ydyn_sP[bNodePrime] * s->imag;
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVSPtempPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_sP[tempNode] + here->HSMHV_ydyn_sP[tempNode] * s->real);
        *(here->HSMHVSPtempPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_sP[tempNode] * s->imag;
      }
      if (flg_nqs) {
	*(here->HSMHVSPqiPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_sP[qiNode] + here->HSMHV_ydyn_sP[qiNode] * s->real);
	*(here->HSMHVSPqiPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_sP[qiNode] * s->imag;
      }
     
      /*bulk prime*/
      *(here->HSMHVBPdpPtr) +=  here->HSMHV_ydc_bP[dNodePrime] + here->HSMHV_ydyn_bP[dNodePrime] * s->real;
      *(here->HSMHVBPdpPtr +1) +=  here->HSMHV_ydyn_bP[dNodePrime] * s->imag;
      *(here->HSMHVBPgpPtr) +=  here->HSMHV_ydc_bP[gNodePrime] + here->HSMHV_ydyn_bP[gNodePrime] * s->real;
      *(here->HSMHVBPgpPtr +1) +=  here->HSMHV_ydyn_bP[gNodePrime] * s->imag;
      *(here->HSMHVBPspPtr) +=  here->HSMHV_ydc_bP[sNodePrime] + here->HSMHV_ydyn_bP[sNodePrime] * s->real;
      *(here->HSMHVBPspPtr +1) +=  here->HSMHV_ydyn_bP[sNodePrime] * s->imag;
      *(here->HSMHVBPbpPtr) +=  here->HSMHV_ydc_bP[bNodePrime] + here->HSMHV_ydyn_bP[bNodePrime] * s->real;
      *(here->HSMHVBPbpPtr +1) +=  here->HSMHV_ydyn_bP[bNodePrime] * s->imag;
      *(here->HSMHVBPbPtr) +=  here->HSMHV_ydc_bP[bNode] + here->HSMHV_ydyn_bP[bNode] * s->real;
      *(here->HSMHVBPbPtr +1) +=  here->HSMHV_ydyn_bP[bNode] * s->imag;
      *(here->HSMHVBPdbPtr) +=  here->HSMHV_ydc_bP[dbNode] + here->HSMHV_ydyn_bP[dbNode] * s->real;
      *(here->HSMHVBPdbPtr +1) +=  here->HSMHV_ydyn_bP[dbNode] * s->imag;
      *(here->HSMHVBPsbPtr) +=  here->HSMHV_ydc_bP[sbNode] + here->HSMHV_ydyn_bP[sbNode] * s->real;
      *(here->HSMHVBPsbPtr +1) +=  here->HSMHV_ydyn_bP[sbNode] * s->imag;
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVBPtempPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_bP[tempNode] + here->HSMHV_ydyn_bP[tempNode] * s->real);
        *(here->HSMHVBPtempPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_bP[tempNode] * s->imag;
      }
      if (flg_nqs) {
	*(here->HSMHVBPqbPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_bP[qbNode] + here->HSMHV_ydyn_bP[qbNode] * s->real);
	*(here->HSMHVBPqbPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_bP[qbNode] * s->imag;
      }
     
      /*bulk*/
      *(here->HSMHVBbpPtr) +=  here->HSMHV_ydc_b[bNodePrime] + here->HSMHV_ydyn_b[bNodePrime] * s->real;
      *(here->HSMHVBbpPtr +1) +=  here->HSMHV_ydyn_b[bNodePrime] * s->imag;
      *(here->HSMHVBbPtr) +=  here->HSMHV_ydc_b[bNode] + here->HSMHV_ydyn_b[bNode] * s->real;
      *(here->HSMHVBbPtr +1) +=  here->HSMHV_ydyn_b[bNode] * s->imag;

      /*drain bulk*/
      *(here->HSMHVDBdPtr)  +=  here->HSMHV_ydc_db[dNode] + here->HSMHV_ydyn_db[dNode] * s->real;
      *(here->HSMHVDBdPtr +1)  +=  here->HSMHV_ydyn_db[dNode] * s->imag;
      *(here->HSMHVDBbpPtr) +=  here->HSMHV_ydc_db[bNodePrime] + here->HSMHV_ydyn_db[bNodePrime] * s->real;
      *(here->HSMHVDBbpPtr +1) +=  here->HSMHV_ydyn_db[bNodePrime] * s->imag;
      *(here->HSMHVDBdbPtr) +=  here->HSMHV_ydc_db[dbNode] + here->HSMHV_ydyn_db[dbNode] * s->real;
      *(here->HSMHVDBdbPtr +1) +=  here->HSMHV_ydyn_db[dbNode] * s->imag;
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVDBtempPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_db[tempNode] + here->HSMHV_ydyn_db[tempNode] * s->real);
        *(here->HSMHVDBtempPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_db[tempNode] * s->imag;
      }
     
      /*source bulk*/
      *(here->HSMHVSBsPtr)  +=  here->HSMHV_ydc_sb[sNode] + here->HSMHV_ydyn_sb[sNode] * s->real;
      *(here->HSMHVSBsPtr +1)  +=  here->HSMHV_ydyn_sb[sNode] * s->imag;
      *(here->HSMHVSBbpPtr) +=  here->HSMHV_ydc_sb[bNodePrime] + here->HSMHV_ydyn_sb[bNodePrime] * s->real;
      *(here->HSMHVSBbpPtr +1) +=  here->HSMHV_ydyn_sb[bNodePrime] * s->imag;
      *(here->HSMHVSBsbPtr) +=  here->HSMHV_ydc_sb[sbNode] + here->HSMHV_ydyn_sb[sbNode] * s->real;
      *(here->HSMHVSBsbPtr +1) +=  here->HSMHV_ydyn_sb[sbNode] * s->imag;
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVSBtempPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_sb[tempNode] + here->HSMHV_ydyn_sb[tempNode] * s->real);
        *(here->HSMHVSBtempPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_sb[tempNode] * s->imag;
      }
     
      /*temp*/
      if( here->HSMHVtempNode > 0) { 
        *(here->HSMHVTempdPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_t[dNode] + here->HSMHV_ydyn_t[dNode] * s->real);
        *(here->HSMHVTempdPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_t[dNode] * s->imag;
        *(here->HSMHVTempdpPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_t[dNodePrime] + here->HSMHV_ydyn_t[dNodePrime] * s->real);
        *(here->HSMHVTempdpPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_t[dNodePrime] * s->imag;
        *(here->HSMHVTempgpPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_t[gNodePrime] + here->HSMHV_ydyn_t[gNodePrime] * s->real);
        *(here->HSMHVTempgpPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_t[gNodePrime] * s->imag;
        *(here->HSMHVTempsPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_t[sNode] + here->HSMHV_ydyn_t[sNode] * s->real);
        *(here->HSMHVTempsPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_t[sNode] * s->imag;
        *(here->HSMHVTempspPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_t[sNodePrime] + here->HSMHV_ydyn_t[sNodePrime] * s->real);
        *(here->HSMHVTempspPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_t[sNodePrime] * s->imag;
        *(here->HSMHVTempbpPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_t[bNodePrime] + here->HSMHV_ydyn_t[bNodePrime] * s->real);
        *(here->HSMHVTempbpPtr +1) +=  model->HSMHV_type * here->HSMHV_ydyn_t[bNodePrime] * s->imag;
        *(here->HSMHVTemptempPtr) +=  here->HSMHV_ydc_t[tempNode] + here->HSMHV_ydyn_t[tempNode] * s->real;
        *(here->HSMHVTemptempPtr +1) +=  here->HSMHV_ydyn_t[tempNode] * s->imag;
      }
      /* additional entries for flat nqs handling */
      if ( flg_nqs ) {
        /*qi*/
	*(here->HSMHVQIdpPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_qi[dNodePrime] + here->HSMHV_ydyn_qi[dNodePrime] * s->real);
	*(here->HSMHVQIdpPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_qi[dNodePrime] * s->imag;
	*(here->HSMHVQIgpPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_qi[gNodePrime] + here->HSMHV_ydyn_qi[gNodePrime] * s->real);
	*(here->HSMHVQIgpPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_qi[gNodePrime] * s->imag;
	*(here->HSMHVQIspPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_qi[sNodePrime] + here->HSMHV_ydyn_qi[sNodePrime] * s->real);
	*(here->HSMHVQIspPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_qi[sNodePrime] * s->imag;
	*(here->HSMHVQIbpPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_qi[bNodePrime] + here->HSMHV_ydyn_qi[bNodePrime] * s->real);
	*(here->HSMHVQIbpPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_qi[bNodePrime] * s->imag;
	*(here->HSMHVQIqiPtr) +=                     here->HSMHV_ydc_qi[qiNode] + here->HSMHV_ydyn_qi[qiNode] * s->real;
	*(here->HSMHVQIqiPtr+1) +=                   here->HSMHV_ydyn_qi[qiNode] * s->imag;
        if ( here->HSMHVtempNode > 0 ) {
	  *(here->HSMHVQItempPtr) +=                 here->HSMHV_ydc_qi[tempNode] + here->HSMHV_ydyn_qi[tempNode] * s->real;
	  *(here->HSMHVQItempPtr+1) +=               here->HSMHV_ydyn_qi[tempNode] * s->imag;
        }

        /*qb*/
	*(here->HSMHVQBdpPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_qb[dNodePrime] + here->HSMHV_ydyn_qb[dNodePrime] * s->real);
	*(here->HSMHVQBdpPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_qb[dNodePrime] * s->imag;
	*(here->HSMHVQBgpPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_qb[gNodePrime] + here->HSMHV_ydyn_qb[gNodePrime] * s->real);
	*(here->HSMHVQBgpPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_qb[gNodePrime] * s->imag;
	*(here->HSMHVQBspPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_qb[sNodePrime] + here->HSMHV_ydyn_qb[sNodePrime] * s->real);
	*(here->HSMHVQBspPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_qb[sNodePrime] * s->imag;
	*(here->HSMHVQBbpPtr) +=  model->HSMHV_type * (here->HSMHV_ydc_qb[bNodePrime] + here->HSMHV_ydyn_qb[bNodePrime] * s->real);
	*(here->HSMHVQBbpPtr+1) +=  model->HSMHV_type * here->HSMHV_ydyn_qb[bNodePrime] * s->imag;
	*(here->HSMHVQBqbPtr) +=                     here->HSMHV_ydc_qb[qbNode] + here->HSMHV_ydyn_qb[qbNode] * s->real;
	*(here->HSMHVQBqbPtr+1) +=                   here->HSMHV_ydyn_qb[qbNode] * s->imag;
        if ( here->HSMHVtempNode > 0 ) {
	  *(here->HSMHVQBtempPtr) +=                 here->HSMHV_ydc_qb[tempNode] + here->HSMHV_ydyn_qb[tempNode] * s->real;
	  *(here->HSMHVQBtempPtr+1) +=               here->HSMHV_ydyn_qb[tempNode] * s->imag;
        }
      }
    }
  }
  return(OK);
}

