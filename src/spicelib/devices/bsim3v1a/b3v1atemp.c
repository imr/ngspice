/***********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1atemp.c
**********/
/* Lmin, Lmax, Wmin, Wmax */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "bsim3v1adef.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

#define Kb 1.3806226e-23
#define KboQ 8.617087e-5  /* Kb / q  where q = 1.60219e-19 */
#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define Charge_q 1.60219e-19


/* ARGSUSED */
int
BSIM3v1Atemp(GENmodel *inModel, CKTcircuit *ckt)
{ 
BSIM3v1Amodel *model = (BSIM3v1Amodel*) inModel;
BSIM3v1Ainstance *here;
struct bsim3v1aSizeDependParam *pSizeDependParamKnot, *pLastKnot, *pParam = NULL;
double tmp1, tmp2, Eg, ni, T0, T1, T2, T3, Ldrn, Wdrn;
double Temp, TRatio, Inv_L, Inv_W, Inv_LW, Vtm0, Tnom;
int Size_Not_Found;

    /*  loop through all the BSIM3v1A device models */
    for (; model != NULL; model = model->BSIM3v1AnextModel)
    {    Temp = ckt->CKTtemp;
         if (model->BSIM3v1AbulkJctPotential < 0.1)  
	     model->BSIM3v1AbulkJctPotential = 0.1;
         if (model->BSIM3v1AsidewallJctPotential < 0.1)
             model->BSIM3v1AsidewallJctPotential = 0.1;
         model->pSizeDependParamKnot = NULL;
	 pLastKnot = NULL;

	 Tnom = model->BSIM3v1Atnom;
	 TRatio = Temp / Tnom;
         
	 /* loop through all the instances of the model */
         for (here = model->BSIM3v1Ainstances; here != NULL;
              here=here->BSIM3v1AnextInstance) 
	 {    
	 
              if (here->BSIM3v1Aowner != ARCHme)
                      continue;
	      
	      pSizeDependParamKnot = model->pSizeDependParamKnot;
	      Size_Not_Found = 1;
	      while ((pSizeDependParamKnot != NULL) && Size_Not_Found)
	      {   if ((here->BSIM3v1Al == pSizeDependParamKnot->Length)
		      && (here->BSIM3v1Aw == pSizeDependParamKnot->Width))
                  {   Size_Not_Found = 0;
		      here->pParam = pSizeDependParamKnot;
		  }
		  else
		  {   pLastKnot = pSizeDependParamKnot;
		      pSizeDependParamKnot = pSizeDependParamKnot->pNext;
		  }
              }

	      if (Size_Not_Found)
	      {   pParam = (struct bsim3v1aSizeDependParam *)tmalloc(
	                    sizeof(struct bsim3v1aSizeDependParam));
                  if (pLastKnot == NULL)
		      model->pSizeDependParamKnot = pParam;
                  else
		      pLastKnot->pNext = pParam;
                  pParam->pNext = NULL;
                  here->pParam = pParam;

		     Ldrn = here->BSIM3v1Al;
		     Wdrn = here->BSIM3v1Aw;
		  
                  T0 = pow(Ldrn, model->BSIM3v1ALln);
                  T1 = pow(Wdrn, model->BSIM3v1ALwn);
                  tmp1 = model->BSIM3v1ALl / T0 + model->BSIM3v1ALw / T1
                       + model->BSIM3v1ALwl / (T0 * T1);
                  pParam->BSIM3v1Adl = model->BSIM3v1ALint + tmp1;
                  pParam->BSIM3v1Adlc = model->BSIM3v1Adlc + tmp1;

                  T2 = pow(Ldrn, model->BSIM3v1AWln);
                  T3 = pow(Wdrn, model->BSIM3v1AWwn);
                  tmp2 = model->BSIM3v1AWl / T2 + model->BSIM3v1AWw / T3
                       + model->BSIM3v1AWwl / (T2 * T3);
                  pParam->BSIM3v1Adw = model->BSIM3v1AWint + tmp2;
                  pParam->BSIM3v1Adwc = model->BSIM3v1Adwc + tmp2;

                  pParam->BSIM3v1Aleff = here->BSIM3v1Al - 2.0 * pParam->BSIM3v1Adl;
                  if (pParam->BSIM3v1Aleff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3v1AmodName;
                      namarray[1] = here->BSIM3v1Aname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3v1A: mosfet %s, model %s: Effective channel length <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v1Aweff = here->BSIM3v1Aw - 2.0 * pParam->BSIM3v1Adw;
                  if (pParam->BSIM3v1Aweff <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3v1AmodName;
                      namarray[1] = here->BSIM3v1Aname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3v1A: mosfet %s, model %s: Effective channel width <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v1AleffCV = here->BSIM3v1Al - 2.0 * pParam->BSIM3v1Adlc;
                  if (pParam->BSIM3v1AleffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3v1AmodName;
                      namarray[1] = here->BSIM3v1Aname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3v1A: mosfet %s, model %s: Effective channel length for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

                  pParam->BSIM3v1AweffCV = here->BSIM3v1Aw - 2.0 * pParam->BSIM3v1Adwc;
                  if (pParam->BSIM3v1AweffCV <= 0.0)
	          {   IFuid namarray[2];
                      namarray[0] = model->BSIM3v1AmodName;
                      namarray[1] = here->BSIM3v1Aname;
                      (*(SPfrontEnd->IFerror))(ERR_FATAL,
                      "BSIM3v1A: mosfet %s, model %s: Effective channel width for C-V <= 0",
                       namarray);
                      return(E_BADPARM);
                  }

	          model->BSIM3v1Avcrit = CONSTvt0 * log(CONSTvt0
				    / (CONSTroot2 * 1.0e-14));
                  model->BSIM3v1Afactor1 = sqrt(EPSSI / EPSOX * model->BSIM3v1Atox);


		  if (model->BSIM3v1AbinUnit == 1)
		  {   Inv_L = 1.0e-6 / pParam->BSIM3v1Aleff;
		      Inv_W = 1.0e-6 / pParam->BSIM3v1Aweff;
		      Inv_LW = 1.0e-12 / (pParam->BSIM3v1Aleff
			     * pParam->BSIM3v1Aweff);
		  }
		  else
		  {   Inv_L = 1.0 / pParam->BSIM3v1Aleff;
		      Inv_W = 1.0 / pParam->BSIM3v1Aweff;
		      Inv_LW = 1.0 / (pParam->BSIM3v1Aleff
			     * pParam->BSIM3v1Aweff);
		  }
		  pParam->BSIM3v1Acdsc = model->BSIM3v1Acdsc
				    + model->BSIM3v1Alcdsc * Inv_L
				    + model->BSIM3v1Awcdsc * Inv_W
				    + model->BSIM3v1Apcdsc * Inv_LW;
		  pParam->BSIM3v1Acdscb = model->BSIM3v1Acdscb
				     + model->BSIM3v1Alcdscb * Inv_L
				     + model->BSIM3v1Awcdscb * Inv_W
				     + model->BSIM3v1Apcdscb * Inv_LW; 
				     
    		  pParam->BSIM3v1Acdscd = model->BSIM3v1Acdscd
				     + model->BSIM3v1Alcdscd * Inv_L
				     + model->BSIM3v1Awcdscd * Inv_W
				     + model->BSIM3v1Apcdscd * Inv_LW; 
				     
		  pParam->BSIM3v1Acit = model->BSIM3v1Acit
				   + model->BSIM3v1Alcit * Inv_L
				   + model->BSIM3v1Awcit * Inv_W
				   + model->BSIM3v1Apcit * Inv_LW;
		  pParam->BSIM3v1Anfactor = model->BSIM3v1Anfactor
				       + model->BSIM3v1Alnfactor * Inv_L
				       + model->BSIM3v1Awnfactor * Inv_W
				       + model->BSIM3v1Apnfactor * Inv_LW;
		  pParam->BSIM3v1Axj = model->BSIM3v1Axj
				  + model->BSIM3v1Alxj * Inv_L
				  + model->BSIM3v1Awxj * Inv_W
				  + model->BSIM3v1Apxj * Inv_LW;
		  pParam->BSIM3v1Avsat = model->BSIM3v1Avsat
				    + model->BSIM3v1Alvsat * Inv_L
				    + model->BSIM3v1Awvsat * Inv_W
				    + model->BSIM3v1Apvsat * Inv_LW;
		  pParam->BSIM3v1Aat = model->BSIM3v1Aat
				  + model->BSIM3v1Alat * Inv_L
				  + model->BSIM3v1Awat * Inv_W
				  + model->BSIM3v1Apat * Inv_LW;
		  pParam->BSIM3v1Aa0 = model->BSIM3v1Aa0
				  + model->BSIM3v1Ala0 * Inv_L
				  + model->BSIM3v1Awa0 * Inv_W
				  + model->BSIM3v1Apa0 * Inv_LW; 
				  
		  pParam->BSIM3v1Aags = model->BSIM3v1Aags
				  + model->BSIM3v1Alags * Inv_L
				  + model->BSIM3v1Awags * Inv_W
				  + model->BSIM3v1Apags * Inv_LW;
				  
		  pParam->BSIM3v1Aa1 = model->BSIM3v1Aa1
				  + model->BSIM3v1Ala1 * Inv_L
				  + model->BSIM3v1Awa1 * Inv_W
				  + model->BSIM3v1Apa1 * Inv_LW;
		  pParam->BSIM3v1Aa2 = model->BSIM3v1Aa2
				  + model->BSIM3v1Ala2 * Inv_L
				  + model->BSIM3v1Awa2 * Inv_W
				  + model->BSIM3v1Apa2 * Inv_LW;
		  pParam->BSIM3v1Aketa = model->BSIM3v1Aketa
				    + model->BSIM3v1Alketa * Inv_L
				    + model->BSIM3v1Awketa * Inv_W
				    + model->BSIM3v1Apketa * Inv_LW;
		  pParam->BSIM3v1Ansub = model->BSIM3v1Ansub
				    + model->BSIM3v1Alnsub * Inv_L
				    + model->BSIM3v1Awnsub * Inv_W
				    + model->BSIM3v1Apnsub * Inv_LW;
		  pParam->BSIM3v1Anpeak = model->BSIM3v1Anpeak
				     + model->BSIM3v1Alnpeak * Inv_L
				     + model->BSIM3v1Awnpeak * Inv_W
				     + model->BSIM3v1Apnpeak * Inv_LW;
		  pParam->BSIM3v1Angate = model->BSIM3v1Angate
				     + model->BSIM3v1Alngate * Inv_L
				     + model->BSIM3v1Awngate * Inv_W
				     + model->BSIM3v1Apngate * Inv_LW;
		  pParam->BSIM3v1Agamma1 = model->BSIM3v1Agamma1
				      + model->BSIM3v1Algamma1 * Inv_L
				      + model->BSIM3v1Awgamma1 * Inv_W
				      + model->BSIM3v1Apgamma1 * Inv_LW;
		  pParam->BSIM3v1Agamma2 = model->BSIM3v1Agamma2
				      + model->BSIM3v1Algamma2 * Inv_L
				      + model->BSIM3v1Awgamma2 * Inv_W
				      + model->BSIM3v1Apgamma2 * Inv_LW;
		  pParam->BSIM3v1Avbx = model->BSIM3v1Avbx
				   + model->BSIM3v1Alvbx * Inv_L
				   + model->BSIM3v1Awvbx * Inv_W
				   + model->BSIM3v1Apvbx * Inv_LW;
		  pParam->BSIM3v1Avbm = model->BSIM3v1Avbm
				   + model->BSIM3v1Alvbm * Inv_L
				   + model->BSIM3v1Awvbm * Inv_W
				   + model->BSIM3v1Apvbm * Inv_LW;
		  pParam->BSIM3v1Axt = model->BSIM3v1Axt
				   + model->BSIM3v1Alxt * Inv_L
				   + model->BSIM3v1Awxt * Inv_W
				   + model->BSIM3v1Apxt * Inv_LW;
		  pParam->BSIM3v1Ak1 = model->BSIM3v1Ak1
				  + model->BSIM3v1Alk1 * Inv_L
				  + model->BSIM3v1Awk1 * Inv_W
				  + model->BSIM3v1Apk1 * Inv_LW;
		  pParam->BSIM3v1Akt1 = model->BSIM3v1Akt1
				   + model->BSIM3v1Alkt1 * Inv_L
				   + model->BSIM3v1Awkt1 * Inv_W
				   + model->BSIM3v1Apkt1 * Inv_LW;
		  pParam->BSIM3v1Akt1l = model->BSIM3v1Akt1l
				    + model->BSIM3v1Alkt1l * Inv_L
				    + model->BSIM3v1Awkt1l * Inv_W
				    + model->BSIM3v1Apkt1l * Inv_LW;
		  pParam->BSIM3v1Ak2 = model->BSIM3v1Ak2
				  + model->BSIM3v1Alk2 * Inv_L
				  + model->BSIM3v1Awk2 * Inv_W
				  + model->BSIM3v1Apk2 * Inv_LW;
		  pParam->BSIM3v1Akt2 = model->BSIM3v1Akt2
				   + model->BSIM3v1Alkt2 * Inv_L
				   + model->BSIM3v1Awkt2 * Inv_W
				   + model->BSIM3v1Apkt2 * Inv_LW;
		  pParam->BSIM3v1Ak3 = model->BSIM3v1Ak3
				  + model->BSIM3v1Alk3 * Inv_L
				  + model->BSIM3v1Awk3 * Inv_W
				  + model->BSIM3v1Apk3 * Inv_LW;
		  pParam->BSIM3v1Ak3b = model->BSIM3v1Ak3b
				   + model->BSIM3v1Alk3b * Inv_L
				   + model->BSIM3v1Awk3b * Inv_W
				   + model->BSIM3v1Apk3b * Inv_LW;
		  pParam->BSIM3v1Aw0 = model->BSIM3v1Aw0
				  + model->BSIM3v1Alw0 * Inv_L
				  + model->BSIM3v1Aww0 * Inv_W
				  + model->BSIM3v1Apw0 * Inv_LW;
		  pParam->BSIM3v1Anlx = model->BSIM3v1Anlx
				   + model->BSIM3v1Alnlx * Inv_L
				   + model->BSIM3v1Awnlx * Inv_W
				   + model->BSIM3v1Apnlx * Inv_LW;
		  pParam->BSIM3v1Advt0 = model->BSIM3v1Advt0
				    + model->BSIM3v1Aldvt0 * Inv_L
				    + model->BSIM3v1Awdvt0 * Inv_W
				    + model->BSIM3v1Apdvt0 * Inv_LW;
		  pParam->BSIM3v1Advt1 = model->BSIM3v1Advt1
				    + model->BSIM3v1Aldvt1 * Inv_L
				    + model->BSIM3v1Awdvt1 * Inv_W
				    + model->BSIM3v1Apdvt1 * Inv_LW;
		  pParam->BSIM3v1Advt2 = model->BSIM3v1Advt2
				    + model->BSIM3v1Aldvt2 * Inv_L
				    + model->BSIM3v1Awdvt2 * Inv_W
				    + model->BSIM3v1Apdvt2 * Inv_LW;
		  pParam->BSIM3v1Advt0w = model->BSIM3v1Advt0w
				    + model->BSIM3v1Aldvt0w * Inv_L
				    + model->BSIM3v1Awdvt0w * Inv_W
				    + model->BSIM3v1Apdvt0w * Inv_LW;
		  pParam->BSIM3v1Advt1w = model->BSIM3v1Advt1w
				    + model->BSIM3v1Aldvt1w * Inv_L
				    + model->BSIM3v1Awdvt1w * Inv_W
				    + model->BSIM3v1Apdvt1w * Inv_LW;
		  pParam->BSIM3v1Advt2w = model->BSIM3v1Advt2w
				    + model->BSIM3v1Aldvt2w * Inv_L
				    + model->BSIM3v1Awdvt2w * Inv_W
				    + model->BSIM3v1Apdvt2w * Inv_LW;
		  pParam->BSIM3v1Adrout = model->BSIM3v1Adrout
				     + model->BSIM3v1Aldrout * Inv_L
				     + model->BSIM3v1Awdrout * Inv_W
				     + model->BSIM3v1Apdrout * Inv_LW;
		  pParam->BSIM3v1Adsub = model->BSIM3v1Adsub
				    + model->BSIM3v1Aldsub * Inv_L
				    + model->BSIM3v1Awdsub * Inv_W
				    + model->BSIM3v1Apdsub * Inv_LW;
		  pParam->BSIM3v1Avth0 = model->BSIM3v1Avth0
				    + model->BSIM3v1Alvth0 * Inv_L
				    + model->BSIM3v1Awvth0 * Inv_W
				    + model->BSIM3v1Apvth0 * Inv_LW;
		  pParam->BSIM3v1Aua = model->BSIM3v1Aua
				  + model->BSIM3v1Alua * Inv_L
				  + model->BSIM3v1Awua * Inv_W
				  + model->BSIM3v1Apua * Inv_LW;
		  pParam->BSIM3v1Aua1 = model->BSIM3v1Aua1
				   + model->BSIM3v1Alua1 * Inv_L
				   + model->BSIM3v1Awua1 * Inv_W
				   + model->BSIM3v1Apua1 * Inv_LW;
		  pParam->BSIM3v1Aub = model->BSIM3v1Aub
				  + model->BSIM3v1Alub * Inv_L
				  + model->BSIM3v1Awub * Inv_W
				  + model->BSIM3v1Apub * Inv_LW;
		  pParam->BSIM3v1Aub1 = model->BSIM3v1Aub1
				   + model->BSIM3v1Alub1 * Inv_L
				   + model->BSIM3v1Awub1 * Inv_W
				   + model->BSIM3v1Apub1 * Inv_LW;
		  pParam->BSIM3v1Auc = model->BSIM3v1Auc
				  + model->BSIM3v1Aluc * Inv_L
				  + model->BSIM3v1Awuc * Inv_W
				  + model->BSIM3v1Apuc * Inv_LW;
		  pParam->BSIM3v1Auc1 = model->BSIM3v1Auc1
				   + model->BSIM3v1Aluc1 * Inv_L
				   + model->BSIM3v1Awuc1 * Inv_W
				   + model->BSIM3v1Apuc1 * Inv_LW;
		  pParam->BSIM3v1Au0 = model->BSIM3v1Au0
				  + model->BSIM3v1Alu0 * Inv_L
				  + model->BSIM3v1Awu0 * Inv_W
				  + model->BSIM3v1Apu0 * Inv_LW;
		  pParam->BSIM3v1Aute = model->BSIM3v1Aute
				   + model->BSIM3v1Alute * Inv_L
				   + model->BSIM3v1Awute * Inv_W
				   + model->BSIM3v1Apute * Inv_LW;
		  pParam->BSIM3v1Avoff = model->BSIM3v1Avoff
				    + model->BSIM3v1Alvoff * Inv_L
				    + model->BSIM3v1Awvoff * Inv_W
				    + model->BSIM3v1Apvoff * Inv_LW;
		  pParam->BSIM3v1Adelta = model->BSIM3v1Adelta
				     + model->BSIM3v1Aldelta * Inv_L
				     + model->BSIM3v1Awdelta * Inv_W
				     + model->BSIM3v1Apdelta * Inv_LW;
		  pParam->BSIM3v1Ardsw = model->BSIM3v1Ardsw
				    + model->BSIM3v1Alrdsw * Inv_L
				    + model->BSIM3v1Awrdsw * Inv_W
				    + model->BSIM3v1Aprdsw * Inv_LW;
		  pParam->BSIM3v1Aprwg = model->BSIM3v1Aprwg
				    + model->BSIM3v1Alprwg * Inv_L
				    + model->BSIM3v1Awprwg * Inv_W
				    + model->BSIM3v1Apprwg * Inv_LW;
		  pParam->BSIM3v1Aprwb = model->BSIM3v1Aprwb
				    + model->BSIM3v1Alprwb * Inv_L
				    + model->BSIM3v1Awprwb * Inv_W
				    + model->BSIM3v1Apprwb * Inv_LW;
		  pParam->BSIM3v1Aprt = model->BSIM3v1Aprt
				    + model->BSIM3v1Alprt * Inv_L
				    + model->BSIM3v1Awprt * Inv_W
				    + model->BSIM3v1Apprt * Inv_LW;
		  pParam->BSIM3v1Aeta0 = model->BSIM3v1Aeta0
				    + model->BSIM3v1Aleta0 * Inv_L
				    + model->BSIM3v1Aweta0 * Inv_W
				    + model->BSIM3v1Apeta0 * Inv_LW;
		  pParam->BSIM3v1Aetab = model->BSIM3v1Aetab
				    + model->BSIM3v1Aletab * Inv_L
				    + model->BSIM3v1Awetab * Inv_W
				    + model->BSIM3v1Apetab * Inv_LW;
		  pParam->BSIM3v1Apclm = model->BSIM3v1Apclm
				    + model->BSIM3v1Alpclm * Inv_L
				    + model->BSIM3v1Awpclm * Inv_W
				    + model->BSIM3v1Appclm * Inv_LW;
		  pParam->BSIM3v1Apdibl1 = model->BSIM3v1Apdibl1
				      + model->BSIM3v1Alpdibl1 * Inv_L
				      + model->BSIM3v1Awpdibl1 * Inv_W
				      + model->BSIM3v1Appdibl1 * Inv_LW;
		  pParam->BSIM3v1Apdibl2 = model->BSIM3v1Apdibl2
				      + model->BSIM3v1Alpdibl2 * Inv_L
				      + model->BSIM3v1Awpdibl2 * Inv_W
				      + model->BSIM3v1Appdibl2 * Inv_LW;
		  pParam->BSIM3v1Apdiblb = model->BSIM3v1Apdiblb
				      + model->BSIM3v1Alpdiblb * Inv_L
				      + model->BSIM3v1Awpdiblb * Inv_W
				      + model->BSIM3v1Appdiblb * Inv_LW;
		  pParam->BSIM3v1Apscbe1 = model->BSIM3v1Apscbe1
				      + model->BSIM3v1Alpscbe1 * Inv_L
				      + model->BSIM3v1Awpscbe1 * Inv_W
				      + model->BSIM3v1Appscbe1 * Inv_LW;
		  pParam->BSIM3v1Apscbe2 = model->BSIM3v1Apscbe2
				      + model->BSIM3v1Alpscbe2 * Inv_L
				      + model->BSIM3v1Awpscbe2 * Inv_W
				      + model->BSIM3v1Appscbe2 * Inv_LW;
		  pParam->BSIM3v1Apvag = model->BSIM3v1Apvag
				    + model->BSIM3v1Alpvag * Inv_L
				    + model->BSIM3v1Awpvag * Inv_W
				    + model->BSIM3v1Appvag * Inv_LW;
		  pParam->BSIM3v1Awr = model->BSIM3v1Awr
				  + model->BSIM3v1Alwr * Inv_L
				  + model->BSIM3v1Awwr * Inv_W
				  + model->BSIM3v1Apwr * Inv_LW;
		  pParam->BSIM3v1Adwg = model->BSIM3v1Adwg
				   + model->BSIM3v1Aldwg * Inv_L
				   + model->BSIM3v1Awdwg * Inv_W
				   + model->BSIM3v1Apdwg * Inv_LW;
		  pParam->BSIM3v1Adwb = model->BSIM3v1Adwb
				   + model->BSIM3v1Aldwb * Inv_L
				   + model->BSIM3v1Awdwb * Inv_W
				   + model->BSIM3v1Apdwb * Inv_LW;
		  pParam->BSIM3v1Ab0 = model->BSIM3v1Ab0
				  + model->BSIM3v1Alb0 * Inv_L
				  + model->BSIM3v1Awb0 * Inv_W
				  + model->BSIM3v1Apb0 * Inv_LW;
		  pParam->BSIM3v1Ab1 = model->BSIM3v1Ab1
				  + model->BSIM3v1Alb1 * Inv_L
				  + model->BSIM3v1Awb1 * Inv_W
				  + model->BSIM3v1Apb1 * Inv_LW;
		  pParam->BSIM3v1Aalpha0 = model->BSIM3v1Aalpha0
				      + model->BSIM3v1Alalpha0 * Inv_L
				      + model->BSIM3v1Awalpha0 * Inv_W
				      + model->BSIM3v1Apalpha0 * Inv_LW;
		  pParam->BSIM3v1Abeta0 = model->BSIM3v1Abeta0
				     + model->BSIM3v1Albeta0 * Inv_L
				     + model->BSIM3v1Awbeta0 * Inv_W
				     + model->BSIM3v1Apbeta0 * Inv_LW;
		  /* CV model */
		  pParam->BSIM3v1Aelm = model->BSIM3v1Aelm
				  + model->BSIM3v1Alelm * Inv_L
				  + model->BSIM3v1Awelm * Inv_W
				  + model->BSIM3v1Apelm * Inv_LW;
		  pParam->BSIM3v1Acgsl = model->BSIM3v1Acgsl
				    + model->BSIM3v1Alcgsl * Inv_L
				    + model->BSIM3v1Awcgsl * Inv_W
				    + model->BSIM3v1Apcgsl * Inv_LW;
		  pParam->BSIM3v1Acgdl = model->BSIM3v1Acgdl
				    + model->BSIM3v1Alcgdl * Inv_L
				    + model->BSIM3v1Awcgdl * Inv_W
				    + model->BSIM3v1Apcgdl * Inv_LW;
		  pParam->BSIM3v1Ackappa = model->BSIM3v1Ackappa
				      + model->BSIM3v1Alckappa * Inv_L
				      + model->BSIM3v1Awckappa * Inv_W
				      + model->BSIM3v1Apckappa * Inv_LW;
		  pParam->BSIM3v1Acf = model->BSIM3v1Acf
				  + model->BSIM3v1Alcf * Inv_L
				  + model->BSIM3v1Awcf * Inv_W
				  + model->BSIM3v1Apcf * Inv_LW;
		  pParam->BSIM3v1Aclc = model->BSIM3v1Aclc
				   + model->BSIM3v1Alclc * Inv_L
				   + model->BSIM3v1Awclc * Inv_W
				   + model->BSIM3v1Apclc * Inv_LW;
		  pParam->BSIM3v1Acle = model->BSIM3v1Acle
				   + model->BSIM3v1Alcle * Inv_L
				   + model->BSIM3v1Awcle * Inv_W
				   + model->BSIM3v1Apcle * Inv_LW;
                  pParam->BSIM3v1AabulkCVfactor = 1.0 + pow((pParam->BSIM3v1Aclc
					     / pParam->BSIM3v1Aleff),
					     pParam->BSIM3v1Acle);

                  pParam->BSIM3v1Acgdo = (model->BSIM3v1Acgdo + pParam->BSIM3v1Acf)
				    * pParam->BSIM3v1AweffCV;
                  pParam->BSIM3v1Acgso = (model->BSIM3v1Acgso + pParam->BSIM3v1Acf)
				    * pParam->BSIM3v1AweffCV;
                  pParam->BSIM3v1Acgbo = model->BSIM3v1Acgbo * pParam->BSIM3v1AleffCV;

	          T0 = (TRatio - 1.0);
	          pParam->BSIM3v1Aua = pParam->BSIM3v1Aua + pParam->BSIM3v1Aua1 * T0;
	          pParam->BSIM3v1Aub = pParam->BSIM3v1Aub + pParam->BSIM3v1Aub1 * T0;
	          pParam->BSIM3v1Auc = pParam->BSIM3v1Auc + pParam->BSIM3v1Auc1 * T0;

                  pParam->BSIM3v1Au0temp = pParam->BSIM3v1Au0
				      * pow(TRatio, pParam->BSIM3v1Aute); 
                  pParam->BSIM3v1Avsattemp = pParam->BSIM3v1Avsat - pParam->BSIM3v1Aat 
			                * T0;
	          pParam->BSIM3v1Ards0 = (pParam->BSIM3v1Ardsw + pParam->BSIM3v1Aprt * T0)
                                    / pow(pParam->BSIM3v1Aweff * 1E6, pParam->BSIM3v1Awr);

                  if (!model->BSIM3v1AnpeakGiven && model->BSIM3v1Agamma1Given)
                  {   T0 = pParam->BSIM3v1Agamma1 * model->BSIM3v1Acox;
                      pParam->BSIM3v1Anpeak = 3.021E22 * T0 * T0;
                  }

	          Vtm0 = KboQ * Tnom;
	          Eg = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
	          ni = 1.45e10 * (Tnom / 300.15) * sqrt(Tnom / 300.15) 
                     * exp(21.5565981 - Eg / (2.0 * Vtm0));

		  pParam->BSIM3v1Aphi = 2.0 * Vtm0 
			           * log(pParam->BSIM3v1Anpeak / ni);

	          pParam->BSIM3v1AsqrtPhi = sqrt(pParam->BSIM3v1Aphi);
	          pParam->BSIM3v1Aphis3 = pParam->BSIM3v1AsqrtPhi * pParam->BSIM3v1Aphi;

                  pParam->BSIM3v1AXdep0 = sqrt(2.0 * EPSSI / (Charge_q
				     * pParam->BSIM3v1Anpeak * 1.0e6))
                                     * pParam->BSIM3v1AsqrtPhi; 
                  pParam->BSIM3v1AsqrtXdep0 = sqrt(pParam->BSIM3v1AXdep0);
                  pParam->BSIM3v1Alitl = sqrt(3.0 * pParam->BSIM3v1Axj
				    * model->BSIM3v1Atox);
                  pParam->BSIM3v1Avbi = Vtm0 * log(1.0e20
			           * pParam->BSIM3v1Anpeak / (ni * ni));
                  pParam->BSIM3v1Acdep0 = sqrt(Charge_q * EPSSI
				     * pParam->BSIM3v1Anpeak * 1.0e6 / 2.0
				     / pParam->BSIM3v1Aphi);
        
                  if (model->BSIM3v1Ak1Given || model->BSIM3v1Ak2Given)
	          {   if (!model->BSIM3v1Ak1Given)
	              {   fprintf(stdout, "Warning: k1 should be specified with k2.\n");
                          pParam->BSIM3v1Ak1 = 0.53;
                      }
                      if (!model->BSIM3v1Ak2Given)
	              {   fprintf(stdout, "Warning: k2 should be specified with k1.\n");
                          pParam->BSIM3v1Ak2 = -0.0186;
                      }
                      if (model->BSIM3v1AnsubGiven)
                          fprintf(stdout, "Warning: nsub is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1AxtGiven)
                          fprintf(stdout, "Warning: xt is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1AvbxGiven)
                          fprintf(stdout, "Warning: vbx is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1AvbmGiven)
                          fprintf(stdout, "Warning: vbm is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1Agamma1Given)
                          fprintf(stdout, "Warning: gamma1 is ignored because k1 or k2 is given.\n");
                      if (model->BSIM3v1Agamma2Given)
                          fprintf(stdout, "Warning: gamma2 is ignored because k1 or k2 is given.\n");
                  }
                  else
	          {   if (!model->BSIM3v1AvbxGiven)
                          pParam->BSIM3v1Avbx = pParam->BSIM3v1Aphi - 7.7348e-4 
                                           * pParam->BSIM3v1Anpeak
					   * pParam->BSIM3v1Axt * pParam->BSIM3v1Axt;
	              if (pParam->BSIM3v1Avbx > 0.0)
		          pParam->BSIM3v1Avbx = -pParam->BSIM3v1Avbx;
	              if (pParam->BSIM3v1Avbm > 0.0)
                          pParam->BSIM3v1Avbm = -pParam->BSIM3v1Avbm;
           
                      if (!model->BSIM3v1Agamma1Given)
                          pParam->BSIM3v1Agamma1 = 5.753e-12
					      * sqrt(pParam->BSIM3v1Anpeak)
                                              / model->BSIM3v1Acox;
                      if (!model->BSIM3v1Agamma2Given)
                          pParam->BSIM3v1Agamma2 = 5.753e-12
					      * sqrt(pParam->BSIM3v1Ansub)
                                              / model->BSIM3v1Acox;

                      T0 = pParam->BSIM3v1Agamma1 - pParam->BSIM3v1Agamma2;
                      T1 = sqrt(pParam->BSIM3v1Aphi - pParam->BSIM3v1Avbx)
			 - pParam->BSIM3v1AsqrtPhi;
                      T2 = sqrt(pParam->BSIM3v1Aphi * (pParam->BSIM3v1Aphi
			 - pParam->BSIM3v1Avbm)) - pParam->BSIM3v1Aphi;
                      pParam->BSIM3v1Ak2 = T0 * T1 / (2.0 * T2 + pParam->BSIM3v1Avbm);
                      pParam->BSIM3v1Ak1 = pParam->BSIM3v1Agamma2 - 2.0
				      * pParam->BSIM3v1Ak2 * sqrt(pParam->BSIM3v1Aphi
				      - pParam->BSIM3v1Avbm);
                  }
 
		  if (pParam->BSIM3v1Ak2 > 0.0)
		  {   T0 = 0.5 * pParam->BSIM3v1Ak1 / pParam->BSIM3v1Ak2;
                      pParam->BSIM3v1Avbsc = 0.9 * (pParam->BSIM3v1Aphi - T0 * T0);
		      if (pParam->BSIM3v1Avbsc > -3.0)
		          pParam->BSIM3v1Avbsc = -3.0;
		      else if (pParam->BSIM3v1Avbsc < -30.0)
		          pParam->BSIM3v1Avbsc = -30.0;
		  }
		  else
		  {   pParam->BSIM3v1Avbsc = -10.0;
		  }

	          model->BSIM3v1Avtm = KboQ * Temp;

	          if (model->BSIM3v1Avth0Given)
                      pParam->BSIM3v1Avfb = model->BSIM3v1Atype * pParam->BSIM3v1Avth0 
                                       - pParam->BSIM3v1Aphi - pParam->BSIM3v1Ak1 
                                       * pParam->BSIM3v1AsqrtPhi;
                  else
                      pParam->BSIM3v1Avth0 = model->BSIM3v1Atype * (-1.0
                                        + pParam->BSIM3v1Aphi + pParam->BSIM3v1Ak1 
                                        * pParam->BSIM3v1AsqrtPhi);

                  T1 = sqrt(EPSSI / EPSOX * model->BSIM3v1Atox
		     * pParam->BSIM3v1AXdep0);
                  T0 = exp(-0.5 * pParam->BSIM3v1Adsub * pParam->BSIM3v1Aleff / T1);
                  pParam->BSIM3v1Atheta0vb0 = (T0 + 2.0 * T0 * T0);

                  T0 = exp(-0.5 * pParam->BSIM3v1Adrout * pParam->BSIM3v1Aleff / T1);
                  T2 = (T0 + 2.0 * T0 * T0);
                  pParam->BSIM3v1AthetaRout = pParam->BSIM3v1Apdibl1 * T2
				         + pParam->BSIM3v1Apdibl2;

                  /* process source/drain series resistance */
                  here->BSIM3v1AdrainConductance = model->BSIM3v1AsheetResistance 
		                              * here->BSIM3v1AdrainSquares;
                  if (here->BSIM3v1AdrainConductance > 0.0)
                      here->BSIM3v1AdrainConductance = 1.0
						  / here->BSIM3v1AdrainConductance;
	          else
                      here->BSIM3v1AdrainConductance = 0.0;
                   
                  here->BSIM3v1AsourceConductance = model->BSIM3v1AsheetResistance 
		                               * here->BSIM3v1AsourceSquares;
                  if (here->BSIM3v1AsourceConductance > 0.0) 
                      here->BSIM3v1AsourceConductance = 1.0
						   / here->BSIM3v1AsourceConductance;
	          else
                      here->BSIM3v1AsourceConductance = 0.0;
	      }
	      here->BSIM3v1Acgso = pParam->BSIM3v1Acgso;
	      here->BSIM3v1Acgdo = pParam->BSIM3v1Acgdo;
         }
    }
    return(OK);
}




