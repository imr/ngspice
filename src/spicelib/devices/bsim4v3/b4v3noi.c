/**** BSIM4.3.0 Released by Xuemei (Jane)  Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3noi.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "bsim4v3def.h"
#include "cktdefs.h"
#include "iferrmsg.h"
#include "noisedef.h"
#include "const.h"


extern void   NevalSrc();
extern double Nintegrate();

/*
 * WDL: 1/f noise model has been smoothed out and enhanced with
 * bulk charge effect as well as physical N* equ. and necessary
 * conversion into the SI unit system.
 */

static double
BSIM4v3Eval1ovFNoise(Vds, model, here, freq, temp)
double Vds, freq, temp;
BSIM4v3model *model;
BSIM4v3instance *here;
{
struct bsim4v3SizeDependParam *pParam;
double cd, esat, DelClm, EffFreq, N0, Nl;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, Ssi;

    pParam = here->pParam;
    cd = fabs(here->BSIM4v3cd);
    esat = 2.0 * pParam->BSIM4v3vsattemp / here->BSIM4v3ueff;
    if(model->BSIM4v3em<=0.0) DelClm = 0.0; /* flicker noise modified -JX  */
    else {
    	T0 = ((((Vds - here->BSIM4v3Vdseff) / pParam->BSIM4v3litl)
       		+ model->BSIM4v3em) / esat);
    	DelClm = pParam->BSIM4v3litl * log (MAX(T0, N_MINLOG));
    }
    EffFreq = pow(freq, model->BSIM4v3ef);
    T1 = CHARGE * CHARGE * CONSTboltz * cd * temp * here->BSIM4v3ueff;
    T2 = 1.0e10 * EffFreq * here->BSIM4v3Abulk * model->BSIM4v3coxe
       * pParam->BSIM4v3leff * pParam->BSIM4v3leff;
    N0 = model->BSIM4v3coxe * here->BSIM4v3Vgsteff / CHARGE;
    Nl = model->BSIM4v3coxe * here->BSIM4v3Vgsteff
       * (1.0 - here->BSIM4v3AbovVgst2Vtm * here->BSIM4v3Vdseff) / CHARGE;

    T3 = model->BSIM4v3oxideTrapDensityA
       * log(MAX(((N0 + here->BSIM4v3nstar) / (Nl + here->BSIM4v3nstar)), N_MINLOG));
    T4 = model->BSIM4v3oxideTrapDensityB * (N0 - Nl);
    T5 = model->BSIM4v3oxideTrapDensityC * 0.5 * (N0 * N0 - Nl * Nl);

    T6 = CONSTboltz * temp * cd * cd;
    T7 = 1.0e10 * EffFreq * pParam->BSIM4v3leff
       * pParam->BSIM4v3leff * pParam->BSIM4v3weff;
    T8 = model->BSIM4v3oxideTrapDensityA + model->BSIM4v3oxideTrapDensityB * Nl
       + model->BSIM4v3oxideTrapDensityC * Nl * Nl;
    T9 = (Nl + here->BSIM4v3nstar) * (Nl + here->BSIM4v3nstar);
    Ssi = T1 / T2 * (T3 + T4 + T5) + T6 / T7 * DelClm * T8 / T9;
    return Ssi;
}


int
BSIM4v3noise (mode, operation, inModel, ckt, data, OnDens)
int mode, operation;
GENmodel *inModel;
CKTcircuit *ckt;
Ndata *data;
double *OnDens;
{
BSIM4v3model *model = (BSIM4v3model *)inModel;
BSIM4v3instance *here;
struct bsim4v3SizeDependParam *pParam;
char name[N_MXVLNTH];
double tempOnoise;
double tempInoise;
double noizDens[BSIM4v3NSRCS];
double lnNdens[BSIM4v3NSRCS];

double N0, Nl;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13;
double Vds, n, ExpArg, Ssi, Swi;
double tmp, gdpr, gspr, npart_theta, npart_beta, igsquare;

int error, i;

    /* define the names of the noise sources */
    static char *BSIM4v3nNames[BSIM4v3NSRCS] =
    {   /* Note that we have to keep the order */
	".rd",              /* noise due to rd */
	".rs",              /* noise due to rs */
        ".rg",              /* noise due to rgeltd */
        ".rbps",            /* noise due to rbps */
        ".rbpd",            /* noise due to rbpd */
        ".rbpb",            /* noise due to rbpb */
        ".rbsb",            /* noise due to rbsb */
        ".rbdb",            /* noise due to rbdb */
	".id",              /* noise due to id */
	".1overf",          /* flicker (1/f) noise */
        ".igs",             /* shot noise due to IGS */
        ".igd",             /* shot noise due to IGD */
        ".igb",             /* shot noise due to IGB */
	""                  /* total transistor noise */
    };

    for (; model != NULL; model = model->BSIM4v3nextModel)
    {    for (here = model->BSIM4v3instances; here != NULL;
	      here = here->BSIM4v3nextInstance)
	 {    pParam = here->pParam;
	      switch (operation)
	      {  case N_OPEN:
		     /* see if we have to to produce a summary report */
		     /* if so, name all the noise generators */

		      if (((NOISEAN*)ckt->CKTcurJob)->NStpsSm != 0)
		      {   switch (mode)
			  {  case N_DENS:
			          for (i = 0; i < BSIM4v3NSRCS; i++)
				  {    (void) sprintf(name, "onoise.%s%s",
					              here->BSIM4v3name,
						      BSIM4v3nNames[i]);
                                       data->namelist = (IFuid *) trealloc(
					     (char *) data->namelist,
					     (data->numPlots + 1)
					     * sizeof(IFuid));
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       (*(SPfrontEnd->IFnewUid)) (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  (IFuid) NULL, name, UID_OTHER,
					  (void **) NULL);
				       /* we've added one more plot */
			          }
			          break;
		             case INT_NOIZ:
			          for (i = 0; i < BSIM4v3NSRCS; i++)
				  {    (void) sprintf(name, "onoise_total.%s%s",
						      here->BSIM4v3name,
						      BSIM4v3nNames[i]);
                                       data->namelist = (IFuid *) trealloc(
					     (char *) data->namelist,
					     (data->numPlots + 1)
					     * sizeof(IFuid));
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       (*(SPfrontEnd->IFnewUid)) (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  (IFuid) NULL, name, UID_OTHER,
					  (void **) NULL);
				       /* we've added one more plot */

			               (void) sprintf(name, "inoise_total.%s%s",
						      here->BSIM4v3name,
						      BSIM4v3nNames[i]);
                                       data->namelist = (IFuid *) trealloc(
					     (char *) data->namelist,
					     (data->numPlots + 1)
					     * sizeof(IFuid));
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       (*(SPfrontEnd->IFnewUid)) (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  (IFuid) NULL, name, UID_OTHER,
					  (void **)NULL);
				       /* we've added one more plot */
			          }
			          break;
		          }
		      }
		      break;
	         case N_CALC:
		      switch (mode)
		      {  case N_DENS:
			      if (model->BSIM4v3tnoiMod == 0)
			      {   if (model->BSIM4v3rdsMod == 0)
				  {   gspr = here->BSIM4v3sourceConductance;
                                      gdpr = here->BSIM4v3drainConductance;
				      if (here->BSIM4v3grdsw > 0.0)
				          tmp = 1.0 / here->BSIM4v3grdsw; /* tmp used below */ 
				      else
					  tmp = 0.0;
				  }
				  else
				  {   gspr = here->BSIM4v3gstot;
                                      gdpr = here->BSIM4v3gdtot;
                                      tmp = 0.0;
				  }
			      }
			      else
			      {   T5 = here->BSIM4v3Vgsteff / here->BSIM4v3EsatL;
				  T5 *= T5;
				  npart_beta = model->BSIM4v3rnoia * (1.0 + T5
					     * model->BSIM4v3tnoia * pParam->BSIM4v3leff);
				  npart_theta = model->BSIM4v3rnoib * (1.0 + T5
                                              * model->BSIM4v3tnoib * pParam->BSIM4v3leff);

				  if (model->BSIM4v3rdsMod == 0)
                                  {   gspr = here->BSIM4v3sourceConductance;
                                      gdpr = here->BSIM4v3drainConductance;
                                  }
                                  else
                                  {   gspr = here->BSIM4v3gstot;
                                      gdpr = here->BSIM4v3gdtot;
                                  }

				  if ((*(ckt->CKTstates[0] + here->BSIM4v3vds)) >= 0.0)
			              gspr = gspr / (1.0 + npart_theta * npart_theta * gspr
				 	   / here->BSIM4v3IdovVds);  /* bugfix */
				  else
				      gdpr = gdpr / (1.0 + npart_theta * npart_theta * gdpr
                                           / here->BSIM4v3IdovVds);
			      } 

		              NevalSrc(&noizDens[BSIM4v3RDNOIZ],
				       &lnNdens[BSIM4v3RDNOIZ], ckt, THERMNOISE,
				       here->BSIM4v3dNodePrime, here->BSIM4v3dNode,
				       gdpr);

		              NevalSrc(&noizDens[BSIM4v3RSNOIZ],
				       &lnNdens[BSIM4v3RSNOIZ], ckt, THERMNOISE,
				       here->BSIM4v3sNodePrime, here->BSIM4v3sNode,
				       gspr);


			      if ((here->BSIM4v3rgateMod == 1) || (here->BSIM4v3rgateMod == 2))
			      {   NevalSrc(&noizDens[BSIM4v3RGNOIZ],
                                       &lnNdens[BSIM4v3RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v3gNodePrime, here->BSIM4v3gNodeExt,
                                       here->BSIM4v3grgeltd);
			      }
			      else if (here->BSIM4v3rgateMod == 3)
			      {   NevalSrc(&noizDens[BSIM4v3RGNOIZ],
                                       &lnNdens[BSIM4v3RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v3gNodeMid, here->BSIM4v3gNodeExt,
                                       here->BSIM4v3grgeltd);
			      }
			      else
			      {    noizDens[BSIM4v3RGNOIZ] = 0.0;
                                   lnNdens[BSIM4v3RGNOIZ] =
                                          log(MAX(noizDens[BSIM4v3RGNOIZ], N_MINLOG));
			      }


                              if (here->BSIM4v3rbodyMod)
                              {   NevalSrc(&noizDens[BSIM4v3RBPSNOIZ],
                                       &lnNdens[BSIM4v3RBPSNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v3bNodePrime, here->BSIM4v3sbNode,
                                       here->BSIM4v3grbps);
                                  NevalSrc(&noizDens[BSIM4v3RBPDNOIZ],
                                       &lnNdens[BSIM4v3RBPDNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v3bNodePrime, here->BSIM4v3dbNode,
                                       here->BSIM4v3grbpd);
                                  NevalSrc(&noizDens[BSIM4v3RBPBNOIZ],
                                       &lnNdens[BSIM4v3RBPBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v3bNodePrime, here->BSIM4v3bNode,
                                       here->BSIM4v3grbpb);
                                  NevalSrc(&noizDens[BSIM4v3RBSBNOIZ],
                                       &lnNdens[BSIM4v3RBSBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v3bNode, here->BSIM4v3sbNode,
                                       here->BSIM4v3grbsb);
                                  NevalSrc(&noizDens[BSIM4v3RBDBNOIZ],
                                       &lnNdens[BSIM4v3RBDBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v3bNode, here->BSIM4v3dbNode,
                                       here->BSIM4v3grbdb);
                              }
                              else
                              {   noizDens[BSIM4v3RBPSNOIZ] = noizDens[BSIM4v3RBPDNOIZ] = 0.0;   
                                  noizDens[BSIM4v3RBPBNOIZ] = 0.0;
                                  noizDens[BSIM4v3RBSBNOIZ] = noizDens[BSIM4v3RBDBNOIZ] = 0.0;
                                  lnNdens[BSIM4v3RBPSNOIZ] =
                                          log(MAX(noizDens[BSIM4v3RBPSNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v3RBPDNOIZ] =
                                          log(MAX(noizDens[BSIM4v3RBPDNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v3RBPBNOIZ] =
                                          log(MAX(noizDens[BSIM4v3RBPBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v3RBSBNOIZ] =
                                          log(MAX(noizDens[BSIM4v3RBSBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v3RBDBNOIZ] =
                                          log(MAX(noizDens[BSIM4v3RBDBNOIZ], N_MINLOG));
                              }


                              switch(model->BSIM4v3tnoiMod)
			      {  case 0:
				      T0 = here->BSIM4v3ueff * fabs(here->BSIM4v3qinv);
				      T1 = T0 * tmp + pParam->BSIM4v3leff
                                         * pParam->BSIM4v3leff;
		                      NevalSrc(&noizDens[BSIM4v3IDNOIZ],
				               &lnNdens[BSIM4v3IDNOIZ], ckt,
					       THERMNOISE, here->BSIM4v3dNodePrime,
                                               here->BSIM4v3sNodePrime,
					       (T0 / T1) * model->BSIM4v3ntnoi);
				      break;
				 case 1:
				      T0 = here->BSIM4v3gm + here->BSIM4v3gmbs + here->BSIM4v3gds;
				      T0 *= T0;
				      igsquare = npart_theta * npart_theta * T0 / here->BSIM4v3IdovVds;
				      T1 = npart_beta * (here->BSIM4v3gm
					 + here->BSIM4v3gmbs) + here->BSIM4v3gds;
				      T2 = T1 * T1 / here->BSIM4v3IdovVds;
                                      NevalSrc(&noizDens[BSIM4v3IDNOIZ],
                                               &lnNdens[BSIM4v3IDNOIZ], ckt,
                                               THERMNOISE, here->BSIM4v3dNodePrime,
                                               here->BSIM4v3sNodePrime, (T2 - igsquare));
                                      break;
			      }

		              NevalSrc(&noizDens[BSIM4v3FLNOIZ], (double*) NULL,
				       ckt, N_GAIN, here->BSIM4v3dNodePrime,
				       here->BSIM4v3sNodePrime, (double) 0.0);

                              switch(model->BSIM4v3fnoiMod)
			      {  case 0:
			              noizDens[BSIM4v3FLNOIZ] *= model->BSIM4v3kf
					    * exp(model->BSIM4v3af
					    * log(MAX(fabs(here->BSIM4v3cd),
					    N_MINLOG)))
					    / (pow(data->freq, model->BSIM4v3ef)
					    * pParam->BSIM4v3leff
				            * pParam->BSIM4v3leff
					    * model->BSIM4v3coxe);
				      break;
			         case 1:
		                      Vds = *(ckt->CKTstates[0] + here->BSIM4v3vds);
			              if (Vds < 0.0)
			                  Vds = -Vds;

                                      Ssi = BSIM4v3Eval1ovFNoise(Vds, model, here,
                                          data->freq, ckt->CKTtemp);
                                      T10 = model->BSIM4v3oxideTrapDensityA
                                          * CONSTboltz * ckt->CKTtemp;
                                      T11 = pParam->BSIM4v3weff * pParam->BSIM4v3leff
                                          * pow(data->freq, model->BSIM4v3ef) * 1.0e10
					  * here->BSIM4v3nstar * here->BSIM4v3nstar;
                                      Swi = T10 / T11 * here->BSIM4v3cd
                                          * here->BSIM4v3cd;
                                      T1 = Swi + Ssi;
                                      if (T1 > 0.0)
                                          noizDens[BSIM4v3FLNOIZ] *= (Ssi * Swi) / T1;
                                      else
                                          noizDens[BSIM4v3FLNOIZ] *= 0.0;
				      break;
			      }

		              lnNdens[BSIM4v3FLNOIZ] =
				     log(MAX(noizDens[BSIM4v3FLNOIZ], N_MINLOG));


                       	if(here->BSIM4v3mode >= 0) {  /* bugfix  */
                              NevalSrc(&noizDens[BSIM4v3IGSNOIZ],
                                   &lnNdens[BSIM4v3IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v3gNodePrime, here->BSIM4v3sNodePrime,
                                   (here->BSIM4v3Igs + here->BSIM4v3Igcs));
                              NevalSrc(&noizDens[BSIM4v3IGDNOIZ],
                                   &lnNdens[BSIM4v3IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v3gNodePrime, here->BSIM4v3dNodePrime,
                                   (here->BSIM4v3Igd + here->BSIM4v3Igcd));
			} else {
                              NevalSrc(&noizDens[BSIM4v3IGSNOIZ],
                                   &lnNdens[BSIM4v3IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v3gNodePrime, here->BSIM4v3sNodePrime,
                                   (here->BSIM4v3Igs + here->BSIM4v3Igcd));
                              NevalSrc(&noizDens[BSIM4v3IGDNOIZ],
                                   &lnNdens[BSIM4v3IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v3gNodePrime, here->BSIM4v3dNodePrime,
                                   (here->BSIM4v3Igd + here->BSIM4v3Igcs));
                        }
                              NevalSrc(&noizDens[BSIM4v3IGBNOIZ],
                                   &lnNdens[BSIM4v3IGBNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v3gNodePrime, here->BSIM4v3bNodePrime,
                                   here->BSIM4v3Igb);


		              noizDens[BSIM4v3TOTNOIZ] = noizDens[BSIM4v3RDNOIZ]
				     + noizDens[BSIM4v3RSNOIZ] + noizDens[BSIM4v3RGNOIZ]
				     + noizDens[BSIM4v3RBPSNOIZ] + noizDens[BSIM4v3RBPDNOIZ]
				     + noizDens[BSIM4v3RBPBNOIZ]
				     + noizDens[BSIM4v3RBSBNOIZ] + noizDens[BSIM4v3RBDBNOIZ]
				     + noizDens[BSIM4v3IDNOIZ] + noizDens[BSIM4v3FLNOIZ]
                                     + noizDens[BSIM4v3IGSNOIZ] + noizDens[BSIM4v3IGDNOIZ]
                                     + noizDens[BSIM4v3IGBNOIZ];
		              lnNdens[BSIM4v3TOTNOIZ] = 
				     log(MAX(noizDens[BSIM4v3TOTNOIZ], N_MINLOG));

		              *OnDens += noizDens[BSIM4v3TOTNOIZ];

		              if (data->delFreq == 0.0)
			      {   /* if we haven't done any previous 
				     integration, we need to initialize our
				     "history" variables.
				    */

			          for (i = 0; i < BSIM4v3NSRCS; i++)
				  {    here->BSIM4v3nVar[LNLSTDENS][i] =
					     lnNdens[i];
			          }

			          /* clear out our integration variables
				     if it's the first pass
				   */
			          if (data->freq ==
				      ((NOISEAN*) ckt->CKTcurJob)->NstartFreq)
				  {   for (i = 0; i < BSIM4v3NSRCS; i++)
				      {    here->BSIM4v3nVar[OUTNOIZ][i] = 0.0;
				           here->BSIM4v3nVar[INNOIZ][i] = 0.0;
			              }
			          }
		              }
			      else
			      {   /* data->delFreq != 0.0,
				     we have to integrate.
				   */
			          for (i = 0; i < BSIM4v3NSRCS; i++)
				  {    if (i != BSIM4v3TOTNOIZ)
				       {   tempOnoise = Nintegrate(noizDens[i],
						lnNdens[i],
				                here->BSIM4v3nVar[LNLSTDENS][i],
						data);
				           tempInoise = Nintegrate(noizDens[i]
						* data->GainSqInv, lnNdens[i]
						+ data->lnGainInv,
				                here->BSIM4v3nVar[LNLSTDENS][i]
						+ data->lnGainInv, data);
				           here->BSIM4v3nVar[LNLSTDENS][i] =
						lnNdens[i];
				           data->outNoiz += tempOnoise;
				           data->inNoise += tempInoise;
				           if (((NOISEAN*)
					       ckt->CKTcurJob)->NStpsSm != 0)
					   {   here->BSIM4v3nVar[OUTNOIZ][i]
						     += tempOnoise;
				               here->BSIM4v3nVar[OUTNOIZ][BSIM4v3TOTNOIZ]
						     += tempOnoise;
				               here->BSIM4v3nVar[INNOIZ][i]
						     += tempInoise;
				               here->BSIM4v3nVar[INNOIZ][BSIM4v3TOTNOIZ]
						     += tempInoise;
                                           }
			               }
			          }
		              }
		              if (data->prtSummary)
			      {   for (i = 0; i < BSIM4v3NSRCS; i++)
				  {    /* print a summary report */
			               data->outpVector[data->outNumber++]
					     = noizDens[i];
			          }
		              }
		              break;
		         case INT_NOIZ:
			      /* already calculated, just output */
		              if (((NOISEAN*)ckt->CKTcurJob)->NStpsSm != 0)
			      {   for (i = 0; i < BSIM4v3NSRCS; i++)
				  {    data->outpVector[data->outNumber++]
					     = here->BSIM4v3nVar[OUTNOIZ][i];
			               data->outpVector[data->outNumber++]
					     = here->BSIM4v3nVar[INNOIZ][i];
			          }
		              }
		              break;
		      }
		      break;
	         case N_CLOSE:
		      /* do nothing, the main calling routine will close */
		      return (OK);
		      break;   /* the plots */
	      }       /* switch (operation) */
	 }    /* for here */
    }    /* for model */

    return(OK);
}
