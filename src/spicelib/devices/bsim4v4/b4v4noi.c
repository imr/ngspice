/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4noi.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 **********/

#include "ngspice.h"
#include "bsim4v4def.h"
#include "cktdefs.h"
#include "iferrmsg.h"
#include "noisedef.h"
#include "suffix.h"
#include "const.h"


extern void   NevalSrc();
extern double Nintegrate();

/*
 * WDL: 1/f noise model has been smoothed out and enhanced with
 * bulk charge effect as well as physical N* equ. and necessary
 * conversion into the SI unit system.
 */

static double
Eval1ovFNoise(Vds, model, here, freq, temp)
double Vds, freq, temp;
BSIM4V4model *model;
BSIM4V4instance *here;
{
struct bsim4SizeDependParam *pParam;
double cd, esat, DelClm, EffFreq, N0, Nl, Leff, Leffsq;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, Ssi;

    pParam = here->pParam;
    cd = fabs(here->BSIM4V4cd);
    Leff = pParam->BSIM4V4leff - 2.0 * model->BSIM4V4lintnoi;
    Leffsq = Leff * Leff;
    esat = 2.0 * here->BSIM4V4vsattemp / here->BSIM4V4ueff;
    if(model->BSIM4V4em<=0.0) DelClm = 0.0; /* flicker noise modified -JX  */
    else {
    	T0 = ((((Vds - here->BSIM4V4Vdseff) / pParam->BSIM4V4litl)
       		+ model->BSIM4V4em) / esat);
    	DelClm = pParam->BSIM4V4litl * log (MAX(T0, N_MINLOG));
    }
    EffFreq = pow(freq, model->BSIM4V4ef);
    T1 = CHARGE * CHARGE * CONSTboltz * cd * temp * here->BSIM4V4ueff;
    T2 = 1.0e10 * EffFreq * here->BSIM4V4Abulk * model->BSIM4V4coxe * Leffsq;
    N0 = model->BSIM4V4coxe * here->BSIM4V4Vgsteff / CHARGE;
    Nl = model->BSIM4V4coxe * here->BSIM4V4Vgsteff
       * (1.0 - here->BSIM4V4AbovVgst2Vtm * here->BSIM4V4Vdseff) / CHARGE;

    T3 = model->BSIM4V4oxideTrapDensityA
       * log(MAX(((N0 + here->BSIM4V4nstar) / (Nl + here->BSIM4V4nstar)), N_MINLOG));
    T4 = model->BSIM4V4oxideTrapDensityB * (N0 - Nl);
    T5 = model->BSIM4V4oxideTrapDensityC * 0.5 * (N0 * N0 - Nl * Nl);

    T6 = CONSTboltz * temp * cd * cd;
    T7 = 1.0e10 * EffFreq * Leffsq * pParam->BSIM4V4weff;
    T8 = model->BSIM4V4oxideTrapDensityA + model->BSIM4V4oxideTrapDensityB * Nl
       + model->BSIM4V4oxideTrapDensityC * Nl * Nl;
    T9 = (Nl + here->BSIM4V4nstar) * (Nl + here->BSIM4V4nstar);
    Ssi = T1 / T2 * (T3 + T4 + T5) + T6 / T7 * DelClm * T8 / T9;
    return Ssi;
}


int
BSIM4V4noise (mode, operation, inModel, ckt, data, OnDens)
int mode, operation;
GENmodel *inModel;
CKTcircuit *ckt;
Ndata *data;
double *OnDens;
{
BSIM4V4model *model = (BSIM4V4model *)inModel;
BSIM4V4instance *here;
struct bsim4SizeDependParam *pParam;
char name[N_MXVLNTH];
double tempOnoise;
double tempInoise;
double noizDens[BSIM4V4NSRCS];
double lnNdens[BSIM4V4NSRCS];

double T0, T1, T2, T5, T10, T11;
double Vds, Ssi, Swi;
double tmp=0.0, gdpr, gspr, npart_theta=0.0, npart_beta=0.0, igsquare;

double m;

int i;

    /* define the names of the noise sources */
    static char *BSIM4V4nNames[BSIM4V4NSRCS] =
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

    for (; model != NULL; model = model->BSIM4V4nextModel)
    {    for (here = model->BSIM4V4instances; here != NULL;
	      here = here->BSIM4V4nextInstance)
	 {    pParam = here->pParam;
	      switch (operation)
	      {  case N_OPEN:
		     /* see if we have to to produce a summary report */
		     /* if so, name all the noise generators */

		      if (((NOISEAN*)ckt->CKTcurJob)->NStpsSm != 0)
		      {   switch (mode)
			  {  case N_DENS:
			          for (i = 0; i < BSIM4V4NSRCS; i++)
				  {    (void) sprintf(name, "onoise.%s%s",
					              here->BSIM4V4name,
						      BSIM4V4nNames[i]);
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
			          for (i = 0; i < BSIM4V4NSRCS; i++)
				  {    (void) sprintf(name, "onoise_total.%s%s",
						      here->BSIM4V4name,
						      BSIM4V4nNames[i]);
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
						      here->BSIM4V4name,
						      BSIM4V4nNames[i]);
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
		      m = here->BSIM4V4m;
		      switch (mode)
		      {  case N_DENS:
			      if (model->BSIM4V4tnoiMod == 0)
			      {   if (model->BSIM4V4rdsMod == 0)
				  {   gspr = here->BSIM4V4sourceConductance;
                                      gdpr = here->BSIM4V4drainConductance;
				      if (here->BSIM4V4grdsw > 0.0)
				          tmp = 1.0 / here->BSIM4V4grdsw; /* tmp used below */ 
				      else
					  tmp = 0.0;
				  }
				  else
				  {   gspr = here->BSIM4V4gstot;
                                      gdpr = here->BSIM4V4gdtot;
                                      tmp = 0.0;
				  }
			      }
			      else
			      {   T5 = here->BSIM4V4Vgsteff / here->BSIM4V4EsatL;
				  T5 *= T5;
				  npart_beta = model->BSIM4V4rnoia * (1.0 + T5
					     * model->BSIM4V4tnoia * pParam->BSIM4V4leff);
				  npart_theta = model->BSIM4V4rnoib * (1.0 + T5
                                              * model->BSIM4V4tnoib * pParam->BSIM4V4leff);

				  if (model->BSIM4V4rdsMod == 0)
                                  {   gspr = here->BSIM4V4sourceConductance;
                                      gdpr = here->BSIM4V4drainConductance;
                                  }
                                  else
                                  {   gspr = here->BSIM4V4gstot;
                                      gdpr = here->BSIM4V4gdtot;
                                  }

				  if ((*(ckt->CKTstates[0] + here->BSIM4V4vds)) >= 0.0)
			              gspr = gspr / (1.0 + npart_theta * npart_theta * gspr
				 	   / here->BSIM4V4IdovVds);  /* bugfix */
				  else
				      gdpr = gdpr / (1.0 + npart_theta * npart_theta * gdpr
                                           / here->BSIM4V4IdovVds);
			      } 

		              NevalSrc(&noizDens[BSIM4V4RDNOIZ],
				       &lnNdens[BSIM4V4RDNOIZ], ckt, THERMNOISE,
				       here->BSIM4V4dNodePrime, here->BSIM4V4dNode,
				       gdpr * m);

		              NevalSrc(&noizDens[BSIM4V4RSNOIZ],
				       &lnNdens[BSIM4V4RSNOIZ], ckt, THERMNOISE,
				       here->BSIM4V4sNodePrime, here->BSIM4V4sNode,
				       gspr * m);


			      if ((here->BSIM4V4rgateMod == 1) || (here->BSIM4V4rgateMod == 2))
			      {   NevalSrc(&noizDens[BSIM4V4RGNOIZ],
                                       &lnNdens[BSIM4V4RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4V4gNodePrime, here->BSIM4V4gNodeExt,
                                       here->BSIM4V4grgeltd * m);
			      }
			      else if (here->BSIM4V4rgateMod == 3)
			      {   NevalSrc(&noizDens[BSIM4V4RGNOIZ],
                                       &lnNdens[BSIM4V4RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4V4gNodeMid, here->BSIM4V4gNodeExt,
                                       here->BSIM4V4grgeltd * m);
			      }
			      else
			      {    noizDens[BSIM4V4RGNOIZ] = 0.0;
                                   lnNdens[BSIM4V4RGNOIZ] =
                                          log(MAX(noizDens[BSIM4V4RGNOIZ], N_MINLOG));
			      }


                              if (here->BSIM4V4rbodyMod)
                              {   NevalSrc(&noizDens[BSIM4V4RBPSNOIZ],
                                       &lnNdens[BSIM4V4RBPSNOIZ], ckt, THERMNOISE,
                                       here->BSIM4V4bNodePrime, here->BSIM4V4sbNode,
                                       here->BSIM4V4grbps * m);
                                  NevalSrc(&noizDens[BSIM4V4RBPDNOIZ],
                                       &lnNdens[BSIM4V4RBPDNOIZ], ckt, THERMNOISE,
                                       here->BSIM4V4bNodePrime, here->BSIM4V4dbNode,
                                       here->BSIM4V4grbpd * m);
                                  NevalSrc(&noizDens[BSIM4V4RBPBNOIZ],
                                       &lnNdens[BSIM4V4RBPBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4V4bNodePrime, here->BSIM4V4bNode,
                                       here->BSIM4V4grbpb * m);
                                  NevalSrc(&noizDens[BSIM4V4RBSBNOIZ],
                                       &lnNdens[BSIM4V4RBSBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4V4bNode, here->BSIM4V4sbNode,
                                       here->BSIM4V4grbsb * m);
                                  NevalSrc(&noizDens[BSIM4V4RBDBNOIZ],
                                       &lnNdens[BSIM4V4RBDBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4V4bNode, here->BSIM4V4dbNode,
                                       here->BSIM4V4grbdb * m);
                              }
                              else
                              {   noizDens[BSIM4V4RBPSNOIZ] = noizDens[BSIM4V4RBPDNOIZ] = 0.0;   
                                  noizDens[BSIM4V4RBPBNOIZ] = 0.0;
                                  noizDens[BSIM4V4RBSBNOIZ] = noizDens[BSIM4V4RBDBNOIZ] = 0.0;
                                  lnNdens[BSIM4V4RBPSNOIZ] =
                                          log(MAX(noizDens[BSIM4V4RBPSNOIZ], N_MINLOG));
                                  lnNdens[BSIM4V4RBPDNOIZ] =
                                          log(MAX(noizDens[BSIM4V4RBPDNOIZ], N_MINLOG));
                                  lnNdens[BSIM4V4RBPBNOIZ] =
                                          log(MAX(noizDens[BSIM4V4RBPBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4V4RBSBNOIZ] =
                                          log(MAX(noizDens[BSIM4V4RBSBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4V4RBDBNOIZ] =
                                          log(MAX(noizDens[BSIM4V4RBDBNOIZ], N_MINLOG));
                              }


                              switch(model->BSIM4V4tnoiMod)
			      {  case 0:
				      T0 = m * here->BSIM4V4ueff * fabs(here->BSIM4V4qinv);
				      T1 = T0 * tmp + pParam->BSIM4V4leff
                                         * pParam->BSIM4V4leff;
		                      NevalSrc(&noizDens[BSIM4V4IDNOIZ],
				               &lnNdens[BSIM4V4IDNOIZ], ckt,
					       THERMNOISE, here->BSIM4V4dNodePrime,
                                               here->BSIM4V4sNodePrime,
					       (T0 / T1) * model->BSIM4V4ntnoi);
				      break;
				 case 1:
				      T0 = m * (here->BSIM4V4gm + here->BSIM4V4gmbs + here->BSIM4V4gds);
				      T0 *= T0;
				      igsquare = npart_theta * npart_theta * T0 / here->BSIM4V4IdovVds;
				      T1 = npart_beta * (here->BSIM4V4gm
					 + here->BSIM4V4gmbs) + here->BSIM4V4gds;
				      T2 = T1 * T1 / here->BSIM4V4IdovVds;
                                      NevalSrc(&noizDens[BSIM4V4IDNOIZ],
                                               &lnNdens[BSIM4V4IDNOIZ], ckt,
                                               THERMNOISE, here->BSIM4V4dNodePrime,
                                               here->BSIM4V4sNodePrime, (T2 - igsquare));
                                      break;
			      }

		              NevalSrc(&noizDens[BSIM4V4FLNOIZ], (double*) NULL,
				       ckt, N_GAIN, here->BSIM4V4dNodePrime,
				       here->BSIM4V4sNodePrime, (double) 0.0);

                              switch(model->BSIM4V4fnoiMod)
			      {  case 0:
			              noizDens[BSIM4V4FLNOIZ] *= m * model->BSIM4V4kf
					    * exp(model->BSIM4V4af
					    * log(MAX(fabs(here->BSIM4V4cd),
					    N_MINLOG)))
					    / (pow(data->freq, model->BSIM4V4ef)
					    * pParam->BSIM4V4leff
				            * pParam->BSIM4V4leff
					    * model->BSIM4V4coxe);
				      break;
			         case 1:
		                      Vds = *(ckt->CKTstates[0] + here->BSIM4V4vds);
			              if (Vds < 0.0)
			                  Vds = -Vds;

                                      Ssi = Eval1ovFNoise(Vds, model, here,
                                          data->freq, ckt->CKTtemp);
                                      T10 = model->BSIM4V4oxideTrapDensityA
                                          * CONSTboltz * ckt->CKTtemp;
                                      T11 = pParam->BSIM4V4weff * pParam->BSIM4V4leff
                                          * pow(data->freq, model->BSIM4V4ef) * 1.0e10
					  * here->BSIM4V4nstar * here->BSIM4V4nstar;
                                      Swi = T10 / T11 * here->BSIM4V4cd
                                          * here->BSIM4V4cd;
                                      T1 = Swi + Ssi;
                                      if (T1 > 0.0)
                                          noizDens[BSIM4V4FLNOIZ] *= m * (Ssi * Swi) / T1;
                                      else
                                          noizDens[BSIM4V4FLNOIZ] *= 0.0;
				      break;
			      }

		              lnNdens[BSIM4V4FLNOIZ] =
				     log(MAX(noizDens[BSIM4V4FLNOIZ], N_MINLOG));


                       	if(here->BSIM4V4mode >= 0) {  /* bugfix  */
                              NevalSrc(&noizDens[BSIM4V4IGSNOIZ],
                                   &lnNdens[BSIM4V4IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4V4gNodePrime, here->BSIM4V4sNodePrime,
                                   m * (here->BSIM4V4Igs + here->BSIM4V4Igcs));
                              NevalSrc(&noizDens[BSIM4V4IGDNOIZ],
                                   &lnNdens[BSIM4V4IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4V4gNodePrime, here->BSIM4V4dNodePrime,
                                   m * (here->BSIM4V4Igd + here->BSIM4V4Igcd));
			} else {
                              NevalSrc(&noizDens[BSIM4V4IGSNOIZ],
                                   &lnNdens[BSIM4V4IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4V4gNodePrime, here->BSIM4V4sNodePrime,
                                   m * (here->BSIM4V4Igs + here->BSIM4V4Igcd));
                              NevalSrc(&noizDens[BSIM4V4IGDNOIZ],
                                   &lnNdens[BSIM4V4IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4V4gNodePrime, here->BSIM4V4dNodePrime,
                                   m * (here->BSIM4V4Igd + here->BSIM4V4Igcs));
                        }
                              NevalSrc(&noizDens[BSIM4V4IGBNOIZ],
                                   &lnNdens[BSIM4V4IGBNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4V4gNodePrime, here->BSIM4V4bNodePrime,
                                   m * here->BSIM4V4Igb);


		              noizDens[BSIM4V4TOTNOIZ] = noizDens[BSIM4V4RDNOIZ]
				     + noizDens[BSIM4V4RSNOIZ] + noizDens[BSIM4V4RGNOIZ]
				     + noizDens[BSIM4V4RBPSNOIZ] + noizDens[BSIM4V4RBPDNOIZ]
				     + noizDens[BSIM4V4RBPBNOIZ]
				     + noizDens[BSIM4V4RBSBNOIZ] + noizDens[BSIM4V4RBDBNOIZ]
				     + noizDens[BSIM4V4IDNOIZ] + noizDens[BSIM4V4FLNOIZ]
                                     + noizDens[BSIM4V4IGSNOIZ] + noizDens[BSIM4V4IGDNOIZ]
                                     + noizDens[BSIM4V4IGBNOIZ];
		              lnNdens[BSIM4V4TOTNOIZ] = 
				     log(MAX(noizDens[BSIM4V4TOTNOIZ], N_MINLOG));

		              *OnDens += noizDens[BSIM4V4TOTNOIZ];

		              if (data->delFreq == 0.0)
			      {   /* if we haven't done any previous 
				     integration, we need to initialize our
				     "history" variables.
				    */

			          for (i = 0; i < BSIM4V4NSRCS; i++)
				  {    here->BSIM4V4nVar[LNLSTDENS][i] =
					     lnNdens[i];
			          }

			          /* clear out our integration variables
				     if it's the first pass
				   */
			          if (data->freq ==
				      ((NOISEAN*) ckt->CKTcurJob)->NstartFreq)
				  {   for (i = 0; i < BSIM4V4NSRCS; i++)
				      {    here->BSIM4V4nVar[OUTNOIZ][i] = 0.0;
				           here->BSIM4V4nVar[INNOIZ][i] = 0.0;
			              }
			          }
		              }
			      else
			      {   /* data->delFreq != 0.0,
				     we have to integrate.
				   */
			          for (i = 0; i < BSIM4V4NSRCS; i++)
				  {    if (i != BSIM4V4TOTNOIZ)
				       {   tempOnoise = Nintegrate(noizDens[i],
						lnNdens[i],
				                here->BSIM4V4nVar[LNLSTDENS][i],
						data);
				           tempInoise = Nintegrate(noizDens[i]
						* data->GainSqInv, lnNdens[i]
						+ data->lnGainInv,
				                here->BSIM4V4nVar[LNLSTDENS][i]
						+ data->lnGainInv, data);
				           here->BSIM4V4nVar[LNLSTDENS][i] =
						lnNdens[i];
				           data->outNoiz += tempOnoise;
				           data->inNoise += tempInoise;
				           if (((NOISEAN*)
					       ckt->CKTcurJob)->NStpsSm != 0)
					   {   here->BSIM4V4nVar[OUTNOIZ][i]
						     += tempOnoise;
				               here->BSIM4V4nVar[OUTNOIZ][BSIM4V4TOTNOIZ]
						     += tempOnoise;
				               here->BSIM4V4nVar[INNOIZ][i]
						     += tempInoise;
				               here->BSIM4V4nVar[INNOIZ][BSIM4V4TOTNOIZ]
						     += tempInoise;
                                           }
			               }
			          }
		              }
		              if (data->prtSummary)
			      {   for (i = 0; i < BSIM4V4NSRCS; i++)
				  {    /* print a summary report */
			               data->outpVector[data->outNumber++]
					     = noizDens[i];
			          }
		              }
		              break;
		         case INT_NOIZ:
			      /* already calculated, just output */
		              if (((NOISEAN*)ckt->CKTcurJob)->NStpsSm != 0)
			      {   for (i = 0; i < BSIM4V4NSRCS; i++)
				  {    data->outpVector[data->outNumber++]
					     = here->BSIM4V4nVar[OUTNOIZ][i];
			               data->outpVector[data->outNumber++]
					     = here->BSIM4V4nVar[INNOIZ][i];
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
