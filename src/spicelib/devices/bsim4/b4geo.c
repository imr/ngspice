/* ******************************************************************************
   *  BSIM4 4.8.2 released by Chetan Kumar Dabhi 01/01/2020                     *
   *  BSIM4 Model Equations                                                     *
   ******************************************************************************

   ******************************************************************************
   *  Copyright (c) 2020 University of California                               *
   *                                                                            *
   *  Project Director: Prof. Chenming Hu.                                      *
   *  Current developers: Chetan Kumar Dabhi   (Ph.D. student, IIT Kanpur)      *
   *                      Prof. Yogesh Chauhan (IIT Kanpur)                     *
   *                      Dr. Pragya Kushwaha  (Postdoc, UC Berkeley)           *
   *                      Dr. Avirup Dasgupta  (Postdoc, UC Berkeley)           *
   *                      Ming-Yen Kao         (Ph.D. student, UC Berkeley)     *
   *  Authors: Gary W. Ng, Weidong Liu, Xuemei Xi, Mohan Dunga, Wenwei Yang     *
   *           Ali Niknejad, Chetan Kumar Dabhi, Yogesh Singh Chauhan,          *
   *           Sayeef Salahuddin, Chenming Hu                                   * 
   ******************************************************************************/

/*
Licensed under Educational Community License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain a copy of the license at
http://opensource.org/licenses/ECL-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations
under the License.
*/

#include "ngspice/ngspice.h"
#include "bsim4def.h"


static int
BSIM4NumFingerDiff(
double nf,
int minSD,
double *nuIntD, double *nuEndD, double *nuIntS, double *nuEndS)
{
int NF;
        NF = (int)nf;
	if ((NF%2) != 0)
	{   *nuEndD = *nuEndS = 1.0;
	    *nuIntD = *nuIntS = 2.0 * MAX((nf - 1.0) / 2.0, 0.0);
	}
	else
	{   if (minSD == 1) /* minimize # of source */
	    {   *nuEndD = 2.0;
		*nuIntD = 2.0 * MAX((nf / 2.0 - 1.0), 0.0);
		*nuEndS = 0.0;
		*nuIntS = nf;
	    }
	    else
	    {   *nuEndD = 0.0;
                *nuIntD = nf;
                *nuEndS = 2.0;
                *nuIntS = 2.0 * MAX((nf / 2.0 - 1.0), 0.0);
	    }
	}
return 0;
}


int
BSIM4PAeffGeo(
double nf,
int geo, int minSD,
double Weffcj, double DMCG, double DMCI, double DMDG,
double *Ps, double *Pd, double *As, double *Ad)
{
double T0, T1, T2;
double ADiso, ADsha, ADmer, ASiso, ASsha, ASmer;
double PDiso, PDsha, PDmer, PSiso, PSsha, PSmer;
double nuIntD = 0.0, nuEndD = 0.0, nuIntS = 0.0, nuEndS = 0.0;

	if (geo < 9) /* For geo = 9 and 10, the numbers of S/D diffusions already known */
	BSIM4NumFingerDiff(nf, minSD, &nuIntD, &nuEndD, &nuIntS, &nuEndS);

	T0 = DMCG + DMCI;
	T1 = DMCG + DMCG;
	T2 = DMDG + DMDG;

	PSiso = PDiso = T0 + T0 + Weffcj;
	PSsha = PDsha = T1;
	PSmer = PDmer = T2;

	ASiso = ADiso = T0 * Weffcj;
	ASsha = ADsha = DMCG * Weffcj;
	ASmer = ADmer = DMDG * Weffcj;

	switch(geo)
	{   case 0:
		*Ps = nuEndS * PSiso + nuIntS * PSsha;
		*Pd = nuEndD * PDiso + nuIntD * PDsha;
		*As = nuEndS * ASiso + nuIntS * ASsha;
		*Ad = nuEndD * ADiso + nuIntD * ADsha;
		break;
	    case 1:
                *Ps = nuEndS * PSiso + nuIntS * PSsha;
                *Pd = (nuEndD + nuIntD) * PDsha;
                *As = nuEndS * ASiso + nuIntS * ASsha;
                *Ad = (nuEndD + nuIntD) * ADsha;
                break;
            case 2:
                *Ps = (nuEndS + nuIntS) * PSsha;
                *Pd = nuEndD * PDiso + nuIntD * PDsha;
                *As = (nuEndS + nuIntS) * ASsha;
                *Ad = nuEndD * ADiso + nuIntD * ADsha;
                break;
            case 3:
                *Ps = (nuEndS + nuIntS) * PSsha;
                *Pd = (nuEndD + nuIntD) * PDsha;
                *As = (nuEndS + nuIntS) * ASsha;
                *Ad = (nuEndD + nuIntD) * ADsha;
                break;
            case 4:
                *Ps = nuEndS * PSiso + nuIntS * PSsha;
                *Pd = nuEndD * PDmer + nuIntD * PDsha;
                *As = nuEndS * ASiso + nuIntS * ASsha;
                *Ad = nuEndD * ADmer + nuIntD * ADsha;
                break;
            case 5:
                *Ps = (nuEndS + nuIntS) * PSsha;
                *Pd = nuEndD * PDmer + nuIntD * PDsha;
                *As = (nuEndS + nuIntS) * ASsha;
                *Ad = nuEndD * ADmer + nuIntD * ADsha;
                break;
            case 6:
                *Ps = nuEndS * PSmer + nuIntS * PSsha;
                *Pd = nuEndD * PDiso + nuIntD * PDsha;
                *As = nuEndS * ASmer + nuIntS * ASsha;
                *Ad = nuEndD * ADiso + nuIntD * ADsha;
                break;
            case 7:
                *Ps = nuEndS * PSmer + nuIntS * PSsha;
                *Pd = (nuEndD + nuIntD) * PDsha;
                *As = nuEndS * ASmer + nuIntS * ASsha;
                *Ad = (nuEndD + nuIntD) * ADsha;
                break;
            case 8:
                *Ps = nuEndS * PSmer + nuIntS * PSsha;
                *Pd = nuEndD * PDmer + nuIntD * PDsha;
                *As = nuEndS * ASmer + nuIntS * ASsha;
                *Ad = nuEndD * ADmer + nuIntD * ADsha;
                break;
            case 9: /* geo = 9 and 10 happen only when nf = even */
                *Ps = PSiso + (nf - 1.0) * PSsha;
                *Pd = nf * PDsha;
                *As = ASiso + (nf - 1.0) * ASsha;
                *Ad = nf * ADsha;
                break;
            case 10:
                *Ps = nf * PSsha;
                *Pd = PDiso + (nf - 1.0) * PDsha;
                *As = nf * ASsha;
                *Ad = ADiso + (nf - 1.0) * ADsha;
                break;
	    default:
		printf("Warning: Specified GEO = %d not matched\n", geo); 
	}
return 0;
}


int
BSIM4RdseffGeo(
double nf,
int geo, int rgeo, int minSD,
double Weffcj, double Rsh, double DMCG, double DMCI, double DMDG,
int Type,
double *Rtot)
{
double Rint=0.0, Rend = 0.0;
double nuIntD = 0.0, nuEndD = 0.0, nuIntS = 0.0, nuEndS = 0.0;

        if (geo < 9) /* since geo = 9 and 10 only happen when nf = even */
        {   BSIM4NumFingerDiff(nf, minSD, &nuIntD, &nuEndD, &nuIntS, &nuEndS);

            /* Internal S/D resistance -- assume shared S or D and all wide contacts */
	    if (Type == 1)
	    {   if (nuIntS == 0.0)
		    Rint = 0.0;
	        else
		    Rint = Rsh * DMCG / ( Weffcj * nuIntS); 
	    }
	    else
	    {  if (nuIntD == 0.0)
                   Rint = 0.0;
               else        
                   Rint = Rsh * DMCG / ( Weffcj * nuIntD);
	    }
	}

        /* End S/D resistance  -- geo dependent */
        switch(geo)
        {   case 0:
		if (Type == 1) BSIM4RdsEndIso(Weffcj, Rsh, DMCG, DMCI, DMDG,
					      nuEndS, rgeo, 1, &Rend);
		else           BSIM4RdsEndIso(Weffcj, Rsh, DMCG, DMCI, DMDG,
			     		      nuEndD, rgeo, 0, &Rend);
                break;
            case 1:
                if (Type == 1) BSIM4RdsEndIso(Weffcj, Rsh, DMCG, DMCI, DMDG,
                                              nuEndS, rgeo, 1, &Rend);
                else           BSIM4RdsEndSha(Weffcj, Rsh, DMCG, DMCI, DMDG,
					      nuEndD, rgeo, 0, &Rend);
                break;
            case 2:
                if (Type == 1) BSIM4RdsEndSha(Weffcj, Rsh, DMCG, DMCI, DMDG,
					      nuEndS, rgeo, 1, &Rend);
                else           BSIM4RdsEndIso(Weffcj, Rsh, DMCG, DMCI, DMDG,
					      nuEndD, rgeo, 0, &Rend);
                break;
            case 3:
                if (Type == 1) BSIM4RdsEndSha(Weffcj, Rsh, DMCG, DMCI, DMDG,
                                              nuEndS, rgeo, 1, &Rend);
                else           BSIM4RdsEndSha(Weffcj, Rsh, DMCG, DMCI, DMDG,
                                              nuEndD, rgeo, 0, &Rend);
                break;
            case 4:
                if (Type == 1) BSIM4RdsEndIso(Weffcj, Rsh, DMCG, DMCI, DMDG,
                                              nuEndS, rgeo, 1, &Rend);
                else           Rend = Rsh * DMDG / Weffcj;
                break;
            case 5:
                if (Type == 1) BSIM4RdsEndSha(Weffcj, Rsh, DMCG, DMCI, DMDG,
                                              nuEndS, rgeo, 1, &Rend);
                else           Rend = Rsh * DMDG / (Weffcj * nuEndD);
                break;
            case 6:
                if (Type == 1) Rend = Rsh * DMDG / Weffcj;
                else           BSIM4RdsEndIso(Weffcj, Rsh, DMCG, DMCI, DMDG,
                                              nuEndD, rgeo, 0, &Rend);
                break;
            case 7:
                if (Type == 1) Rend = Rsh * DMDG / (Weffcj * nuEndS);
                else           BSIM4RdsEndSha(Weffcj, Rsh, DMCG, DMCI, DMDG,
                                              nuEndD, rgeo, 0, &Rend);
                break;
            case 8:
                Rend = Rsh * DMDG / Weffcj;	
                break;
            case 9: /* all wide contacts assumed for geo = 9 and 10 */
		if (Type == 1)
		{   Rend = 0.5 * Rsh * DMCG / Weffcj;
		    if (nf == 2.0)
		        Rint = 0.0;
		    else
		        Rint = Rsh * DMCG / (Weffcj * (nf - 2.0));
		}
		else
		{   Rend = 0.0;
                    Rint = Rsh * DMCG / (Weffcj * nf);
		}
                break;
            case 10:
                if (Type == 1)
                {   Rend = 0.0;
                    Rint = Rsh * DMCG / (Weffcj * nf);
                }
                else
                {   Rend = 0.5 * Rsh * DMCG / Weffcj;;
                    if (nf == 2.0)
                        Rint = 0.0;
                    else
                        Rint = Rsh * DMCG / (Weffcj * (nf - 2.0));
                }
                break;
            default:
                printf("Warning: Specified GEO = %d not matched\n", geo);
        }

	if (Rint <= 0.0)
	    *Rtot = Rend;
	else if (Rend <= 0.0)
	    *Rtot = Rint;
	else
	    *Rtot = Rint * Rend / (Rint + Rend);
if(*Rtot==0.0)
	printf("Warning: Zero resistance returned from RdseffGeo\n");
return 0;
}


int
BSIM4RdsEndIso(
double Weffcj, double Rsh, double DMCG, double DMCI, double DMDG,
double nuEnd,
int rgeo, int Type,
double *Rend)
{	
        NG_IGNORE(DMDG);

	if (Type == 1)
	{   switch(rgeo)
            {	case 1:
		case 2:
		case 5:
		    if (nuEnd == 0.0)
		        *Rend = 0.0;
		    else
                        *Rend = Rsh * DMCG / (Weffcj * nuEnd);
		    break;
                case 3:
                case 4:
                case 6:
		    if ((DMCG + DMCI) == 0.0)
                         printf("(DMCG + DMCI) can not be equal to zero\n");
                    if ((nuEnd == 0.0)||((DMCG+DMCI)==0.0))
                        *Rend = 0.0;
                    else
                        *Rend = Rsh * Weffcj / (3.0 * nuEnd * (DMCG + DMCI));
                    break;
		default:
		    printf("Warning: Specified RGEO = %d not matched\n", rgeo);
            }
	}
	else
	{  switch(rgeo)
            {   case 1:
                case 3:
                case 7:
                    if (nuEnd == 0.0)
                        *Rend = 0.0;
                    else
                        *Rend = Rsh * DMCG / (Weffcj * nuEnd);
                    break;
                case 2:
                case 4:
                case 8:
                    if ((DMCG + DMCI) == 0.0)
                         printf("(DMCG + DMCI) can not be equal to zero\n");
                    if ((nuEnd == 0.0)||((DMCG + DMCI)==0.0))
                        *Rend = 0.0;
                    else
                        *Rend = Rsh * Weffcj / (3.0 * nuEnd * (DMCG + DMCI));
                    break;
                default:
                    printf("Warning: Specified RGEO = %d not matched\n", rgeo);
            }
	}
return 0;
}


int
BSIM4RdsEndSha(
double Weffcj, double Rsh, double DMCG, double DMCI, double DMDG,
double nuEnd,
int rgeo, int Type,
double *Rend)
{
        NG_IGNORE(DMCI);
        NG_IGNORE(DMDG);

        if (Type == 1)
        {   switch(rgeo)
            {   case 1:
                case 2:
                case 5:
                    if (nuEnd == 0.0)
                        *Rend = 0.0;
                    else
                        *Rend = Rsh * DMCG / (Weffcj * nuEnd);
                    break;
                case 3:
                case 4:
                case 6:
                    if (DMCG == 0.0)
                        printf("DMCG can not be equal to zero\n");
                    if (nuEnd == 0.0)
                        *Rend = 0.0;
                    else
                        *Rend = Rsh * Weffcj / (6.0 * nuEnd * DMCG);
                    break;
                default:
                    printf("Warning: Specified RGEO = %d not matched\n", rgeo);
            }
        }
        else
        {  switch(rgeo)
            {   case 1:
                case 3:
                case 7:
                    if (nuEnd == 0.0)
                        *Rend = 0.0;
                    else
                        *Rend = Rsh * DMCG / (Weffcj * nuEnd);
                    break;
                case 2:
                case 4:
                case 8:
                    if (DMCG == 0.0)
                        printf("DMCG can not be equal to zero\n");
                    if (nuEnd == 0.0)
                        *Rend = 0.0;
                    else
                        *Rend = Rsh * Weffcj / (6.0 * nuEnd * DMCG);
                    break;
                default:
                    printf("Warning: Specified RGEO = %d not matched\n", rgeo);
            }
        }
return 0;
}
