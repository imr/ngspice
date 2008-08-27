#include <config.h>
#include <ngspice.h>

#include <fteext.h>
#include <wordlist.h>

#include "vectors.h"
#include <math.h>

typedef struct measure
{
  char *result;

  char *m_vec;		// name of the output variable which determines the beginning of the measurement
  char *m_vec2;
  int m_rise;
  int m_fall;
  int m_cross;
  float m_val;		// value of the m_ver at which the counter for crossing, rises or falls is incremented by one
  float m_td;		// amount of delay before the measurement should start
  float m_from;
  float m_to;
  float m_at;
  float m_measured;
  float m_measured_at;

} measure;

enum AnalysisType {
	AT_DELAY, AT_TRIG,	
	AT_FIND, AT_WHEN,
	AT_AVG, AT_MIN, AT_MAX, AT_RMS, AT_PP,
	AT_INTEG, AT_DERIV,
	AT_ERR, AT_ERR1, AT_ERR2, AT_ERR3
};

void com_measure_when(struct measure *meas) {

	int i, first;
        int riseCnt =0;
        int fallCnt =0;
        int crossCnt =0;
        int section = -1;
	float value, prevValue;
	float timeValue, prevTimeValue;
	
	enum ValSide { S_ABOVE_VAL, S_BELOW_VAL };
	enum ValEdge { E_RISING, E_FALLING };
	
	struct dvec *d, *dTime;
	
	d = vec_get(meas->m_vec);
	dTime = plot_cur->pl_scale;

        if (d == NULL) {
                fprintf(cp_err, "Error: no such vector as %s.\n", meas->m_vec);
                return;
        }

	if (dTime == NULL) {
                fprintf(cp_err, "Error: no such vector as time.\n");
                return;
        }

	prevValue =0;
	prevTimeValue =0;
	first =0;

	for (i=0; i < d->v_length; i++) {

		value = d->v_realdata[i];
		timeValue = dTime->v_realdata[i];

		if (timeValue < meas->m_td)
			continue;
			
		if (first == 1) {
			// initialise
			crossCnt =0;
			if (value < meas->m_val) {
				section = S_BELOW_VAL;
		              	if ( (prevValue <= meas->m_val) && (value >= meas->m_val) ) {
					fallCnt =1;
					crossCnt =1;
				}

			} else {
				section = S_ABOVE_VAL;
				if ( (prevValue <= meas->m_val) && (value >= meas->m_val) ) {
					riseCnt =1;
					crossCnt =1;
				}
			}
			printf("");
		}
	
		if (first > 1) {

			if ( (section == S_BELOW_VAL) && (value >= meas->m_val) ) {
					section = S_ABOVE_VAL;
					crossCnt++;
					riseCnt++;

			} else if ( (section == S_ABOVE_VAL) && (value <= meas->m_val) ) {
					section = S_BELOW_VAL;
					crossCnt++;
					fallCnt++;
			} 

			if  ((crossCnt == meas->m_cross) || (riseCnt == meas->m_rise) || (fallCnt == meas->m_fall)) {
					meas->m_measured = prevTimeValue + (meas->m_val - prevValue) * (timeValue - prevTimeValue) / (value - prevValue);
                                        return;

			}
		}
		first ++;

		prevValue = value;
		prevTimeValue = timeValue;
	}

	meas->m_measured = 0.0e0;
        return;      
}

void measure_at(struct measure *meas, float at) {
	
	int i;
	float value, pvalue, svalue, psvalue;
	struct dvec *d, *dScale;

	psvalue = pvalue = 0;
        d = vec_get(meas->m_vec);
        dScale = plot_cur->pl_scale;

        if (d == NULL) {
                fprintf(cp_err, "Error: no such vector as %s.\n", meas->m_vec);
                return;
        }

        if (dScale == NULL) {
                fprintf(cp_err, "Error: no such vector time.\n");
                return;
        }

	for (i=0; i < d->v_length; i++) {
                value = d->v_realdata[i];
                svalue = dScale->v_realdata[i];

  		if ( (i > 0) && (psvalue <= at) && (svalue >= at) ) {
			meas->m_measured = pvalue + (at - psvalue) * (value - pvalue) / (svalue - psvalue);
		//	meas->m_measured = value;
			return;
		}

		psvalue = svalue;
		pvalue = value;
        }

        meas->m_measured = 0.0e0;
        return;
}

void measure_avg( ) {
	// AVG (Average):
	// Calculates the area under the 'out_var' divided by the periods of intrest
	return;
}

void measure_minMaxAvg( struct measure *meas, int minMax ) {
      
        int i, avgCnt;
        struct dvec *d, *dScale;
        float value, svalue, mValue, mValueAt;
        int first;

        mValue =0;
        mValueAt = svalue =0;
        meas->m_measured = 0.0e0;
        meas->m_measured_at = 0.0e0;
        first =0;
	avgCnt =0;

        d = vec_get(meas->m_vec);
        if (d == NULL) {
                fprintf(cp_err, "Error: no such vector as %s.\n", meas->m_vec);
                return;
        }

        dScale = vec_get("time");
        if (d == NULL) {
                fprintf(cp_err, "Error: no such vector as time.\n");
                return;
        }

        for (i=0; i < d->v_length; i++) {
                value = d->v_realdata[i];
                svalue = dScale->v_realdata[i];

                if (svalue < meas->m_from)
                        continue;

                if ((meas->m_to != 0.0e0) && (svalue > meas->m_to) )
                        break;

		if (first ==0) {
                        mValue = value;
                        mValueAt = svalue;
                        first =1;
			
		} else  {
			switch (minMax) {
				case AT_MIN: {
		                        if (value <= mValue) {
                		                mValue = value;
		                                mValueAt = svalue;
		                        }
					break;
				}
				case AT_MAX: {
					if (value >= mValue) {
		                                mValue = value;
                		                mValueAt = svalue;
		                        }
					break;
				}
				case AT_AVG:
				case AT_RMS: {
					mValue = mValue + value;
					avgCnt ++;			
					break;
				}
			}
			
		}
        }

	switch (minMax)
	{	
		case AT_AVG: {
			meas->m_measured = (mValue / avgCnt);
	                meas->m_measured_at = svalue;
			break;
		}
		case AT_RMS: {
	//		printf(" mValue %e svalue %e avgCnt %i, ", mValue, svalue, avgCnt);
                        meas->m_measured = sqrt(mValue) / avgCnt;
                        meas->m_measured_at = svalue;					
			break;
		}
		case AT_MIN:
		case AT_MAX: {
			meas->m_measured = mValue;
        	        meas->m_measured_at = mValueAt;
			break;
		}
	}

        return;
}

void measure_rms( ) {
	// RMS (root mean squared):
	// Calculates the square root of the area under the 'out_var2' curve
	//  divided be the period of interest
	return;
}

void measure_integ( ) {
	// INTEGRAL INTEG
	return;
}

void measure_deriv( ) {
	// DERIVATIVE DERIV
	return;
}

// ERR Equations
void measure_ERR( ) {
	return;
}

void measure_ERR1( ) {
        return;
}

void measure_ERR2( ) {
        return;
}

void measure_ERR3( ) {
        return;
}

void measure_errMessage(char *mName, char *mFunction, char *trigTarg, char *errMsg) {

	printf("\tmeasure '%s'  failed\n", mName);
	printf("Error: measure  %s  %s(%s) :\n", mName, mFunction, trigTarg);
	printf("\t%s\n",errMsg);
	return;
}

void com_dotmeasure( ) {

        // simulation info
//      printf("*%s\n", plot_cur->pl_title);
//      printf("\t %s, %s\n", plot_cur->pl_name, plot_cur->pl_date); // missing temp

	return;
}

int measure_valid_vector(char *vec) {

        struct dvec *d;

        d = vec_get(vec);
        if (d == NULL)
                return 0;
        
	return 1;
}

int measure_parse_stdParams (struct measure *meas, wordlist *wl, wordlist *wlBreak, char *errbuf) {
	
	int pCnt;	
	char *p, *pName, *pValue;
        double *engVal, engVal1;

	pCnt =0;
	while (wl != wlBreak) {
		p = wl->wl_word;
		pName = strtok(p, "=");
		pValue = strtok(NULL, "=");

		if (pValue == NULL) {
			sprintf(errbuf,"bad syntax of ??\n");
			return 0;
		}

		if (!(engVal = ft_numparse(&pValue, FALSE))) {
			sprintf(errbuf,"bad syntax of ??\n");
			return 0;
        	}

		engVal1 = *engVal;

		if(strcasecmp(pName,"RISE")==0) {
          		meas->m_rise = engVal1;
            		meas->m_fall = -1;
           		meas->m_cross = -1;
		} else if(strcasecmp(pName,"FALL")==0) {
               		meas->m_fall = engVal1;
                 	meas->m_rise = -1;
			meas->m_cross = -1;
		} else if(strcasecmp(pName,"CROSS")==0) {
           		meas->m_cross = engVal1;
                 	meas->m_rise = -1;
             		meas->m_fall = -1;
              	} else if(strcasecmp(pName,"VAL")==0) {
                      	meas->m_val = engVal1;
             	} else if(strcasecmp(pName,"TD")==0) {
                       	meas->m_td = engVal1;
             	} else if(strcasecmp(pName,"FROM")==0) {
                	meas->m_from = engVal1;
              	} else if(strcasecmp(pName,"TO")==0) {
                  	meas->m_to = engVal1;
		} else if(strcasecmp(pName,"AT")==0) {
			meas->m_at = engVal1;
              	} else {
                      	sprintf(errbuf,"no such parameter as '%s'\n",pName);
                      	return 0;
             	}

		pCnt ++;
		wl = wl->wl_next;
	}	

	if (pCnt == 0) {
                sprintf(errbuf,"bad syntax of ??\n");
                return 0;
        }

	// valid vector
        if (measure_valid_vector(meas->m_vec)==0) {
                sprintf(errbuf,"no such vector as '%s'\n", meas->m_vec);
                return 0;
        }

	// valid vector2
	if (meas->m_vec2 != NULL) {
	        if (measure_valid_vector(meas->m_vec2)==0) {
	                sprintf(errbuf,"no such vector as '%s'\n", meas->m_vec2);
	                return 0;
	        }
	}

	return 1;
}

int measure_parse_find (struct measure *meas, wordlist *wl, wordlist *wlBreak, char *errbuf) {
	
	int pCnt;
	char *p, *pName, *pVal;
	double *engVal, engVal1;

        meas->m_vec = NULL;
        meas->m_vec2 = NULL;
        meas->m_val = -1;
        meas->m_cross = -1;
        meas->m_fall = -1;
        meas->m_rise = -1;
        meas->m_td = 0;
        meas->m_from = 0.0e0;
        meas->m_to = 0.0e0;
	meas->m_at = -1;

	pCnt =0;
	while(wl != wlBreak) {
		p = wl->wl_word;
		
		if (pCnt == 0 ) {
//			meas->m_vec =(char *)malloc(strlen(wl->wl_word)+1);
  //                      strcpy(meas->m_vec, cp_unquote(wl->wl_word));
     		meas->m_vec= cp_unquote(wl->wl_word);
		} else if (pCnt == 1) {

			pName = strtok(p, "=");
                        pVal = strtok(NULL, "=");

			if (pVal == NULL) {
                                sprintf(errbuf,"bad syntax of WHEN\n");
                                return 0;
                        }
	
			if (strcasecmp(pName,"AT")==0) {	

				if (!(engVal = ft_numparse(&pVal, FALSE))) {
                               		sprintf(errbuf,"bad syntax of WHEN\n");
                               		return 0;
                        	}

                        	engVal1 = *engVal;

				meas->m_at = engVal1;				

			} else {
				 sprintf(errbuf,"bad syntax of WHEN\n");
                                 return 0;
			}
		} else {
			if (measure_parse_stdParams(meas, wl, NULL, errbuf) == 0)
                                return 0;
		}
		
		wl = wl->wl_next;
		pCnt ++;
	}

	return 1;
}

int measure_parse_when (struct measure *meas, wordlist *wl, char *errBuf) {

        int pCnt;
        char *p, *pVar1, *pVar2;

        meas->m_vec = NULL;
        meas->m_vec2 = NULL;
        meas->m_val = -1;
        meas->m_cross = -1;
        meas->m_fall = -1;
        meas->m_rise = -1;
        meas->m_td = 0;
        meas->m_from = 0.0e0;
        meas->m_to = 0.0e0;
	meas->m_at = -1;

        pCnt =0;
        while (wl) {
                p= wl->wl_word;

                if (pCnt == 0) {
                        pVar1 = strtok(p, "=");
                        pVar2 = strtok(NULL, "=");

                        if (pVar2 == NULL) {
                                sprintf(errBuf,"bad syntax\n");
                                return 0;
                        }

                        meas->m_vec = pVar1;
                        if (measure_valid_vector(pVar2)==1)
                                meas->m_vec2 = pVar2;
                        else
                                meas->m_val = atof(pVar2);
                } else {
                        if (measure_parse_stdParams(meas, wl, NULL, errBuf) == 0) 
				return 0;
                        break;
                }

                wl = wl->wl_next;
                pCnt ++;
        }
        return 1;
}


int measure_parse_trigtarg (struct measure *meas, wordlist *words, wordlist *wlTarg, char *trigTarg, char *errbuf) {

	int pcnt;
	char *p;
	   	
        meas->m_vec = NULL;
	meas->m_vec2 = NULL;
        meas->m_cross = -1;
        meas->m_fall = -1;
        meas->m_rise = -1;
        meas->m_td = 0;
	meas->m_from = 0.0e0;
	meas->m_to = 0.0e0;
	meas->m_at = -1;

	pcnt =0;
        while (words != wlTarg) {
                p = words->wl_word;

                if (pcnt ==0) {
//                        meas->m_vec =(char *)malloc(strlen(words->wl_word)+1);
  //                      strcpy(meas->m_vec, cp_unquote(words->wl_word));
			meas->m_vec= cp_unquote(words->wl_word);
                } else {

			if (measure_parse_stdParams(meas, words, wlTarg, errbuf) == 0)
                                return 0;
                        break;

                }

                words = words->wl_next;
                pcnt ++;
        }

        if (pcnt == 0) {
                sprintf(errbuf,"bad syntax of '%s'\n", trigTarg);
                return 0;
        }

	// valid vector                 
   	if (measure_valid_vector(meas->m_vec)==0) {
              	sprintf(errbuf,"no such vector as '%s'\n", meas->m_vec);
            	return 0;
       	}

	return 1;
}

float
get_measure2(wordlist *wl)
{
	wordlist *words, *wlTarg, *wlWhen;
 	char errbuf[100];
        char *mType = NULL;             // analysis type
	char *mName = NULL;             // name given to the measured output
	char *mFunction = NULL;
	int mFunctionType, wl_cnt;
	char *p;
	
	mFunctionType = -1;
		
	if (!wl) {
	    printf("usage: measure .....\n");
	    return 0.0e0;
	}

	if (!plot_cur || !plot_cur->pl_dvecs || !plot_cur->pl_scale) {
        	fprintf(cp_err, "Error: no vectors available\n");
	        return 0.0e0;
	}

	if (!ciprefix("tran", plot_cur->pl_typename)) {
     	   	fprintf(cp_err, "Error: measure limited to transient analysis\n");
	        return 0.0e0;
	}

	words =wl;
	wlTarg = NULL;
	wlWhen = NULL;

	if (!words) {
                fprintf(cp_err, "Error: no assignment found.\n");
                return 0.0e0;
        }

	wl_cnt = 0;
	while (words) {

		switch(wl_cnt)
		{
			case 0:
				mType = cp_unquote(words->wl_word);
				break;
			case 1:
				mName = cp_unquote(words->wl_word);
				break;
			case 2:
			{
				mFunction = cp_unquote(words->wl_word);
		                // Functions
		                if (strcasecmp(mFunction,"DELAY")==0)
		                        mFunctionType = AT_DELAY;
		                else if (strcasecmp(mFunction,"TRIG")==0)
		                        mFunctionType = AT_DELAY;
		                else if (strcasecmp(mFunction,"FIND")==0)
		                        mFunctionType = AT_FIND;
		                else if (strcasecmp(mFunction,"WHEN")==0)
		                        mFunctionType = AT_WHEN;
		                else if (strcasecmp(mFunction,"AVG")==0)
		                        mFunctionType = AT_AVG;
		                else if (strcasecmp(mFunction,"MIN")==0)
		                        mFunctionType = AT_MIN;
		                else if (strcasecmp(mFunction,"MAX")==0)
		                        mFunctionType = AT_MAX;
		                else if (strcasecmp(mFunction,"RMS")==0)
		                        mFunctionType = AT_RMS;
		                else if (strcasecmp(mFunction,"PP")==0)
		                        mFunctionType = AT_PP;
		                else if (strcasecmp(mFunction,"INTEG")==0)
		                        mFunctionType = AT_INTEG;
		                else if (strcasecmp(mFunction,"DERIV")==0)
		                        mFunctionType = AT_DERIV;
		                else if (strcasecmp(mFunction,"ERR")==0)
		                        mFunctionType = AT_ERR;
		                else if (strcasecmp(mFunction,"ERR1")==0)
		                        mFunctionType = AT_ERR1;
		                else if (strcasecmp(mFunction,"ERR2") == 0)
		                        mFunctionType = AT_ERR2;
		                else if (strcasecmp(mFunction,"ERR3") == 0)
		                        mFunctionType = AT_ERR3;
		                else {
		                        printf("\tmeasure '%s'  failed\n", mName);
		                        printf("Error: measure  %s  :\n", mName);
		                        printf("\tno such function as '%s'\n", mFunction);
		                        return 0.0e0;
		                }
				break;
			}
			default:
			{
				p = words->wl_word;
	                        
	                        if (strcasecmp(p,"targ")==0)
					wlTarg = words;
				
				if (strcasecmp(p,"when")==0)
					wlWhen = words;
                
				break;
			}
		}
		wl_cnt ++;
		words = words->wl_next;
	}	

	if (wl_cnt < 3) {
		printf("\tmeasure '%s'  failed\n", mName);
		printf("Error: measure  %s  :\n", mName);
		printf("\tinvalid num params\n");
		return 0.0e0;
	}

	//------------------------


	words =wl;

       	if (words)
          	words = words->wl_next; // skip
        if (words)
              	words = words->wl_next; // results name
        if (words)
               	words = words->wl_next; // Function 


	// switch here
	switch(mFunctionType)
		{
		case AT_DELAY:
		case AT_TRIG: 
		{		
			// trig parameters
			measure *measTrig, *measTarg;
			measTrig = (struct measure*)malloc(sizeof(struct measure));           
	                measTarg = (struct measure*)malloc(sizeof(struct measure));
									
			if (measure_parse_trigtarg(measTrig, words , wlTarg, "trig", errbuf)==0) {
				measure_errMessage(mName, mFunction, "TRIG", errbuf);
				return 0.0e0;
			}

			if ((measTrig->m_rise == -1) && (measTrig->m_fall == -1) && (measTrig->m_cross == -1)) {
                		sprintf(errbuf,"rise, fall or cross must be given\n");
				measure_errMessage(mName, mFunction, "TRIG", errbuf);
                		return 0.0e0;
        		}

			while (words != wlTarg)
				words = words->wl_next; // hack
		
			if (words)
				words = words->wl_next; // skip targ
				
		        if (measure_parse_trigtarg(measTarg, words , NULL, "targ", errbuf)==0) {
				measure_errMessage(mName, mFunction, "TARG", errbuf);
		                return 0.0e0;
		       	}

	        	if ((measTarg->m_rise == -1) && (measTarg->m_fall == -1) && (measTarg->m_cross == -1)) {
                                sprintf(errbuf,"rise, fall or cross must be given\n");
                                measure_errMessage(mName, mFunction, "TARG", errbuf);
                                return 0.0e0;
                        }

			// measure trig
			if (measTrig->m_at == -1)
				com_measure_when(measTrig);
			else
				measTrig->m_measured = measTrig->m_at;


			if (measTrig->m_measured == 0.0e0) {
				sprintf(errbuf,"out of interval\n");
		                measure_errMessage(mName, mFunction, "TRIG", errbuf);
		                return 0.0e0;
			}
			// measure targ
			com_measure_when(measTarg);
		
			if (measTarg->m_measured == 0.0e0) {
				sprintf(errbuf,"out of interval\n");
		                measure_errMessage(mName, mFunction, "TARG", errbuf);
		                return 0.0e0;
		        }
		
			// print results
		        printf("%-20s=  %e targ=  %e trig=  %e\n", mName, (measTarg->m_measured - measTrig->m_measured), measTarg->m_measured, measTrig->m_measured);
		
			return (measTarg->m_measured - measTrig->m_measured);
		}
                case AT_FIND:
		{
                        measure *meas, *measFind;
                        meas = (struct measure*)malloc(sizeof(struct measure));
			measFind = (struct measure*)malloc(sizeof(struct measure));

			if (measure_parse_find(meas, words, wlWhen, errbuf) == 0) {
				measure_errMessage(mName, mFunction, "FIND", errbuf);
                                return 0.0e0;
			}

			if (meas->m_at == -1 ) { 
				// find .. when statment

				while (words != wlWhen)
                            		words = words->wl_next; // hack

                        	if (words)
                                	words = words->wl_next; // skip targ

				if (measure_parse_when(measFind, words, errbuf) ==0) {
                                	measure_errMessage(mName, mFunction, "WHEN", errbuf);
                                	return 0.0e0;
				}

				com_measure_when(measFind);

                 		if (measFind->m_measured == 0.0e0) {
                              		sprintf(errbuf,"out of interval\n");
                        		measure_errMessage(mName, mFunction, "WHEN", errbuf);
                           		return 0.0e0;
                        	}
				
				measure_at(measFind, measFind->m_measured);
				meas->m_measured = measFind->m_measured; 
                      
			} else {
				measure_at(meas, meas->m_at);
			}
			
                        if (meas->m_measured == 0.0e0) {
                                sprintf(errbuf,"out of interval\n");
                                measure_errMessage(mName, mFunction, "WHEN", errbuf);
                                return 0.0e0;
                        }

                        // print results
                        printf("%-20s=  %e\n", mName, meas->m_measured);
			return meas->m_measured;
		}
                case AT_WHEN:
		{
			measure *meas;
			meas = (struct measure*)malloc(sizeof(struct measure));

			if (measure_parse_when(meas, words, errbuf) ==0) {
                     	      	measure_errMessage(mName, mFunction, "WHEN", errbuf);
                        	return 0.0e0;
			}

			com_measure_when(meas);

			if (meas->m_measured == 0.0e0) {
                                sprintf(errbuf,"out of interval\n");
                                measure_errMessage(mName, mFunction, "WHEN", errbuf);
                                return 0.0e0;
                        }
			
			// print results
                        printf("%-20s=  %e\n", mName, meas->m_measured);

			return (meas->m_measured);
	        }
		case AT_RMS:
                        printf("\tmeasure '%s'  failed\n", mName);
                        printf("Error: measure  %s  :\n", mName);
                        printf("\tfunction '%s' currently not supported\n", mFunction);
                        break;
                case AT_AVG:
		{
			// trig parameters
                        measure *meas;
                        meas = (struct measure*)malloc(sizeof(struct measure));

                        if (measure_parse_trigtarg(meas, words , NULL, "trig", errbuf)==0) {
                                measure_errMessage(mName, mFunction, "TRIG", errbuf);
                                return 0.0e0;
                        }

                        // measure
                   	measure_minMaxAvg(meas, mFunctionType);

                        if (meas->m_measured == 0.0e0) {
                                sprintf(errbuf,"out of interval\n");
                                measure_errMessage(mName, mFunction, "TRIG", errbuf); // ??
                                return 0.0e0;
                        }

			if (meas->m_at == -1)
				meas->m_at = 0.0e0;

                        // print results
                        printf("%-20s=  %e from=  %e to=  %e\n", mName, meas->m_measured, meas->m_at, meas->m_measured_at);
                        return meas->m_measured;

		}
                case AT_MIN:
		case AT_MAX:
		{
		        // trig parameters
                        measure *measTrig;
                        measTrig = (struct measure*)malloc(sizeof(struct measure));

                        if (measure_parse_trigtarg(measTrig, words , NULL, "trig", errbuf)==0) {
                                measure_errMessage(mName, mFunction, "TRIG", errbuf);
                                return 0.0e0;
                        }
			
			// measure
			if (mFunctionType == AT_MIN)
 	                      	measure_minMaxAvg(measTrig, AT_MIN);
			else 
				measure_minMaxAvg(measTrig, AT_MAX);


                        if (measTrig->m_measured == 0.0e0) {
                                sprintf(errbuf,"out of interval\n");
                                measure_errMessage(mName, mFunction, "TRIG", errbuf); // ??
                                return 0.0e0;
                        }

                        // print results
                        printf("%-20s=  %e at=  %e\n", mName, measTrig->m_measured, measTrig->m_measured_at);
			return measTrig->m_measured;
		}
                case AT_PP:
		{			
		        float minValue, maxValue;

                        measure *measTrig;
                        measTrig = (struct measure*)malloc(sizeof(struct measure));

                        if (measure_parse_trigtarg(measTrig, words , NULL, "trig", errbuf)==0) {
                                measure_errMessage(mName, mFunction, "TRIG", errbuf);
                                return 0.0e0;
                        }

			// measure min
                        measure_minMaxAvg(measTrig, AT_MIN);
                        if (measTrig->m_measured == 0.0e0) {
                                sprintf(errbuf,"out of interval\n");
                                measure_errMessage(mName, mFunction, "TRIG", errbuf); // ??
                                return 0.0e0;
                        }
			minValue = measTrig->m_measured;

			// measure max
                        measure_minMaxAvg(measTrig, AT_MAX);
   	                if (measTrig->m_measured == 0.0e0) {
                                sprintf(errbuf,"out of interval\n");
                                measure_errMessage(mName, mFunction, "TRIG", errbuf); // ??
                                return 0.0e0;
                        }
                        maxValue = measTrig->m_measured;

                        // print results
                        printf("%-20s=  %e from=  %e to=  %e\n", mName, (maxValue - minValue), measTrig->m_from, measTrig->m_to);
			return (maxValue - minValue);
		}
                case AT_INTEG:
                case AT_DERIV:
                case AT_ERR:
                case AT_ERR1:
                case AT_ERR2:
                case AT_ERR3:
		{
                        printf("\tmeasure '%s'  failed\n", mName);
                        printf("Error: measure  %s  :\n", mName);
                        printf("\tfunction '%s' currently not supported\n", mFunction);
                        break;
		}
        }
	return 0.0e0;
}

void com_measure2(wordlist *wl) {
	get_measure2(wl);
	return;
}

