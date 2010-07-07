/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "iferrmsg.h"
#include "inpdefs.h"
#include "inp.h"

/*--------------------------------------------------------------
 * This fcn takes the model card & examines it.  Depending upon
 * model type, it parses the model line, and then calls
 * INPmakeMod to stick the model name into the model list.
 * Note that multi-line models are handled in the calling fcn
 * (INPpas1).
 *-------------------------------------------------------------*/
char *INPdomodel(CKTcircuit *ckt, card * image, INPtables * tab)
{

    char *modname;
    int type = -1;
    int lev, error1=0;
    char ver[100];
    char *typename;
    char *err = (char *) NULL;
    char *line;
    char *val;
    double rval=0, lval=0;

    line = image->line;
    
#ifdef TRACE
    printf("In INPdomodel, examining line %s . . . \n", line); 
#endif    
    
    INPgetTok(&line, &modname, 1);	/* throw away '.model' */
    tfree(modname);
    INPgetTok(&line, &modname, 1);      /* get model name */
    INPinsert(&modname, tab);           /* stick model name into table */
    INPgetTok(&line, &typename, 1);     /* get model type */
    
    /*  -----  Check if model is a BJT --------- */
    if ((strcmp(typename, "npn") == 0) || (strcmp(typename, "pnp") == 0)) {
	err = INPfindLev(line,&lev);
	switch(lev) {
		case 0:
		case 1:
			type = INPtypelook("BJT");
			if (type < 0) {
	  		  err = INPmkTemp(
	  		          "Device type BJT not available in this binary\n");
			}
		break;
		case 2:
			 type = INPtypelook("BJT2");
                if(type < 0) {
                    err = INPmkTemp(
                            "Device type BJT2 not available in this binary\n");
                }
                break;
		case 4:
			 type = INPtypelook("VBIC");
                if(type < 0) {
                    err = INPmkTemp(
                            "Device type VBIC not available in this binary\n");
                }
                break;
#ifdef ADMS
		case 6:
			 type = INPtypelook("mextram");
                if(type < 0) {
                    err = INPmkTemp(
                            "Device type MEXTRAM not available in this binary\n");
                }
                break;
		case 7:
			 type = INPtypelook("hicum0");
                if(type < 0) {
                    err = INPmkTemp(
                            "Device type HICUM0 not available in this binary\n");
                }
                break;
		case 8:
			 type = INPtypelook("hicum2");
                if(type < 0) {
                    err = INPmkTemp(
                            "Device type HICUM2 not available in this binary\n");
                }
                break;
#endif
		default: /* placeholder; use level 4 for the next model */
		err = INPmkTemp(
		  "Only BJT levels 1-2, 4,6,7,8 are supported in this binary\n");
                break;

	}	
	INPmakeMod(modname, type, image);
    } /* end if ((strcmp(typename, "npn") == 0) || (strcmp(typename, "pnp") == 0)) */

    /*  --------  Check if model is a diode --------- */    
    else if (strcmp(typename, "d") == 0) {
	type = INPtypelook("Diode");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type Diode not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } /*   else if (strcmp(typename, "d") == 0) {  */
    
    /*  --------  Check if model is a jfet --------- */
    else if ((strcmp(typename, "njf") == 0)
	       || (strcmp(typename, "pjf") == 0)) {
	err = INPfindLev(line, &lev);
	switch (lev) {
	case 0:
	case 1:
	    type = INPtypelook("JFET");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type JFET not available in this binary\n");
	    }
	    break;
	case 2:
	    type = INPtypelook("JFET2");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type JFET2 not available in this binary\n");
	    }
	    break;
	default:		/* placeholder; use level 3 for the next model */
	    err =
		INPmkTemp
		("Only JFET device levels 1-2 are supported in this binary\n");
	    break;
	}
	INPmakeMod(modname, type, image);
    }  /*   end  else if ((strcmp(typename, "njf") == 0) */
    
    /*  --------  Check if model is a MES or an HFET --------- */
    else if ((strcmp(typename, "nmf")   == 0)
	       || (strcmp(typename, "pmf")   == 0)
	       || (strcmp(typename, "nhfet") == 0)
	       || (strcmp(typename, "phfet") == 0)) {
	  err = INPfindLev( line, &lev );
	  switch ( lev ) 
	  {	
		case 0:
        	case 1:
        		type = INPtypelook("MES");
        		if (type < 0)
        		{
        			err = INPmkTemp("Device type MES not available\n");
        		}
        		break;
        	case 2:
        		type = INPtypelook("MESA");
        		if (type < 0)
        		{
        			err = INPmkTemp("Device type MESA not availabe\n");
        		}
        		break;
		case 3:
        		type = INPtypelook("MESA");
        		if (type < 0)
        		{
        			err = INPmkTemp("Device type MESA not availabe\n");
        		}
        		break;
        	case 4:
        		type = INPtypelook("MESA");
        		if ( type < 0)
        		{
        			err = INPmkTemp(" Device type MESA not available\n");
        		}
        		break;
        	case 5:
        		type = INPtypelook("HFET1");
        		if ( type < 0)
        		{
        			err = INPmkTemp(" Device type HFET1 not available\n");
        		}
        		break;
        	case 6:
        		type = INPtypelook("HFET2");
        		if ( type < 0)
        		{
        			err = INPmkTemp(" Device type HFET2 not available in this binary\n");
        		}
        		break;

        	default:
        		err = INPmkTemp("only mesfet device level 1-4 and hfet level 5-6 supported\n");
        		break;
        }
	INPmakeMod(modname, type, image);
    } 
    
    /*  --------  Check if model is a Uniform Distrib. RC line --------- */
    else if (strcmp(typename, "urc") == 0) {
	type = INPtypelook("URC");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type URC not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } 
    
    /*  --------  Check if model is a MOSFET --------- */
    else if ((strcmp(typename, "nmos") == 0)
	       || (strcmp(typename, "pmos") == 0)
	       || (strcmp(typename, "nsoi") == 0)
	       || (strcmp(typename, "psoi") == 0)) {
	err = INPfindLev(line, &lev);
	switch (lev) {
	case 0:
	case 1:
	    type = INPtypelook("Mos1");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type MOS1 not available in this binary\n");
	    }
	    break;
	case 2:
	    type = INPtypelook("Mos2");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type MOS2 not available in this binary\n");
	    }
	    break;
	case 3:
	    type = INPtypelook("Mos3");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type MOS3 not available in this binary\n");
	    }
	    break;
	case 4:
	    type = INPtypelook("BSIM1");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type BSIM1 not available in this binary\n");
	    }
	    break;
	case 5:
	    type = INPtypelook("BSIM2");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type BSIM2 not available in this binary\n");
	    }
	    break;
	case 6:
	    type = INPtypelook("Mos6");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type MOS6 not available in this binary\n");
	    }
	    break;
	case 7:
	    type = INPtypelook("MOS7");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type MOS7 not available in this binary\n");
	    }
	    break;
	case 8: case 49:
	    err = INPfindVer(line, ver);

	    if ( strcmp(ver, "3.0") == 0 ) {
	      type = INPtypelook("BSIM3v0");
	    }
	    if ( strcmp(ver, "3.1") == 0 ) {
	      type = INPtypelook("BSIM3v1");
	    }
	    if ( strcmp(ver, "3.1s") == 0 ) {
	      type = INPtypelook("BSIM3v1S");
	    }
	    if ( strcmp(ver, "3.1a") == 0 ) {
	      type = INPtypelook("BSIM3v1A");
	    }
	    if ( prefix("3.2", ver)) { /* version string ver has to start with 3.2 */
	      type = INPtypelook("BSIM3v32");
	    }
	    if ( (strstr(ver, "default")) 
	        || (prefix("3.3", ver)) ) {
	      type = INPtypelook("BSIM3");
	    }
	    if (type < 0) {
	       err = (char *) tmalloc(sizeof(char) * (60 + strlen(ver)));
	       sprintf(err,"Device type BSIM3 version %s not available in this binary\n",ver);
	    }
	    break;
	case  9:
	    type = INPtypelook("Mos9");
	    if(type < 0) {
		err = INPmkTemp
		    ("Device type MOS9 not available in this binary\n");
	    }
	    break;
	case 14: case 54:
	    err = INPfindVer(line, ver); /* mapping of minor versions are included */
	    if ((prefix("4.0", ver)) || (prefix("4.1", ver)) || (prefix("4.2", ver))) {
	      type = INPtypelook("BSIM4v2");
	    }
	    if (prefix("4.3", ver)) {
	      type = INPtypelook("BSIM4v3");
	    }
	    if (prefix("4.4", ver)) {
	      type = INPtypelook("BSIM4v4");
	    }
	    if (prefix("4.5", ver)) {
	      type = INPtypelook("BSIM4v5");
	    }
	    if ( (strstr(ver, "default")) 
	        || (prefix("4.6", ver)) ) {
	      type = INPtypelook("BSIM4");
	    }
	    if (type < 0) {
	       err = (char *) tmalloc(sizeof(char) * (60 + strlen(ver)));
	       sprintf(err,"Device type BSIM4 version %s not available in this binary\n",ver);
	    }
	    break;
	case 15:
	    type = INPtypelook("BSIM5");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type BSIM5 not available in this binary\n");
	    }
	    break;        
	case 16:
	    type = INPtypelook("BSIM6");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type BSIM6 not available in this binary\n");}
	    break;
#ifdef ADMS
	case 44:
		type = INPtypelook("ekv");
		if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type EKV not available in this binary\n");
	    }
	    break;    
	case 45:
		type = INPtypelook("psp102");
		if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type PSP102 not available in this binary\n");
	    }
                break;
#endif
	case 55:
	    type = INPtypelook("B3SOIFD");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Placeholder: Device type B3SOIFD not available in this binary\n");
	    }
	    break;	
        case 56:
	    type = INPtypelook("B3SOIDD");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Placeholder: Device type B3SOIDD not available in this binary\n");
	    }
	    break;	    
	case 57:
	    type = INPtypelook("B3SOIPD");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Placeholder: Device type B3SOIPD not available in this binary\n");
	    }
	    break;	        
	case 10: case 58:
	    type = INPtypelook("B4SOI");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type B4SOI V4.0 not available in this binary\n");
	    }
	    break;  	    
	
	case 62:
	    type = INPtypelook("SOI3");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type SOI3 not available in this binary (STAG release)\n");
	    }
	    break;
	case 64:
	    type = INPtypelook("HiSIM1");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Placeholder: Device type HiSIM1 not available in this binary\n");
	    }
	    break;
	default:		/* placeholder; use level xxx for the next model */
	    err =
		INPmkTemp
#ifdef ADMS
		("Only MOS device levels 1-6,8-10,14,44,45,49,54-58,62,64 are supported in this binary\n");
#else
		("Only MOS device levels 1-6,8-10,14,49,54-58,62,64 are supported in this binary\n");
#endif		
	    break;
	}
	INPmakeMod(modname, type, image);
    } 
#ifdef  NDEV    
    /*  --------  Check if model is a numerical device --------- */
    else if (strcmp(typename, "ndev") == 0) {
	type = INPtypelook("NDEV");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type NDEV not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } 
#endif    
    /*  --------  Check if model is a resistor --------- */
    else if (strcmp(typename, "r") == 0) {
	type = INPtypelook("Resistor");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type Resistor not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } 
    
    /*  --------  Check if model is a transmission line of some sort --------- */
    else if(strcmp(typename,"txl") == 0) {
      INPgetTok(&line,&val,1);
      while (*line != '\0') {
	if (*val == 'R' || *val == 'r') {
	  INPgetTok(&line,&val,1);
          rval = INPevaluate(&val, &error1, 1);
	}
	if ((strcmp(val,"L") == 0)  || (strcmp(val,"l") == 0)) {
	  INPgetTok(&line,&val,1);
          lval = INPevaluate(&val, &error1, 1);
	}
	INPgetTok(&line,&val,1);
      }
      if(lval)
	rval = rval/lval;
      if (rval > 1.6e10) {
	type = INPtypelook("TransLine");
	INPmakeMod(modname,type,image);
      }
      if (rval > 1.6e9) {
	type = INPtypelook("CplLines");
	INPmakeMod(modname,type,image);
      }
      else {
	type = INPtypelook("TransLine");
	INPmakeMod(modname,type,image);
      }
      if(type < 0) {
	err = INPmkTemp(
			"Device type TransLine not available in this binary\n");
      }
      
    } 

    /*  --------  Check if model is a coupled transmission line --------- */
    else if(strcmp(typename,"cpl") == 0) {
      type = INPtypelook("CplLines");
      if(type < 0) {
	err = INPmkTemp(
			"Device type CplLines not available in this binary\n");
      }
      INPmakeMod(modname,type,image);

    } 
    
    
    /*  --------  Check if model is a cap --------- */
    else if (strcmp(typename, "c") == 0) {
	type = INPtypelook("Capacitor");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type Capacitor not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } 

    /*  --------  Check if model is an ind --------- */
    else if (strcmp(typename, "l") == 0) {
	type = INPtypelook("Inductor");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type Inductor not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    }    
    
    
    /*  --------  Check if model is a switch --------- */
    else if (strcmp(typename, "sw") == 0) {
	type = INPtypelook("Switch");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type Switch not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } 
    
    /*  --------  Check if model is a Current Controlled Switch --------- */
    else if (strcmp(typename, "csw") == 0) {
	type = INPtypelook("CSwitch");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type CSwitch not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } 
    
    /*  --------  Check if model is a Lossy TransLine --------- */
    else if (strcmp(typename, "ltra") == 0) {
	type = INPtypelook("LTRA");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type LTRA not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } 

#ifdef CIDER     
    else if(strcmp(typename,"numd") == 0) {
         err = INPfindLev(line,&lev);
    switch( lev ) {
    case 1:
    default:
        type = INPtypelook("NUMD");
        if(type < 0) {
           err = 
           INPmkTemp
           ("Device type NUMD not available in this binary\n");
         }
        break;
    case 2:
        type = INPtypelook("NUMD2");
        if(type < 0) {
           err = 
           INPmkTemp
           ("Device type NUMD2 not available in this binary\n");
        }
        break;
	}
    INPmakeMod(modname,type,image);
    } else if(strcmp(typename,"nbjt") == 0) {
        err = INPfindLev(line,&lev);
    switch( lev ) {
    case 1:
    default:
        type = INPtypelook("NBJT");
        if(type < 0) {
           err = 
           INPmkTemp
           ("Device type NBJT not available in this binary\n");
        }
        break;
    case 2:
        type = INPtypelook("NBJT2");
        if(type < 0) {
           err = 
           INPmkTemp
           ("Device type NBJT2 not available in this binary\n");
        }
        break;
     }
     INPmakeMod(modname,type,image);
    } else if(strcmp(typename,"numos") == 0) {
        type = INPtypelook("NUMOS");
        if(type < 0) {
            err = 
            INPmkTemp
            ("Device type NUMOS not available in this binary\n");
         }
        INPmakeMod(modname,type,image);
    } 
#endif /* CIDER */

    /*  type poly added by SDB  . . . */
#ifdef XSPICE
    /*  --------  Check if model is a poly (specific to xspice) --------- */
    else if ( (strcmp(typename, "poly") == 0) ||
	      (strcmp(typename, "POLY") == 0) ) {
	type = INPtypelook("POLY");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type POLY not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } 
#endif    
    
    /*  --------  Default action  --------- */    
    else {
#ifndef XSPICE    
	type = -1;
	err = (char *) MALLOC(35 + strlen(typename));
	(void) sprintf(err, "unknown model type %s - ignored\n", typename);
#else	
      /* gtri - modify - wbk - 10/23/90 - modify to look for code models */

#ifdef TRACE
      printf("In INPdomodel, found unknown model type, typename = %s . . .\n", typename); 
#endif

      /* look for this model type and put it in the table of models */
      type = INPtypelook(typename);
      if(type < 0) {
	err = (char *) MALLOC(35 + strlen(typename));
	sprintf(err,"Unknown model type %s - ignored\n",typename);

#ifdef TRACE
	printf("In INPdomodel, ignoring unknown model typ typename = %s . . .\n", typename); 
#endif

      }
      else {

#ifdef TRACE
	printf("In INPdomodel, adding unknown model typename = %s to model list. . .\n", typename); 
#endif

	INPmakeMod(modname,type,image);
      }
      
      /* gtri - end - wbk - 10/23/90 */
#endif
      
    }
    tfree(typename);
    return (err);
}
