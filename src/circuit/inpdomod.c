/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include <stdio.h>
#include "iferrmsg.h"
#include "inpdefs.h"
#include "inp.h"


char *INPdomodel(void *ckt, card * image, INPtables * tab)
{

    char *modname;
    int type = -1;
    int lev;
    char *typename;
    char *err = (char *) NULL;
    char *line;

    line = image->line;
    INPgetTok(&line, &modname, 1);	/* throw away '.model' */
    INPgetTok(&line, &modname, 1);
    INPinsert(&modname, tab);
    INPgetTok(&line, &typename, 1);
    if ((strcmp(typename, "npn") == 0) || (strcmp(typename, "pnp") == 0)) {
	type = INPtypelook("BJT");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type BJT not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } else if (strcmp(typename, "d") == 0) {
	type = INPtypelook("Diode");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type Diode not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } else if ((strcmp(typename, "njf") == 0)
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
    } else if ((strcmp(typename, "nmf") == 0)
	       || (strcmp(typename, "pmf") == 0)) {
	type = INPtypelook("MES");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type MES not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } else if (strcmp(typename, "urc") == 0) {
	type = INPtypelook("URC");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type URC not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } else if ((strcmp(typename, "nmos") == 0)
	       || (strcmp(typename, "pmos") == 0)) {
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
	case 8:
	    type = INPtypelook("BSIM3");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type BSIM3 not available in this binary\n");
	    }
	    break;
	case 9:
	    type = INPtypelook("BSIMsoi");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type BSIMsoi not available in this binary\n");
	    }
	    break;
	case 14:
	    type = INPtypelook("BSIM4");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type BSIM4 not available in this binary\n");}
	    break;
	case 15:
	    type = INPtypelook("BSIM5");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Placeholder: Device type BSIM5 not available in this binary\n");
	    }
	    break;
	case 16:
	    type = INPtypelook("BSIM6");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Placeholder: Device type BSIM6 not available in this binary\n");
	    }
	    break;
	case 49:
	    type = INPtypelook("BSIM3V1");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type BSIM3V1 not available in this binary\n");
	    }
	    break;


	case 50:
	    type = INPtypelook("BSIM3V2");
	    if (type < 0) {
		err =
		    INPmkTemp
		    ("Device type BSIM3V2 not available in this binary\n");
	    }
	    break;
	default:		/* placeholder; use level xxx for the next model */
	    err =
		INPmkTemp
		("Only MOS device levels 1-6,8,14 are supported in this binary\n");
	    break;
	}
	INPmakeMod(modname, type, image);
    } else if (strcmp(typename, "r") == 0) {
	type = INPtypelook("Resistor");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type Resistor not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } else if (strcmp(typename, "c") == 0) {
	type = INPtypelook("Capacitor");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type Capacitor not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } else if (strcmp(typename, "sw") == 0) {
	type = INPtypelook("Switch");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type Switch not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } else if (strcmp(typename, "csw") == 0) {
	type = INPtypelook("CSwitch");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type CSwitch not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } else if (strcmp(typename, "ltra") == 0) {
	type = INPtypelook("LTRA");
	if (type < 0) {
	    err =
		INPmkTemp
		("Device type LTRA not available in this binary\n");
	}
	INPmakeMod(modname, type, image);
    } else {
	type = -1;
	err = (char *) MALLOC(35 + strlen(typename));
	(void) sprintf(err, "unknown model type %s - ignored\n", typename);
    }
    return (err);
}
