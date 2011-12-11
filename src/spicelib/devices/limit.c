/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/devdefs.h"

/* limiting routines for junction voltages */

double DvRevMax = 0.5, DvFwdMax = 50.0e-3;

double 
limitResistorVoltage( double vnew, double vold, int *icheck )
{
    double vlim, vinc;

    if( vnew > vold ) {
	vinc = 0.5;
	vlim = vold + vinc;
	if( vnew < vlim ) {
	    *icheck = 0;
	    return( vnew );
	} else {
	    *icheck = 1;
	    return( vlim );
	}
    } else if ( vnew < vold ) {
	vinc = 0.5;
	vlim = vold - vinc;
	if( vnew > vlim ) {
	    *icheck = 0;
	    return( vnew );
	} else {
	    *icheck = 1;
	    return( vlim );
	}
    } else { /* vnew == vold */
	*icheck = 0;
	return( vnew );
    }
    /* NOTREACHED */
}

double limitJunctionVoltage( double vnew, double vold, int *icheck )
{
    double vlim, vinc;

    if( vold >= 0.0 ) {
	if( vnew > vold ) {
	    if( vold > 0.65 ) 
		vinc = DvFwdMax;
	    else 
		vinc = 2.0 * DvFwdMax;
	    vlim = vold + vinc;
	    if( vnew < vlim ) {
		*icheck = 0;
		return( vnew );
	    }
	    else {
		*icheck = 1;
		return( vlim );
	    }
	}
	else if ( vnew == vold )  {
	    *icheck = 0;
	    return ( vnew );
	}
	else {
	    /* vnew is less than vold */
	    /* case 1: vnew is positive */
	    /* case 2: vnew is negative, ensure it goes thru zero */
	    if( vnew < 0.0 && vold <= 0.05 && vold > 0.0) {
		/* limit to zero when vnew < 0.0 and vold |= 0.0 */
		*icheck = 1;
		return( 0.0 );
	    }
	    vinc = 2.0 * DvFwdMax;
	    vlim = vold - vinc;
	    if( vlim > vnew ) {
		*icheck = 1;
		return( vlim );
	    }
	    else {
		*icheck = 0;
		return( vnew );
	    }
	}
    }
    else {
	if( vnew < vold ) {
	    vlim = vold - DvRevMax;
	    if ( vlim > vnew ) {
		*icheck = 1;
		return( vlim );
	    }
	    else
	    {
		*icheck = 0;
		return( vnew );
	    }
	}
	else {
	    /* vnew > vold. check if vnew less than 0.0 */
	    if( vnew < 0.0 ) {
		vlim = vold + 1.0;
		if( vnew < vlim ) {
		    *icheck = 0;
		    return( vnew );
		}
		else {
		    *icheck = 1;
		    return( vlim );
		}
	    }
	    else {
		vlim = vold + 2.0 * DvRevMax;
		*icheck = 1;
		if( vlim > 0.0 )
		    return( 0.0 );
		else
		    return( vlim );
	    }
	}
    }
}
	    
	    
double limitVbe( double vnew, double vold, int *icheck )
{
    double vlim, vinc;
    if( vold >= 0.0 ) {
	if( vnew > vold ) {
	    if( vold > 0.9 ) 
		/* deep high-level injection conditions */
		vinc = 0.01;
	    else if( vold > 0.85 ) 
		vinc = 0.025;
	    else if( vold > 0.65 ) 
		vinc = 0.05;
	    else 
		vinc = 0.10;
	    vlim = vold + vinc;
	    if( vnew < vlim ) {
		*icheck = 0;
		return( vnew );
	    }
	    else {
		*icheck = 1;
		return( vlim );
	    }
	}
	else if ( vnew == vold )  {
	    *icheck = 0;
	    return ( vnew );
	}
	else {
	    /* vnew is less than vold */
	    /* case 1: vnew is positive */
	    /* case 2: vnew is negative, ensure it goes thru zero */
	    if( vnew < 0.0 && vold <= 0.05 && vold > 0.0) {
		/* limit to zero when vnew < 0.0 and vold |= 0.0 */
		*icheck = 1;
		return( 0.0 );
	    }
	    vinc = 0.1;
	    vlim = vold - vinc;
	    if( vlim > vnew ) {
		*icheck = 1;
		return( vlim );
	    }
	    else {
		*icheck = 0;
		return( vnew );
	    }
	}
    }
    else {
	if( vnew < vold ) {
	    vlim = vold - 0.1;  /* XXX Originally 0.1 */
	    if ( vlim > vnew ) {
		*icheck = 1;
		return( vlim );
	    }
	    else {
		*icheck = 0;
		return( vnew );
	    }
	}
	else {
	    /* vnew > vold. check if vnew less than 0.0 */
	    if( vnew < 0.0 ) {
		vlim = vold + 1.0;
		if( vnew < vlim ) {
		    *icheck = 0;
		    return( vnew );
		}
		else {
		    *icheck = 1;
		    return( vlim );
		}
	    }
	    else {
		vlim = vold + 1.0;
		*icheck = 1;
		if( vlim > 0.0 )
		    return( 0.0 );
		else
		    return( vlim );
	    }
	}
    }
}

double limitVce( double vnew, double vold, int *icheck )
{
    double vlim;
    if( vold >= 0.0 ) {
	if( vnew > vold ) {
	    vlim = vold + 1.0;
	    if( vnew < vlim ) {
		*icheck = 0;
		return( vnew );
	    } else {
		*icheck = 1;
		return( vlim );
	    }
	} else {
	    vlim = vold - 0.5; /* XXX Originally 0.2 */
	    if( vlim > vnew ) {
		*icheck = 1;
		return( vlim );
	    } else {
		*icheck = 0;
		return( vnew );
	    }
	}
    } else {
	if( vnew < vold ) {
	    vlim = vold - 1.0;
	    if ( vlim > vnew ) {
		*icheck = 1;
		return( vlim );
	    } else {
		*icheck = 0;
		return( vnew );
	    }
	} else {
	    /* vnew > vold. check if vnew less than 0.0 */
	    if( vnew < 0.0 ) {
		vlim = vold + 1.0;
		if( vnew < vlim ) {
		    *icheck = 0;
		    return( vnew );
		} else {
		    *icheck = 1;
		    return( vlim );
		}
	    } else {
		vlim = vold + 1.0;
		*icheck = 1;
		if( vlim > 0.0 )
		    return( 0.0 );
		else
		    return( vlim );
	    }
	}
    }
}

double limitVgb( double vnew, double vold, int *icheck )
{
    double vlim;
    if( vold >= 0.0 ) {
	if( vnew > vold ) {
	    vlim = vold + 1.0;
	    if( vnew < vlim ) {
		*icheck = 0;
		return( vnew );
	    } else {
		*icheck = 1;
		return( vlim );
	    }
	} else {
	    vlim = vold - 0.2;
	    if( vlim > vnew ) {
		*icheck = 1;
		return( vlim );
	    } else {
		*icheck = 0;
		return( vnew );
	    }
	}
    } else {
	if( vnew < vold ) {
	    vlim = vold - 1.0;
	    if ( vlim > vnew ) {
		*icheck = 1;
		return( vlim );
	    } else {
		*icheck = 0;
		return( vnew );
	    }
	} else {
	    /* vnew > vold. check if vnew less than 0.0 */
	    if( vnew < 0.0 ) {
		vlim = vold + 1.0;
		if( vnew < vlim ) {
		    *icheck = 0;
		    return( vnew );
		} else {
		    *icheck = 1;
		    return( vlim );
		}
	    } else {
		vlim = vold + 1.0;
		*icheck = 1;
		if( vlim > 0.0 )
		    return( 0.0 );
		else
		    return( vlim );
	    }
	}
    }
}
