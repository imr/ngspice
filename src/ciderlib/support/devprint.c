/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

/* Device-type Dependent Printing Routines */

#include "ngspice/ngspice.h"
#include "ngspice/optndefs.h"
#include "ngspice/cidersupt.h"

void
printVoltages(FILE *file, char *mName, char *iName, int devType, 
              int numVolt, double v1, double delV1, double v2, 
	      double delV2, double v3, double delV3 )
/* FILE *file   output filestream */
/* char *mName	name of model     */
/* char *iName  name of instance  */
{
  fprintf( file, "\n" );
  switch ( devType ) {
    case OPTN_RESISTOR:
      fprintf( file, "RES %s:%s voltage:\n", mName, iName );
      fprintf( file, "    Vpn =% .4e delVpn =% .4e\n", v1, delV1 );
      break;
    case OPTN_CAPACITOR:
      fprintf( file, "CAP %s:%s voltage:\n", mName, iName );
      fprintf( file, "    Vpn =% .4e delVpn =% .4e\n", v1, delV1 );
      break;
    case OPTN_DIODE:
      fprintf( file, "DIO %s:%s voltage:\n", mName, iName );
      fprintf( file, "    Vpn =% .4e delVpn =% .4e\n", v1, delV1 );
      break;
    case OPTN_MOSCAP:
      fprintf( file, "MOS %s:%s voltage:\n", mName, iName );
      fprintf( file, "    Vgb =% .4e delVgb =% .4e\n", v1, delV1 );
      break;
    case OPTN_BIPOLAR:
      fprintf( file, "BJT %s:%s voltages:\n", mName, iName );
      if ( numVolt == 3 ) {
	fprintf( file, "    Vce =% .4e delVce =% .4e\n",
	    v1 - v3, delV1 - delV3 );
	fprintf( file, "    Vbe =% .4e delVbe =% .4e\n",
	    v2 - v3, delV2 - delV3 );
	fprintf( file, "    Vcs =% .4e delVcs =% .4e\n", v1, delV1 );
      } else {
	fprintf( file, "    Vce =% .4e delVce =% .4e\n", v1, delV1 );
	fprintf( file, "    Vbe =% .4e delVbe =% .4e\n", v2, delV2 );
      }
      break;
    case OPTN_MOSFET:
      fprintf( file, "MOS %s:%s voltages:\n", mName, iName );
      fprintf( file, "    Vdb =% .4e delVdb =% .4e\n", v1, delV1 );
      fprintf( file, "    Vgb =% .4e delVgb =% .4e\n", v2, delV2 );
      fprintf( file, "    Vsb =% .4e delVsb =% .4e\n", v3, delV3 );
      break;
    case OPTN_JFET:
      if ( numVolt == 3 ) {
	fprintf( file, "JFET %s:%s voltages:\n", mName, iName );
	fprintf( file, "    Vdb =% .4e delVdb =% .4e\n", v1, delV1 );
	fprintf( file, "    Vgb =% .4e delVgb =% .4e\n", v2, delV2 );
	fprintf( file, "    Vsb =% .4e delVsb =% .4e\n", v3, delV3 );
      } else {
	fprintf( file, "JFET %s:%s voltages:\n", mName, iName );
	fprintf( file, "    Vds =% .4e delVds =% .4e\n", v1, delV1 );
	fprintf( file, "    Vgs =% .4e delVgs =% .4e\n", v2, delV2 );
      }
      break;
    case OPTN_SOIBJT:
    case OPTN_SOIMOS:
    case OPTN_MESFET:
    default:
      break;
  }
}
