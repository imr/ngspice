/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/* Functions to read SUPREM (Binary or Ascii) & ASCII input files */

#include "ngspice/ngspice.h"
#include "ngspice/profile.h"
#include "ngspice/cidersupt.h"

void
readAsciiData( char *fileName, int impType, DOPtable **ppTable )
{
    FILE *fpAscii;
    int index;
    double x, y;
    int numPoints;
    DOPtable *tmpTable;
    double **profileData;
    double sign;

    /* Open Input File */
    if ((fpAscii = fopen( fileName, "r" )) == NULL) {
      perror( fileName );
      exit(-1);
    }

    /* Get sign of concentrations */
    if (impType == IMP_P_TYPE) {
      sign = -1.0;
    } else {
      sign = 1.0;
    }

    /* read the number of points */
    fscanf( fpAscii, "%d", &numPoints );

    /* allocate 2-D array to read in data of x-coordinate and N(x) */
    XCALLOC( profileData, double *, 2 );
    for( index = 0; index <= 1; index++ ) {
	XCALLOC( profileData[ index ], double, 1 + numPoints );
    }
    /* the number of points is stored as profileData[0][0] */
    profileData[0][0] = numPoints;

    for( index = 1; index <= numPoints; index++ ) {
	fscanf( fpAscii, "%lf   %lf ", &x, &y ); 
	profileData[ 0 ][ index ] = x;
	profileData[ 1 ][ index ] = sign * ABS(y);
    }

    /* Now create a new lookup table */
    XCALLOC( tmpTable, DOPtable, 1 );
    if ( *ppTable == NULL ) {
      /* First Entry */
      tmpTable->impId = 1;
      tmpTable->dopData = profileData;
      tmpTable->next = NULL;
      *ppTable = tmpTable;
    } else {
      tmpTable->impId = (*ppTable)->impId + 1;
      tmpTable->dopData = profileData;
      tmpTable->next = *ppTable;
      *ppTable = tmpTable;
    }

    /* for debugging print the data that has been just read */
    /*
    for( index = 1; index <= numPoints; index++ ) {
	printf("\n %e     %e", profileData[ 0 ][ index ], profileData[ 1 ][ index ]);
    }
    */
    fclose(fpAscii);
    return;
}

/* interface routine based on notes provided by Steve Hansen of Stanford  */

/*
 * The material types are:
 *	1 = single crystal silicon
 *	2 = silicon dioxide
 *	3 = poly-crystalline silicon
 *	4 = silicon nitride
 *	5 = aluminum

 * The impurity types are:
 *	1 = boron
 *	2 = phosphorus
 *	3 = arsenic
 *	4 = antimony

 * The crystalline orientations are:
 * 	1 = <111>
 * 	2 = <100>
 * 	3 = <110>

 * The layer thinkness, poly-crystalline grain size, node spacing and
 * distance from the surface are all in microns.

 * The integrated dopant concentration and the phophorus implant dose are
 * in atoms per square centimeter.

 * The interior polycrystalline grain concentration and the impurity
 * concentrations at each node are in atoms per cubic centimeter.
 */


void
readSupremData(char *fileName, int fileType, int impType, DOPtable **ppTable)
{
#define MAX_GRID 500
    float x[ MAX_GRID ], conc[ MAX_GRID ];

    int index;
    DOPtable *tmpTable;
    double **profileData;
    int numNodes;

    /* read the Suprem data file */
    if ( fileType == 0 ) { /* BINARY FILE */
      SUPbinRead( fileName, x, conc, &impType, &numNodes );
    }
    else {
      SUPascRead( fileName, x, conc, &impType, &numNodes );
    }

    /* allocate 2-D array to read in data of x-coordinate and N(x) */
    XCALLOC( profileData, double *, 2 );
    for( index = 0; index <= 1; index++ ) {
	XCALLOC( profileData[ index ], double, 1 + numNodes );
    }
    /* the number of points is stored as profileData[0][0] */
    profileData[0][0] = numNodes;

    for( index = 1; index <= numNodes; index++ ) {
	profileData[ 0 ][ index ] = x[ index ];
	profileData[ 1 ][ index ] = conc[ index ];
    }

    /* Now create a new lookup table */
    XCALLOC( tmpTable, DOPtable, 1 );
    if ( *ppTable == NULL ) {
      /* First Entry */
      tmpTable->impId = 1;
      tmpTable->dopData = profileData;
      tmpTable->next = NULL;
      *ppTable = tmpTable;
    } else {
      tmpTable->impId = (*ppTable)->impId + 1;
      tmpTable->dopData = profileData;
      tmpTable->next = *ppTable;
      *ppTable = tmpTable;
    }

    /* for debugging print the data that has been just read */
    /*
    for( index = 1; index <= numNodes; index++ ) {
	printf("%e     %e\n", profileData[ 0 ][ index ], profileData[ 1 ][ index ]);
    }
    */
    return;
}


/* main program to debug readSupremData */

/*
main(ac, av)
char **av;
{
    void readSupremData();
    DOPtable *supTable = NULL;
    double **supInput;
    int numPoints, index;
    char *impName;
    int impType;

    switch (ac) {
	case 1: 
	    printf( "Usage: %s suprem-file ...\n", av[0] );
	    exit(-1);
	    break;
	default:
	    break;
    }
      
    for ( index = 1; index < ac; index++ ) {
      for ( impType=1; impType <= 4; impType++ ) {
	readSupremData( av[index], 1, impType, &supTable );
      }
    }
    for ( ; supTable ISNOT NULL; supTable = supTable->next ) {
      fprintf( stdout, "\"Impurity Number: %d\n", supTable->impId );
      supInput = supTable->dopData;
      numPoints = supInput[0][0];
      for( index = 1; index <= numPoints; index++ ) {
	printf("%e\t%e\n",
	    supInput[ 0 ][ index ], ABS(supInput[ 1 ][ index ]) + 1e-20 );
      }
      fprintf( stdout, "\n" );
    }
}
*/
