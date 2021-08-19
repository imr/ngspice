/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/* Functions to read SUPREM (Binary or Ascii) & ASCII input files */
#include <errno.h>
#include <string.h>

#include "ngspice/cidersupt.h"
#include "ngspice/cpextern.h"
#include "ngspice/ngspice.h"
#include "ngspice/profile.h"


static void free_profile_data(double **p);
static double **alloc_profile_data(size_t n);


int readAsciiData(const char *fileName, int impType, DOPtable **ppTable)
{
    int xrc = 0;
    FILE *fpAscii = (FILE *) NULL;
    int index;
    double x, y;
    int numPoints;
    DOPtable *tmpTable = (DOPtable *) NULL;
    double **profileData = (double **) NULL;
    double sign;

    /* Open Input File */
    if ((fpAscii = fopen( fileName, "r" )) == NULL) {
        (void) fprintf(cp_err, "unable to open SUPREM file \"%s\": %s\n",
                fileName, strerror(errno));
        xrc = -1;
        return xrc; /* return immediately, nothing to free */
    }

    /* Get sign of concentrations */
    if (impType == IMP_P_TYPE) {
      sign = -1.0;
    } else {
      sign = 1.0;
    }

    /* read the number of points */
    if (fscanf(fpAscii, "%d", &numPoints) != 1) {
        (void) fprintf(cp_err, "unable to read point count "
                "from SUPREM file \"%s\"\n",
                fileName);
      xrc = -1;
      goto EXITPOINT;
    }


    /* allocate 2-D array to read in data of x-coordinate and N(x) */
    profileData = alloc_profile_data((size_t) numPoints + 1);

    /* the number of points is stored as profileData[0][0] */
    profileData[0][0] = numPoints;

    for (index = 1; index <= numPoints; index++ ) {
        if (fscanf(fpAscii, "%lf   %lf ", &x, &y) != 2) { 
            (void) fprintf(cp_err, "unable to read point %d"
                    "from SUPREM file \"%s\"\n",
                    index + 1, fileName);
          xrc = -1;
          goto EXITPOINT;
        }
        profileData[0][index] = x;
        profileData[1][index] = sign * ABS(y);
    } /* end of loop over points */

    /* Now create a new lookup table */
    XCALLOC(tmpTable, DOPtable, 1);
    if (*ppTable == NULL) {
      /* First Entry */
      tmpTable->impId = 1;
      tmpTable->dopData = profileData;
      tmpTable->next = NULL;
      *ppTable = tmpTable;
    }
    else {
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
EXITPOINT:
    if (fpAscii != (FILE *) NULL) { /* close data file if open */
        fclose(fpAscii);
    }

    /* Free resources on error */
    if (xrc != 0) {
        free_profile_data(profileData);
        free(tmpTable);
    }

    return xrc;
} /* end of function readAsciiData  */



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


int readSupremData(const char *fileName, int fileType, int impType,
        DOPtable **ppTable)
{
    int xrc = 0;
#define MAX_GRID 500
    float x[ MAX_GRID ], conc[ MAX_GRID ];

    int index;
    DOPtable *tmpTable = (DOPtable *) NULL;
    double **profileData = (double **) NULL;
    int numNodes;

    /* read the Suprem data file */
    if ( fileType == 0 ) { /* BINARY FILE */
        xrc = SUPbinRead(fileName, x, conc, &impType, &numNodes);
    }
    else {
        xrc = SUPascRead( fileName, x, conc, &impType, &numNodes );
    }
    if (xrc != 0) {
        (void) fprintf(cp_err, "Data input failed.\n");
        xrc = -1;
        return xrc; /* return immediately, nothing to free */
    }

    /* allocate 2-D array to read in data of x-coordinate and N(x) */
    profileData = alloc_profile_data((size_t) numNodes + 1);

    /* the number of points is stored as profileData[0][0] */
    profileData[0][0] = numNodes;

    for( index = 1; index <= numNodes; index++ ) {
        profileData[ 0 ][ index ] = x[ index ];
        profileData[ 1 ][ index ] = conc[ index ];
    }

    /* Now create a new lookup table */
    XCALLOC(tmpTable, DOPtable, 1);
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

    if (xrc != 0) {
        free_profile_data(profileData);
        free(tmpTable);
    }
    return xrc;
} /* end of function readSupremData */



/* Allocate a profile data */
static double **alloc_profile_data(size_t n)
{
    double **p;
    XCALLOC(p, double *, 2);
    XCALLOC(p[0], double, n);
    XCALLOC(p[1], double, n);
    return p;
} /* end of function alloc_profile_data */



/* Free a profile data */
static void free_profile_data(double **p)
{
    /* Immediate exit if no allocation */
    if (p == (double **) NULL) {
        return;
    }

    free(p[0]);
    free(p[1]);
    free(p);
} /* end of function alloc_profile_data */








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
