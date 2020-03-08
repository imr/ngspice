/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 *   Translated FORTRAN subroutine to read the SUPREM-3 binary file
 *   the data is read and stored in two arrays x and conc
 */

#include <errno.h>
#include <string.h>

#include "ngspice/cidersupt.h"
#include "ngspice/cpextern.h"
#include "ngspice/ngspice.h"

#define MAXMAT 10
#define MAXIMP 4
#define MAXLAYER 10
#define MAXGRID 500

#define GFREAD( fp, ptr, type, num )\
    if (num && (fread( ptr, sizeof(type), (size_t) num,\
            fp ) != (size_t) num)) {\
        xrc = -1;\
        goto EXITPOINT;\
    }

#define DEBUG	if (0)

int SUPbinRead(const char *inFile, float *x, float *conc, int *impId,
        int *numNod)
{
    int xrc = 0;
  int idata, recordMark;
  int ldata;
  int i, j;
  float rdata;
  char cdata[21];
  int numLay, numImp, numGrid;
  int impTyp[4], matTyp[10], topNod[10], siIndex, offset;
  float xStart;
  float layerTh[10];
  float con[500];
  FILE *fpSuprem = (FILE *) NULL;

  /* Clear Concentration Array */
  for ( i=0; i < MAXGRID; i++ ) {
    conc[i] = 0.0;
  }

    /* Open Input File */
    if ((fpSuprem = fopen( inFile, "r" )) == NULL) {
        (void) fprintf(cp_err, "Unable to read file \"%s\": %s.\n",
                inFile, strerror(errno));
        xrc = -1;
        goto EXITPOINT;
    }

/*
 * The first record contains the number of layers (I4), the number of 
 * impurities (I4), and the number of nodes (I4) present in the structure.
 */
  GFREAD( fpSuprem, &recordMark, int, 1 );
  GFREAD( fpSuprem, &numLay, int, 1 );
  GFREAD( fpSuprem, &numImp, int, 1 );
  GFREAD( fpSuprem, &numGrid, int, 1 );
  DEBUG fprintf(stderr,"rec 1: %d %d %d\n", numLay, numImp, numGrid);
  GFREAD( fpSuprem, &recordMark, int, 1 );

/*
 * The second record contains, for each layer, the material type (I4), the
 * layer thickness (R4), and the pointer to the top node of the layer (I4).
 */
  GFREAD( fpSuprem, &recordMark, int, 1 );
  for ( i=0; i < numLay; i++ ) {
    GFREAD( fpSuprem, &matTyp[i], int, 1 );
    GFREAD( fpSuprem, &layerTh[i], float, 1 );
    GFREAD( fpSuprem, &topNod[i], int, 1 );
    DEBUG fprintf(stderr,"rec 2: %d %f %d\n", matTyp[i], layerTh[i], topNod[i] );
  }
  GFREAD( fpSuprem, &recordMark, int, 1 );

/*
 * The third record contains, for each layer, the material name (A20).
 */
  /* Put a null at the end */
  cdata[20] = '\0';
  GFREAD( fpSuprem, &recordMark, int, 1 );
  for ( i=0; i < numLay; i++ ) {
    GFREAD( fpSuprem, cdata, char, 20 );
    DEBUG fprintf(stderr,"rec 3: %s\n", cdata );
  }
  GFREAD( fpSuprem, &recordMark, int, 1 );

/*
 * The fourth record contains, for each layer, the crystalline orientation
 * (I4), and the poly-crystalline grain size in microns (R4).
 */
  GFREAD( fpSuprem, &recordMark, int, 1 );
  for ( i=0; i < numLay; i++ ) {
    GFREAD( fpSuprem, &idata, int, 1 );
    GFREAD( fpSuprem, &rdata, float, 1 );
    DEBUG fprintf(stderr,"rec 4: %d %f\n", idata, rdata );
  }
  GFREAD( fpSuprem, &recordMark, int, 1 );
     
/*
 * The fifth record contains, for each impurity, the type of impurity (I4).
 */
  GFREAD( fpSuprem, &recordMark, int, 1 );
  for ( i=0; i < numImp; i++ ) {
    GFREAD( fpSuprem, &impTyp[i], int, 1 );
    DEBUG fprintf(stderr,"rec 5: %d\n", impTyp[i] );
  }
  GFREAD( fpSuprem, &recordMark, int, 1 );

/*
 * The sixth record contains, for each impurity, the impurity name (A20).
 */
  GFREAD( fpSuprem, &recordMark, int, 1 );
  for ( i=0; i < numImp; i++ ) {
    GFREAD( fpSuprem, cdata, char, 20 );
    DEBUG fprintf(stderr,"rec 6: %s\n", cdata );
  }
  GFREAD( fpSuprem, &recordMark, int, 1 );

/*
 * The seventh record contains, for each layer by each impurity, the
 * integrated dopant (R4), and the interior concentration of the
 * polysilicon grains (R4).
 */
  GFREAD( fpSuprem, &recordMark, int, 1 );
  for ( j=0; j < numLay; j++ ) {
    for ( i=0; i < numImp; i++ ) {
      GFREAD( fpSuprem, &rdata, float, 1 );
      DEBUG fprintf(stderr,"rec 7: %e", rdata );
      GFREAD( fpSuprem, &rdata, float, 1 );
      DEBUG fprintf(stderr," %e\n", rdata );
    }
  }
  GFREAD( fpSuprem, &recordMark, int, 1 );

/*
 * The eighth record contains, for each node in the structure, the distance
 * to the next deepest node (R4).
 */
  GFREAD( fpSuprem, &recordMark, int, 1 );
  for ( i=0; i < numGrid; i++ ) {
    GFREAD( fpSuprem, &rdata, float, 1 );
  }
  DEBUG fprintf(stderr,"rec 8: %f\n", rdata );
  GFREAD( fpSuprem, &recordMark, int, 1 );

/*
 * The ninth record contains, for each node in the structure, the distance
 * from the surface (R4).
 */
  GFREAD( fpSuprem, &recordMark, int, 1 );
  GFREAD( fpSuprem, &x[1], float, numGrid );
  DEBUG fprintf(stderr,"rec 9: %f\n", x[1] );
  GFREAD( fpSuprem, &recordMark, int, 1 );

/*
 * Next, for each impurity there is a record containing the impurity's
 * chemical concentration at each node (R4) and a record containing the
 * impurity's active concentration at each node (R4).
 */

  for ( j=0; j < numImp; j++ ) {
/* chemical concentration - not required */
    GFREAD( fpSuprem, &recordMark, int, 1 );
    GFREAD( fpSuprem, &con[1], float, numGrid );
    DEBUG fprintf(stderr,"rec 10: %e\n", con[1] );
    GFREAD( fpSuprem, &recordMark, int, 1 );
        
/* store active concentration */
    GFREAD( fpSuprem, &recordMark, int, 1 );
    GFREAD( fpSuprem, &con[1], float, numGrid );
    DEBUG fprintf(stderr,"rec 11: %e\n", con[1] );
    GFREAD( fpSuprem, &recordMark, int, 1 );

    if (impTyp[j] == *impId) {
/*...Boron...*/
      if (impTyp[j] == 1) {
	for ( i=1; i <= numGrid; i++ ) conc[i] = - con[i];
      } else {
/*...All Other Impurities: P, As, Sb ...*/
	for ( i=1; i <= numGrid; i++ ) conc[i] = con[i];
      }
    }
  }

/*
 * The last record in the file contains some random stuff that might be
 * useful to some people, the temperature in degrees Kelvin of the last
 * diffusion step (R4), the phosphorus implant dose (R4), the arsenic
 * implant flag (L4).
 */
  GFREAD( fpSuprem, &recordMark, int, 1 );
  GFREAD( fpSuprem, &rdata, float, 1 );
  DEBUG fprintf(stderr,"rec 12: %f", rdata );
  GFREAD( fpSuprem, &rdata, float, 1 );
  DEBUG fprintf(stderr," %e", rdata );
  GFREAD( fpSuprem, &ldata, int, 1 );
  DEBUG fprintf(stderr," %d\n", ldata );
  GFREAD( fpSuprem, &recordMark, int, 1 );

    if (fclose(fpSuprem) != 0) {
        (void) fprintf(cp_err, "Unable to close file \"%s\": %s.\n",
                inFile, strerror(errno));
        xrc = -1;
        goto EXITPOINT;
    }
    fpSuprem = (FILE *) NULL;

     /* shift silicon layer to beginning of array */
    for (j = numLay; --j >= 0; ) {
        if (matTyp[j] == 1) {
            break;
        }
    }

    if (j < 0) {
        (void) fprintf(cp_err, "internal error in %s!\n", __FUNCTION__);
        xrc = -1;
        goto EXITPOINT;
    }


  siIndex = j;

  offset = topNod[ siIndex ] - 1;
  numGrid -= offset;
  xStart = x[1 + offset];
  for ( i=1; i <= numGrid; i++ ) {
    x[i] = x[i + offset] - xStart;
    conc[i] = conc[i + offset];
  }

/* Store number of valid nodes using pointer */
  *numNod = numGrid;

EXITPOINT:
    if (fpSuprem != (FILE *) NULL) {
        if (fclose(fpSuprem) != 0) {
            (void) fprintf(cp_err, "Unable to close \"%s\" at exit: %s\n",
                    inFile, strerror(errno));
            xrc = -1;
        }
    }

    return xrc;
} /* end of function SUPbinRead */



int SUPascRead(const char *inFile, float *x, float *conc, int *impId,
        int *numNod)
{
    int xrc = 0;
  int i, j;
  char cdata[21];
  int numLay, numImp, numGrid;
  int impTyp[4], matTyp[10], topNod[10], siIndex, offset;
  float xStart;
  float layerTh[10];
  float con[500];
  FILE *fpSuprem;

  /* Clear Concentration Array */
  for ( i=0; i < MAXGRID; i++ ) {
    conc[i] = 0.0;
  }

    /* Open Input File */
    if ((fpSuprem = fopen( inFile, "r" )) == NULL) {
        (void) fprintf(cp_err, "Unable to open file \"%s\": %s.\n",
                inFile, strerror(errno));
        xrc = -1;
        goto EXITPOINT;
    }

/*
 * The first line contains the number of layers (I4), the number of 
 * impurities (I4), and the number of nodes (I4) present in the structure.
 */
    if (fscanf( fpSuprem, "%d %d %d\n", &numLay, &numImp, &numGrid) != 3) {
        (void) fprintf(cp_err, "Unable to read file first line of \"%s\"\n",
                inFile);
        xrc = -1;
        goto EXITPOINT;
    }

  DEBUG fprintf( stderr, "set 1: %d %d %d\n", numLay, numImp, numGrid);

    /*
     * The second set of lines contains, for each layer, the material name
     * (A20), the material type (I4), the layer thickness (R4), and the
     * pointer to the top node of the layer (I4), and an unknown int and
     * unknown float.
     *
     * The material type code:
     *   1 - Si
     *   2 - SiO2
     *   3 - Poly
     *   4 - Si3N4
     *   5 - Alum
     */
    for (i = 0; i < numLay; ++i) {
        int idata;
        float rdata;
        if (fscanf(fpSuprem, "%s\n %d %e %d %d %e\n",
                cdata, &matTyp[i], &layerTh[i], &topNod[i],
                &idata, &rdata) != 6) {
            (void) fprintf(cp_err, "Unable to read layer %d "
                    "from file \"%s\".\n",
                    i + 1, inFile);
            xrc = -1;
            goto EXITPOINT;
        }

        DEBUG fprintf(stderr,"set 2: %s: %d %f %d\n",
                cdata, matTyp[i], layerTh[i], topNod[i]);
    } /* end of loop over layers */

    /*
     * The third set of lines contains, for each impurity, the name of the
     * impurity (A20) and the type of impurity (I4).
     */
    for (i = 0; i < numImp; ++i) {
        if (fscanf( fpSuprem, "%s\n %d\n", cdata, &impTyp[i]) != 2) {
            (void) fprintf(cp_err, "Unable to read impurity %d "
                    "from file \"%s\".\n",
                    i + 1, inFile);
            xrc = -1;
            goto EXITPOINT;
        }
        DEBUG fprintf(stderr,"set 3: %s: %d\n", cdata, impTyp[i]);
    } /* end of loop over impurities */

    /*
     * The fourth set of lines contains, for each layer by each impurity,
     * the integrated dopant (R4), and the interior concentration of the
     * polysilicon grains (R4).
     */
    for (j = 0; j < numLay; ++j) {
        for (i = 0; i < numImp; ++i) {
            float rdata;
            if (fscanf(fpSuprem, "%e%e", &rdata, &rdata) != 2) {
                (void) fprintf(cp_err, "Unable to read integrated dopant "
                        "and interior concentration of layer %d and "
                        "impurity %d from file \"%s\".\n",
                        j + 1, i + 1, inFile);
                xrc = -1;
                goto EXITPOINT;
            }
        }
    }
    DEBUG fprintf(stderr,"set 4:\n" );

    /*
     * The fifth set of lines contains, for each node in the structure,
     * the distance to the next deepest node (R4), the distance from the
     * surface (R4), and, for each impurity type, the impurity's
     * chemical concentration (R4) and the impurity's active concentration (R4).
     */
    for (i = 1; i <= numGrid; ++i) {
        float rdata;
        if (fscanf(fpSuprem, "%e %e", &rdata, &x[i]) != 2) {
            (void) fprintf(cp_err, "Unable to read grid %d "
                    "from file \"%s\".\n",
                    i + 1, inFile);
            xrc = -1;
            goto EXITPOINT;
        }

        for (j = 0; j < numImp; j++) {
            float junk;
            /* chemical concentration - not required */
            if (fscanf(fpSuprem, "%e", &junk) != 1) {
                (void) fprintf(cp_err, "Unable to chemical concentration "
                        "%d of layer %d "
                        "from file \"%s\".\n",
                        j + 1, i + 1, inFile);
                xrc = -1;
                goto EXITPOINT;
            }

            /* store active concentration */
            if (fscanf(fpSuprem, "%e", &con[i]) != 1) {
                (void) fprintf(cp_err, "Unable to active concentration "
                        "%d of layer %d "
                        "from file \"%s\".\n",
                        j + 1, i + 1, inFile);
                xrc = -1;
                goto EXITPOINT;
            }

            /* orient sign properly */
            if (impTyp[j] == *impId) {
                /*...Boron...*/
                if (impTyp[j] == 1) {
                    conc[i] = - con[i];
                }
                else {
                    /*...All Other Impurities: P, As, Sb ...*/
                    conc[i] = con[i];
                }
            }
        }
    } /* end of loop over num grid */
    DEBUG fprintf( stderr, "set 5: %e %e\n", x[1], conc[1] );

    /*
     * The last line in the file contains some random stuff that might be
     * useful to some people, the temperature in degrees Kelvin of the last
     * diffusion step (R4), the phosphorus implant dose (R4), the arsenic
     * implant flag (L4).  However, we can just ignore that stuff.
     */
    if (fclose(fpSuprem) != 0) {
        (void) fprintf(cp_err, "Unable to close file \"%s\": %s.\n",
                inFile, strerror(errno));
        xrc = -1;
        goto EXITPOINT;
    }
    fpSuprem = (FILE *) NULL;

    /* shift silicon layer to beginning of array */
    for (j = numLay; --j >= 0; ) {
        if (matTyp[j] == 1) {
            break;
        }
    }

    if (j < 0) {
        (void) fprintf(cp_err, "internal error in %s!\n", __FUNCTION__);
        xrc = -1;
        goto EXITPOINT;
    }

    siIndex = j;

    offset = topNod[siIndex] - 1;
    numGrid -= offset;
    xStart = x[1 + offset];
    for (i = 1; i <= numGrid; i++) {
        x[i] = x[i + offset] - xStart;
        conc[i] = conc[i + offset];
    }

    /* Store number of valid nodes using pointer */
    *numNod = numGrid;

EXITPOINT:
    if (fpSuprem != (FILE *) NULL) {
        if (fclose(fpSuprem) != 0) {
            (void) fprintf(cp_err, "Unable to close \"%s\" at exit: %s\n",
                    inFile, strerror(errno));
            xrc = -1;
        }
    }

    return xrc;
} /* end of function SUPascRead */



