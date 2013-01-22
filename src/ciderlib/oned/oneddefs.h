/*
 * 2001 Paolo Nenzi
 */
 
#ifndef ngspice_ONEDDEFS_H
#define ngspice_ONEDDEFS_H

/* Debug statements */

extern BOOLEAN ONEacDebug;
extern BOOLEAN ONEdcDebug;
extern BOOLEAN ONEtranDebug;
extern BOOLEAN ONEjacDebug;

/* Now some defines for the one dimensional simulator
 * library.
 * Theese defines were gathered from all the code in
 * oned directory.
 */
 
#define LEVEL_ALPHA_SI 3.1e-8	/* From de Graaf & Klaasen, pg. 12 */
#define MIN_DELV 1e-3
#define NORM_RED_MAXITERS 10





#endif
