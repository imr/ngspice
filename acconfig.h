
/* Define the location of NGSPICEDATADIR */ 
#define NGSPICEDATADIR none

/* Define the location of NGSPICEBINDIR */ 
#define NGSPICEBINDIR none

/* Define the build date */ 
#define NGSPICEBUILDDATE none

/* Define if we want garbage collection enabled */ 
#undef HAVE_LIBGC

/* Define if we have termcap */ 
#undef HAVE_TERMCAP

/* Define if we want NOBYPASS */ 
#undef NOBYPASS

/* Define if we want to bypass cbd/cbs calculation for non varying vbs/vbd */
#undef CAPBYPASS
 
/* Define if we want to bypass cbd/cbs calculation if Czero is zero */
#undef CAPZEROBYPASS 
 
/* Experimental code never implemented to damp Newton iterations */
#undef NODELIMITING

/* Define if we want predictor algorithm */ 
#undef PREDICTOR

/* Define if we want spice2 sensitivity analysis */ 
#undef WANT_SENSE2

/* Define if we want some experimental code */ 
#undef EXPERIMENTAL_CODE

/* Define if we want to enable experimental devices */
#undef EXP_DEV

/* Define if we want noise integration code */
#undef INT_NOISE

/* Undefine HAVE_EKV since it is not included in the standard distribution */
#undef HAVE_EKV

/* Define if we have GNU readline */   
#undef HAVE_GNUREADLINE

/* We do not want spurios debug info into non-developer code */
#undef FTEDEBUG

/* Do not trigger unwanted traps by default */
#undef NEWTRUNC
  
/* Define if we wanto debug sensititvity analysis */
#undef SENSDEBUG

/* Define i we want stepdebug */
#undef STEPDEBUG
  
/* Define to use always exp/log for bulk diode calculations in mosfet */
#undef NOSQRT

/*The xspice enhancements*/
#undef XSPICE

/* The CIDER enhancements */
#undef CIDER
 
/* Spice cluster support */
#undef CLUSTER

/* Generate MS WINDOWS executable */
#undef HAS_WINDOWS

/* get system memory and time */
#undef HAVE__MEMAVL
