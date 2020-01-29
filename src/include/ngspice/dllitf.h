/* 
   DLL load interface
   (c)2000 Arpad Buermen
*/

#ifndef ngspice_DLLITF_H
#define ngspice_DLLITF_H

#include "ngspice/mifproto.h"
#include "ngspice/cmproto.h"


/* This structure contains pointers to core SPICE OPUS functions used in CMs and UDNs.
   A pointer to this structure is passed to the dll when the dll is loaded. */

struct coreInfo_t {
	/* MIF stuff */
	void      ((*dllitf_MIF_INP2A)(CKTcircuit *, INPtables *, struct card *));
    char *    ((*dllitf_MIFgetMod)(CKTcircuit *, const char *, INPmodel  **,
            INPtables *));
	IFvalue * ((*dllitf_MIFgetValue)(CKTcircuit *, char **, int, INPtables *, char **));
	int		  ((*dllitf_MIFsetup)(SMPmatrix *, GENmodel *, CKTcircuit *, int *));
	int		  ((*dllitf_MIFunsetup)(GENmodel *, CKTcircuit *));
	int       ((*dllitf_MIFload)(GENmodel *, CKTcircuit *));
	int       ((*dllitf_MIFmParam)(int, IFvalue *, GENmodel *));
	int       ((*dllitf_MIFask)(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *));
	int       ((*dllitf_MIFmAsk)(CKTcircuit *, GENmodel *, int, IFvalue *));
	int       ((*dllitf_MIFtrunc)(GENmodel *, CKTcircuit *, double *));
	int       ((*dllitf_MIFconvTest)(GENmodel *, CKTcircuit *));
	int       ((*dllitf_MIFdelete)(GENinstance *));
	int       ((*dllitf_MIFmDelete)(GENmodel *));
	void      ((*dllitf_MIFdestroy)(void));
	char *    ((*dllitf_MIFgettok)(char **));
	char *    ((*dllitf_MIFget_token)(char **, Mif_Token_Type_t *));
	Mif_Cntl_Src_Type_t ((*dllitf_MIFget_cntl_src_type)(Mif_Port_Type_t, Mif_Port_Type_t));
	char *    ((*dllitf_MIFcopy)(const char *));
	/* CM stuff */
	void      ((*dllitf_cm_climit_fcn)(double, double, double, double, double, double, 
		                               double, double, int, double *, double *, double *, 
								       double *));
	void      ((*dllitf_cm_smooth_corner)(double, double, double, double, double, double,
	                                      double *, double *));
	void      ((*dllitf_cm_smooth_discontinuity)(double, double, double, double, double, 
		                                         double *, double *));
	double    ((*dllitf_cm_smooth_pwl)(double, double *, double *, int, double, double *));
	double    ((*dllitf_cm_analog_ramp_factor)(void));
	void      ((*dllitf_cm_analog_alloc)(int, int));
	void *    ((*dllitf_cm_analog_get_ptr)(int, int));
	int       ((*dllitf_cm_analog_integrate)(double, double *, double *));
	int       ((*dllitf_cm_analog_converge)(double *));
    int       ((*dllitf_cm_analog_set_temp_bkpt)(double));
    int       ((*dllitf_cm_analog_set_perm_bkpt)(double));
    void      ((*dllitf_cm_analog_not_converged)(void));
    void      ((*dllitf_cm_analog_auto_partial)(void));
	void      ((*dllitf_cm_event_alloc)(int, int));
	void *    ((*dllitf_cm_event_get_ptr)(int, int));
	int       ((*dllitf_cm_event_queue)(double));
	char *    ((*dllitf_cm_message_get_errmsg)(void));
	int       ((*dllitf_cm_message_send)(const char *));
	double    ((*dllitf_cm_netlist_get_c)(void));
	double    ((*dllitf_cm_netlist_get_l)(void));
	Complex_t ((*dllitf_cm_complex_set)(double, double));
	Complex_t ((*dllitf_cm_complex_add)(Complex_t, Complex_t));
	Complex_t ((*dllitf_cm_complex_subtract)(Complex_t, Complex_t));
	Complex_t ((*dllitf_cm_complex_multiply)(Complex_t, Complex_t));
	Complex_t ((*dllitf_cm_complex_divide)(Complex_t, Complex_t));
	char *    ((*dllitf_cm_get_path)(void));
	CKTcircuit *((*dllitf_cm_get_circuit)(void));
	FILE *    ((*dllitf_cm_stream_out)(void));
	FILE *    ((*dllitf_cm_stream_in)(void));
	FILE *    ((*dllitf_cm_stream_err)(void));
  /*Other stuff*/
	void *    ((*dllitf_malloc_pj)(size_t));
	void *    ((*dllitf_calloc_pj)(size_t, size_t));
	void *    ((*dllitf_realloc_pj)(void *, size_t));
	void      ((*dllitf_free_pj)(void *));
	void *    ((*dllitf_tmalloc)(size_t));
	void *    ((*dllitf_trealloc)(void *, size_t));
	void      ((*dllitf_txfree)(void *));

    /***   VERSION 2 ADDITIONS   ***/
    const char *(*dllitf_ngspice_version)(void);
                                 /* version string of ngspice using the cm */
    void *(*dllitf_tmalloc_raw)(size_t size);
                                 /* mutex-protected malloc() that will not
                                  * cause program termination on failure */
    void *(*dllitf_tcalloc_raw)(size_t num, size_t size);
                                 /* mutex-protected calloc() that will not
                                  * cause program termination on failure */
    void *(*dllitf_trealloc_raw)(void *p, size_t size);
                                 /* mutex-protected realloc() that will not
                                  * cause program termination on failure */
    char *(*dllitf_tstrdup)(const char *sz);
                                 /* mutex-protected strdup() that WILL
                                  * cause program termination on failure */
    char *(*dllitf_tstrdup_raw)(const char *sz);
                                 /* mutex-protected strdup() that will not
                                  * cause program termination on failure */
};

extern const struct coreInfo_t  coreInfo;

#endif /* include guard */
