//////////////////////////////////////////////////////////////////////////////
// Build cmextrn.h, cminfo.h, udnextrn.h and udninfo.h from udnpath.lst and
// modpath.lst using 'cmpp -lst'. Then compile this file and link it with
// cm and udn object files to produce a dll that can be loaded by the
// spice opus simulator at run-time.
//
// Author: Arpad Buermen
////////////////////////////////////////////////////////////////////////////// 
#include "dlinfo.h"

#include <devdefs.h>
#include <evtudn.h>
#include "port.h"
#include "misc.h"
#include "cmextrn.h"
#include "udnextrn.h"
#include "dllitf.h"

// This one is automatically set by the compiler
char inf_Date[]=__DATE__;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// Do not modify anything below this line
//////////////////////////////////////////////////////////////////////////////

SPICEdev *cmDEVices[] = {
#include "cminfo.h"
	NULL
};

int cmDEVicesCNT = sizeof(cmDEVices)/sizeof(SPICEdev *)-1;

Evt_Udn_Info_t  *cmEVTudns[] = {
#include "udninfo.h"
	NULL
};

int cmEVTudnCNT = sizeof(cmEVTudns)/sizeof(Evt_Udn_Info_t *)-1;

// Pointer to core info structure containing pointers to core functions.
struct coreInfo_t *coreitf;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// Functions that return pointers to structures.
//////////////////////////////////////////////////////////////////////////////
#ifdef CM_WINDOWS
#undef CM_EXPORT
#define CM_EXPORT __declspec(dllexport)
#endif

#ifdef CM_LINUX
#undef CM_EXPORT
#define CM_EXPORT extern
#endif

int cmopusvers = 2;

// This one checks the opus dynamic link version.
CM_EXPORT void *CMsysvers() {
	return (void *)&cmopusvers;
}

// This one returns the title
CM_EXPORT void *CMtitle() {
	return (void *)inf_Title;
}

// This one returns the version
CM_EXPORT void *CMversion() {
	return (void *)inf_Version;
}

// This one returns the date
CM_EXPORT void *CMdate() {
	return (void *)inf_Date;
}

// This one returns the description
CM_EXPORT void *CMdescription() {
	return (void *)inf_Description;
}

// This one returns the author
CM_EXPORT void *CMauthor() {
	return (void *)inf_Author;
}

// This one returns the copyright
CM_EXPORT void *CMcopyright() {
	return (void *)inf_Copyright;
}

// This one returns the device table
CM_EXPORT void *CMdevs() {
	return (void *)cmDEVices;
}

// This one returns the device count
CM_EXPORT void *CMdevNum() {
	return (void *)&cmDEVicesCNT;
}

// This one returns the UDN table
CM_EXPORT void *CMudns() {
	return (void *)cmEVTudns;
}

// This one returns the UDN count
CM_EXPORT void *CMudnNum() {
	return (void *)&cmEVTudnCNT;
}

// This one returns the pointer to the pointer to the core interface structure
CM_EXPORT void *CMgetCoreItfPtr() {
	return (void *)(&coreitf);
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// These functions call the real core functions of SPICE OPUS using the
// pointers in coreitf structure.
//////////////////////////////////////////////////////////////////////////////
void MIF_INP2A(
    GENERIC      *ckt,      /* circuit structure to put mod/inst structs in */
    INPtables    *tab,      /* symbol table for node names, etc.            */
    card         *current   /* the card we are to parse                     */
	) {
	(coreitf->dllitf_MIF_INP2A)(ckt,tab,current);
}

char * MIFgetMod(
    GENERIC   *ckt,
    char      **name,
    INPmodel  **model,
    INPtables *tab 
	) {
	return (coreitf->dllitf_MIFgetMod)(ckt,name,model,tab);
}

IFvalue * MIFgetValue(
    GENERIC   *ckt,
    char      **line,
    int       type,
    INPtables *tab,
    char      **err 
	) {
	return (coreitf->dllitf_MIFgetValue)(ckt,line,type,tab,err);
}


int MIFsetup(
    SMPmatrix     *matrix,
    GENmodel      *inModel,
    CKTcircuit    *ckt,
    int           *state 
	) {
	return (coreitf->dllitf_MIFsetup)(matrix,inModel,ckt,state);
}

int MIFunsetup(
    GENmodel      *inModel,
    CKTcircuit    *ckt
) {
        return (coreitf->dllitf_MIFunsetup)(inModel,ckt);
}

int MIFload(
    GENmodel      *inModel,
    CKTcircuit    *ckt 
	) {
	return (coreitf->dllitf_MIFload)(inModel,ckt);
}


int MIFmParam(
    int param_index,
    IFvalue *value,
    GENmodel *inModel 
	) {
	return (coreitf->dllitf_MIFmParam)(param_index,value,inModel);
}

int MIFask(
    CKTcircuit *ckt,
    GENinstance *inst,
    int param_index,
    IFvalue *value,
    IFvalue *select
	) {
	return (coreitf->dllitf_MIFask)(ckt,inst,param_index,value,select);
}

int MIFmAsk(
    CKTcircuit *ckt,
    GENmodel *inModel,
    int param_index,
    IFvalue *value
	) {
	return (coreitf->dllitf_MIFmAsk)(ckt,inModel,param_index,value);
}

int MIFtrunc(
    GENmodel   *inModel,
    CKTcircuit *ckt,
    double     *timeStep
	) {
	return (coreitf->dllitf_MIFtrunc)(inModel,ckt,timeStep);
}

int MIFconvTest(
    GENmodel   *inModel,
    CKTcircuit *ckt
	) {
	return (coreitf->dllitf_MIFconvTest)(inModel,ckt);
}

int MIFdelete(
    GENmodel *inModel,
    IFuid    name,
    GENinstance  **inst
	) {
	return (coreitf->dllitf_MIFdelete)(inModel,name,inst);
}

int MIFmDelete(
    GENmodel **inModel,
    IFuid    modname,
    GENmodel *model
	) {
	return (coreitf->dllitf_MIFmDelete)(inModel,modname,model);
}

void MIFdestroy(
    GENmodel **inModel
	) {
	(coreitf->dllitf_MIFdestroy)(inModel);
}

char  *MIFgettok(
    char **s
	) {
	return (coreitf->dllitf_MIFgettok)(s);
}


char  *MIFget_token(
    char **s,
    Mif_Token_Type_t *type
	) {
	return (coreitf->dllitf_MIFget_token)(s,type);
}


Mif_Cntl_Src_Type_t MIFget_cntl_src_type(
    Mif_Port_Type_t in_port_type,
    Mif_Port_Type_t out_port_type
	) {
	return (coreitf->dllitf_MIFget_cntl_src_type)(in_port_type,out_port_type);
}

char *MIFcopy(char *c) {
	return (coreitf->dllitf_MIFcopy)(c);
}


void cm_climit_fcn(double in, double in_offset, double cntl_upper, 
                   double cntl_lower, double lower_delta, 
                   double upper_delta, double limit_range, 
                   double gain, int percent, double *out_final,
                   double *pout_pin_final, double *pout_pcntl_lower_final,
                   double *pout_pcntl_upper_final) {
	(coreitf->dllitf_cm_climit_fcn)(in,in_offset,cntl_upper,cntl_lower,lower_delta,
		                            upper_delta,limit_range,gain,percent,out_final,
									pout_pin_final,pout_pcntl_lower_final,
									pout_pcntl_upper_final);
}



void cm_smooth_corner(double x_input, double x_center, double y_center,
                 double domain, double lower_slope, double upper_slope,
                 double *y_output, double *dy_dx) {
	(coreitf->dllitf_cm_smooth_corner)(x_input,x_center,y_center,domain,lower_slope,
		                               upper_slope,y_output,dy_dx);
}

void cm_smooth_discontinuity(double x_input, double x_lower, double y_lower,
                 double x_upper, double y_upper, 
                 double *y_output, double *dy_dx) {
	(coreitf->dllitf_cm_smooth_discontinuity)(x_input,x_lower,y_lower,x_upper,y_upper,
		                                      y_output,dy_dx);
}

double cm_smooth_pwl(double x_input, double *x, double *y, int size,
					 double input_domain, double *dout_din) {
	return (coreitf->dllitf_cm_smooth_pwl)(x_input,x,y,size,input_domain,dout_din);
}

double cm_analog_ramp_factor(void) {
	return (coreitf->dllitf_cm_analog_ramp_factor)();
}

void *cm_analog_alloc(int tag, int bytes) {
	return (coreitf->dllitf_cm_analog_alloc)(tag,bytes);
}

void *cm_analog_get_ptr(int tag, int timepoint) {
	return (coreitf->dllitf_cm_analog_get_ptr)(tag,timepoint);
}

int  cm_analog_integrate(double integrand, double *integral, double *partial) {
	return (coreitf->dllitf_cm_analog_integrate)(integrand,integral,partial);
}

int  cm_analog_converge(double *state) {
	return (coreitf->dllitf_cm_analog_converge)(state);
}

int  cm_analog_set_temp_bkpt(double time) {
	return (coreitf->dllitf_cm_analog_set_temp_bkpt)(time);
}

int  cm_analog_set_perm_bkpt(double time) {
	return (coreitf->dllitf_cm_analog_set_perm_bkpt)(time);
}

void cm_analog_not_converged(void) {
	(coreitf->dllitf_cm_analog_not_converged)();
}

void cm_analog_auto_partial(void) {
	(coreitf->dllitf_cm_analog_auto_partial)();
}

void *cm_event_alloc(int tag, int bytes){
	return (coreitf->dllitf_cm_event_alloc)(tag,bytes);
}

void *cm_event_get_ptr(int tag, int timepoint) {
	return (coreitf->dllitf_cm_event_get_ptr)(tag,timepoint);
}

int  cm_event_queue(double time) {
	return (coreitf->dllitf_cm_event_queue)(time);
}

char *cm_message_get_errmsg(void) {
	return (coreitf->dllitf_cm_message_get_errmsg)();
}

int  cm_message_send(char *msg) {
	return (coreitf->dllitf_cm_message_send)(msg);
}

double cm_netlist_get_c(void) {
	return (coreitf->dllitf_cm_netlist_get_c)();
}

double cm_netlist_get_l(void) {
	return (coreitf->dllitf_cm_netlist_get_l)();
}

Complex_t cm_complex_set(double real, double imag) {
	return (coreitf->dllitf_cm_complex_set)(real,imag);
}

Complex_t cm_complex_add(Complex_t x, Complex_t y) {
	return (coreitf->dllitf_cm_complex_add)(x,y);
}

Complex_t cm_complex_subtract(Complex_t x, Complex_t y) {
	return (coreitf->dllitf_cm_complex_subtract)(x,y);
}

Complex_t cm_complex_multiply(Complex_t x, Complex_t y) {
	return (coreitf->dllitf_cm_complex_multiply)(x,y);
}

Complex_t cm_complex_divide(Complex_t x, Complex_t y) {
	return (coreitf->dllitf_cm_complex_divide)(x,y);
}

FILE * cm_stream_out(void) {
	return (coreitf->dllitf_cm_stream_out)();
}

FILE * cm_stream_in(void) {
	return (coreitf->dllitf_cm_stream_in)();
}

FILE * cm_stream_err(void) {
	return (coreitf->dllitf_cm_stream_err)();
}

char * tmalloc_internal(size_t s, int clean, const char *f, int l, int sw) {
	return (coreitf->dllitf_tmalloc)(s,clean,f,l,sw);
}

char * trealloc_internal(char *ptr, size_t s, const char *f, int l, int sw) {
	return (coreitf->dllitf_trealloc)(ptr,s,f,l,sw);
}

void tfree_internal(char *ptr, const char *f, int l, int sw) {
	(coreitf->dllitf_tfree)(ptr,f,l,sw);
}
