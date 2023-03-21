/* ----------------------------------------------------------------------
 Build cmextrn.h, cminfo.h, udnextrn.h and udninfo.h from udnpath.lst and
 modpath.lst using 'cmpp -lst'. Then compile this file and link it with
 cm and udn object files to produce a dll that can be loaded by ngspice
 at run-time.
 Copyright 2000 The ngspice team
 3 - Clause BSD license
 (see COPYING or https://opensource.org/licenses/BSD-3-Clause)
 Author: Arpad Buermen
------------------------------------------------------------------------ */

#include  <stdarg.h>
#include  <stdlib.h>
#include  <string.h>

#include "ngspice/cpextern.h"
#include "ngspice/devdefs.h"
#include "ngspice/dstring.h"
#include "ngspice/dllitf.h"
#include "ngspice/evtudn.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inertial.h"
#include "cmextrn.h"
#include "udnextrn.h"



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

/* Instantiation of pointer to core info structure containing pointers
 * to core functions. */
struct coreInfo_t *coreitf = (struct coreInfo_t *) NULL;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// Functions that return pointers to structures.
//////////////////////////////////////////////////////////////////////////////

#if defined (__MINGW32__) || defined (__CYGWIN__) || defined (_MSC_VER)
#define CM_EXPORT __declspec(dllexport)
#else
  /* use with gcc flag -fvisibility=hidden */
  #if __GNUC__ >= 4
    #define CM_EXPORT __attribute__ ((visibility ("default")))
    #define CM_EXPORT_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define CM_EXPORT
    #define CM_EXPORT_LOCAL
  #endif
#endif

extern CM_EXPORT void *CMdevs(void);
extern CM_EXPORT void *CMdevNum(void);
extern CM_EXPORT void *CMudns(void);
extern CM_EXPORT void *CMudnNum(void);
extern CM_EXPORT void *CMgetCoreItfPtr(void);


// This one returns the device table
CM_EXPORT void *CMdevs(void) {
	return (void *)cmDEVices;
}

// This one returns the device count
CM_EXPORT void *CMdevNum(void) {
	return (void *)&cmDEVicesCNT;
}

// This one returns the UDN table
CM_EXPORT void *CMudns(void) {
	return (void *)cmEVTudns;
}

// This one returns the UDN count
CM_EXPORT void *CMudnNum(void) {
	return (void *)&cmEVTudnCNT;
}

// This one returns the pointer to the pointer to the core interface structure
CM_EXPORT void *CMgetCoreItfPtr(void) {
	return (void *)(&coreitf);
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// These functions call the real core functions of SPICE OPUS using the
// pointers in coreitf structure.
//////////////////////////////////////////////////////////////////////////////
void MIF_INP2A(
    CKTcircuit   *ckt,      /* circuit structure to put mod/inst structs in */
    INPtables    *tab,      /* symbol table for node names, etc.            */
    struct card  *current   /* the card we are to parse                     */
	) {
	(coreitf->dllitf_MIF_INP2A)(ckt,tab,current);
}

char * MIFgetMod(
    CKTcircuit *ckt,
    char      *name,
    INPmodel  **model,
    INPtables *tab 
	) {
	return (coreitf->dllitf_MIFgetMod)(ckt,name,model,tab);
}

IFvalue * MIFgetValue(
    CKTcircuit *ckt,
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
    GENinstance  *inst
	) {
	return (coreitf->dllitf_MIFdelete)(inst);
}

int MIFmDelete(
    GENmodel *gen_model
	) {
	return (coreitf->dllitf_MIFmDelete)(gen_model);
}

void MIFdestroy(
    void
	) {
	(coreitf->dllitf_MIFdestroy)();
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

void cm_analog_alloc(int tag, int bytes) {
	(coreitf->dllitf_cm_analog_alloc)(tag,bytes);
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

void cm_event_alloc(int tag, int bytes){
	(coreitf->dllitf_cm_event_alloc)(tag,bytes);
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

const char *cm_get_node_name(const char *port, unsigned int index) {
    return coreitf->dllitf_cm_get_node_name(port, index);
}

bool cm_probe_node(unsigned int  conn_index,
                   unsigned int  port_index,
                   void         *value) {
    return coreitf->dllitf_cm_probe_node(conn_index, port_index, value);
}

bool cm_schedule_output(unsigned int conn_index, unsigned int port_index,
                        double delay, void *vp)
{
    return (coreitf->dllitf_cm_schedule_output)(conn_index, port_index,
                                                delay, vp);
}

bool cm_getvar(char *name, enum cp_types type, void *retval, size_t rsize)
{
    return (coreitf->dllitf_cm_getvar)(name, type, retval, rsize);
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

char * cm_get_path(void) {
	return (coreitf->dllitf_cm_get_path)();
}

CKTcircuit *cm_get_circuit(void) {
	return (coreitf->dllitf_cm_get_circuit)();
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

void * malloc_pj(size_t s) {
	return (coreitf->dllitf_malloc_pj)(s);
}

void * calloc_pj(size_t s1, size_t s2) {
	return (coreitf->dllitf_calloc_pj)(s1,s2);
}

void * realloc_pj(const void *ptr, size_t s) {
	return (coreitf->dllitf_realloc_pj)(ptr,s);
}

void free_pj(const void *ptr) {
	(coreitf->dllitf_free_pj)(ptr);
}

void * tmalloc(size_t s) {
	return (coreitf->dllitf_tmalloc)(s);
}

void * trealloc(const void *ptr, size_t s) {
	return (coreitf->dllitf_trealloc)(ptr,s);
}

void txfree(const void *ptr) {
	(coreitf->dllitf_txfree)(ptr);
}


/*
fopen_with_path()
Opens an input file <infile>. Called from d_state, file_source, d_source.
Firstly retrieves the path Infile_Path of the ngspice input netlist.
Then searches for (and opens) <infile> an a sequence from
Infile_Path/<infile>
NGSPICE_INPUT_DIR/<infile>, where the path is given by the environmental variable
<infile>, where the path is the current directory
*/
#define DFLT_BUF_SIZE   256
FILE *fopen_with_path(const char *path, const char *mode)
{
    FILE *fp;

    if((path[0] != '/') && (path[1] != ':')) { /* path absolue (probably) */
//        const char *x = getenv("ngspice_vpath");
        const char *x = cm_get_path();
        if (x) {
            DS_CREATE(ds, DFLT_BUF_SIZE);

            /* Build file <cm_get_path(()>/path> */
            if (ds_cat_printf(&ds, "%s/%s", x, path) != 0) {
                cm_message_printf(
                        "Unable to build cm_get_path() path for opening file.");
                ds_free(&ds);
                return (FILE *) NULL;
            }

            /* Try opening file. If fail, try using NGSPICE_INPUT_DIR
             * env variable location */
            if ((fp = fopen(ds_get_buf(&ds), mode)) == (FILE *) NULL) {
                char *y = getenv("NGSPICE_INPUT_DIR");
                if (y && *y) { /* have env var and not "" */
                    int rc_ds = 0;
                    /* Build <env var>/path and try opening. If the env var
                     * ends with a slash, do not add a second slash */
                    ds_clear(&ds);
                    rc_ds |= ds_cat_str(&ds, y);

                    /* Add slash if not present. Note that check for
                     * length > 0 is done on the remote chance that the
                     * ds_cat_str() failed. */
                    const size_t len = ds_get_length(&ds);
                    if (len > 0 && ds_get_buf(&ds)[len - 1] != '/') {
                        rc_ds |= ds_cat_char(&ds, '/'); /* add dir sep */
                    }
                    rc_ds |= ds_cat_str(&ds, path); /* add input path */

                    /* Ensure path built OK */
                    if (rc_ds != 0) {
                        cm_message_printf(
                                "Unable to build NGSPICE_INPUT_DIR "
                                "path for opening file.");
                        ds_free(&ds);
                        return (FILE *) NULL;
                    }

                    /* Try opening file name that was built */
                    if ((fp = fopen(ds_get_buf(&ds),
                            mode)) != (FILE *) NULL) {
                        ds_free(&ds);
                        return fp;
                    }
                }
            } /* end of open using prefix from cm_get_path() failed */
            else { /* Opened OK */
                ds_free(&ds);
                return fp;
            }
            ds_free(&ds); /* free dstring resources, if any */
        }
    } /* end of case that path is not absolute */

    /* If not opened yet, try opening exactly as given */
    fp =  fopen(path, mode);

    return fp;
} /* end of function fopen_with_path */



int
cm_message_printf(const char *fmt, ...)
{
    char buf[1024];
    char *p = buf;
    int size = sizeof(buf);
    int rv;

    for (;;) {

        int nchars;
        va_list ap;

        va_start(ap, fmt);
        nchars = vsnprintf(p, (size_t) size, fmt, ap);
        va_end(ap);

        if (nchars == -1) {     // compatibility to old implementations
            size *= 2;
        } else if (size < nchars + 1) {
            size = nchars + 1;
        } else {
            break;
        }

        if (p == buf)
            p = tmalloc((size_t) size * sizeof(char));
        else
            p = trealloc(p, (size_t) size * sizeof(char));
    }

    rv = cm_message_send(p);
    if (p != buf)
        txfree(p);
    return rv;
}

/* Function used for inertial delay in digital logic models. */

Mif_Boolean_t cm_is_inertial(enum param_vals param)
{
    int cvar;

    /* Get the value of the control variable. */

    if (cm_getvar("digital_delay_type", CP_NUM, &cvar, sizeof cvar)) {
        if (cvar >= OVERRIDE_TRANSPORT) {
            /* Parameter override. */

            return cvar > OVERRIDE_TRANSPORT;
        }
        if (param == Not_set) // Not specified
            return cvar != DEFAULT_TRANSPORT;
        return param != Off;
    }
    return param == On;
}
