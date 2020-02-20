#include "ngspice/ngspice.h"
#include "ngspice/mif.h"
#include "ngspice/cm.h"
#include "ngspice/cpextern.h"
#include "ngspice/dllitf.h"
#include "ngspice/alloc.h"

#ifdef HAVE_LIBGC
static void no_free(const void *p) {
  NG_IGNORE(p);
}
#endif

/* Returns the version string for ngspice */
static const char *get_ngspice_version(void)
{
    const char *buf = VERSION;
    return buf;
} /* end of function get_ngspice_version */


/* Returns stdout, stdin, and stderr */
static FILE *get_stdout(void)
{
    return cp_out;
}
static FILE *get_stdin(void)
{
    return cp_in;
}
static FILE *get_stderr(void)
{
    return cp_err;
}



const struct coreInfo_t coreInfo =
{
  MIF_INP2A,
  MIFgetMod,
  MIFgetValue,
  MIFsetup,
  MIFunsetup,
  MIFload,
  MIFmParam,
  MIFask,
  MIFmAsk,
  MIFtrunc,
  MIFconvTest,
  MIFdelete,
  MIFmDelete,
  MIFdestroy,
  MIFgettok,
  MIFget_token,
  MIFget_cntl_src_type,
  MIFcopy,
  cm_climit_fcn,
  cm_smooth_corner,
  cm_smooth_discontinuity,
  cm_smooth_pwl,
  cm_analog_ramp_factor,
  cm_analog_alloc,
  cm_analog_get_ptr,
  cm_analog_integrate,
  cm_analog_converge,
  cm_analog_set_temp_bkpt,
  cm_analog_set_perm_bkpt,
  cm_analog_not_converged,
  cm_analog_auto_partial,
  cm_event_alloc,
  cm_event_get_ptr,
  cm_event_queue,
  cm_message_get_errmsg,
  cm_message_send,
  cm_netlist_get_c,
  cm_netlist_get_l,
  cm_complex_set,
  cm_complex_add,
  cm_complex_subtract,
  cm_complex_multiply,
  cm_complex_divide,
  cm_get_path,
  cm_get_circuit,
    &get_stdout,
    &get_stdin,
    &get_stderr,
    &tmalloc,
    &tcalloc,
    &trealloc,
    &txfree,
    &tmalloc,
    &trealloc,
    &txfree,
};
