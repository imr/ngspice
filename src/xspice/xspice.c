#include "ngspice/ngspice.h"
#include "ngspice/mif.h"
#include "ngspice/cm.h"
#include "ngspice/dllitf.h"

/*how annoying!, needed for structure below*/
static void *tcalloc(size_t a, size_t b) {
  return tmalloc(a*b);          /* FIXME, tcalloc must zero !?!? */
}

static void no_free(const void *p) {
  NG_IGNORE(p);
}

static FILE * no_file(void) {
  return NULL;
}

struct coreInfo_t  coreInfo =
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
  no_file,
  no_file,
  no_file,
#ifndef HAVE_LIBGC
  tmalloc,
  tcalloc,
  trealloc,
  txfree,
  tmalloc,
  trealloc,
  txfree
#else
  GC_malloc,
  tcalloc,
  GC_realloc,
  no_free,
  GC_malloc,
  GC_realloc,
  no_free
#endif
};
