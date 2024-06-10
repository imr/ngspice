/* Dummy implementations of the functions defined in vpi_user_dummy.h,
 * used to make a DLL to replace libvvp.DLL when linking, so that
 * Icarus Verilog need not be installed when linking components for it.
 */

#include <stdint.h>
#include "vpi_user_dummy.h"

PLI_INT32 vpi_printf(const char *, ...) {return 0;}
PLI_INT32 vpi_get_vlog_info(struct t_vpi_vlog_info *p) {return 0;}
void      vpi_get_time(vpiHandle h, struct t_vpi_time *p) {}
vpiHandle vpi_register_cb(struct t_cb_data *p) {return (void *)0;}
PLI_INT32 vpi_remove_cb(vpiHandle h) {return 0;}
PLI_INT32 vpi_free_object(vpiHandle h) {return 0;}

vpiHandle vpi_put_value(vpiHandle h, struct t_vpi_value *p1,
                        struct t_vpi_time *p2, PLI_INT32 i) {return (void *)0;}
char      *vpi_get_str(PLI_INT32 i, vpiHandle h) {return (char *)0;}
PLI_INT32  vpi_get(int, vpiHandle)  {return 0;}

vpiHandle  vpi_iterate(PLI_INT32, vpiHandle) {return (void *)0;}
vpiHandle  vpi_scan(vpiHandle) {return (void *)0;}
vpiHandle  vpi_handle_by_name(const char *, vpiHandle) {return (void *)0;}
void vpi_control(PLI_INT32 operation, ...) {}
