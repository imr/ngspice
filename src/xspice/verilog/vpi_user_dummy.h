/* A minimal extract from the Verilog Standard's vpi_user.h, so that
 * Icarus Verilog support for ngspice can be compiled
 * without an installed copy.
 */

typedef char     PLI_BYTE8;
typedef uint32_t PLI_UINT32;
typedef int32_t  PLI_INT32;

struct t_vpi_time {
    PLI_INT32  type;
    PLI_UINT32 high;
    PLI_UINT32 low;
    double     real;
};

#define vpiScaledRealTime 1
#define vpiSimTime        2
#define vpiSuppressTime   3

struct t_vpi_vecval {
    PLI_INT32 aval, bval;
};

struct t_vpi_value {
    PLI_INT32 format;
    union {
        char                     *str;
        PLI_INT32                 scalar;
        PLI_INT32                 integer;
        double                    real;
        struct t_vpi_time        *time;
        struct t_vpi_vecval      *vector;
        struct t_vpi_strengthval *strength;
        char                     *misc;
    }         value;
};

#define vpiIntVal    6
#define vpiVectorVal 9

typedef struct __vpiHandle *vpiHandle;

struct t_cb_data {
    PLI_INT32           reason;
    PLI_INT32         (*cb_rtn)(struct t_cb_data *);
    vpiHandle           obj;
    struct t_vpi_time  *time;
    struct t_vpi_value *value;
    PLI_INT32           index;
    const PLI_BYTE8    *user_data;
};

#define cbValueChange        1
#define cbReadWriteSynch     6
#define cbReadOnlySynch      7
#define cbNextSimTime        8
#define cbAfterDelay         9
#define cbStartOfSimulation 11

struct t_vpi_vlog_info
{
    PLI_INT32   argc;
    char      **argv;
    char       *product;
    char       *version;
};

extern PLI_INT32 vpi_printf(const char *, ...);
extern PLI_INT32 vpi_get_vlog_info(struct t_vpi_vlog_info *);
extern void      vpi_get_time(vpiHandle, struct t_vpi_time *);
extern vpiHandle vpi_register_cb(struct t_cb_data *);
extern PLI_INT32 vpi_remove_cb(vpiHandle);
extern PLI_INT32 vpi_free_object(vpiHandle);

#define vpiNoDelay 1
extern vpiHandle vpi_put_value(vpiHandle, struct t_vpi_value *,
			       struct t_vpi_time *, PLI_INT32);

#define vpiType           1
#define vpiName           2
#define vpiSize           4
#define vpiTimeUnit      11
#define vpiTimePrecision 12
#define vpiDirection     20

#define vpiInput  1
#define vpiOutput 2
#define vpiInout  3

#define vpiModule 32
#define vpiPort   44

extern char      *vpi_get_str(PLI_INT32, vpiHandle);
extern PLI_INT32  vpi_get(int, vpiHandle);
extern vpiHandle  vpi_iterate(PLI_INT32, vpiHandle);
extern vpiHandle  vpi_scan(vpiHandle);
extern vpiHandle  vpi_handle_by_name(const char *, vpiHandle);

#define vpiFinish 67

extern void vpi_control(PLI_INT32 operation, ...);
