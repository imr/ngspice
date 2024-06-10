#ifndef _ICARUS_SHIM_H_
#define _ICARUS_SHIM_H_

/* This is the interface definition file for the shim library (ivlng.so)
 * and associated Verilog VPI module (icarus_shim.vpi).
 *
 * Together, the two libraries allow execution of Verilog code using the
 * shared library version of Icarus Verilog's VVP program inside a SPICE
 * simulation performed by ngspice.
 *
 * Use of these components starts with a SPICE netlist containing an A-device
 * (XSPICE device) whose model card specifies the d_cosim code model and
 * parameter 'simulation="some_path/ivlng.so"'.  During initialisation,
 * the shim library finds the VVP file to be run as a model card parameter,
 * loads libvvp.so, and creates a thread to run it, passing the path
 * to the VVP file and options to load the VPI module.
 *
 * The VPI module is called on loading: its first task is to obtain the
 * list of ports for the top-level Verilog module. After that the VPI code
 * controls the execution of the Verilog code by blocking execution until
 * commanded to proceed by the d_cosim instance, always regaining control via
 * a VPI callback before the Verilog code moves ahead of the SPICE simulation.
 */

/* Structure holding pointers to functions in libvvp.so, the Icarus runtime. */

struct vvp_ptrs {
    void (*add_module_path)(const char *path);
#define VVP_FN_0 "vpip_add_module_path"
    void (*init)(const char *logfile_name, int argc, char*argv[]);
#define VVP_FN_1 "vvp_init"
    void (*no_signals)(void);
#define VVP_FN_2 "vvp_no_signals"
    void (*load_module)(const char *name);
#define VVP_FN_3 "vpip_load_module"
    int (*run)(const char *design_path);
#define VVP_FN_4 "vvp_run"
};

/* Data stored for each port. */

struct ngvp_port {
    uint16_t            bits;           // How many bits?
    uint16_t            flags;          // I/O pending.
    uint32_t            position;       // Number of bits before this port.
    struct {                            // Like struct t_vpi_vecval.
        int32_t           aval;
        int32_t           bval;
    }                   previous, new;  // Previous and new values.
    struct __vpiHandle *handle;         // Handle to the port's variable.
    struct ng_vvp      *ctx;            // Pointer back to parent.
};

#define IN_PENDING  1
#define OUT_PENDING 2

/* Data strucure used to share context between the ngspice and VVP threads. */

struct ng_vvp {
    struct cr_ctx       cr_ctx;         // Coroutine context.
    int                 stop;           // Indicates simulation is over.
    struct co_info     *cosim_context;
    uint32_t            ins;            // Port counts by type.
    uint32_t            outs;
    uint32_t            inouts;
    double              base_time;      // SPICE time on entry.
    double              tick_length;    // VVP's time unit.
    struct __vpiHandle *stop_cb;        // Handle to end-of-tick callback.
    volatile uint32_t   in_pending;     // Counts of changed ports.
    volatile uint32_t   out_pending;
    struct ngvp_port   *ports;          // Port information array.
    void               *vvp_handle;     // dlopen() handle for libvvp.
};

/* Function to find the current d_cosim instance data. Called by VPI code
 * and valid only during initialisation: while ngspice thread is waiting.
 */

struct ng_vvp *Get_ng_vvp(void);

#endif // _ICARUS_SHIM_H_
