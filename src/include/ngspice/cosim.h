/* Header file for the shim code between d_cosim and a co-simulator. */

#if __cplusplus
extern "C" {
#endif

/* A value of this type controls how the step() function is called.
 * The normal method is to call step() to advance to the time of a
 * queued input, then supply the input, then call step again to
 * advance to the next input time or the end of the current SPICE
 * timestep.  But Verilator does nothing without an input change,
 * so step() must be called after input.
 */

typedef enum {Normal, After_input} Cosim_method;

/* Structure used by Cosim_setup() to pass and return
 * co-simulation interface information.
 */

struct co_info {
    /* The co-simulator must set the number of ports in Cosim_setup(). */

    unsigned int    in_count;
    unsigned int    out_count;
    unsigned int    inout_count;

    /* The co-simulator may specify a function to be called just before
     * it is unloaded at the end of a simulation run.
     */

    void          (*cleanup)(struct co_info *);

    /* Function called by SPICE to advance the co-simulation.
     * A pointer to this structure is passed, so it has access to its handle
     * and the target simulation time, vtime.  The co-simulator should
     * pause the step when output is produced and update vtime.
     */

    void          (*step)(struct co_info *pinfo); // Advance simulation.

    /* Function called by SPICE to pass input to input and inout ports.
     * (Inouts after inputs.)
     * Called as:
     *    struct co_info info;
     *    (*in_fn)(&info, bit_number, &value);
     * Function provided by co-simulator.
     */

    void          (*in_fn)(struct co_info *, unsigned int, Digital_t *);

    /* Function called by co-simulator to report output on
     * output and inout ports. (Inouts after outputs.)
     * Called as:
     *    struct co_info *p_info;
     *    (*out_fn)(p_info, bit_number, &value);
     * It will usually be called inside a call to step().
     */

    void          (*out_fn)(struct co_info *, unsigned int, Digital_t *);
    void           *handle;        // Co-simulator's private handle
    double          vtime;         // Time in the co-simulation.
    Cosim_method    method;        // May be set in Cosim_setup;
};

extern void  Cosim_setup(struct co_info *pinfo); // This must exist.
extern void  Cosim_step(struct co_info *pinfo);  // Exists for Verilator.

#if __cplusplus
}
#endif
