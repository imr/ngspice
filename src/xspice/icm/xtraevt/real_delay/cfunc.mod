

#define CLK_STATE       0


void ucm_real_delay (ARGS)
{

    double              *in;
    double              *out;

    Digital_State_t     *state;
    Digital_State_t     *old_state;


    if(INIT) {
        cm_event_alloc(CLK_STATE, sizeof(Digital_State_t));
        state = (Digital_State_t *) cm_event_get_ptr(CLK_STATE, 0);
        old_state = state;
        *state = INPUT_STATE(clk);
    }
    else {
        state = (Digital_State_t *) cm_event_get_ptr(CLK_STATE, 0);
        old_state = (Digital_State_t *) cm_event_get_ptr(CLK_STATE, 1);
    }

    if(ANALYSIS != TRANSIENT)
        OUTPUT_CHANGED(out) = FALSE;
    else {
        *state = INPUT_STATE(clk);
        if(*state == *old_state)
            OUTPUT_CHANGED(out) = FALSE;
        else if(*state != ONE)
            OUTPUT_CHANGED(out) = FALSE;
        else {
            in = (double *) INPUT(in);
            out = (double *) OUTPUT(out);
            *out = *in;
            OUTPUT_DELAY(out) = PARAM(delay);
        }
    }
}




