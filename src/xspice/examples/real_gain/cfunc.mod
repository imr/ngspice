
void ucm_real_gain (ARGS)
{
    double      *in;
    double      *out;

    double      in_offset;
    double      gain;
    double      out_offset;
    double      delay;
    double      ic;


    /* Get the input and output pointers */
    in = INPUT(in);
    out = OUTPUT(out);

    /* Get the parameters */
    in_offset  = PARAM(in_offset);
    gain       = PARAM(gain);
    out_offset = PARAM(out_offset);
    delay      = PARAM(delay);
    ic         = PARAM(ic);


    /* Assign the output and delay */    
    if(ANALYSIS == DC) {
        *out = ic;
        if(INIT)
            cm_event_queue(delay);
    }
    else {
        *out = gain * (*in + in_offset) + out_offset;
        OUTPUT_DELAY(out) = delay;
    }
}
