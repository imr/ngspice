
void ucm_d_to_real (ARGS)
{

    Digital_State_t     in;

    double              *out;
    double              delay;
    double              zero;
    double              one;
    double              ena;


    in = INPUT_STATE(in);
    if(PORT_NULL(enable))
        ena = 1.0;
    else if(INPUT_STATE(enable) == ONE)
        ena = 1.0;
    else
        ena = 0.0;
    out = OUTPUT(out);

    zero  = PARAM(zero);
    one   = PARAM(one);
    delay = PARAM(delay);


    if(in == ZERO)
        *out = zero * ena;
    else if(in == UNKNOWN)
        *out = (zero + one) / 2.0 * ena;
    else
        *out = one * ena;

    if(TIME > 0.0)
        OUTPUT_DELAY(out) = delay;

}
