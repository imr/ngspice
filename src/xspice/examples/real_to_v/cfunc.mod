

#define TS 0
#define VS 1


void ucm_real_to_v (ARGS)
{

    double *t, *v;
    double *in;

    double out;


    in = INPUT(in);

    if(INIT) {
        cm_event_alloc(TS, 2 * sizeof(double));
        cm_event_alloc(VS, 2 * sizeof(double));
        t = (void *) cm_event_get_ptr(TS, 0);
        v = (void *) cm_event_get_ptr(VS, 0);
        t[0] = -2.0;
        t[1] = -1.0;
        v[0] = *in;
        v[1] = *in;
    }
    else {
        t = (void *) cm_event_get_ptr(TS, 0);
        v = (void *) cm_event_get_ptr(VS, 0);
    }

    switch(CALL_TYPE) {

    case ANALOG:
        if(TIME == 0.0) {
            OUTPUT(out) = *in;
            v[0] = *in;
            v[1] = *in;
        }
        else {
            if(TIME <= t[0])
                OUTPUT(out) = v[0];
            else if(TIME >= t[1])
                OUTPUT(out) = v[1];
            else {
                OUTPUT(out) = v[0] + (v[1] - v[0]) *
                                (TIME - t[0]) / (t[1] - t[0]);
            }
        }
        break;

    case EVENT:
        if(TIME == 0.0)
            return;
        if(TIME >= t[1]) {
            v[0] = v[1];
            v[1] = *in;
            t[0] = TIME;
            t[1] = TIME + PARAM(transition_time);
        }
        else {
            v[0] = v[0] + (v[1] - v[0]) *
                                (TIME - t[0]) / (t[1] - t[0]);
            v[1] = *in;
            t[0] = TIME;
            t[1] = TIME + PARAM(transition_time);
        }
        break;

    }
}




