
void *malloc(unsigned);

#define OUT_STATE 0
#define NXT_TIME  1
#define NUM_NOTES 128


/* A numerically controlled oscillator.  Output frequencies */
/* are determined according to the MIDI note number at input */

void ucm_nco (ARGS)
{

    double      *freq;

    int         *output_state;
    double      *next_time;

    int         i;
    int         index;
    int         scale_factor;

    double      half_period;


    if(INIT) {

        /* Setup storage for the toggled output state */
        cm_event_alloc(OUT_STATE, sizeof(int));
        cm_event_alloc(NXT_TIME, sizeof(double));
        output_state = (int *) cm_event_get_ptr(OUT_STATE, 0);
        next_time = (double *) cm_event_get_ptr(NXT_TIME, 0);

        /* Allocate storage for frequencies */
        STATIC_VAR(freq) = malloc(NUM_NOTES * sizeof(double));
        freq = STATIC_VAR(freq);

        /* Initialize the frequency array */
        for(i = 0; i < NUM_NOTES; i++) {
            if(i == 0)
                freq[0] = 8.17578 * PARAM(mult_factor);
            else
                freq[i] = freq[i-1] * 1.059463094;
        }
    }
    else {

        /* Get old output state */
        output_state = (int *) cm_event_get_ptr(OUT_STATE, 0);
        next_time = (double *) cm_event_get_ptr(NXT_TIME, 0);
    }


    /* Convert the input bits to an integer */
    index = 0;
    scale_factor = 64;
    for(i = 0; i < 7; i++) {
        if(INPUT_STATE(in[i]) == ONE)
            index += scale_factor;
        scale_factor /= 2;
    }

    /* Look up the frequency and compute half its period */
    freq = STATIC_VAR(freq);
    half_period = 1.0 / freq[index];


    /* Queue up events and output the new state */
    if(TIME == 0.0) {
        *next_time = half_period;
        cm_event_queue(*next_time);
        OUTPUT_STATE(out) = *output_state;
    }
    else {
        if(TIME == *next_time) {
            *next_time = TIME + half_period;
            cm_event_queue(*next_time);
            *output_state = 1 - *output_state;
            OUTPUT_STATE(out) = *output_state;
            OUTPUT_DELAY(out) = PARAM(delay);
        }
        else
            OUTPUT_CHANGED(out) = FALSE;
    }

}




