/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE adc_bridge/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405

AUTHORS
    6 June 1991     Jeffrey P. Murray

MODIFICATIONS
    26 Sept 1991    Jeffrey P. Murray

SUMMARY

    This file contains the functional description of the adc_bridge code model.

INTERFACES

    FILE                 ROUTINE CALLED

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()

    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()
                         int  cm_event_queue()

REFERENCED FILES
    Inputs from and outputs to ARGS structure.

NON-STANDARD FEATURES
    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

/*=== CONSTANTS ========================*/

/*=== MACROS ===========================*/

/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         

/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/

/*==============================================================================

FUNCTION cm_adc_bridge()

AUTHORS
    6 June 1991     Jeffrey P. Murray

MODIFICATIONS
    26 Sept 1991    Jeffrey P. Murray

SUMMARY
    This function implements the adc_bridge code model.

INTERFACES

    FILE                 ROUTINE CALLED

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()

    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()
                         int  cm_event_queue()

RETURNED VALUE
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    NONE

NON-STANDARD FEATURES
    NONE

==============================================================================*/

/*=== CM_ADC_BRIDGE ROUTINE ===*/

/************************************************
*      The following is the model for the       *
*   analog-to-digital nodebridge for the        *
*   ATESSE Version 2.0 system.                  *
*                                               *
*   Created 6/6/91                              *
*   Last Modified 7/26/91         J.P.Murray    *
************************************************/

static Digital_State_t get_out_value(double in, double low, double high)
{
    if (in >= high)
        return ONE;
    else if (in <= low)
        return ZERO;
    return UNKNOWN;
}

void cm_adc_bridge(ARGS)
{
    double         in_low,  /* analog output value corresponding to '0'
                               digital input    */
                  in_high;  /* analog output value corresponding to '1' 
                               digital input    */
    int                 i,  /* generic loop counter index */
                     size;  /* number of input & output ports */

   Digital_State_t   *out,  /* base address of array holding all output
                               values plus their previous values */
                     test;  /* temp holding variable for digital states */

    /* determine "width" of the node bridge... */

    size = PORT_SIZE(in);
    in_high = PARAM(in_high);
    in_low = PARAM(in_low);

    if (INIT) {  /*** Test for INIT == TRUE. If so, allocate storage, etc. ***/
        /* Allocate storage for outputs */

        cm_event_alloc(0, size * (int) sizeof(Digital_State_t));

        /* Get discrete addresses */

        out = (Digital_State_t *) cm_event_get_ptr(0,0);

        /* Ensure output on first call. */

        for (i = 0; i < size; i++)
            out[i] = UNKNOWN + 1;
        return;
    }

    /*** This is not an initialization pass...retrieve storage
         addresses and calculate new outputs, if required. ***/

    out = (Digital_State_t *) cm_event_get_ptr(0, 0);

    switch (CALL_TYPE) {
        case ANALOG:    /** analog call...check for breakpoint calls. **/
            /* loop through all inputs... */

            for (i = 0; i < size; i++) {
                test = get_out_value(INPUT(in[i]), in_low, in_high);
                if (test !=  out[i]) {
                    /* call for event breakpoint... */

                    cm_event_queue(TIME);
                    break;
                }
            }
            break;

        case EVENT:    /** discrete call...lots to do **/
            /* loop through all inputs... */

            for (i = 0; i < size; i++) {
                test = get_out_value(INPUT(in[i]), in_low, in_high);
                if (test !=  out[i]) {
                    /* Post changed value. */

                    OUTPUT_STATE(out[i]) = test;
                    switch (test) {
                    case ZERO:
                        OUTPUT_DELAY(out[i]) = PARAM(fall_delay);
                        break;
                    case ONE:
                        OUTPUT_DELAY(out[i]) = PARAM(rise_delay);
                        break;
                    default:
                        if (out[i] == ZERO)
                            OUTPUT_DELAY(out[i]) = PARAM(rise_delay);
                        else
                            OUTPUT_DELAY(out[i]) = PARAM(fall_delay);
                        break;
                    }
                    out[i] = test;
                    /* Regardless, output the strength */

                    OUTPUT_STRENGTH(out[i]) = STRONG;
                } else {
                    /* no change since last time */

                    OUTPUT_CHANGED(out[i]) = FALSE;
                }
            }
            break;
        default:
            break;
    }
}
