
void ucm_print_param_types (ARGS)
{
    int i;

    if(INIT) {
        /* Print scalar parameters */
        printf("\nScalar parameters\n\n");
        printf("integer = %d\n", PARAM(integer));
        printf("real    = %e\n", PARAM(real));
        printf("complex = <%e %e>\n", PARAM(complex).real,
                                      PARAM(complex).imag);
        printf("string  = %s\n", PARAM(string));

        /* Print vector parameters */
        printf("\nVector parameters\n\n");
        for(i = 0; i < PARAM_SIZE(integer_array); i++)
            printf("integer = %d\n", PARAM(integer_array[i]));
        for(i = 0; i < PARAM_SIZE(real_array); i++)
            printf("real    = %e\n", PARAM(real_array[i]));
        for(i = 0; i < PARAM_SIZE(complex_array); i++)
            printf("complex = <%e %e>\n", PARAM(complex_array[i]).real,
                                          PARAM(complex_array[i]).imag);
        for(i = 0; i < PARAM_SIZE(string_array); i++)
            printf("string  = %s\n", PARAM(string_array[i]));

    }
}
