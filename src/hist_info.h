/* Header for saving and retrieving history strings */
#ifndef HIST_INFO_H
#define HIST_INFO_H

struct History_info;

/* Options for history logging */
struct History_info_opt {
    size_t n_byte_struct; /* sizeof(struct History_info_opt)
                           * Used for possible versioning */
    unsigned int n_str_init; /* initial number of allocated string locators
                              * Value must be at least 2 */
    unsigned int n_str_max; /* max number of string locators
                             * Value must be at least n_str_init when
                             * initializing but not when changing options.
                             * Then it must be at least 2 */
    size_t n_byte_str_buf_init; /* initial size of the string buffer.
                                 * Value must be at least 2 */

    /* If the amount of the buffer for storing strings that is in use
     * multiplied by oversize_factor is less than the allocated amount,
     * the buffer for storing strings is reduced by a factor of 2.
     * Value must be at least 4.
     *
     * The first check for an oversized buffer is made after
     * n_insert_first_oversize_check calls to history_add().
     * A value of 0 suppresses checks.
     *
     * Subsequent checks are made after n_insert_per_oversize_check calls
     * to history_add().
     * A value of 0 suppresses checks after the first one.
     */
    unsigned int oversize_factor;
    unsigned int n_insert_first_oversize_check;
    unsigned int n_insert_per_oversize_check;
};


struct History_info *history_init(struct History_info_opt *p_hi_opt);
int history_add(struct History_info **pp_hi,
        unsigned int n_char_str, const char *str);
void history_free(struct History_info *p_hi);\
const char *history_get_newest(struct History_info *p_hi,
        unsigned int *p_n_char_str);
const char *history_get_next(struct History_info *p_hi,
        unsigned int *p_n_char_str);
const char *history_get_prev(struct History_info *p_hi,
        unsigned int *p_n_char_str);
void history_reset_pos(struct History_info *p_hi);
int history_getopt(const struct History_info *p_hi,
        struct History_info_opt *p_hi_opt);
int history_setopt(struct History_info **pp_hi,
        struct History_info_opt *p_hi_opt);

#endif /* HIST_INFO_H include guard */
