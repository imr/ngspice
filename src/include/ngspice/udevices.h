#ifndef INCLUDED_UDEVICES
#define INCLUDED_UDEVICES

BOOL u_process_instance(char *line);
BOOL u_process_model_line(char *line);
BOOL u_check_instance(char *line);
void initialize_udevice(char *subckt_line);
struct card *replacement_udevice_cards(void);
void cleanup_udevice(void);
void u_add_instance(char *str);
void u_add_logicexp_model(char *tmodel, char *xspice_gate, char *model_name);
void u_remember_pin(char *name, int type);

#endif
