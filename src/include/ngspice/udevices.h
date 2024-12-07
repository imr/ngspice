#ifndef INCLUDED_UDEVICES
#define INCLUDED_UDEVICES

struct udevices_info {
    int mntymx;
    bool shorter_delays;
};

bool u_process_instance(char *line);
bool u_process_model_line(char *line, bool global);
bool u_check_instance(char *line);
void initialize_udevice(char *subckt_line);
struct card *replacement_udevice_cards(void);
void cleanup_udevice(bool global);
void u_add_instance(char *str);
void u_add_logicexp_model(char *tmodel, char *xspice_gate, char *model_name);
void u_remember_pin(char *name, int type);
struct udevices_info u_get_udevices_info(void);
void u_subckt_line(char *subckt_line);

#endif
