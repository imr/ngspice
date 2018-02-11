// GENmodule and GENinstance for each device specific instance and module struct
//
// (compile "git grep -l -e 'modPtr\\|nextModel' -- '*.h' | xargs -P8 -n1 spatch --defined XSPICE --sp-file klag-5.cocci --in-place && emacs -q --no-site-file --batch --load klag-5.el && git commit -am 'GENmodel, GENinstance, change layout, #3/4, coccinelle semantic patch'")

// (compile "spatch --sp-file klag-5.cocci src/spicelib/devices/bsim3v32/bsim3v32def.h")
// (compile "spatch --sp-file klag-5.cocci src/spicelib/devices/cap/capdefs.h")
// (compile "spatch --sp-file klag-5.cocci src/spicelib/devices/asrc/asrcdefs.h")
// (compile "spatch --sp-file klag-5.cocci src/include/ngspice/numgen.h")
// (compile "spatch --sp-file klag-5.cocci src/include/ngspice/mifdefs.h")
// (compile "spatch --sp-file klag-5.cocci src/spicelib/devices/soi3/soi3defs.h")


@struc1@
identifier Xstruct !~ "GENinstance$";
identifier XmodPtr =~ "modPtr$";
identifier XnextInstance =~ "nextInstance$";
identifier Xname =~ "name$";
identifier Xstates =~ "states?$";
type T1, T2;
typedef IFuid;
type xx;
@@
(
typedef struct Xstruct {
- T1 XmodPtr;
- T2 XnextInstance;
- IFuid Xname;
- int Xstates;
+
+ struct GENinstance gen;
+ /* emacs-place-here */
+
  ...
} xx ;
+/* emacs-from-here */
+#define XmodPtr(inst)        ((T1)((inst)->gen.GENmodPtr))
+#define XnextInstance(inst)  ((T2)((inst)->gen.GENnextInstance))
+#define Xname                gen.GENname
+#define Xstates              gen.GENstate
+/* emacs-upto-here */
|
struct Xstruct {
- T1 XmodPtr;
- T2 XnextInstance;
- IFuid Xname;
- int Xstates;
+
+ struct GENinstance gen;
+ /* emacs-place-here */
+
  ...
};
+/* emacs-from-here */
+#define XmodPtr(inst)        ((T1)((inst)->gen.GENmodPtr))
+#define XnextInstance(inst)  ((T2)((inst)->gen.GENnextInstance))
+#define Xname                gen.GENname
+#define Xstates              gen.GENstate
+/* emacs-upto-here */
)



@struc2@
identifier Xstruct !~ "GENmodel$";
identifier XmodType =~ "modType$";
identifier XnextModel =~ "nextModel$";
identifier Xinstances =~ "instances$";
identifier XmodName =~ "modName$";
type T1, T2;
typedef IFuid;
type xx;
@@
(
typedef struct Xstruct {
- int XmodType;
- T1 XnextModel;
- T2 Xinstances;
- IFuid XmodName;
+
+ struct GENmodel gen;
+ /* emacs-place-here */
+
  ...
} xx ;
+/* emacs-from-here */
+#define XmodType            gen.GENmodType
+#define XnextModel(inst)    ((T1)((inst)->gen.GENnextModel))
+#define Xinstances(inst)    ((T2)((inst)->gen.GENinstances))
+#define XmodName            gen.GENmodName
+/* emacs-upto-here */
|
struct Xstruct {
- int XmodType;
- T1 XnextModel;
- T2 Xinstances;
- IFuid XmodName;
+
+ struct GENmodel gen;
+ /* emacs-place-here */
+
  ...
};
+/* emacs-from-here */
+#define XmodType            gen.GENmodType
+#define XnextModel(inst)    ((T1)((inst)->gen.GENnextModel))
+#define Xinstances(inst)    ((T2)((inst)->gen.GENinstances))
+#define XmodName            gen.GENmodName
+/* emacs-upto-here */
)
