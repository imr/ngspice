// DEVdestroy()
//   drop the model loop
//   drop the instance loop + FREE
//
// (compile "git grep -l -e 'destroy' -- '*.[ch]' | xargs -P8 -n1 spatch --sp-file klag-3.cocci --in-place && git commit -am 'DEVdestroy(), change API, #1/2, coccinelle semantic patch'")
//
// (compile "spatch --sp-file klag-3.cocci src/spicelib/devices/bjt/*.[ch]")
// (compile "spatch --sp-file klag-3.cocci src/spicelib/devices/res/resdest.c")
// (compile "spatch --sp-file klag-3.cocci src/spicelib/devices/bjt/bjtdest.c")
// (compile "spatch --sp-file klag-3.cocci src/spicelib/devices/asrc/asrcdest.c")
// (compile "spatch --sp-file klag-3.cocci src/spicelib/devices/bsim1/b1dest.c")


@unify_DEVdestroy@
identifier fn =~ "destroy$";
typedef GENmodel;
identifier XinModel;
identifier Xmod, Xinst, Xnext_mod, Xnext_inst;
identifier XXXnextModel, XXXinstances, XXXnextInstance;
type XXXmodel, XXXinstance;
@@
  void fn (
- GENmodel **XinModel
+ void
  )
  {
-     XXXmodel *Xmod = *(XXXmodel**) XinModel;
-
-     while (Xmod) {
-         XXXmodel *Xnext_mod = Xmod->XXXnextModel;
-         XXXinstance *Xinst = Xmod->XXXinstances;
-         while (Xinst) {
-            XXXinstance *Xnext_inst = Xinst->XXXnextInstance;
-            FREE(Xinst);
-            Xinst = Xnext_inst;
-         }
-         FREE(Xmod);
-         Xmod = Xnext_mod;
-     }
-
-     *XinModel = NULL;
  }


@unify_DEVdestroy_h@
identifier fn =~ "destroy$";
typedef GENmodel;
@@
  extern void fn (
- GENmodel **
+ void
  );
