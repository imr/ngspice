;;; cross compile .cm code model files
;;;   to fix the broken ones from ngspice-26_140112.zip

(compile "git checkout ngspice-26-fix-cm")

;;; replace
;;;   x86_64-unknown-linux-gnu
;;; with your build machines "canonical host triplet"
;;;   (invoke config.guess if you don't know)

;;; replace
;;;   i686-w64-mingw32
;;; with your cross compilers target triplet
;;;   (invoke the cross compiler gcc with `-dumpmachine')

;;; 64 bit would be
;;;   x86_64-w64-mingw32

;;; on debian these compilers can be found in package
;;;   mingw-w64
;;; (even the 32 compiler !)

(compile "
  ./autogen.sh
  rm -rf tmp-build
  mkdir -p tmp-build tmp-output
  ( cd tmp-build && ../configure \
      --build=x86_64-unknown-linux-gnu --host=i686-w64-mingw32 \
      --with-windows --enable-xspice --enable-cider --disable-debug )
  LC_ALL=C make -C tmp-build/src/xspice -j6
  LC_ALL=C make -C tmp-build/src/xspice -j6 DESTDIR=$(pwd)/tmp-output install
  tar -zcf tmp-output.tgz -C tmp-output .
  ")

;;; the .cm files are now to be found in
(ffap "tmp-output.tgz")
