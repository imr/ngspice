# Building ngspice with CMake

CMake builds ngspice on Linux, macOS and Windows (MSVC, MinGW). It never
builds in the source directory; everything goes into a build directory, and
deleting that directory undoes the build.

```
cmake -S . -B build [options]     configure
cmake --build build               compile
cmake --install build             install
```

## Prerequisites

| | packages |
|---|---|
| Debian, Ubuntu | `build-essential cmake bison flex libfftw3-dev libx11-dev libxaw7-dev libxmu-dev libxext-dev libreadline-dev`, optionally `ninja-build` |
| Fedora, RHEL | `gcc gcc-c++ make cmake bison flex fftw-devel libX11-devel libXaw-devel readline-devel`, optionally `ninja-build` |
| macOS | Xcode command line tools, then `brew install cmake ninja bison flex fftw readline` |
| MSYS2 UCRT64 | `mingw-w64-ucrt-x86_64-{gcc,cmake,ninja,fftw}` and `bison flex` |
| MSYS2 MINGW64 | `mingw-w64-x86_64-{gcc,cmake,ninja,fftw}` and `bison flex` |
| Visual Studio | VS 2022 with the C++ workload, winflexbison, the FFTW DLL package (see [Windows dependencies](#windows-dependencies)) |

macOS ships a bison too old for ngspice; put the Homebrew one first in `PATH`
or pass `-DBISON_EXECUTABLE=$(brew --prefix bison)/bin/bison`.

## Linux and macOS

`-G Ninja` is optional and faster; without it CMake writes makefiles.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # configure
cmake --build build -j$(nproc)                   # build
ctest --test-dir build -j8                       # test
sudo cmake --install build                       # install
cmake --build build --target clean               # clean, keep configuration
rm -rf build                                     # clean everything
```

macOS: `-j$(sysctl -n hw.ncpu)`. Other compiler:
`-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++` in a fresh build
directory.

## Windows, MSYS2

Works in the UCRT64 and the MINGW64 shell, not in the plain MSYS shell.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -j8
cmake --install build
cmake --build build --target clean
rm -rf build
```

## Windows, Visual Studio

The configuration is chosen when building, not when configuring, and
`--config` is needed every time: CMake defaults `cmake --build` to Debug and
`cmake --install` to Release, so omitting it installs a binary that was never
built.

```cmd
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release
cmake --install build --config Release
cmake --build build --config Release --target clean
rmdir /s /q build
```

`build\ngspice.sln` can also be opened in Visual Studio.

## Windows, VS Developer Command Prompt

Use the **x64 Native Tools Command Prompt**; the plain developer prompt
targets x86 (17.13.2), and 32 bit is unmaintained. `echo
%VSCMD_ARG_TGT_ARCH%` shows which. The build type is set when configuring, as
on Linux.

```cmd
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -j8
cmake --install build
cmake --build build --target clean
rmdir /s /q build
```

Ninja ships with Visual Studio. `-G "NMake Makefiles"` works but is serial.

## Options

Given when configuring, remembered in `build/CMakeCache.txt`.
`cmake -LH build` lists all of them with their current values.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNGSPICE_ENABLE_CIDER=ON
```

| option | default | effect |
|---|---|---|
| `CMAKE_INSTALL_PREFIX` | `/usr/local`, `C:/Spice64` | Install location. Must be set when configuring. |
| `CMAKE_BUILD_TYPE` | `Release` | `Debug` is unoptimised and defines `NGDEBUG`. Visual Studio uses `--config`. |
| `NGSPICE_ENABLE_XSPICE` | ON | XSPICE extensions and code models. |
| `NGSPICE_ENABLE_CODEMODELS` | ON | Build the `.cm` files. |
| `NGSPICE_ENABLE_OSDI` | ON | OSDI / OpenVAF model loading. |
| `NGSPICE_ENABLE_RFSPICE` | ON | S-parameter analysis. |
| `NGSPICE_ENABLE_KLU` | ON | KLU solver; OFF falls back to SPARSE 1.3. |
| `NGSPICE_KLU_PROVIDER` | `bundled` | `system` links SuiteSparse instead. |
| `NGSPICE_WITH_FFTW3` | ON | OFF uses the internal FFT. |
| `NGSPICE_ENABLE_OPENMP` | ON | Parallel device loading. |
| `NGSPICE_ENABLE_X11` | ON on Unix | X11 plot window. |
| `NGSPICE_WITH_READLINE` | `auto` | `readline`, `editline` or `none`. |
| `NGSPICE_ENABLE_CIDER` | OFF | CIDER numerical device models. |
| `NGSPICE_ENABLE_PSS` | OFF | Periodic steady state analysis. |

| `NGSPICE_ENABLE_SENSE2` | OFF | Old sense2 sensitivity analysis. |
| `NGSPICE_ENABLE_NDEV` | OFF | NDEV external device interface. |
| `NGSPICE_BUILD_SHARED` | OFF | `libngspice` instead of the binary. |
| `NGSPICE_WINGUI` | ON | Windows: OFF gives the console binary. |
| `NGSPICE_DISABLE_UTF8` | OFF | Windows: non-Unicode build (`EXT_ASC`). |
| `NGSPICE_ENABLE_RELPATH` | OFF | Unix: `spinit` relative to the current directory. Windows must not use this. |
| `NGSPICE_TEST_LEVEL` | `check` | `all` also registers the dist-only tests. |
| `NGSPICE_TEST_TIMEOUT` | 120 | Seconds per netlist. |
| `NGSPICE_ENABLE_TESTS` | ON | Register tests with CTest. |
| `NGSPICE_QUIET_CONFIG` | OFF | Shorten the ~110 system checks to two lines. |

### Visual Studio solution configurations

| solution | CMake |
|---|---|
| `console_debug` | `-DNGSPICE_WINGUI=OFF -DNGSPICE_ENABLE_OPENMP=OFF` + Debug |
| `console_release` | the same + Release |
| `console_release_omp` | `-DNGSPICE_WINGUI=OFF` + Release |
| `Debug` / `Release` | `-DNGSPICE_ENABLE_OPENMP=OFF` + Debug / Release |
| `Release_omp` | Release |
| shared `Debug` / `Release` | `+ -DNGSPICE_BUILD_SHARED=ON -DNGSPICE_WITH_FFTW3=OFF -DNGSPICE_ENABLE_OPENMP=OFF` |
| shared `*-fftw` | as above but with FFTW3 |
| shared `Release_OMP-fftw` | `-DNGSPICE_BUILD_SHARED=ON` + Release |

## Install

```
<prefix>/bin/ngspice
<prefix>/lib/ngspice/*.cm
<prefix>/share/ngspice/scripts/{spinit,setplot,spectrum}
```

Defaults: `/usr/local` on Unix, `C:/Spice64` on Windows (`C:/Spice64d` for
Debug, to be given by hand with a multi-config generator).

The prefix must be set when configuring, since it is compiled in and written
into `spinit`. `cmake --install --prefix <other>` is refused. For packaging
use `DESTDIR=/tmp/stage cmake --install build`.

Windows installs are movable: `config.h` holds `../share/ngspice` and
`../lib/ngspice`, resolved against the directory of `ngspice.exe`.

## Tests

```sh
ctest --test-dir build -j8 --output-on-failure
ctest --test-dir build -j8 -C Release      # multi-config generators
```

The default registers what `make check` runs: 44 tests, about a second.
`-DNGSPICE_TEST_LEVEL=all` adds the directories that autotools ships but does
not run; some of those have stale reference output. Tests are labelled with
their directory plus `check` or `extra`, so `ctest -L regression` works.

The CMC model QA suites (`bsim3`, `bsim4`, `bsimsoi`, `hisim`, `hicum2`,
`hisimhv1`, `hisimhv2`) run via `tests/bin/check_cmc.sh` rather than
`TESTS = *.cir` and are not registered yet.

## Init files

If a build runs but no XSPICE model is found, `spinit` was not read. Search
order, from `src/frontend/cpitf.c` and `src/misc/ivars.c`:

1. `$SPICE_SCRIPTS/spinit`
2. `$SPICE_LIB_DIR/scripts/spinit`
3. `SPICE_LIB_DIR` defaults to the directory of the executable plus
   `../share/ngspice` on Windows, to `NGSPICEDATADIR` from `config.h` on Unix
4. Windows only: `./spinit`
5. `set no_spinit` skips it

Code model paths in `spinit` go to `dlopen()`: relative to the executable on
Windows, to the current directory on Unix — hence absolute paths there.

`.spiceinit` is searched next to the netlist, then in `$SPICE_USERINIT_DIR`,
the current directory, `$HOME`, `$USERPROFILE`; `spice.rc` works as an
alternative name, `-n` skips it.

Running from the build tree: `SPICE_SCRIPTS=build/scripts build/src/ngspice`.

<a name="windows-dependencies"></a>
## Windows dependencies

Expected next to the ngspice directory, as the visualc projects expect;
`PATH` is used as well.

| | expected at | override |
|---|---|---|
| winflexbison | `<ngspice>/../flex-bison` | `-DNGSPICE_FLEX_BISON_DIR=`, `-DBISON_EXECUTABLE=`, `-DFLEX_EXECUTABLE=` |
| FFTW3 | `<ngspice>/../fftw-3.3-dll64` | `-DNGSPICE_FFTW3_DIR=`, `-DFFTW3_INCLUDE_DIR=`, `-DFFTW3_LIBRARY=` |

The FFTW package has no import library; CMake generates one from
`libfftw3-3.def` and copies the DLL next to `ngspice.exe`.

## If something goes wrong

| Symptom | Cause |
|---|---|
| `Does not match the generator used previously` | Generator, compiler and architecture are cached. Delete the build directory. |
| Debug binary although Release was wanted | Visual Studio needs `--config Release`, `ctest -C Release`. |
| `<config> was not built` at install time | Build and install used different `--config`. |
| `FFTW3 ... not found`, path says `dll32` | The compiler targets 32 bit. Use the x64 prompt and a fresh build directory. |
| `bison >= 3.0 is required` | Install bison and flex, or unpack winflexbison next to the ngspice directory. |
| Installed ngspice does not find `spinit` | `CMAKE_INSTALL_PREFIX` was not set when configuring. |
| Configuring appears to hang | ~110 compiler checks; with MSVC that takes a minute. `-DNGSPICE_QUIET_CONFIG=ON` shortens the output. |

## For maintainers

`cmake/NgspiceSourceLists.cmake` is generated from the `Makefile.am` files,
which stay the reference. After changing any `src/**/Makefile.am`:

```sh
python3 cmake/am2cmake.py
```

Automake conditionals are translated via `AM_COND_MAP` in that script; an
unknown one aborts the run. Device directories are picked up automatically.

Verified: Linux with GCC and clang (makefiles and Ninja), MSYS2 UCRT64 and
MINGW64, Visual Studio and NMake including XSPICE code models, install and
`DESTDIR` staging, `NGSPICE_BUILD_SHARED` against a client program.
Untested: macOS, `NGSPICE_ENABLE_CIDER`, `NGSPICE_KLU_PROVIDER=system`,
cross compilation.
