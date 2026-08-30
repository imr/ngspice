# Building ngspice with CMake

ngspice can be built with CMake on Linux, macOS and Windows. This document
describes that build.

CMake never builds in the source directory. All generated files go into a
build directory of your choice (`build` below), and deleting that directory
removes every trace of the build.

Three steps are always the same:

```
cmake -S . -B build [options]     configure -- checks the system, writes build files
cmake --build build               compile
cmake --install build             install (optional)
```

## Prerequisites

| | packages |
|---|---|
| Debian, Ubuntu | `build-essential cmake bison flex libfftw3-dev libx11-dev libxaw7-dev libxmu-dev libxext-dev libreadline-dev`, optionally `ninja-build` |
| Fedora, RHEL | `gcc gcc-c++ make cmake bison flex fftw-devel libX11-devel libXaw-devel readline-devel`, optionally `ninja-build` |
| macOS | Xcode command line tools, then `brew install cmake ninja bison flex fftw readline` |
| MSYS2 UCRT64 | `mingw-w64-ucrt-x86_64-{gcc,cmake,ninja,fftw}` and `bison flex` |
| Visual Studio | VS 2022 with the C++ workload; winflexbison and the FFTW DLL package (see [Windows dependencies](#windows-dependencies)) |

macOS ships a bison too old for ngspice, so the Homebrew one must come first
in `PATH`, or be named with `-DBISON_EXECUTABLE=$(brew --prefix bison)/bin/bison`.

## Linux and macOS, GCC or Clang

Default generator, produces makefiles.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # configure
cmake --build build -j$(nproc)                   # build
ctest --test-dir build -j8                       # test
sudo cmake --install build                       # install
cmake --build build --target clean               # clean, keeps the configuration
rm -rf build                                     # clean everything
```

On macOS use `-j$(sysctl -n hw.ncpu)`.

To choose a compiler explicitly, on the configure line of a fresh build
directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
```

## Linux and macOS with Ninja

Same as above, but faster and parallel by default. Add `-G Ninja` when
configuring:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -j8
sudo cmake --install build
cmake --build build --target clean
rm -rf build
```

## Windows, MSYS2 UCRT64

Run these in the **UCRT64** shell, not the plain MSYS shell.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -j8
cmake --install build
cmake --build build --target clean
rm -rf build
```

## Windows, Visual Studio

The Visual Studio generator handles all configurations at once, so the
configuration is chosen when building, testing and installing, not when
configuring. Give `--config` to build, test and install alike -- CMake
hardcodes Debug for `cmake --build` and Release for `cmake --install` when it
is omitted, so leaving it out installs a binary that was never built. This is
CMake behaviour and cannot be changed from `CMakeLists.txt`. All other
generators build Release by default.

```cmd
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release
cmake --install build --config Release
cmake --build build --config Release --target clean
rmdir /s /q build
```

`-A x64` selects the architecture; use `-A Win32` for a 32 bit build.
`build\ngspice.sln` can also be opened in Visual Studio directly.

## Windows, VS Developer Command Prompt

Use the **x64 Native Tools Command Prompt for VS 2022**; the plain developer
prompt targets 32 bit. Here the configuration is fixed when configuring, as
on Linux.

```cmd
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -j8
cmake --install build
cmake --build build --target clean
rmdir /s /q build
```

Ninja ships with Visual Studio. `-G "NMake Makefiles"` also works but cannot
build in parallel.

## The Visual Studio solution configurations

The configurations of `visualc/*.sln` map onto option combinations. On a
multi-config generator the build type comes from `--config`, elsewhere from
`-DCMAKE_BUILD_TYPE`.

Executable:

| solution configuration | CMake |
|---|---|
| `console_debug` | `-DNGSPICE_WINGUI=OFF -DNGSPICE_ENABLE_OPENMP=OFF` + Debug |
| `console_release` | `-DNGSPICE_WINGUI=OFF -DNGSPICE_ENABLE_OPENMP=OFF` + Release |
| `console_release_omp` | `-DNGSPICE_WINGUI=OFF` + Release |
| `Debug` | `-DNGSPICE_ENABLE_OPENMP=OFF` + Debug |
| `Release` | `-DNGSPICE_ENABLE_OPENMP=OFF` + Release |
| `Release_omp` | Release (defaults are GUI and OpenMP on Windows) |

Shared library, all with `-DNGSPICE_BUILD_SHARED=ON`:

| solution configuration | CMake |
|---|---|
| `Debug` | `-DNGSPICE_WITH_FFTW3=OFF -DNGSPICE_ENABLE_OPENMP=OFF` + Debug |
| `Debug-fftw` | `-DNGSPICE_ENABLE_OPENMP=OFF` + Debug |
| `Release` | `-DNGSPICE_WITH_FFTW3=OFF -DNGSPICE_ENABLE_OPENMP=OFF` + Release |
| `Release-fftw` | `-DNGSPICE_ENABLE_OPENMP=OFF` + Release |
| `Release_OMP-fftw` | Release |

`NGSPICE_WINGUI` only exists on Windows; elsewhere the console build is the
only one. All of these work on Linux too, apart from `NGSPICE_WINGUI`.

## Options

Options are given when configuring, e.g.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNGSPICE_ENABLE_CIDER=ON
```

They are remembered in `build/CMakeCache.txt`, so they only need to be
repeated when they change. `cmake -LH build` lists all of them with their
current values.

### Where things go

| | |
|---|---|
| `-DCMAKE_INSTALL_PREFIX=<dir>` | Install location, default `/usr/local`. Must be set when **configuring**: the path is compiled into the binary and written into `spinit`. |
| `-DNGSPICE_ENABLE_RELPATH=ON` | Unix only: look for `spinit` relative to the current directory instead of at an absolute path. Windows is relocatable anyway and must not use this. |
| `-DCMAKE_BUILD_TYPE=Release` | Optimised build (default). `Debug` builds unoptimised with debug info and defines `NGDEBUG`. Ignored by the Visual Studio generator, which uses `--config` instead. |

### Simulator features

| | |
|---|---|
| `-DNGSPICE_ENABLE_XSPICE=OFF` | Drop the XSPICE mixed-signal extensions and the code models. Default ON. |
| `-DNGSPICE_ENABLE_CODEMODELS=OFF` | Keep XSPICE but do not build the `.cm` files. Default ON. |
| `-DNGSPICE_ENABLE_CIDER=ON` | CIDER numerical device models. Default OFF. |
| `-DNGSPICE_ENABLE_OSDI=OFF` | OSDI / OpenVAF model loading. Default ON. |
| `-DNGSPICE_ENABLE_PSS=ON` | Periodic steady state analysis. Default OFF. |
| `-DNGSPICE_ENABLE_RFSPICE=ON` | S-parameter analysis. Default OFF. |
| `-DNGSPICE_ENABLE_SENSE2=ON` | Old sense2 sensitivity analysis. Default OFF. |
| `-DNGSPICE_ENABLE_NDEV=ON` | NDEV external device interface. Default OFF. |

### Libraries and performance

| | |
|---|---|
| `-DNGSPICE_ENABLE_KLU=OFF` | Disable the KLU sparse solver and fall back to SPARSE 1.3. Default ON. |
| `-DNGSPICE_KLU_PROVIDER=system` | Link the system SuiteSparse instead of the bundled `src/maths/KLU`. Default `bundled`. |
| `-DNGSPICE_WITH_FFTW3=OFF` | Use the internal FFT instead of FFTW3. Default ON. |
| `-DNGSPICE_ENABLE_OPENMP=OFF` | Disable parallel device loading. Default ON. |
| `-DNGSPICE_WITH_READLINE=editline` | Line editing: `auto` (default), `readline`, `editline` or `none`. |
| `-DNGSPICE_ENABLE_X11=OFF` | Build without the X11 plot window. Default ON on Unix, always off on Windows. |
| `-DNGSPICE_WINGUI=OFF` | Windows only: console binary instead of the GUI one. Default ON. |

### Build products

| | |
|---|---|
| `-DNGSPICE_BUILD_SHARED=ON` | Build `libngspice`, the shared library API, instead of the `ngspice` binary. Use a second build directory if both are needed. Default OFF. |
| `-DNGSPICE_DISABLE_UTF8=ON` | Non-Unicode Windows build (`EXT_ASC`). Default OFF. |

### Tests and output

| | |
|---|---|
| `-DNGSPICE_TEST_LEVEL=all` | Also register the test directories that `make check` does not run. Default `check`. |
| `-DNGSPICE_TEST_TIMEOUT=<s>` | Abort a single netlist after this many seconds. Default 120. |
| `-DNGSPICE_ENABLE_TESTS=OFF` | Do not register any tests. Default ON. |
| `-DNGSPICE_QUIET_CONFIG=ON` | Reduce the ~110 system checks to two summary lines. |

## Tests

```sh
ctest --test-dir build -j8 --output-on-failure
```

By default the tests that `make check` runs are registered: 44 tests, about a
second. Every test is labelled with its directory plus `check` or `extra`:

```sh
ctest --test-dir build -L regression      # one subtree
ctest --test-dir build -R func            # by name
ctest --test-dir build --print-labels
```

`-DNGSPICE_TEST_LEVEL=all` adds the directories that are shipped but not run
by autotools; some of those have stale reference output.

The CMC model QA suites (`bsim3`, `bsim4`, `bsimsoi`, `hisim`, `hicum2`,
`hisimhv1`, `hisimhv2`) are driven by `tests/bin/check_cmc.sh` rather than by
`TESTS = *.cir`, and are not registered yet.

## Install

```
<prefix>/bin/ngspice
<prefix>/lib/ngspice/*.cm
<prefix>/share/ngspice/scripts/{spinit,setplot,spectrum}
```

### Default location

| | |
|---|---|
| Linux, macOS | `/usr/local` |
| Windows 64 bit | `C:/Spice64`, Debug `C:/Spice64d` |
| Windows 32 bit | `C:/Spice`, Debug `C:/Spiced` |

These are the locations `visualc/make-install-vngspice.bat` uses. Windows
installs are movable without any option: `config.h` gets the relative strings
`../share/ngspice` and `../lib/ngspice`, and ngspice resolves both against the
directory of `ngspice.exe`, not against the working directory. Do **not** set
`NGSPICE_ENABLE_RELPATH` on Windows -- it switches `spinit` lookup to a path
relative to the current directory, which then only works when started from
`<prefix>/bin`.

A multi-config generator does not know the configuration while configuring,
so a debug install needs the prefix spelled out:

```cmd
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_INSTALL_PREFIX=C:/Spice64d
cmake --build build --config Debug
cmake --install build --config Debug
```

### Other locations

Without `NGSPICE_ENABLE_RELPATH` the prefix must be given when configuring,
because it is compiled into the binary and written into `spinit`:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$HOME/ngspice-47
cmake --build build -j$(nproc)
cmake --install build
```

`cmake --install --prefix <other>` would install into a different place while
the binary still looks in the configured one; that is refused with a warning.
For packaging use `DESTDIR`, which stages the tree without touching the
compiled-in paths:

```sh
DESTDIR=/tmp/stage cmake --install build
```

## Where ngspice looks for its init files

Useful when a build runs but no XSPICE model is found: that means `spinit`
was not read. The order is fixed in `src/frontend/cpitf.c`,
`src/misc/ivars.c` and `src/main.c`, not in the build system.

`spinit`, the system init file that loads the code models:

1. `$SPICE_SCRIPTS/spinit` if that variable is set
2. otherwise `$SPICE_LIB_DIR/scripts/spinit`
3. `SPICE_LIB_DIR` itself defaults to
   * the string `../share/ngspice`, resolved against the directory of
     `ngspice.exe` -- Windows, and what this build configures
   * the literal `../share/ngspice`, resolved against the **current
     directory** -- only with `NGSPICE_ENABLE_RELPATH`
   * the absolute `NGSPICEDATADIR` from `config.h` -- Unix
4. on Windows only, a last fallback to `./spinit` in the current directory
5. `set no_spinit` skips all of it

The code model paths inside `spinit` are passed to `dlopen()`. On Windows a
relative path there is resolved against the directory of `ngspice.exe`
(`src/spicelib/devices/dev.c`), on Unix against the current directory, which
is why Unix gets absolute paths and Windows relative ones.

`.spiceinit`, the user init file, is searched in this order, each time with
`spice.rc` as an alternative name:

1. the directory of the netlist on the command line
2. `$SPICE_USERINIT_DIR`
3. the current directory
4. `$HOME`
5. `$USERPROFILE`

`-n` / `--no-spiceinit` skips it.

Running from the build directory without installing:

```sh
SPICE_SCRIPTS=build/scripts build/src/ngspice circuit.cir
```

`build/scripts/spinit` points at `build/codemodels`; `build/install/spinit`
is the copy meant for installation. Neither is placed at the top of the build
tree, because of the `./spinit` fallback in step 4.

<a name="windows-dependencies"></a>
## Windows dependencies

bison, flex and FFTW3 are expected next to the ngspice directory, the same
convention the Visual Studio project files use. With `D:\spice\ngspice` that
means `D:\spice\flex-bison` and `D:\spice\fftw-3.3-dll64`.

| | expected at | override |
|---|---|---|
| winflexbison | `<ngspice>/../flex-bison` | `-DNGSPICE_FLEX_BISON_DIR=<dir>`, or `-DBISON_EXECUTABLE=` / `-DFLEX_EXECUTABLE=` |
| FFTW3 | `<ngspice>/../fftw-3.3-dll64`, `-dll32` for 32 bit | `-DNGSPICE_FFTW3_DIR=<dir>`, or `-DFFTW3_INCLUDE_DIR=` / `-DFFTW3_LIBRARY=` |

Anything found in `PATH` is used as well. The FFTW package contains no import
library; CMake generates one from `libfftw3-3.def` and copies the DLL next to
`ngspice.exe`.

## If something goes wrong

| Symptom | Cause |
|---|---|
| `Does not match the generator used previously` | Generator, compiler and architecture are cached. Delete the build directory and configure again. |
| Debug binary although Release was wanted | Visual Studio needs `--config Release` when building, `-C Release` for `ctest`, `--config Release` when installing. |
| `<config> was not built` at install time | Build and install used different configurations. Pass the same `--config` to both. |
| `FFTW3 ... not found`, path says `dll32` | The compiler targets 32 bit. Use the x64 Native Tools prompt and a fresh build directory. The `target ...` line of the summary shows the width. |
| `bison >= 3.0 is required` | Install bison and flex, or unpack winflexbison next to the ngspice directory. |
| Installed ngspice does not find `spinit` | `CMAKE_INSTALL_PREFIX` was not set when configuring, or the tree was moved. See Install, or use `NGSPICE_ENABLE_RELPATH`. |
| Configuring appears to hang | About 110 compiler checks run, one line each. With MSVC that takes a minute; `-DNGSPICE_QUIET_CONFIG=ON` shortens the output. |

## For maintainers

The source lists in `cmake/NgspiceSourceLists.cmake` are generated from the
`Makefile.am` files, which stay the reference. After changing any
`src/**/Makefile.am`:

```sh
python3 cmake/am2cmake.py
```

Automake conditionals are translated via the `AM_COND_MAP` table in that
script; an unknown conditional aborts the run, so a new one cannot silently
drop sources. Device directories are picked up automatically, so a new device
model needs no CMake change.

Verified: Linux with GCC and clang (makefiles and Ninja), MSYS2/UCRT64 and
Visual Studio including loading of the XSPICE code models, NMake, install and
`DESTDIR` staging, multi-config generators, `NGSPICE_BUILD_SHARED` against a
client program.

Untested: macOS, `NGSPICE_ENABLE_CIDER`, `NGSPICE_KLU_PROVIDER=system`,
cross compilation.
