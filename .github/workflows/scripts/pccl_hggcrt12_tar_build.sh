#!/bin/bash
set -e
# Validate required environment variables.
if [ -z "${source_home}" ]; then
    echo "Error: source_home environment variable is not set"
    exit 1
fi
if [ -z "${build_output}" ]; then
    echo "Error: build_output environment variable is not set"
    exit 1
fi

# Matrix builds override pkg_name/sdk_url/sdk_name so every distribution keeps a distinct
# tarball name and installs the SDK build that matches its own OS and HGGCRT runtime.
pkg_name=${pkg_name:-pccl_hggcrt12_afa3}
# The matrix also ships the pccl-tests perf binaries from this same run, so they always
# match the library revision that was just built. Default derives from pkg_name.
pccl_tests_pkg_name=${pccl_tests_pkg_name:-pccl_tests_${pkg_name#pccl_}}
if [ -z "${sdk_url}" ]; then
    echo "Error: sdk_url environment variable is not set"
    exit 1
fi
if [ -z "${sdk_name}" ]; then
    echo "Error: sdk_name environment variable is not set"
    exit 1
fi

# alios7u2-min ships only Alibaba intranet yum sources: every enabled repo in /etc/yum.repos.d
# points at http://yum.tbsite.net (10.101.0.66), which a CI runner pod cannot route to, so each
# yum call dies after a few seconds. The image is a CentOS 7 derivative, so repoint it at the
# public CentOS 7.9 vault mirror plus EPEL, which the SDK's own dependency step also needs.
prepare_yum_sources() {
    local os_id os_version
    os_id=$(. /etc/os-release 2>/dev/null && echo "${ID:-}")
    os_version=$(. /etc/os-release 2>/dev/null && echo "${VERSION_ID:-}")
    case "${os_id}" in
        alios) ;;
        alinux)
            # alinux3, alinux4 and the alios8u image, which reports itself as ALinux 3.
            prepare_alinux_sources "${os_version}"
            return 0
            ;;
        *) return 0 ;;
    esac
    echo "alios base image detected: replacing intranet yum sources with public mirrors"
    mkdir -p /etc/yum.repos.d/disabled
    mv /etc/yum.repos.d/*.repo /etc/yum.repos.d/disabled/ 2>/dev/null || true
    # Plain HTTP on purpose: these images carry no ca-certificates before this install runs.
    cat > /etc/yum.repos.d/ci-public.repo <<'REPO'
[ci-centos7-vault]
name=CentOS 7.9.2009 vault (public mirror)
baseurl=http://mirrors.aliyun.com/centos-vault/7.9.2009/os/x86_64/
enabled=1
gpgcheck=0

[ci-epel7]
name=EPEL 7 (public mirror)
baseurl=http://mirrors.aliyun.com/epel/7/x86_64/
enabled=1
gpgcheck=0
REPO
    yum clean all >/dev/null 2>&1 || true
}

# Two callers need an interpreter here and they do not agree on its name: build.sh's argparse.bash
# accepts `python` or `python3`, while pccl's src/collectives/device/Makefile runs the symmetric
# kernel generator through `which python3` and, when that fails, only prints
#   ERROR: Building NCCL requires a Python 3 installation invocable as 'python3'
# Since the generated rules.mk is pulled in with `-include`, that failure does not stop make: the
# build continues without a single symmetric kernel and links a libpccl.so that has no
# ncclSymkKernel*. An image where /usr/bin/python is python2 (alios8u) used to satisfy the `python`
# half of the old check while `which python3` still found nothing. Publish a *Python 3* binary under
# the exact name 'python3', whatever the image calls it.
is_python3() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)' >/dev/null 2>&1
}

expose_python3() {
    local candidate
    local py
    PYTHON3_BIN=""
    case ":${PATH}:" in
        *:/usr/local/bin:*) ;;
        *) export PATH="/usr/local/bin:${PATH}" ;;
    esac
    if py=$(command -v python3 2>/dev/null) && is_python3 "${py}"; then
        PYTHON3_BIN="${py}"
        return 0
    fi
    for candidate in /usr/libexec/platform-python /usr/bin/python3 /usr/bin/python3.* /usr/bin/python; do
        [ -x "${candidate}" ] || continue
        is_python3 "${candidate}" || continue
        if ln -sf "${candidate}" /usr/local/bin/python3 2>/dev/null && is_python3 /usr/local/bin/python3; then
            PYTHON3_BIN=/usr/local/bin/python3
            echo "exposed ${candidate} as /usr/local/bin/python3"
            return 0
        fi
    done
    echo "::warning::no Python 3 interpreter found, pccl's symmetric kernel generator cannot run"
    return 1
}

# pccl's makefiles locate the compiler with `which $(HGCC)` (makefiles/common.mk), and the minimal
# Alibaba Cloud Linux image ships no `which` binary at all, so that probe silently produced an empty
# string and the build stopped at
#   ../makefiles/common.mk:28: *** "/workspace/source/PPU_SDK/bin/hgcc Not Found. ..."  Stop.
# one second after the same hgcc had answered `hgcc --version`. Install the package where it exists
# and fall back to a small shim, so the build never depends on that package again.
ensure_which() {
    if which sh >/dev/null 2>&1; then
        return 0
    fi
    echo "no working 'which' on this image, but pccl's makefiles need it to find hgcc"
    if command -v yum >/dev/null 2>&1; then
        yum install -y --setopt=timeout=60 --setopt=retries=3 which >/dev/null 2>&1 || true
    elif command -v apt-get >/dev/null 2>&1; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends which \
            </dev/null >/dev/null 2>&1 || true
    fi
    if ! which sh >/dev/null 2>&1; then
        cat > /usr/local/bin/which <<'SHIM'
#!/bin/sh
# Just enough of util-linux' which for a build script: `which <name>` or `which <abs path>`.
status=0
for candidate in "$@"; do
    case "${candidate}" in
        -*) continue ;;
    esac
    if [ -x "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        continue
    fi
    resolved=$(command -v "${candidate}" 2>/dev/null) || { status=1; continue; }
    printf '%s\n' "${resolved}"
done
exit "${status}"
SHIM
        chmod 0755 /usr/local/bin/which
        hash -r
    fi
    if which sh >/dev/null 2>&1; then
        echo "using $(command -v which)"
    else
        echo "::warning::still no usable 'which'; pccl's makefiles may not find hgcc"
    fi
}

# The Alibaba Cloud Linux images resolve their packages through yum.tbsite.net or
# mirrors.cloud.aliyuncs.com; neither is reachable from the GitHub runners. Repoint them at the
# public Aliyun mirror, whose layout moved between major versions: 3 keeps the packages directly
# under <channel>/x86_64/ while 4 nests them one level deeper in <channel>/x86_64/os/.
prepare_alinux_sources() {
    local version="${1:-}"
    local suffix=""
    case "${version}" in
        3*) ;;
        4*) suffix="/os" ;;
        *)
            echo "::warning::unexpected Alibaba Cloud Linux version ${version:-<empty>}; keeping the baked-in yum sources"
            return 0
            ;;
    esac
    echo "Alibaba Cloud Linux ${version} detected: replacing the intranet yum sources with mirrors.aliyun.com"
    mkdir -p /etc/yum.repos.d/disabled
    mv /etc/yum.repos.d/*.repo /etc/yum.repos.d/disabled/ 2>/dev/null || true
    # Plain HTTP on purpose: these images carry no ca-certificates before this install runs.
    cat > /etc/yum.repos.d/ci-public.repo <<REPO
[ci-alinux-os]
name=Alibaba Cloud Linux ${version} os (public mirror)
baseurl=http://mirrors.aliyun.com/alinux/${version}/os/\$basearch${suffix}
enabled=1
gpgcheck=0

[ci-alinux-updates]
name=Alibaba Cloud Linux ${version} updates (public mirror)
baseurl=http://mirrors.aliyun.com/alinux/${version}/updates/\$basearch${suffix}
enabled=1
gpgcheck=0

[ci-alinux-plus]
name=Alibaba Cloud Linux ${version} plus (public mirror)
baseurl=http://mirrors.aliyun.com/alinux/${version}/plus/\$basearch${suffix}
enabled=1
gpgcheck=0
REPO
    rm -rf /var/cache/yum /var/cache/dnf
}

# pccl's makefiles always pass -std=c++14 (makefiles/common.mk), but a CentOS 7 based image ships
# gcc 4.8.5, which predates that switch, so the host compile died with
#   g++: error: unrecognized command line option '-std=c++14'
# and every object file after the first one failed the same way. Nothing in the el7 repositories
# carries a newer g++, but SCL's devtoolset does, so install one and put it on PATH. Images whose
# compiler already understands the flag return early.
# scl-utils in the CentOS 7 vault pulls policycoreutils-python, whose libsemanage-python asks for
# the strictly equal libsemanage-2.5-14.el7 while alios7 installs libsemanage-2.5-14.1.aliOS7, so
# the transaction stays unsatisfiable whatever multilib escape is used:
#   Error: Package: libsemanage-python-2.5-14.el7.x86_64 (ci-centos7-vault)
#          Requires: libsemanage = 2.5-14.el7
#          Installed: libsemanage-2.5-14.1.aliOS7.x86_64 (@alios.7u2.base.x86_64)
# yumdownloader fetches packages without resolving their dependencies and rpm then installs them
# with --nodeps, which skips exactly that chain; the compiler itself only needs glibc and binutils,
# both of which the image already has.
install_toolset_from_rpms() {
    local part
    local rpm_dir=/tmp/toolset-rpms
    local toolset="$1"
    if ! command -v yumdownloader >/dev/null 2>&1; then
        yum install -y --setopt=timeout=60 --setopt=retries=3 yum-utils \
            >> /tmp/devtoolset-install.log 2>&1 || true
    fi
    if ! command -v yumdownloader >/dev/null 2>&1; then
        return 1
    fi
    rm -rf "${rpm_dir}"
    mkdir -p "${rpm_dir}"
    # binutils belongs to the same set: a devtoolset g++ hands the debug format to the assembler as
    # `--gdwarf-N`, so pairing it with the image's ancient as fails on every single object with
    #   as: unrecognized option `--gdwarf-4'
    # yum pulls binutils in as a dependency, yumdownloader --nodeps does not.
    for part in runtime gcc gcc-c++ libstdc++-devel binutils; do
        yumdownloader --destdir="${rpm_dir}" --exclude='*.i686' "${toolset}-${part}" \
            >> /tmp/devtoolset-install.log 2>&1 || \
            echo "::warning::the SCL mirror carries no ${toolset}-${part} package"
    done
    if [ -z "$(ls -A "${rpm_dir}" 2>/dev/null)" ]; then
        return 1
    fi
    echo "installing $(ls -1 "${rpm_dir}" | tr '\n' ' ')"
    rpm -Uvh --nodeps --replacefiles "${rpm_dir}"/*.rpm >> /tmp/devtoolset-install.log 2>&1 || true
    [ -f "/opt/rh/${toolset}/enable" ]
}

ensure_cxx14_compiler() {
    local probe=/tmp/cxx14-probe
    local toolset
    printf 'int main() { return 0; }\n' > "${probe}.cc"
    if g++ -std=c++14 -o "${probe}" "${probe}.cc" >/dev/null 2>&1; then
        rm -f "${probe}" "${probe}.cc"
        echo "compiler $(g++ -dumpversion 2>/dev/null) already accepts -std=c++14"
        return 0
    fi
    rm -f "${probe}" "${probe}.cc"
    echo "compiler $(g++ -dumpversion 2>/dev/null) rejects -std=c++14; looking for a newer one"
    # devtoolset-9 ships binutils 2.32, whose assembler rejects the DWARF-5 directives a newer
    # compiler emits (as: Error: unknown .loc sub-directive `view' / unknown pseudo-op: `.lvu26'),
    # while devtoolset-10 and -11 bring binutils 2.35/2.36.1 and their assembler understands them.
    for toolset in devtoolset-11 devtoolset-10 devtoolset-9 devtoolset-8; do
        if [ ! -f "/opt/rh/${toolset}/enable" ] && command -v yum >/dev/null 2>&1; then
            # SCL lives in the CentOS 7 vault; aliyun mirrors it whereas vault.centos.org does not
            # answer on the CI network.
            cat > /etc/yum.repos.d/ci-sclo.repo <<'REPO'
[ci-centos7-sclo-rh]
name=CentOS 7.9.2009 SCLo rh (public mirror)
baseurl=http://mirrors.aliyun.com/centos-vault/7.9.2009/sclo/x86_64/rh/
enabled=1
gpgcheck=0
REPO
            # The alios7 image mixes its own packages with CentOS 7 ones: libpng.x86_64 is
            # ...alios7 while libpng.i686 is plain el7, so yum compares the 32-bit halves against
            # the repositories and aborts before downloading anything with
            #   Protected multilib versions: libpng-2:1.6.37-12.el7.i686 != libpng-...x86_64
            # Nothing here needs 32-bit code, so drop the i686 candidates; if a dependency still
            # insists on one, fall back to yum's own escape hatch for the same check.
            if ! yum install -y --setopt=timeout=60 --setopt=retries=3 --exclude='*.i686' \
                    "${toolset}-gcc-c++" > /tmp/devtoolset-install.log 2>&1; then
                echo "::warning::${toolset}-gcc-c++ did not install cleanly:"
                tail -n 20 /tmp/devtoolset-install.log || true
                # Anonymous access to these job logs is not possible, so surface the reason here
                # instead of leaving an unexplained ::error:: at the end of the loop.
                if ! install_toolset_from_rpms "${toolset}"; then
                    tail -n 20 /tmp/devtoolset-install.log || true
                fi
            fi
        fi
        if [ -f "/opt/rh/${toolset}/enable" ]; then
            # shellcheck disable=SC1090
            source "/opt/rh/${toolset}/enable"
            hash -r
            printf 'int main() { return 0; }\n' > "${probe}.cc"
            if g++ -std=c++14 -o "${probe}" "${probe}.cc" >/dev/null 2>&1; then
                rm -f "${probe}" "${probe}.cc"
                echo "using ${toolset}: $(g++ --version | head -n 1)"
                echo "assembler $(command -v as): $(as --version 2>/dev/null | head -n 1)"
                return 0
            fi
            rm -f "${probe}" "${probe}.cc"
        fi
    done
    echo "::error::no compiler on this image accepts -std=c++14; the build cannot start"
    return 1
}

# The PPU SDK tools (hgcc, clang-13, lld, hglink) are built with gcc >= 5 and ask the loader for
# GLIBCXX_3.4.22 / CXXABI_1.3.11, but a CentOS/RHEL 7 based image ships gcc 4.8.5's libstdc++,
# which stops at GLIBCXX_3.4.19, so hgcc dies while loading:
#   .../PPU_SDK/bin/hgcc: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found
# and the makefile reports it as a missing/broken compiler.
# No repository of that image carries a newer shared libstdc++ (the SCL devtoolset ships static
# archives only), so take conda-forge's libstdcxx-ng: it targets the same glibc 2.17, provides
# GLIBCXX_3.4.30/CXXABI_1.3.13, and is only injected through LD_LIBRARY_PATH, so the linker
# still resolves -lstdc++ against /usr/lib64 and the tarballs keep the old ABI requirement.
ensure_modern_libstdcxx() {
    local system_lib=/usr/lib64/libstdc++.so.6
    local runtime_dir=/opt/ppu-libstdcxx
    local sdk_probe="${source_home}/PPU_SDK/bin/hgcc"
    local newest
    [ -f "${system_lib}" ] || return 0
    # ensure_cxx14_compiler may already have put a newer libstdc++ on the loader path (devtoolset
    # ships one under /opt/rh), so only download one when the SDK tools still cannot start.
    if [ -x "${sdk_probe}" ] && "${sdk_probe}" --version >/dev/null 2>&1; then
        echo "${sdk_probe} starts with the current loader path; no extra libstdc++ needed"
        return 0
    fi
    newest=$(grep -ao 'GLIBCXX_3\.4\.[0-9]*' "${system_lib}" | sed 's/.*\.//' | sort -n | tail -n 1)
    if [ -z "${newest}" ] || [ "${newest}" -ge 20 ]; then
        return 0
    fi
    echo "system libstdc++ stops at GLIBCXX_3.4.${newest}; the PPU SDK tools need 3.4.22"
    if [ ! -f "${runtime_dir}/libstdc++.so.6" ]; then
        command -v bzip2 >/dev/null 2>&1 || \
            yum install -y --setopt=timeout=60 --setopt=retries=3 bzip2
        wget -nv --no-check-certificate --timeout=120 --tries=3 -O /tmp/libstdcxx-ng.tar.bz2 \
            "https://conda.anaconda.org/conda-forge/linux-64/libstdcxx-ng-12.2.0-h46fd767_19.tar.bz2" || {
            echo "::error::could not download a libstdc++ providing GLIBCXX_3.4.20+"
            return 1
        }
        rm -rf /tmp/libstdcxx-ng && mkdir -p /tmp/libstdcxx-ng "${runtime_dir}"
        tar xjf /tmp/libstdcxx-ng.tar.bz2 -C /tmp/libstdcxx-ng lib/libstdc++.so.6.0.30
        cp -f /tmp/libstdcxx-ng/lib/libstdc++.so.6.0.30 "${runtime_dir}/libstdc++.so.6"
        chmod 0755 "${runtime_dir}/libstdc++.so.6"
    fi
    export LD_LIBRARY_PATH="${runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "prepended ${runtime_dir} to LD_LIBRARY_PATH (GLIBCXX up to $(grep -ao 'GLIBCXX_3\.4\.[0-9]*' "${runtime_dir}/libstdc++.so.6" | sed 's/.*\.//' | sort -n | tail -n 1))"
    if [ -x "${sdk_probe}" ] && ! "${sdk_probe}" --version >/dev/null 2>&1; then
        echo "::error::${sdk_probe} still cannot start:"
        "${sdk_probe}" --version 2>&1 | head -n 5 || true
        return 1
    fi
}

# The PPU SDK installer is a Perl script that starts with `use Tie::File`, and the Alibaba Cloud
# Linux 4 image runs a /usr/local perl 5.36 that ships no such module while the distribution keeps
# it in a vendor directory that 5.36 does not search, so the whole SDK install died with:
#   can't locate Tie/File.pm in @INC (@INC contains: /usr/local/lib/perl5/5.36 ...)
#   BEGIN failed--compilation aborted at ./install-ppu.pl line 5.
# Tie::File is pure Perl, so extending @INC through PERL5LIB -- which every perl binary inherits,
# including the one behind the installer's `#!` line -- is enough; install the distribution package
# only when none of the existing directories has the module.
ensure_sdk_installer_perl() {
    local dir
    local perl_dirs="/usr/share/perl5/vendor_perl /usr/share/perl5 /usr/lib64/perl5/vendor_perl /usr/lib64/perl5 /usr/local/share/perl5 /usr/local/lib64/perl5"
    command -v perl >/dev/null 2>&1 || return 0
    if perl -MTie::File -e 1 >/dev/null 2>&1; then
        return 0
    fi
    echo "this perl cannot load Tie::File, which the PPU SDK installer needs"
    for dir in ${perl_dirs}; do
        if [ -d "${dir}" ]; then
            PERL5LIB="${dir}${PERL5LIB:+:${PERL5LIB}}"
        fi
    done
    if [ -n "${PERL5LIB}" ]; then
        export PERL5LIB
    fi
    if perl -MTie::File -e 1 >/dev/null 2>&1; then
        echo "Tie::File is reachable with PERL5LIB=${PERL5LIB}"
        return 0
    fi
    if command -v yum >/dev/null 2>&1; then
        yum install -y --setopt=timeout=60 --setopt=retries=3 perl-Tie-File \
            > /tmp/perl-tie-file-install.log 2>&1 || true
    fi
    if perl -MTie::File -e 1 >/dev/null 2>&1; then
        echo "Tie::File installed from the distribution"
        return 0
    fi
    echo "::warning::Tie::File is still missing, the SDK installer will probably fail:"
    tail -n 10 /tmp/perl-tie-file-install.log 2>/dev/null || true
}

# An undefined-reference failure at link time leaves just a couple of lines in the pod log, which
# cannot be read without repository admin rights, so print the inputs that decide which PPU
# libraries get linked (src/Makefile gates -llt_oki / -llt_oki_device_* behind HGGC_HAS_OKI, and
# src/collectives/device/Makefile picks the generated symmetric sources up through `-include`, so
# a make run that starts on an empty build/ parses LIB_OBJS_SYM_GEN before rules.mk exists and only
# the restart that follows generate.py fills the list in. When that restart does not happen the
# kernel tables never reach colldevice.a and the last link step of the build reports
#   libpccl.so: undefined reference to `ncclSymkKernelList'
#   libpccl.so: undefined reference to `ncclSymkKernelList_fast'
#   libpccl.so: undefined reference to `ncclSymkKernelCount'
# Generate the sources before the first make run so rules.mk is already there when make parses.
prepare_symmetric_sources() {
    local devdir="${source_home}/pccl/src/collectives/device"
    local gensrc="${source_home}/pccl/build/obj/collectives/device/gensrc/symmetric"
    if [ ! -f "${devdir}/symmetric/generate.py" ] || [ -f "${gensrc}/rules.mk" ]; then
        return 0
    fi
    mkdir -p "${gensrc}"
    if ( cd "${devdir}" && "${PYTHON3_BIN:-python3}" ./symmetric/generate.py "${gensrc}" ); then
        echo "pre-generated $(ls -1 "${gensrc}" | wc -l) symmetric kernel sources"
    else
        # generate.py truncates the directory before filling it, and the device Makefile treats an
        # existing directory as up to date, so leaving a half-written one behind would silently
        # drop every symmetric object; remove it and let make's own rule (and its Python banner)
        # run instead.
        rm -rf "${gensrc}"
        echo "::warning::could not pre-generate the symmetric kernel sources"
    fi
}

# An undefined-reference failure at link time leaves just a couple of lines in the pod log, which
# cannot be read without repository admin rights, so print the inputs that decide which PPU
# libraries get linked (src/Makefile gates -llt_oki / -llt_oki_device_* behind HGGC_HAS_OKI, and
# makefiles/common.mk derives that from `hgcc --version`) as soon as the build fails.
dump_pccl_link_diagnostics() {
    local archive
    local hgcc_bin="${source_home}/PPU_SDK/bin/hgcc"
    local parsed
    echo "::error::the pccl build failed; collecting link diagnostics"
    if [ -x "${hgcc_bin}" ]; then
        echo "hgcc --version:"
        "${hgcc_bin}" --version 2>&1 | head -n 3 || true
        parsed=$("${hgcc_bin}" --version 2>/dev/null | grep version | sed 's/.*version //' | sed 's/\,.*//')
        echo "version parsed by makefiles/common.mk: '${parsed}'"
    else
        echo "${hgcc_bin} is missing"
    fi
    echo "PPU runtimes in PPU_SDK/lib: $(ls -1 "${source_home}/PPU_SDK/lib" 2>/dev/null | grep -E 'lt_oki|hggc_wrapper|hg_wrapper|hggcrt' | tr '\n' ' ')"
    find "${source_home}" -maxdepth 6 -name colldevice.a 2>/dev/null | while read -r archive; do
        echo "device archive ${archive}: $(ar t "${archive}" 2>/dev/null | wc -l) members, $(ar t "${archive}" 2>/dev/null | grep -c symmetric) of them symmetric"
    done
    local sym_root="${source_home}/pccl/build/obj/collectives/device"
    echo "symmetric gensrc: $(ls -1 "${sym_root}/gensrc/symmetric" 2>/dev/null | wc -l) files under ${sym_root}/gensrc/symmetric"
    echo "symmetric genobj: $(ls -1 "${sym_root}/genobj/symmetric" 2>/dev/null | wc -l) files under ${sym_root}/genobj/symmetric"
}

# The docker_base images are plain OS images: they carry neither the PPU SDK nor the build
# toolchain, so install what the build needs before anything else. Some images point apt at an
# internal-only proxy, hence the mirror rewrite below.
install_build_dependencies() {
    local missing=""
    for tool in wget make gcc g++ cmake git python3; do
        command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
    done
    command -v mpicxx >/dev/null 2>&1 || missing="$missing openmpi"
    if [ -z "$missing" ]; then
        echo "build dependencies already installed"
        return 0
    fi
    echo "missing build dependencies:${missing}"
    if command -v apt-get >/dev/null 2>&1; then
        local codename
        codename=$(. /etc/os-release && echo "${VERSION_CODENAME:-}")
        if [ -n "${codename}" ] && grep -Rqs "art.eng.t-head.cn" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null; then
            echo "Replacing art.eng.t-head.cn apt sources with mirrors.aliyun.com"
            rm -f /etc/apt/sources.list.d/*.sources
            printf '%s\n' \
                "deb http://mirrors.aliyun.com/ubuntu/ ${codename} main restricted universe multiverse" \
                "deb http://mirrors.aliyun.com/ubuntu/ ${codename}-updates main restricted universe multiverse" \
                "deb http://mirrors.aliyun.com/ubuntu/ ${codename}-backports main restricted universe multiverse" \
                "deb http://mirrors.aliyun.com/ubuntu/ ${codename}-security main restricted universe multiverse" \
                > /etc/apt/sources.list
        else
            if [ -f /etc/apt/sources.list ]; then
                sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list
                sed -i 's|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list
            fi
            for f in /etc/apt/sources.list.d/*.sources; do
                [ -f "$f" ] || continue
                sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' "$f"
                sed -i 's|security.ubuntu.com|mirrors.aliyun.com|g' "$f"
            done
        fi
        # apt pulls in tzdata, whose postinst asks debconf for a geographic area. The worker pod has
        # no TTY, so debconf falls back to the Readline frontend and blocks on stdin forever
        # ("Configuring tzdata / Please select the geographic area in which you live"), which is how
        # the ubuntu2004 jobs sat there for 1h26m until they were cancelled. Force the
        # noninteractive frontend, pre-seed the timezone and keep stdin off /dev/null for every apt
        # call so no maintainer script can ever prompt.
        export DEBIAN_FRONTEND=noninteractive
        export TZ="${TZ:-Etc/UTC}"
        echo "Etc/UTC" > /etc/timezone
        ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime
        # A plain `apt-get update` blocks forever when a mirror stops answering; bound every
        # transfer, force IPv4 and cap the whole run so a stuck mirror fails fast instead of hanging.
        timeout -k 30 900 apt-get -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 \
            -o Acquire::Retries=3 -o Acquire::ForceIPv4=true update </dev/null
        # ca-certificates is needed for the HTTPS download of the SDK below (keep the trust store
        # current, the bundled one predates the issuer of download.t-head.cn's certificate), and
        # python3 is required by build.sh through argparse.bash: without it the build aborts with
        # "Neither python nor python3 is installed on this system".
        timeout -k 30 1800 apt-get -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 \
            -o Acquire::Retries=3 -o Acquire::ForceIPv4=true \
            -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
            install -y --no-install-recommends wget make gcc g++ cmake git ca-certificates python3 libopenmpi-dev openmpi-bin tzdata </dev/null
        update-ca-certificates >/dev/null 2>&1 || true
    elif command -v yum >/dev/null 2>&1; then
        prepare_yum_sources
        # `python3` is not a package name on alinux3 (RHEL 8 based: the interpreter only exists as
        # /usr/libexec/platform-python), so naming it in the toolchain line made yum abort with
        # "No package python3 available" before a single tool was installed. Install the
        # toolchain first, then resolve the interpreter separately.
        # `tar`/`gzip` are missing from alinux3's minimal image too, and both the source staging of
        # the worker pod and the final tarball packaging need them.
        yum install -y --setopt=timeout=60 --setopt=retries=3 \
            tar gzip wget make gcc gcc-c++ cmake git ca-certificates openmpi-devel
        if ! yum install -y python3 >/dev/null 2>&1 && ! yum install -y python36 >/dev/null 2>&1; then
            echo "python3/python36 are not packaged here; falling back to the bundled interpreter"
        fi
        update-ca-trust >/dev/null 2>&1 || true
        # RHEL-family images keep the MPI wrappers outside the default PATH.
        export PATH="${PATH}:/usr/lib64/openmpi/bin:/usr/mpi/gcc/openmpi/bin"
    else
        echo "Error: no supported package manager is available to install build dependencies"
        exit 1
    fi
    # Report here, rather than 10 minutes into the build, what would break it later on. The test is
    # deliberately about `python3` and about Python 3 itself: see the note on expose_python3.
    expose_python3 || true
    if [ -z "${PYTHON3_BIN:-}" ]; then
        echo "::error::no python3 on PATH, but pccl's device Makefile runs generate.py through 'python3'"
        exit 1
    fi
    echo "python3 -> ${PYTHON3_BIN} ($("${PYTHON3_BIN}" --version 2>&1))"
    command -v mpicxx >/dev/null 2>&1 || echo "::warning::mpicxx is still missing after installing the toolchain"
}

echo "######################### $(date +%Y%m%d%H%M%S) Install build dependencies ##########################"
install_build_dependencies

ensure_which

echo "######################### $(date +%Y%m%d%H%M%S) Ensure a -std=c++14 aware compiler ##########################"
ensure_cxx14_compiler || exit 1

echo "######################### $(date +%Y%m%d%H%M%S) Source PPU_SDK ##########################"
# The plain OS images carry no PPU SDK, so the .run package (~1.2 GB) has to be downloaded and
# installed here. Both commands used to be silent (`wget -q` plus `--silent`), so a failure at this
# point left no trace in the job log. Keep the transfer and the installer visible: download to the
# pod's local disk instead of the NAS-backed source tree and print size / installer output on error.
sdk_package="/tmp/${sdk_name}"
sdk_install_log="/tmp/${sdk_name}.log"
echo "SDK package: ${sdk_name}"
echo "SDK url    : ${sdk_url}"
df -h /tmp "${source_home}" 2>/dev/null || true

rm -f "${sdk_package}"
# The image trust store cannot verify download.t-head.cn: its chain is issued by "GlobalSign GCC
# R46 OV TLS CA 2025", which the bundled bundle misses, so wget aborted with "unable to locally
# verify the issuer's authority". The host is our internal artifactory and the payload is
# size-checked below, so skip chain validation rather than fail the build on it.
# Bound every phase of the transfer as well: an unbounded wget on a stalled connection turns a
# slow download into a silent 180-minute job timeout.
wget -nv --no-check-certificate --timeout=120 --dns-timeout=60 --connect-timeout=30 \
    --read-timeout=180 --tries=3 -O "${sdk_package}" "${sdk_url}" || {
    echo "::error::failed to download ${sdk_url}"
    exit 1
}
sdk_bytes=$(stat -c %s "${sdk_package}" 2>/dev/null || echo 0)
echo "downloaded ${sdk_bytes} bytes to ${sdk_package}"
if [ "${sdk_bytes}" -lt 100000000 ]; then
    echo "::error::${sdk_name} is truncated (${sdk_bytes} bytes)"
    exit 1
fi

ensure_sdk_installer_perl

# The package is a makeself archive that unpacks ~4.6 GB into $TMPDIR and then runs
# install-ppu.pl. Anything that asks for input would block forever on the worker's stdin, so
# detach stdin and cap the step instead of relying on the 180-minute job timeout.
set +e
timeout -k 60 2400 bash "${sdk_package}" --prefix="${source_home}" --silent </dev/null >"${sdk_install_log}" 2>&1
sdk_rc=$?
set -e
if [ "${sdk_rc}" -eq 124 ] || [ "${sdk_rc}" -eq 137 ]; then
    echo "::error::installing ${sdk_name} timed out after 2400s"
    df -h /tmp "${source_home}" 2>/dev/null || true
    tail -n 80 "${sdk_install_log}" 2>/dev/null || true
    exit 1
fi
if [ "${sdk_rc}" -ne 0 ]; then
    echo "::error::installing ${sdk_name} failed (exit ${sdk_rc})"
    tail -n 80 "${sdk_install_log}" 2>/dev/null || true
    exit "${sdk_rc}"
fi
if [ ! -f "${source_home}/PPU_SDK/envsetup.sh" ]; then
    echo "::error::${source_home}/PPU_SDK/envsetup.sh is missing after installing ${sdk_name}"
    ls -la "${source_home}" 2>&1 || true
    tail -n 80 "${sdk_install_log}" 2>/dev/null || true
    exit 1
fi
source ${source_home}/PPU_SDK/envsetup.sh
# hgcc is built against a newer libstdc++ than this image ships; fix it before the build.
ensure_modern_libstdcxx || exit 1
cd ${source_home}/pccl
# build.sh --rebuild deletes build/ itself and only then starts make, which is why the symmetric
# kernel sources have to be generated up front: do the cleanup here and hand build.sh a clean tree
# that already carries them (see prepare_symmetric_sources).
make clean >/dev/null 2>&1 || true
rm -rf build
prepare_symmetric_sources
# The SDK ships prebuilt C++ libraries (libhg_wrapper.so) that reference GLIBCXX_3.4.21, while
# alios7's /usr/lib64/libstdc++.so.6 stops at GLIBCXX_3.4.19 and the SCL devtoolset brings no
# shared libstdc++ of its own, so -lstdc++ still resolves against the old system copy and ld
# rejects every 3.4.21 symbol the prebuilt library needs:
#   libhg_wrapper.so: undefined reference to `std::__cxx11::basic_stringstream<...>@GLIBCXX_3.4.21'
# A host that runs these libraries needs a libstdc++ new enough for the SDK anyway, so let the link
# step trust the shared libraries instead of failing on them; the objects compiled here keep their
# full symbol checking.
export LDFLAGS="${LDFLAGS:+${LDFLAGS} }-Wl,--allow-shlib-undefined"
set +e
if [ -n "${package_version}" ]; then
    echo "Building with package version: ${package_version}"
    bash build.sh --use_sdk --package_version ${package_version} 2>&1 | tee /tmp/pccl-build.log
else
    echo "Building without package version (default)"
    VERSION_FILE="${source_home}/pccl/makefiles/version.mk"
    PCCL_MAJOR=$(grep "^PCCL_MAJOR" ${VERSION_FILE} | sed 's/.*:=\s*//')
    PCCL_MINOR=$(grep "^PCCL_MINOR" ${VERSION_FILE} | sed 's/.*:=\s*//')
    PCCL_PATCH=$(grep "^PCCL_PATCH" ${VERSION_FILE} | sed 's/.*:=\s*//')
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    GENERATED_VERSION="${PCCL_MAJOR}.${PCCL_MINOR}.${PCCL_PATCH}-${TIMESTAMP}"
    bash build.sh --use_sdk --package_version ${GENERATED_VERSION} 2>&1 | tee /tmp/pccl-build.log
fi
build_rc=${PIPESTATUS[0]}
set -e
if [ "${build_rc}" -ne 0 ]; then
    # The symmetric kernel tables only reach colldevice.a when generate.py ran before make parsed
    # the device Makefile; surface what the build itself said about that step.
    grep -nE "Generating |pre-generated|rules\.mk|python3|ncclSymk" /tmp/pccl-build.log 2>/dev/null \
        | tail -n 30 || true
    dump_pccl_link_diagnostics
    exit "${build_rc}"
fi

# A -shared link accepts undefined symbols, so a library that lost the generated symmetric kernels
# still links and only its consumers fail (pcclras here, the *_perf binaries in pccl-tests). Check
# the library itself so a green run cannot ship one without the kernel tables.
assert_symmetric_tables() {
    local lib="${source_home}/pccl/build/lib/libpccl.so"
    local sym
    if [ ! -e "${lib}" ] || ! command -v nm >/dev/null 2>&1; then
        echo "::warning::cannot verify ${lib} (missing library or nm)"
        return 0
    fi
    for sym in ncclSymkKernelList ncclSymkKernelList_fast ncclSymkKernelCount; do
        if ! nm -D --defined-only "${lib}" 2>/dev/null | grep -qw "${sym}"; then
            echo "::error::${lib} does not define ${sym}; the generated symmetric kernels never reached colldevice.a"
            dump_pccl_link_diagnostics
            exit 1
        fi
    done
    echo "libpccl.so defines the symmetric kernel tables"
}
assert_symmetric_tables

mkdir -p ${build_output}/comm_internal/pccl
mkdir -p ${build_output}/comm_release/pccl
mkdir -p ${build_output}/comm_internal/pccl/lib
mkdir -p ${build_output}/comm_release/pccl/lib

find ${source_home}/pccl/build/lib -name "lib*.so*" ! -name "*.dbginfo" | xargs -i cp -arv {} ${build_output}/comm_internal/pccl/lib/
find ${source_home}/pccl/build/lib -name "lib*.so*" ! -name "*.dbginfo" | xargs -i cp -arv {} ${build_output}/comm_release/pccl/lib/

mkdir -p ${build_output}/comm_internal/pccl/include
mkdir -p ${build_output}/comm_release/pccl/include

cp -arv ${source_home}/pccl/build/include/* ${build_output}/comm_internal/pccl/include/
cp -arv ${source_home}/pccl/build/include/* ${build_output}/comm_release/pccl/include/

mkdir -p ${build_output}/comm_internal/pccl/bin
mkdir -p ${build_output}/comm_release/pccl/bin

cp -arv ${source_home}/pccl/build/bin/pcclras ${build_output}/comm_internal/pccl/bin/
cp -arv ${source_home}/pccl/build/bin/pcclras ${build_output}/comm_release/pccl/bin/
cp -arv ${source_home}/pccl/envsetup.sh ${build_output}/comm_internal/pccl
cp -arv ${source_home}/pccl/envsetup.sh ${build_output}/comm_release/pccl

cp ${source_home}/pccl/LICENSE.txt ${build_output}/comm_internal/pccl/LICENSE.txt
cp ${source_home}/pccl/LICENSE.txt ${build_output}/comm_release/pccl/LICENSE.txt

cp ${source_home}/pccl/build/release.yaml ${build_output}/comm_internal/pccl/release.yaml
cp ${source_home}/pccl/build/release.yaml ${build_output}/comm_release/pccl/release.yaml

cd ${build_output}/comm_internal/ 
tar -zcvf ${build_output}/comm_internal/"${pkg_name}.tar.gz" *
ls | grep -v "${pkg_name}.tar.gz" | xargs rm -rf
cd ${build_output}/comm_release/ 
tar -zcvf ${build_output}/comm_release/"${pkg_name}.tar.gz" *
ls | grep -v "${pkg_name}.tar.gz" | xargs rm -rf
echo "BUILD PCCL FINISHED"

# pccl-tests ships from this same run: the perf binaries are linked against the libpccl.so built a
# few lines above, so packaging both here guarantees they belong to the same commit, SDK and
# toolchain instead of two workflows that can drift apart. The stage reuses ${source_home}/pccl/build
# rather than rebuilding pccl, and it is skipped (with a warning) when the job did not check
# pccl-tests out, which keeps this script usable on its own.
build_pccl_tests_variant() {
    local log="/tmp/$1.log"
    local rc
    shift
    set +e
    bash build.sh "$@" 2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "${rc}" -ne 0 ]; then
        grep -nE "undefined reference|multiple definition|error:|Error [0-9]|No such file" "${log}" 2>/dev/null | tail -n 40 || true
        exit "${rc}"
    fi
}

package_pccl_tests() {
    local repo="${source_home}/pccl-tests"
    local bins="all_gather_perf all_reduce_perf broadcast_perf reduce_scatter_perf reduce_perf"
    local component bin
    bins="${bins} alltoall_perf gather_perf scatter_perf sendrecv_perf hypercube_perf"
    if [ ! -f "${repo}/build.sh" ]; then
        echo "::warning::${repo} is not checked out, skipping the pccl-tests tarball"
        return 0
    fi
    if [ ! -f "${source_home}/pccl/build/lib/libpccl.so" ]; then
        echo "::error::${source_home}/pccl/build/lib/libpccl.so is missing, pccl-tests cannot link against this build"
        exit 1
    fi
    echo "######################### $(date +%Y%m%d%H%M%S) Build ${pccl_tests_pkg_name} ##########################"
    export PCCL_HOME="${source_home}/pccl/build"
    # The perf binaries are linked by hgcc, which never reads LDFLAGS, so the GLIBCXX exemption the
    # library link above relies on has to be handed to the driver explicitly. hgcc rejects the -Wl,
    # spelling ("unknown argument"), hence -Xlinker; src/Makefile only appends to HGLDFLAGS, which
    # keeps the value coming from here.
    export HGLDFLAGS="${HGLDFLAGS:+${HGLDFLAGS} }-Xlinker --allow-shlib-undefined"
    for component in comm_internal comm_release; do
        mkdir -p "${build_output}/${component}/comm_tools/single_process"
        mkdir -p "${build_output}/${component}/comm_tools/multi_process"
        mkdir -p "${build_output}/${component}/comm_tools/licenses"
    done
    cd "${repo}"
    build_pccl_tests_variant pccl-tests-build --rebuild
    for component in comm_internal comm_release; do
        for bin in ${bins}; do
            cp "${repo}/build/${bin}" "${build_output}/${component}/comm_tools/single_process/"
        done
    done
    build_pccl_tests_variant pccl-tests-mpi-build --rebuild --use_mpi
    for component in comm_internal comm_release; do
        for bin in ${bins}; do
            cp "${repo}/build/${bin}" "${build_output}/${component}/comm_tools/multi_process/"
        done
        cp "${repo}/LICENSE.txt" "${build_output}/${component}/comm_tools/licenses/nccl-tests-LICENSE.txt"
        # Archive only this stage's own subtree: the pccl tarball already sits next to it in the
        # same directory, and the pccl-tests script that used to own this output directory worked
        # with `tar *` plus a cleanup that would have deleted it.
        ( cd "${build_output}/${component}" && tar -zcvf "${pccl_tests_pkg_name}.tar.gz" comm_tools )
        rm -rf "${build_output}/${component}/comm_tools"
        if [ ! -f "${build_output}/${component}/${pccl_tests_pkg_name}.tar.gz" ]; then
            echo "::error::${build_output}/${component}/${pccl_tests_pkg_name}.tar.gz was not produced"
            exit 1
        fi
    done
    echo "BUILD PCCL-TESTS FINISHED"
}

package_pccl_tests
