# meson.build
project('pccl-internal-unit-tests', 'cpp', default_options: ['cpp_std=c++17'])

fs = import('fs')

ppu_archs = get_option('ppu_archs')

# 定义编译选项
cpp_args = ['-Wno-unused-function',
            '-Wno-unused-parameter',
            '-Wno-sign-compare',
            '-DPPU=1',
            # '-DENABLE_TRACE',
            # '-DDEBUG_SEARCH',
            ]

cmd = run_command('sh', '-c', 'echo "$PPU_SDK"', check: true)
PPU_SDK = cmd.stdout().strip()
if PPU_SDK == ''
    error('Source /usr/local/PPU_SDK/envsetup.sh before building tests')
endif

# Internal structures must use the same SDK feature guards as libpccl.
sdk_version = run_command('sh', '-c', '''
version=$("$PPU_SDK/bin/hgcc" --version 2>/dev/null | sed -n 's/.*version \([^,]*\).*/\1/p' | head -n1)
if [ -z "$version" ]; then
    version=$(echo "$PPU_SDK" | sed -n 's|.*/PPU_SDK_v*\([0-9][0-9.]*\).*|\1|p')
fi
# Match makefiles/common.mk for older SDKs installed at an unversioned path.
if [ -z "$version" ]; then
    version=0.0.0
fi
echo "$version" | sed 's/[^0-9.].*//'
''', check: true).stdout().strip().split('.')
if sdk_version.length() < 2
    error('Cannot determine HGGC version for internal test ABI')
endif
sdk_major = sdk_version[0].to_int()
sdk_minor = sdk_version[1].to_int()
sdk_patch = sdk_version.length() > 2 ? sdk_version[2].to_int() : 0
cpp_args += ['-DHGGC_MAJOR=' + sdk_major.to_string(),
             '-DHGGC_MINOR=' + sdk_minor.to_string(),
             '-DPCCL_HGGC_VERSION=' + (sdk_major * 10000 + sdk_minor * 100 + sdk_patch).to_string()]

repo_root = meson.project_source_root() / '..'
gtest_home = repo_root / '3rd_party/gtest'

# Ensure that local built libpccl.so is linked first
local_pccl_so_path = repo_root / 'build/lib/libpccl.so'
if fs.is_file(local_pccl_so_path)
    pccl_link_args = [local_pccl_so_path, '-Wl,--disable-new-dtags', '-Wl,-rpath,' + repo_root / 'build/lib']
else
    error('Build PCCL in this worktree before building graphSearchTest')
endif

# Add include paths
incdirs = include_directories(
  '../build/include',
  '../3rd_party/gtest/include',
  '../src/include',
  '../src/include/drv_eit',
  '../src/include/plugin',
  '../src/graph',
  '../src',
  './',
  PPU_SDK + '/include'
)

# Define the sources
sources = [
  './src/graph_search_tests.cc',
]
add_project_arguments(cpp_args, language: 'cpp')
# Define the executable
executable('graphSearchTest',
  sources,
  include_directories: incdirs,
  link_args: pccl_link_args + ['-lrt', '-ldl', gtest_home / 'lib/libgtest.a', '-L'+PPU_SDK+'/lib', '-lhggcrt1', '-lhggc', '-pthread'],
  cpp_args: cpp_args )

# Install the executable
install_dir = join_paths(get_option('prefix'), 'bin')
