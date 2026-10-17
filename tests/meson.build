# meson.build
project('pccl-internal-unit-tests', 'cpp')

fs = import('fs')

ppu_archs = get_option('ppu_archs')

# 定义编译选项
cpp_args = ['-std=c++17',
            '-Wno-unused-function',
            '-Wno-unsed-parameter',
            '-Wno-sign-compare',
            '-DPPU=1',
            # '-DENABLE_TRACE',
            # '-DDEBUG_SEARCH',
            ]

cmd = run_command('sh', '-c', 'echo $PPU_SDK')
PPU_SDK = cmd.stdout().strip()

gtest_home ='../3rd_party/gtest'

# Ensure that local built libpccl.so is linked first
local_pccl_so_path = meson.project_build_root() + '/../../build/lib/libpccl.so.2'
if fs.is_file(local_pccl_so_path)
    pccl_link_args = [local_pccl_so_path, '-Wl,--disable-new-dtags', '-Wl,-rpath,$ORIGIN/../../build/lib']
else
    pccl_link_args = ['-lpccl']
endif

# Add include paths
incdirs = include_directories(
  gtest_home + '/include',
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
  link_args: pccl_link_args + ['-lrt', '-ldl', '-L'+gtest_home+'/lib', '-lgtest', '-L'+PPU_SDK+'/lib', '-lhggcrt1', '-lhggc', '-pthread'],
  cpp_args: cpp_args )

pxn_peer_selection = executable('pxnPeerSelectionTest',
  './src/pxn_peer_selection_tests.cc',
  include_directories: incdirs,
  link_args: pccl_link_args + ['-lrt', '-ldl', '-L'+PPU_SDK+'/lib', '-lhggcrt1', '-lhggc', '-pthread'],
  cpp_args: cpp_args)
test('pxnPeerSelection', pxn_peer_selection,
  args: files('topo_and_reference_graph/8PPU1.0_4CX7_2PIXPPUsPerNET_OAM_4Links_PXN/topo.xml'))

# Install the executable
install_dir = join_paths(get_option('prefix'), 'bin')
