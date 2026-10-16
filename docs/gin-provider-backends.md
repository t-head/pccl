# GIN provider backends

The built-in IB GIN uses the existing `PCCL_GIN_TYPE` setting. Its default,
`-1`, chooses a built-in GPU-direct backend for a recognized, homogeneous
provider and otherwise tries the verbs proxy. Explicit values `2`, `3`, and
`4` select proxy, MLX5 GDA, and SNIC GDA respectively. A requested GDA backend
must have matching devices and GPU-memory support; otherwise initialization
fails.

The existing compile-time macro `PCCL_GIN_GDAKI_ENABLE` defaults to `1` and
controls all built-in GDA device implementations, including MLX5 and SNIC.
Setting it to `0` excludes both providers from device implementation includes,
the default backend mask, and device dispatch. `PCCL_GIN_PROXY_ENABLE` remains
independent. These macros control application device code compilation;
runtime backend selection must match the implementations compiled in.

The verbs device name is mapped to a provider in `net_ib/gin_provider.cc`.
The built-in registry associates each GPU-direct provider with its host GIN
implementation, device type and memory-registration requirement. The common
IB GIN connection code does not construct provider-specific QPs or GPU WQEs:
those stay in `net_ib/providers/mlx5/` and `net_ib/providers/troce/`.
The existing `GDAKI` type and symbol names are kept for ABI
compatibility; GDA is the broader operating mode, not an MLX5-only concept.

The host and device trees use matching provider directories:

```text
src/transport/net_ib/
  gin.cc, gin.h                     common IB GIN connection and lifecycle
  gin_provider.cc, gin_provider.h   provider registry and device mapping
  providers/mlx5/                  MLX5 host adapter, resources, DOCA-GPUNetIO
  providers/troce/                 SNIC host adapter, resources, troce wrapper

src/include/pccl_device/gin/
  gin_device_api.h                 includes the enabled backend implementations
  gin_device_common.h              common API templates and descriptors
  gin_device_host_common.h         provider-independent host/device types
  providers/mlx5/                  MLX5 device API and host/device context layout
  providers/troce/                 SNIC device API, context layout, WQE/CQE format
  proxy/                          generic proxy device implementation
```

QP/CQ layouts, doorbells, completion progress and memory-registration
constraints belong to the provider that implements them. In particular,
`gin_mlx5_device_host_common.h` and `gin_troce_device_host_common.h` are shared
between the host and device halves of one provider; they are not common
cross-provider ABIs. The generic layer defines signal/counter semantics,
while each provider implements their progress and ordering requirements.
The existing device template specializations and backend identifiers are
unchanged by this directory layout. Applications continue to include
`pccl_device.h`; direct includes of backend implementation headers must use
the new provider paths.

| Provider | Recognized verbs names | Built-in GPU-direct backend | Auto mode |
| --- | --- | --- | --- |
| MLX5 | `mlx5*`, `ibp*` | MLX5 GDA | GDA |
| SNIC | `troce*` | SNIC GDA | GDA |
| EIC | `fic2*`, `vsolar*` | None in this tree | Proxy, when supported |
| TSOE | `tsoe*` | None in this tree | Proxy, when supported |
| Unknown | Other names | None | Proxy, when supported |

An external GIN plugin retains its existing selection priority and does not
use the built-in IB registry. Thus a closed EIC plugin can
continue to provide its own implementation. Adding a built-in GPU-direct
provider requires a host backend, a device API backend with a distinct device
type, and a registry entry; provider detection alone never enables GPU-direct
operation. A communicator
whose selected IB devices have different providers uses proxy in auto mode,
because the built-in GDA backends do not share a mixed-provider connection
contract.

GIN connections retain the existing topology-based device selection and
`PCCL_GIN_NCONNECTIONS` setting. Connections are assigned round-robin over
the devices selected by the topology, with the existing
`NCCL_GIN_MAX_CONNECTIONS` limit.
