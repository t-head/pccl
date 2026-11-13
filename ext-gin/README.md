# Bundled GIN plugins

GIN host plugins are built with PCCL and placed in `build/lib/`.
`troce/` exports `pcclGinPlugin_v13` from `libpccl-gin-troce.so`.

These plugins reuse PCCL internal IB helpers and must be built and shipped with
that PCCL version. They do not provide an independent cross-version ABI for
those internal dependencies. Shared IB device state remains owned by `libpccl`.
Device implementations remain in the installed PCCL device headers.

The default search order is `libpccl-gin.so`, `libpccl-gin-troce.so`,
GIN support from the NET plugin, then internal IB. An explicit `PCCL_GIN_PLUGIN`
list replaces the default library candidates; the existing NET discovery
remains active. `PCCL_GIN_PLUGIN=troce` selects the TROCE library explicitly.
The library must be on the dynamic loader search path, as with other plugins.
