# Third-party dependencies

The native normalizer statically links the dependencies below. Their source is
fetched at the immutable revision or archive checksum declared in
`WORKSPACE.bazel`; no third-party source archive is committed here.

| Component | Pinned revision/version | License |
|---|---|---|
| Abseil | 20220623.1 | Apache-2.0 |
| gflags | 2.2.2 | BSD-3-Clause |
| glog | 0.4.0 | BSD-3-Clause |
| gRPC build dependencies | 1.50.1 | Apache-2.0 |
| GoogleTest | 1.11.0 | BSD-3-Clause |
| OpenFst | `fc23b4cf529429284b874a26f28b15c6cc94f404` | Apache-2.0 |
| protobuf | version selected by gRPC 1.50.1 | BSD-3-Clause |
| pugixml | 1.11.4 | MIT |
| RE2 | `ccdbd348259434cca6c66c66a35c5ef71ff3ee1a` | BSD-3-Clause |
| Sparrowhawk | `d9d68e7155b2b2cd64f578144c32b4e09674f43d` | Apache-2.0 |
| Thrax | `21407b7ccbd5dbaa45f34b26f286bc28d57c47fb` | Apache-2.0 |

Boost and zlib headers/libraries are supplied by the Ubuntu 22.04 builder under
their respective Boost Software License 1.0 and zlib license. The original
license files remain authoritative and are present in the fetched source or
builder packages.
