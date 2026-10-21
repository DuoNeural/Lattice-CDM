#!/usr/bin/env bash
# Local compile-time compatibility tools for CUDA 12.5 on this Arch host.
# No system packages or runtime libc files are changed.
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p toolchains
stage() {
  local name="$1" url="$2" checksum="$3"
  if [[ ! -f "toolchains/$name.pkg.tar.zst" ]]; then
    curl -fL --retry 2 "$url" -o "toolchains/$name.pkg.tar.zst"
  fi
  printf '%s  %s\n' "$checksum" "toolchains/$name.pkg.tar.zst" | sha256sum -c -
  if [[ ! -d "toolchains/$name/usr" ]]; then
    mkdir -p "toolchains/$name"
    tar -xf "toolchains/$name.pkg.tar.zst" -C "toolchains/$name"
  fi
}
stage gcc13 https://archive.archlinux.org/packages/g/gcc/gcc-13.2.1-6-x86_64.pkg.tar.zst dd14d1b971f90ef764a1ea9e5605a9a3518a34f2636d559dce075402528958dc
stage glibc239 https://archive.archlinux.org/packages/g/glibc/glibc-2.39-1-x86_64.pkg.tar.zst 17ff75d9476c690f50911a6c23d844ffc760471ae5e8e69b6a1dcde86113332a
# Current runtime libraries are ABI-compatible; GCC moved these linker names.
ln -sfn /usr/lib/libgcc_s.so.1 toolchains/gcc13/usr/lib/libgcc_s.so
ln -sfn /usr/lib/libstdc++.so.6 toolchains/gcc13/usr/lib/libstdc++.so
