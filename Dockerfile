FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git ocl-icd-libopencl1 && rm -rf /var/lib/apt/lists/*

# GPU user space only, same versions as the host. No oneAPI toolkit in this image.
RUN set -eux; mkdir /tmp/gpu; cd /tmp/gpu; \
    CR=https://github.com/intel/compute-runtime/releases/download/26.35.39758.10; \
    IGC=https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5; \
    LZ=https://github.com/oneapi-src/level-zero/releases/download/v1.32.0; \
    for u in $CR/intel-opencl-icd_26.35.39758.10-0_amd64.deb $CR/libze-intel-gpu1_26.35.39758.10-0_amd64.deb $CR/intel-ocloc_26.35.39758.10-0_amd64.deb $CR/libigdgmm12_22.10.0_amd64.deb "$IGC/intel-igc-core-2_2.41.5+22716_amd64.deb" "$IGC/intel-igc-opencl-2_2.41.5+22716_amd64.deb" "$LZ/libze1_1.32.0+u24.04_amd64.deb"; do curl -fsSLO --retry 3 "$u"; done; \
    printf '%s\n' \
      '61712caaddeba3d38e4f79e2a0fb23fea25596ca2d72c3144c6eea2331ec4301  intel-opencl-icd_26.35.39758.10-0_amd64.deb' \
      'c19a641b953d55aebbf1d51bec364a84bf629f985e02fbbe6dc70224c0e88470  libze-intel-gpu1_26.35.39758.10-0_amd64.deb' \
      'c64bff586edf2bd9b49f3e4c9c2be25e05c43466e4dafa2a2c3948d498c736b7  intel-ocloc_26.35.39758.10-0_amd64.deb' \
      '6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54  libigdgmm12_22.10.0_amd64.deb' \
      '0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628  intel-igc-core-2_2.41.5+22716_amd64.deb' \
      '779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058  intel-igc-opencl-2_2.41.5+22716_amd64.deb' \
      '3c846af24f84a89150f6a4c6adcb4ea4ebef74dc119fe44f4e269bfaa72c7ba6  libze1_1.32.0+u24.04_amd64.deb' | sha256sum -c -; \
    apt-get update; apt-get install -y --no-install-recommends ./*.deb; rm -rf /tmp/gpu /var/lib/apt/lists/*

# Python environment from the hash lock.
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN curl -fsSL https://github.com/astral-sh/uv/releases/download/0.13.6/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz --strip-components=1 -C /usr/local/bin
COPY requirements-xpu.lock /tmp/requirements-xpu.lock
RUN uv venv /opt/venv --python 3.11 && uv pip sync --python /opt/venv/bin/python --require-hashes \
      --index-strategy unsafe-best-match --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/xpu \
      /tmp/requirements-xpu.lock

# Upstream at an exact commit: the patches are exact string matches.
RUN mkdir -p /opt/app/TRELLIS.2 && cd /opt/app/TRELLIS.2 && git init -q \
    && git remote add origin https://github.com/microsoft/TRELLIS.2.git \
    && git fetch -q --depth 1 origin 75fbf0183001ed9876c8dbb35de6b68552ee08bd && git checkout -q FETCH_HEAD && rm -rf .git

# Our code last, so edits rebuild in seconds.
COPY . /opt/app
WORKDIR /opt/app
RUN /opt/venv/bin/python patches/xpu_compat.py && /opt/venv/bin/python scripts/check_licenses.py > /opt/app/third-party-licenses.txt
RUN mkdir -p /cache && chown -R 1000:1000 /cache /opt/app
USER 1000:1000
