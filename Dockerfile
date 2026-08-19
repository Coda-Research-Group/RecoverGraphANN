# The reviewer path. This file *is* the "from a freshly installed Ubuntu LTS" walkthrough:
# if it builds, the README's native instructions are still correct.
#
#   docker build --build-arg ISA=native -t recovergraphann .
#   docker run --rm -v "$PWD/data:/app/data" recovergraphann make quick
#
# Build it on the machine you intend to run on — do not pull a prebuilt image. FlatNav and
# RoarGraph compile against the build host's instruction set, so a binary built elsewhere is
# either slower than it should be or will not run at all.
#
#   ISA=native   modern x86-64 with AVX2+FMA (the default)
#   ISA=scalar   pre-AVX2 hosts, including the paper's Intel Xeon E5-2620 (Sandy Bridge)
#
# RoarGraph needs x86-64. On arm64 the image builds without it; see the README support matrix.
FROM ubuntu:24.04

ARG ISA=native
ARG CONDA_DIR=/opt/conda
ARG ENV_NAME=rgann

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=${CONDA_DIR}/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    CONDA_ENV=${ENV_NAME}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ cmake git curl ca-certificates \
        libomp-dev libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Miniforge rather than Anaconda: conda-forge only, no commercial licence question.
RUN curl --fail --location --output /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && conda clean --all --yes

WORKDIR /app

# Environment first, so a source edit does not invalidate the expensive solve.
COPY environment.yml ./
RUN conda env create -n ${ENV_NAME} -f environment.yml && conda clean --all --yes

# Then the pinned native sources.
COPY third_party/ ./third_party/
COPY patches/ ./patches/
COPY scripts/install_flatnav.sh ./scripts/

RUN conda run -n ${ENV_NAME} --no-capture-output \
        pip install --no-cache-dir ./third_party/hnswlib

ENV FLATNAV_SCALAR_BUILD=${ISA}
RUN if [ "${ISA}" = "scalar" ]; then export FLATNAV_SCALAR_BUILD=1; else export FLATNAV_SCALAR_BUILD=0; fi \
    && RGANN_ENV=${ENV_NAME} bash scripts/install_flatnav.sh

# RoarGraph is x86-64 only: its distance kernels are written against x86 SIMD intrinsics.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        conda run -n ${ENV_NAME} --no-capture-output pip install --no-cache-dir pybind11 && \
        conda run -n ${ENV_NAME} --no-capture-output \
            pip install --no-build-isolation -e ./third_party/RoarGraph/pyroar ; \
    else \
        echo "skipping RoarGraph: needs x86-64, got $(uname -m)" ; \
    fi

COPY . .
RUN conda run -n ${ENV_NAME} --no-capture-output pip install --no-cache-dir -e .

# Datasets are ~635 MB and are NOT baked in. Mount them:
#   docker run -v "$PWD/data:/app/data" ...
# or run `bash scripts/download_data.sh` inside the container.
VOLUME ["/app/data", "/app/results", "/app/figures"]

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "rgann", "--no-capture-output"]
CMD ["make", "quick"]
