FROM mambaorg/micromamba:2.3.2

USER root
ARG APT_PACKAGES=""
RUN if [ -n "$APT_PACKAGES" ]; then \
      apt-get update && \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $APT_PACKAGES && \
      rm -rf /var/lib/apt/lists/*; \
    fi

USER $MAMBA_USER
ARG CONDA_PACKAGES
RUN micromamba install --yes --name base --channel conda-forge \
      "python>=3.10,<3.12" numpy=2.* h5py=3.* pyyaml=6.* $CONDA_PACKAGES && \
    micromamba list --name base --explicit > /tmp/conda-explicit.txt && \
    micromamba clean --all --yes

USER root
COPY healthcheck.py /opt/software-bench/healthcheck.py
RUN chmod 0555 /opt/software-bench/healthcheck.py && \
    mv /tmp/conda-explicit.txt /opt/software-bench/conda-explicit.txt && \
    mkdir -p /workspace

ARG EXECUTABLES
ENV SOFTWARE_BENCH_EXECUTABLES=$EXECUTABLES
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /workspace
HEALTHCHECK --interval=30s --timeout=20s --start-period=10s --retries=3 \
  CMD ["python", "/opt/software-bench/healthcheck.py"]
CMD ["/bin/bash"]
