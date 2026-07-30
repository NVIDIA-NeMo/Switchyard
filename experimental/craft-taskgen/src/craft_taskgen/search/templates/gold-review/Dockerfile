FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# System packages + Claude Code
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip ca-certificates curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    curl -fsSL https://claude.ai/install.sh | bash && \
    ln -sf /root/.local/bin/claude /usr/local/bin/claude && \
    apt-get remove -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

# Gold-review skill (copied by adapter)
COPY skills/ /root/.claude/skills/

# Review data: tasks to review + agent trajectories (copied by adapter)
COPY data/ /app/data/

RUN mkdir -p /app
WORKDIR /app
