# Use base image
FROM polarbear:latest
 
# Install sudo and other system packages
RUN apt-get update && \
    apt-get install -y \
    sudo \
    wget \
    git \
    htop \
    less \
    tmux

# Install required python packages
RUN pip install \
    ipython \
    black

# Don't use cache from here on, to avoid using someone else's user and group ID
# This works because ADD will always fetch the file/URL and the above URL generates random data on each request, Docker then compares the result to see if it can use the cache.
RUN wget -q -O /tmp/skipcache "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" && \
    rm /tmp/skipcache

# Variables from build-arg
ARG USER_ID
ARG GROUP_ID
ARG USERNAME

# Create a new group
RUN groupadd -g ${GROUP_ID} ${USERNAME}

# Create the new user and add to the group
RUN useradd -m -d /home/${USERNAME} -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME}

# Add the user to the sudo group with an empty password
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    echo "${USERNAME}:" | chpasswd -e

# Switch to the non-root user
USER ${USER_ID}:${GROUP_ID}