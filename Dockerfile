FROM python:3.11-slim

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    cmake \
    pkg-config \
    libssl-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libfreetype6-dev \
    libpng-dev \
    openssh-server \
    debconf-utils \
    # PyQt6の依存関係を追加
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb1 \
    libx11-xcb1 \
    libdbus-1-3 \
    libfontconfig1 \
    libxkbcommon0 \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# xorgのインストール
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    echo keyboard-configuration keyboard-configuration/layout select 'Japanese' | debconf-set-selections && \
    echo keyboard-configuration keyboard-configuration/layoutcode select 'jp' | debconf-set-selections && \
    apt-get update && apt-get install -y --no-install-recommends xorg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# SSHの設定
RUN mkdir /var/run/sshd
RUN echo 'root:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# 作業ディレクトリを設定
WORKDIR /app

# 環境変数を設定
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=host.docker.internal:10.0
ENV QT_X11_NO_MITSHM=1

# 必要なPythonパッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install PyQt6 matplotlib numpy

# プロジェクトファイルをコピー
COPY . .

# PYTHONPATHを設定
ENV PYTHONPATH=/app

# SSHサーバーを起動
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

RUN echo "cd /app/geo_simulation_project" >> /root/.bashrc