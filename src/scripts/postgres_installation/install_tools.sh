#!/usr/bin/env bash

set -e
set -x

if [[ -e FLAG_INSTALL_TOOLS_DONE ]]
then
  echo "skip tools installation"
else
  sudo add-apt-repository ppa:deadsnakes/ppa -y
  sudo apt update
  sudo apt install python3.9 -y
  sudo apt install python3.9-dev -y
  sudo apt install python3.9-distutils -y
  sudo apt install python3.9-venv -y
  ##sudo apt install python3.9-pip -y
  sudo apt install mysql-client -y
  sudo apt install htop
  touch FLAG_INSTALL_TOOLS_DONE
fi