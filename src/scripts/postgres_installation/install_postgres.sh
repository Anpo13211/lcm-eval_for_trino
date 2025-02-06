#!/usr/bin/env bash

set -e
set -x

if [[ -e FLAG_INSTALL_DONE ]]
then
    echo "skip installation"
else

  wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
  echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/postgresql-pgdg.list > /dev/null
  sudo apt update
  sudo apt install -y postgresql-14
  sudo apt install -y postgresql-server-dev-14
  sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'bM2YGRAX*bG_QAilUid§2iD';"
  sudo service postgresql restart
  sudo apt install gcc
  sudo apt install make
  wget https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/REL14_1_4_0.tar.gz
  tar xzvf REL14_1_4_0.tar.gz
  cd pg_hint_plan-REL14_1_4_0
  make
  sudo make install
  cd ..
  rm REL14_1_4_0.tar.gz
  sudo service postgresql restart
  sudo cp cost-eval/src/conf/modified-postgresql14.conf /etc/postgresql/14/main/postgresql.conf
  sudo cpcost-eval/src/conf/pg_hba.conf /etc/postgresql/14/main/pg_hba.conf
  sudo service postgresql restart
  touch FLAG_INSTALL_DONE
fi