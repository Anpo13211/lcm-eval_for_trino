#!/usr/bin/env bash
set -e
set -x
# resize partitions to 900GB
if [[ -e RESIZE_DONE ]]
then
    echo "skip disk resize"
else
    sudo timedatectl set-timezone Europe/Berlin
    sudo swapoff --all
    sudo sysctl vm.swappiness=0
    echo -e "vm.swappiness = 0" | sudo tee -a /etc/sysctl.conf
    sudo sed -i 's/.*swap.*//g' /etc/fstab
    sudo update-grub
    sudo update-initramfs -u
    sudo parted << EOF

rm 3
EOF
    DISK=/dev/$(lsblk | tail -n +2 | grep -Po "^[^└├\s]*" | head -n 1 )
    DEVICE=$(df -h / | grep -Po '/dev/([^\s]*)')
    printf "yes\n400000M\n" | sudo parted $DISK ---pretend-input-tty resizepart 1 400000M
    touch RESIZE_DONE
    sudo reboot
fi
