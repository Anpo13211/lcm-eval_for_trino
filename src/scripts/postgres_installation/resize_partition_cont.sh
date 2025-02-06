#!/usr/bin/env bash
set -e
set -x
# resize partitions to 900GB
if [[ -e FLAG_DISK_DONE ]]
then
    echo "skip update_fstab"
else
    DISK=/dev/$(lsblk | tail -n +2 | grep -Po "^[^└├\s]*" | head -n 1 )
    DEVICE=$(df -h / | grep -Po '/dev/([^\s]*)')
    sudo resize2fs $DEVICE
    FREE=$(df -k . |awk '{print $4}' | tail -n 1)
    if [[ "$FREE" -gt 100000000 ]]; # at least 100G must be free
    then
        echo "disk resize ok ($FREE)"
    else
        echo "disk resize failed - only $FREE b available"
        exit
    fi
    touch FLAG_DISK_DONE
fi
