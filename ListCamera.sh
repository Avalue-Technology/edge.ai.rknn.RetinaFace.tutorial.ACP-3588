#!/bin/bash
for i in /dev/video*; do
	echo "====== $i ======"
	udevadm info --query=all --name=$i | grep -E 'ID_MODEL|ID_VENDOR|DEVNAME'
done

