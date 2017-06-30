#!/bin/bash

END=3
for ((i=0;i<=END;i++)); do
   sudo nvidia-smi -pm ENABLED -i $i
   sudo nvidia-smi -ac 2505,562 -i $i
   sudo nvidia-smi --auto-boost-default=DISABLED -i $i
done

