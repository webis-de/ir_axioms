#!/bin/bash -e

srun -c 25  --mem=100G \
	--container-writable \
	--container-image=pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel \
	--container-name=sigir22-ir-axioms --pty \
	--container-mounts=${PWD}:/workspace,/mnt/ceph/storage/data-tmp/2021/kibi9872/.ir_datasets:/root/.ir_datasets \
	--chdir ${PWD} \
	bash -c 'cd ~/ecir22-zero-shot/ && jupyter-lab --ip 0.0.0.0 --allow-root'

