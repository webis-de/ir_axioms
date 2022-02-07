#!/bin/bash -e

srun-capreolus-notebook:
	srun -c 25  --mem=100G \
		--container-writable \
		--container-image=pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel \
		--container-name=sigir22-ir-axioms --pty \
		--chdir ${PWD} \
		bash -c 'cd ~/ecir22-zero-shot/ && jupyter-lab --ip 0.0.0.0 --allow-root'
