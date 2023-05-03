FROM webis/ir-axioms:1.0.0-base

RUN pip3 uninstall -y tira ir_axioms && \
	pip3 install git+https://github.com/tira-io/tira.git@development#subdirectory=python-client && \
	rm -Rf /ir_axioms

COPY . /ir_axioms

RUN cd /ir_axioms && pip install -e .[pyterrier]

