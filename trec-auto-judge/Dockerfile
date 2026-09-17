# docker build -t mam10eks/trec-auto-judge-base:ir-axioms-0.0.1 -f judges/ir_axioms/Dockerfile .
# The base image is build from 
FROM mam10eks/trec-auto-judge-base:dev-0.0.1

RUN . /auto-judge/.venv/bin/activate \
	&& apt-get update && apt-get install -y curl \
	&& curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
	&& python3 /tmp/get-pip.py \
	&& rm /tmp/get-pip.py

RUN . /auto-judge/.venv/bin/activate \
	&& python3 -m pip uninstall -y ir_axioms \
	&& python3 -m pip install 'ir_axioms>=1.1.2'

RUN . /auto-judge/.venv/bin/activate \
	&& python3 -m spacy download en_core_web_sm \
	&& python3 -c 'from tira.third_party_integrations import ensure_pyterrier_is_loaded; ensure_pyterrier_is_loaded(patch_ir_datasets=True, is_offline=False)' \
	&& python3 -c 'from language_tool_python import LanguageTool; LanguageTool(language="en-US")'

ADD trec-auto-judge /auto-judge/judges/ir_axioms/
ADD .git /auto-judge/

ENV PYTHONPATH=/auto-judge

ENV HF_HUB_OFFLINE=1

