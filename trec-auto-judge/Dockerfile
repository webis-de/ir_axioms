# docker build -t mam10eks/trec-auto-judge-base:ir-axioms-0.0.1 -f judges/ir_axioms/Dockerfile .
# The base image is build from 
FROM mam10eks/trec-auto-judge-base:dev-0.0.1

RUN . /auto-judge/.venv/bin/activate \
	&& apt-get update && apt-get install -y curl \
	&& curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
	&& python3 /tmp/get-pip.py \
	&& rm /tmp/get-pip.py

# Add this repository's ir_axioms source (instead of installing the PyPI
# release) so that the image always reflects the checked-out code.
# The .git directory is needed alongside pyproject.toml so that
# setuptools-scm can determine the package version.
ADD ir_axioms /auto-judge/ir_axioms-src/ir_axioms
ADD pyproject.toml /auto-judge/ir_axioms-src/pyproject.toml
ADD README.md /auto-judge/ir_axioms-src/README.md
ADD .git /auto-judge/ir_axioms-src/.git

RUN . /auto-judge/.venv/bin/activate \
	&& python3 -m pip uninstall -y ir_axioms \
	&& python3 -m pip install /auto-judge/ir_axioms-src

RUN . /auto-judge/.venv/bin/activate \
	&& python3 -m spacy download en_core_web_sm \
	&& python3 -c 'from tira.third_party_integrations import ensure_pyterrier_is_loaded; ensure_pyterrier_is_loaded(patch_ir_datasets=True, is_offline=False)' \
	&& python3 -c 'from language_tool_python import LanguageTool; LanguageTool(language="en-US")'

ADD trec-auto-judge /auto-judge/judges/ir_axioms/

ENV PYTHONPATH=/auto-judge

ENV HF_HUB_OFFLINE=1

