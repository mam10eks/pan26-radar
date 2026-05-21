FROM python:3.12

COPY . /opt/pan26_genai_evaluator
WORKDIR /opt/pan26_genai_evaluator
RUN set -x \
    && python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install --no-cache . \
    && rm -rf ./build ./*.egg-info

ENTRYPOINT ["/usr/local/bin/pan26-evaluator"]
