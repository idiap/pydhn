# SPDX-FileCopyrightText: 2023 Idiap Research Institute, EPFL
#
# SPDX-License-Identifier: AGPL-3.0-only

FROM python:3.8-slim-buster

COPY . pydhn
WORKDIR /pydhn
RUN pip3 install -e .[test]
CMD ["python", "-m", "unittest", "discover" ,"tests"]
