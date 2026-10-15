# Third-Party Notices

Original project contributions are licensed under the root MIT `LICENSE`.
This does not replace the licenses of the following upstream components.

## Time-Series-Library

Source: <https://github.com/thuml/Time-Series-Library>.

The iTransformer and Autoformer modules and their applicable shared layers
originate from TSLib. The iTransformer integration adds attribution and removes
an unused import; it does not replace the official architecture. Training,
preprocessing, and recursive evaluation are this project's experimental protocol.
Keep `third_party/licenses/Time-Series-Library-MIT.txt` with this code.

## GluonTS Time Features

Source: <https://github.com/awslabs/gluonts>.

`utils/timefeatures.py` retains its original Amazon copyright and Apache-2.0
header. The complete Apache license is in `third_party/licenses/Apache-2.0.txt`.

## Chronos-2

Source: <https://github.com/amazon-science/chronos-forecasting>.

The official package supplies the forecasting architecture and model APIs.
The experiment uses official revision
`f951d9aefa06f5389b2ed6b0e51fd5a1a4cf194b`, pinned in `requirements.txt`;
its installed version string is 2.2.2. The architecture, dataset, pipeline,
configuration, and other checked Chronos-2 core files match that revision.
The package is Apache-2.0 licensed. This repository does not redistribute the
complete package or the `amazon/chronos-2` model weights.

The private load data and manuscript materials are not part of this code release.
