<div align="center">
  <img src="documents/logo_transparent.png" alt="Our project logo" title="'Translator' in German" /><br>
  <p>
      German to English translator system
      <br />
      <a href="https://yoogottamk.github.io/anlp-project"><strong>Explore the docs »</strong></a>
      <br />
      <br />
      <a href="https://github.com/othneildrew/Best-README-Template">Run demo WebAPP</a>
      <br />
      <a href="https://github.com/yoogottamk/anlp-project-nmt/issues/new">Report Bug</a>
      <br />
      <a href="https://github.com/yoogottamk/anlp-project-nmt/issues/new">Request Feature</a>
  </p>
</div>

<!--
Uncomment when repo public
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
-->

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#team-information">Team Information</a>
    </li>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## Team Information
| Name | Roll Number| Branch |
| --- | --- | --- |
| Gaurang Tandon | 2018101091 | CSE |
| Yoogottam Khandelwal | 2018101019 | CSE |

<!-- ABOUT THE PROJECT -->
## About The Project

Webapp IMAGE HERE

**_Übersetzerin_** is our comprehensive and robust ML pipeline built for bidirectional text translation between German and English languages. Key features:

1. A webapp to easily and interactively experiment with any user input sentence
2. A readthedocs website to host complete API reference for our code
3. **Two** machine-learning models (seq2seq and transformer based) used by us to perform the translation task

### Built With

* [🤗 Huggingface](https://huggingface.co/)
* [Bootstrap](https://getbootstrap.com)
* [MDBootstrap](https://mdbootstrap.com/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Repository structure

```
.
├── anlp_project     # code for our ML pipeline
├── documents        # documentation for our project
├── README.md        # you are here
├── scripts          # script to download dataset easily
├── requirements.txt # pip install this
└── setup.py         # used to setup our pip package for easy distributions
```

The source code has been ordered in a logical manner instead of just putting everything into a single file.

```
anlp_project
├── config.py                        # loads all model hyperparameters
├── datasets                         # module for dataset loader processes
├── hparams.yaml                     # centralized file for all model hyperparameters
├── inference.py                     # runs inference
# TODO
```

### Prerequisites

We use Python's `pip` package manager. Perform these steps in root directory of project:

* Install all dependencies:
  ```sh
  pip install -r requirements.txt
  ```

### Installation

#### Training and inference

Our package is provided as a Python module. Perform these steps in root directory of project:

* Install the python package
  ```sh
  pip install -e .
  ```

#### Webapp

* `cd webapp`
* `pip install -r requirements.txt`
* `python app.py`

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Model training

#### Seq2Seq training
#### T5 training

### Webapp

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

We really appreciate more contributions to keep the spirit of open source thriving!  Feel free to create issues or pull requests on whatever topic you see fit. Don't forget to give the project a star! Thanks again!

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/yoogottamk/anlp-project-nmt?style=for-the-badge
[contributors-url]: https://github.com/yoogottamk/anlp-project-nmt/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/yoogottamk/anlp-project-nmt?style=for-the-badge
[stars-url]: https://github.com/yoogottamk/anlp-project-nmt/stargazers
[license-shield]: https://img.shields.io/github/license/yoogottamk/anlp-project-nmt?style=for-the-badge
[license-url]: https://github.com/yoogottamk/anlp-project-nmt/blob/master/LICENSE.txt
