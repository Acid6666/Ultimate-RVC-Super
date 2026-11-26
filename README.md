# Ultimate RVC

[![PyPI version](https://badge.fury.io/py/ultimate-rvc.svg)](https://badge.fury.io/py/ultimate-rvc)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JackismyShephard/ultimate-rvc/blob/main/notebooks/ultimate_rvc_colab.ipynb)
[![Discord Server](https://dcbadge.limes.pink/api/server/https://discord.gg/T4ejEz8HtX?style=flat&compact=true&theme=default-inverted)](https://discord.gg/https://discord.gg/T4ejEz8HtX)
[![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/JackismyShephard/ultimate-rvc)

An extension of [AiCoverGen](https://github.com/SociallyIneptWeeb/AICoverGen), which provides several new features and improvements, enabling users to generate audio-related content using RVC with ease. Ideal for people who want to incorporate singing functionality into their AI assistant/chatbot/vtuber, hear their favourite character sing their favourite song or have their favorite character read their favorite books aloud.

![Ultimate RVC Web Interface](images/webui_generate.png?raw=true)

## Youtube Showcase

* [Ultimate RVC: The Most Powerful AI Voice Cloning Tool Yet!](https://www.youtube.com/watch?v=dVpCF68JpNg)
* [Ultimate-RVC on Your PC | Turn Any Voice Into Yours!](https://www.youtube.com/watch?v=IxFgz50y1AY)
* [Train Any Voice with Ultimate RVC | Fast & Accurate Voice Cloning](https://www.youtube.com/watch?v=O7VeYLL3H-c)

Courtesy of [Social & Apps](https://www.youtube.com/@socialapps1194)

## New Features

* Easy and automated setup using launcher scripts for both Windows and Debian-based linux systems
* Significants improvements to voice conversion quality and speed. New features include support for additional pitch extraction methods such as FCPE, different embedder models and pre/post-processing options such as autotuning and noise reduction.
* TTS functionality, which allows you to generate speech from text using any RVC-based voice model. With this feature, you can do things such as generate audio books using your favourite character's voice.
* A voice model training suite, which allows you to train your own voice models using a wide range of options, such as different datasets, embedder models, and training configurations.
* Caching system which saves intermediate audio files as needed, thereby reducing inference time as much as possible. For example, if song A has already been converted using model B and now you want to convert song A using model C, then vocal extraction can be skipped and inference time reduced drastically
* Ability to listen to intermediate audio files in the UI. This is useful for getting an idea of what is happening in each step of a given generation pipeline.
* "multi-step" generation tabs: Here you can try out each step of a given generation pipeline in isolation. For example, if you already have extracted vocals available and only want to convert these using your voice model, then you can do that in a dedicated "multi-step" tab for song cover generation. Besides, these "multi-step" generation tabs are also useful for experimenting with settings for each step in a given generation pipeline.
* Lots of visual and performance improvements resulting from updating from Gradio 3 to Gradio 5 and from Python 3.9 to Python 3.13
* A redistributable package on PyPI, which allows you to easily access the Ultimate RVC project from any Python 3.13 environment.
* Support for saving and loading of custom configurations for the Ultimate RVC web application. This allows you to easily switch between different configurations without having to manually change settings each time.

## Custom additions in this fork

### FastAPI gateway with on-demand GPU leasing

`urvc_fastapi.py` is a FastAPI gateway that starts the Gradio app only when a request
arrives, leases a GPU from an external manager at `127.0.0.1:7050`, and releases it
after the configured idle timeout. `gpu_client.py` and `gradio_pool.py` implement the
lease and process lifecycle. On Linux, start everything with `./run.sh` (or install the
`urvc.service` systemd unit). `start_service.sh` remains as the simpler, direct Gradio
launcher.

The gateway listens on `127.0.0.1:7866` by default. Set `URVC_GATEWAY_HOST` /
`URVC_GATEWAY_PORT` and `URVC_GRADIO_HOST` / `URVC_GRADIO_PORT` to change the bind
addresses; the defaults are loopback-only and safe to expose through a reverse proxy.

### Importing models from Quark Drive (夸克网盘)

The **Models → Quark Drive** tab can search your whole Quark Drive by keyword, browse
folders, or resolve share links, then download selected `.pth` / `.index` / `.zip`
files and install them as voice models.

1. Open **Settings → Quark Drive**, paste the `Cookie` header copied from a
   `drive.quark.cn` request in your browser's developer tools, and save it.
2. Open **Models → Quark Drive**, enter a keyword and click **List files**.
3. Select the files you want, enter a model name, and click **Download as model**.

The cookie is stored locally in `config/quark_cookie.json` with `0600` permissions and
is excluded from version control. This integration uses Quark's private web endpoints
rather than an official API, so it may stop working if those endpoints change.

## Online Platforms

For those without a powerful enough NVIDIA GPU, you may try out Ultimate RV using [Google Colab](https://colab.research.google.com/github/JackismyShephard/ultimate-rvc/blob/main/notebooks/ultimate_rvc_colab.ipynb). Additionally, Ultimate RVC is also hosted on [Huggingface Spaces](https://huggingface.co/spaces/JackismyShephard/ultimate-rvc), although GPU acceleration is not available there. For those who want to run Ultimate RVC locally, follow the setup guide below.

## Local Setup

The Ultimate RVC project currently supports Windows and Debian-based Linux distributions, namely Ubuntu 22.04 and Ubuntu 24.04. Support for other platforms is not guaranteed.

To setup the project follow the steps below and execute the provided commands in an appropriate terminal. On Windows this terminal should be **powershell**, while on Debian-based linux distributions it should be a **bash**-compliant shell.

### Install Git

Follow the [Git installation instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your computer.

### Set execution policy (Windows only)

To execute the subsequent commands on Windows, it is necessary to first grant
powershell permission to run scripts. This can be done at a user level as follows:

```console
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Clone Ultimate RVC repository

```console
git clone https://github.com/JackismyShephard/ultimate-rvc
cd ultimate-rvc
```

### Install dependencies

```console
./urvc install 
```

Note that on Linux, this command will install the CUDA 12.8 toolkit system-wide, if it is not already available. In case you have problems, you may need to install the toolkit manually.

#### Using Aliyun mirrors (for users in China)

If `astral.sh`, `pypi.org` or `download.pytorch.org` are slow or blocked, install `uv` from
the Aliyun PyPI mirror and set these variables **before** running `./urvc install`:

```bash
python3 -m pip install uv==0.9.11 -i https://mirrors.aliyun.com/pypi/simple/
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu128/
./urvc install
```

All Python packages (including the bundled Node.js binary from `nodejs-wheel-binaries`)
then come from Aliyun, and the CUDA 12.8 PyTorch wheels come from Aliyun's
`pytorch-wheels/cu128` mirror. `./urvc` skips its own `astral.sh` download whenever `uv`
is already installed.

Aliyun does not mirror GitHub repository clones or NVIDIA's CUDA apt repository. If
those two steps are slow, use your own accelerator or install CUDA 12.8 from a local
installer; the Python dependency installation itself no longer needs overseas access.

### Start the app

```console
./urvc run
```

Once the output message `Running on local URL:  http://127.0.0.1:7860` appears, you can click on the link to open a tab with the web app.

### Update to latest version

```console
./urvc update
```

### Development mode

When developing new features or debugging, it is recommended to run the app in development mode. This enables hot reloading, which means that the app will automatically reload when changes are made to the code.

```console
./urvc dev
```

## Usage

### Manage models

#### Download models

![ ](images/webui_dl_model.png?raw=true)

Navigate to the `Download` subtab under the `Models` tab, and paste the download link to an RVC model and give it a unique name.
You may search the [AI Hub Discord](https://discord.gg/aihub) where already trained voice models are available for download.
The downloaded zip file should contain the .pth model file and optionally also a .index file.

Once the two input fields are filled in, simply click `Download`. Once the output message says `[NAME] Model successfully downloaded!`, you should be able to use the downloaded model in either the `Generate`>`song covers` or `Generate`>`speech` tab.

#### Upload models

![ ](images/webui_upload_model.png?raw=true)

For people who have trained RVC models locally and would like to use them for voice conversion.
Navigate to the `Upload` subtab under the `Models` tab, and follow the instructions there.
Once the output message says `Model with name [NAME] successfully uploaded!`, you should be able to use the uploaded model in either the `Generate`>`song covers` or `Generate`>`speech` tab.

### Generate song covers

#### One-click generation

![ ](images/webui_generate.png?raw=true)

* From the `Source type` dropdown, choose the source type from which you want to retrieve the song to convert.
* In the `Source` input field either paste the URL of a song on YouTube or upload an audio file, depending on the source type selected.
* From the `Voice model` dropdown menu, select the voice model to use.
* More Options can be viewed by clicking `Options`.

Once all options are filled in, click `Generate` and the AI generated song cover should appear in less than a few minutes, depending on your GPU.

## PyPI package

The Ultimate RVC project is also available as a [distributable package](https://pypi.org/project/ultimate-rvc/) on [PyPI](https://pypi.org/).

### Installation

The package can be installed with CUDA support using pip in a **Python 3.12-3.13**-based environment as follows:

```console
pip install ultimate-rvc[cuda] --extra-index-url https://download.pytorch.org/whl/cu128
```

With Aliyun mirrors:

```console
pip install ultimate-rvc[cuda] -i https://mirrors.aliyun.com/pypi/simple/ -f https://mirrors.aliyun.com/pytorch-wheels/cu128/
```

### CLI Usage

The `ultimate-rvc` package can be used as a Python library but is primarily intended to be used as a command line tool. The package exposes two top-level commands:

* `urvc` which lets the user generate song covers directly from their terminal
* `urvc-web` which starts a local instance of the Ultimate RVC web application

For more information on either command supply the option `--help`.

## Environment Variables

The behaviour of the Ultimate RVC project can be customized via a number of environment variables. Currently these environment variables control only logging behaviour and data directory locations. They are as follows:

* `URVC_CONSOLE_LOG_LEVEL`: The log level for console logging. If not set, defaults to `ERROR`.
* `URVC_FILE_LOG_LEVEL`: The log level for file logging. If not set, defaults to `INFO`.
* `URVC_LOGS_DIR`: The directory in which log files will be stored. If not set, logs will be stored in a `logs` directory in the current working directory.
* `URVC_NO_LOGGING`: If set to `1`, logging will be disabled.
* `URVC_MODELS_DIR`: The directory in which models will be stored. If not set, models will be stored in a `models` directory in the current working directory.
* `URVC_AUDIO_DIR`: The directory in which audio files will be stored. If not set, audio files will be stored in an `audio` directory in the current working directory.
* `URVC_TEMP_DIR`: The directory in which temporary files will be stored. If not set, temporary files will be stored in a `temp` directory in the current working directory.
* `URVC_CONFIG_DIR`: The directory in which configuration files will be stored. If not set, configuration files will be stored in a `configs` directory in the current working directory.
* `URVC_VOICE_MODELS_DIR`: The directory in which voice models will be stored. If not set, voice models will be stored in a `voice_models` subdirectory of the `URVC_MODELS_DIR` directory.
* `YT_COOKIEFILE`: The path to a file containing cookies to use when downloading audio from YouTube via the web UI. If not set, no cookies will be used.
* `URVC_ACCELERATOR`: The type of hardware accelerator to use when running Ultimate RVC directly via the shell scripts in this repository. Currently supported options are `cuda` and `rocm`, with `cuda` being the default. Note that `rocm` is not supported on Windows and experimental on linux.
* `URVC_CONFIG`: The name of a configuration with custom values for settings to load when starting the Ultimate RVC web application. If not set, the default configuration for Ultimate RVC will be used. The configuration should be located in the `configs` directory of the Ultimate RVC project. If it does not exist, an error will be raised.
* `NODE_PATH`: The path to custom Node.js installation to use when running the Ultimate RVC web application and downloading from YouTube. If not set, a default Node.js installation bundled with Ultimate RVC will be used.
* `URVC_GATEWAY_HOST` / `URVC_GATEWAY_PORT`: Bind address and port of the FastAPI gateway. Defaults to `127.0.0.1:7866`.
* `URVC_GRADIO_HOST` / `URVC_GRADIO_PORT`: Address and port used by the gateway to reach Gradio. Defaults to `127.0.0.1:7865`.
* `URVC_GPU_MANAGER_URL`: Base URL of the external GPU manager. Defaults to `http://127.0.0.1:7050`.
* `URVC_GPU_COUNT` / `URVC_GPU_HEARTBEAT_INTERVAL`: Number of GPUs to lease per task and heartbeat interval in seconds. Defaults to `1` and `60`.
* `URVC_IDLE_TIMEOUT`: Seconds of inactivity after which the gateway stops Gradio and releases the GPU. Defaults to `600`.
* `URVC_GRADIO_MANAGED` / `URVC_GRADIO_PERSISTENT`: Set to `1` (default) to let the gateway manage the Gradio process, optionally keeping it alive while returning the GPU lease.
* `URVC_GRADIO_IDLE_TIMEOUT`: Gradio's own idle cleanup timeout in seconds. Defaults to `0` (disabled) so cleanup happens only when the gateway confirms there are no active tasks.
* `URVC_CONTROL_PORT`: Local control port used to trigger model unloading inside the Gradio process. Defaults to `7867`.
* `URVC_GRADIO_STARTUP_TIMEOUT`: Seconds to wait for Gradio to become ready. Defaults to `300`.

## Help & Community

If you experience (reproducible) bugs or have specific feature requests we encourage you to file a new issue here on GitHub. For broader discussions on the development of Ultimate RVC, as well as connecting with other users, we encourage you to join our Discord [server](https://discord.gg/T4ejEz8HtX) or check out the [Discussions](https://github.com/JackismyShephard/ultimate-rvc/discussions) page here on GitHub.

## Terms of Use

The use of the converted voice for the following purposes is prohibited.

* Criticizing or attacking individuals.

* Advocating for or opposing specific political positions, religions, or ideologies.

* Publicly displaying strongly stimulating expressions without proper zoning.

* Selling of voice models and generated voice clips.

* Impersonation of the original owner of the voice with malicious intentions to harm/hurt others.

* Fraudulent purposes that lead to identity theft or fraudulent phone calls.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.
