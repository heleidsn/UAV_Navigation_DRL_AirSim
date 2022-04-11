# UAV_Navigation_DRL_AirSim

This is a new repo used for training UAV navigation (local path planning) policy using DRL methods.

<p align="center">
  <img src="resources/figures/result_3d_NH_simple_dynamics.gif" width = "400" height = "225"/>
  <img src="resources/figures/result_3d_NH_multirotor.gif" width = "400" height = "225"/>
  <!-- <img src="resources/figures/sim_city_fixed_wing.gif" width = "400" height = "225"/>
  <img src="resources/figures/sim_city_flapping_5hz.gif" width = "400" height = "225"/> -->
</p>

## ChangeLog

- 2020-03-11
  - Add wandb support
- 2020-03-10
  - Remove [gym_airsim_multirotor](https://github.com/heleidsn/gym_airsim_multirotor) submodule
  - Add gym_env as envrionment, include MultirotorSimple, Multirotor and FixedwingSimple dynamics
  - Add train with plot
  - Add SimpleAvoid UE4 environment

## Requirements

- Python 3.8
- [AirSim](https://microsoft.github.io/AirSim/) v1.6.0
- pytorch 1.10.1 with gpu
- gym-0.21.0
- Pyqt5 5.15.6
- keyboard 0.13.5

## Submodules

- [Stable-baselines3](https://github.com/heleidsn/stable-baselines3) v1.4.0

## Install CUDA and PyTorch (Win10)

- Download [CUDA11.6](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)
- `pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
- You can use `scripts/test/torch_gpu_cpu_test.py` to test your PyTorch and CUDA

## Usage

1. Clone this repo and the submodules

   1. `git clone https://github.com/heleidsn/UAV_Navigation_DRL_AirSim.git --recursive`
2. Install gym_env

   1. `cd gym_env`
   2. `pip install -e .`
3. Install customized stable-baselines3

   1. `cd stable-baselines3`
   2. `pip install -e .`
4. Download a AirSim environment, such as Blocks from [Here](https://github.com/microsoft/AirSim/releases/tag/v1.6.0-windows) and run it
5. Start training

   1. `cd UAV_Navigation_DRL_AirSim`
   2. `python scripts/start_train_with_plot.py`
6. Evaluation

   1. `cd UAV_Navigation_DRL_AirSim`
   2. `python scripts/start_evaluate_with_plot.py`

## Configs

This repo using config file to control training conditions.

Now we provide 3 training envrionment and 3 dynamics.

**env_name**

* SimpleAvoid

  * This is a custom UE4 environment used for simple obstacle avoidance test. You can download it from [google drive](https://drive.google.com/file/d/1QgkZY5-GXRr93QTV-s2d2OCoVSndADAM/view?usp=sharing).

  <p align="center">
    <img src="resources/env_maps/simple_world_1.png" width = "350" height = "200"/>
    <img src="resources/env_maps/simple_world_45.png" width = "350" height = "200"/>
  </p>
* City_400_400

  * A custom UE4 environment used for fixedwing obstacle avoidance test.

  <p align="center">
    <img src="resources/env_maps/city_400.png" width = "350" height = "200"/>
    <img src="resources/env_maps/city_400_1.png" width = "350" height = "200"/>
  </p>

* Random obstacles
  * Some envs with random obstacles. Contributed by [Chris-cch](https://github.com/Chris-cch). You can download [here](https://mailnwpueducn-my.sharepoint.com/personal/chenchanghao_mail_nwpu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchenchanghao%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2Fnew%5Fenv&ga=1).

  <p align="center">
    <img src="resources/env_maps/random_1.png" width = "350" height = "200"/>
    <img src="resources/env_maps/random_2.png" width = "350" height = "200"/>
  </p>

* Other Airsim build in envrionment (AirSimNH and CityEnviron):

  <p align="center">
    <img src="resources\env_maps\NH.png" width = "350" height = "200"/>
    <img src="resources\env_maps\city.png" width = "350" height = "200"/>
  </p>

**dynamic_name**

* SimpleMultirotor
* Multirotor
* SimpleFixedwing

## GUI for training and evaluation

![img](resources/figures/gui_for_train_and_eval.png)

## Wandb support

[Wandb](https://wandb.ai/site) is a central dashboard to keep track of your hyperparameters, system metrics. You can find examples [here](https://docs.wandb.ai/guides/integrations/other/stable-baselines-3).

> Note: If you use wandb, please run python as **administators**.

## Results

Training result using TD3 with no_cnn policy![img](resources/figures/training_result_simple_no_cnn.png)

## Benchmark

2D depth navigation Benchmark for 3 different algorithms and 5 different policies in SimpleAvoid environment:

<img src="resources\results\policy-PPO.jpg" alt="drawing" width="800"/>

<img src="resources\results\policy-SAC.jpg" alt="drawing" width="800"/>

<img src="resources\results\policy-TD3.jpg" alt="drawing" width="800"/>

## Settings

Note: 

To speed up image collection, you can set `ViewMode `to `NoDisplay`.

For multirotor with [simple_flight](https://microsoft.github.io/AirSim/simple_flight/) controller, please set `SimMode `to `Multirotor`. You can also set `ClockSpeed `over than 1 to speed up simulation (Only useful in `Multirotor `mode).

Also, it's better to put your environment files in your SSD rather than HDD.

```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "ComputerVision",
  "ViewMode": "NoDisplay",
  "ClockSpeed": 1,
  "SubWindows": [
    {"WindowID": 0, "CameraID": 0, "ImageType": 0, "Visible": true},
    {"WindowID": 1, "CameraID": 0, "ImageType": 3, "Visible": false},
    {"WindowID": 2, "CameraID": 0, "ImageType": 3, "Visible": true}
    ],
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 3,
        "Width": 100,
        "Height": 80,
        "FOV_Degrees": 90,
        "AutoExposureSpeed": 100,
        "AutoExposureBias": 0,
        "AutoExposureMaxBrightness": 0.64,
        "AutoExposureMinBrightness": 0.03,
        "MotionBlurAmount": 0,
        "TargetGamma": 1.0,
        "ProjectionMode": "",
        "OrthoWidth": 5.12
      },
      {
        "ImageType": 0,
        "Width": 256,
        "Height": 144,
        "FOV_Degrees": 90,
        "AutoExposureSpeed": 100,
        "AutoExposureBias": 0,
        "AutoExposureMaxBrightness": 0.64,
        "AutoExposureMinBrightness": 0.03,
        "MotionBlurAmount": 0,
        "TargetGamma": 1.0,
        "ProjectionMode": "",
        "OrthoWidth": 5.12
      }
    ]
  }
}
```
