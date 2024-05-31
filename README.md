# GenZI: Zero-Shot 3D Human-Scene Interaction Generation
By [Lei Li](https://craigleili.github.io/) and [Angela Dai](https://www.3dunderstanding.org/team.html). (CVPR 2024)

![pipeline](asset/teaser.gif)

Can we synthesize 3D humans interacting with scenes without learning from any 3D human-scene interaction data? We propose GenZI, the first zero-shot approach to generating 3D human-scene interactions. Key to GenZI is our distillation of interaction priors from large vision-language models (VLMs), which have learned a rich semantic space of 2D human-scene compositions.

Given a natural language description and a coarse point location of the desired interaction in a 3D scene, we first leverage VLMs to imagine plausible 2D human interactions inpainted into multiple rendered views of the scene. We then formulate a robust iterative optimization to synthesize the pose and shape of a 3D human model in the scene, guided by consistency with the 2D interaction hypotheses.

In contrast to existing learning-based approaches, GenZI circumvents the conventional need for captured 3D interaction data, and allows for flexible control of the 3D interaction synthesis with easy-to-use text prompts. Extensive experiments show that our zero-shot approach has high flexibility and generality, making it applicable to diverse scene types, including both indoor and outdoor environments.


## Links

- [Paper](https://arxiv.org/pdf/2311.17737)
- [Video](https://youtu.be/ozfs6E0JIMY)
- [Website](https://craigleili.github.io/projects/genzi/)

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{li2024genzi,
    title     = {{GenZI}: Zero-Shot {3D} Human-Scene Interaction Generation},
    author    = {Li, Lei and Dai, Angela},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024}
}
```

## Install
```shell
# Create a new conda environment
conda create -n genzi python=3.8
conda activate genzi

# Install PyTorch: assume cuda 11.7
conda install -y pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install other libraries
pip install -r requirements.txt
pip install -U --no-deps smplx==0.1.28 git+https://github.com/nghorbani/human_body_prior.git
conda install -y open3d-admin::open3d=0.10.0.0
# Install PyTorch3D: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# Install AlphaPose: https://github.com/MVIG-SJTU/AlphaPose (try Cython==0.29.35 setuptools==65.7.0 if anything complains)
# Install torch-mesh-isect: https://github.com/vchoutas/torch-mesh-isect
# (Try numpy==1.23.4 if numpy complains)
```

## Data

The 3D scene data and our generation results can be found [here](https://1drv.ms/u/s!Alg6Vpe53dEDgrcBClkV5NvqydM9Xg?e=nsyPeU).
Extract the content of the zipped file to the root directory of the code.

## Running

```shell
# See more running settings therein
bash run.sh
```

## References
1. [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face.
1. Fang et al. [AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time](https://github.com/MVIG-SJTU/AlphaPose).
1. Pavlakos et al. [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://github.com/vchoutas/smplx).
1. Hassan et al. [Resolving 3D Human Pose Ambiguities with 3D Scene Constraints](https://github.com/mohamedhassanmus/prox).
1. Zhao et al. [Compositional Human-Scene Interaction Synthesis with Semantic Control](https://github.com/zkf1997/COINS).

---

This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[![Creative Commons License](https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
