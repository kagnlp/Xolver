# Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team

### [Project Page](https://kagnlp.github.io/xolver.github.io/) | [Paper]() 

This is a preliminary implementation of the paper "Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team". More tasks and settings will be released soon. You may see some additional Xolver logs [here](https://drive.google.com/drive/folders/1O-KYcgQcEniIGfxbUcQjyZLjAzUJkr0s?usp=sharing).

[Md Tanzib Hosain](https://scholar.google.com/citations?user=3YexY9gAAAAJ&hl=en),
[Salman Rahman](https://scholar.google.com/citations?user=vr7uTc8AAAAJ&hl=en&oi=ao),
[Md Kishor Morol](https://scholar.google.com/citations?user=pjn3jg4AAAAJ&hl=en),
[Md Rizwan Parvez](https://scholar.google.com/citations?user=KhC8rtcAAAAJ&hl=en)

## Running Project

To clone the project, run:

	git clone https://github.com/kraritt/Xolver && cd Xolver

The code for running GSM, AIME, MATH and LiveCodeBench tasks may be found in the following subfolders

* ./gsm/ contains code for running GSM
* ./aime/ contains code for running AIME
* ./math/ contains code for running MATH
* ./lcb/ contains code for running LiveCodeBench results

To prepare a new conda or python virtual environment, run:
	`pip install -r requirements.txt`

**GSM:**

To generate and evaluated answer for GSM problems through Xolver, run:
	`python gsm/gsm.py`

You can download the GSM dataset [here](https://huggingface.co/datasets/openai/gsm8k)

 **AIME:**

To generate and evaluated answer for AIME problems through Xolver, run:
	`python aime/aime.py`

You can download the AIME datasets [here](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) and [here](https://huggingface.co/datasets/yentinglin/aime_2025) 

**MATH:**

To generate and evaluated answer for MATH problems through Xolver, run:
	`python math/math.py`

 You can download the MATH dataset [here](https://huggingface.co/datasets/di-zhang-fdu/MATH500)

 **LiveCodeBench:**

To generate and evaluated answer for LiveCodeBench problems through Xolver, run:
	`python lcb/lcb.py`

 You can download the LiveCodeBench dataset [here](https://huggingface.co/datasets/livecodebench/code_generation)


You can download the math external retrieval corpus [here](https://huggingface.co/datasets/nvidia/OpenMathReasoning) and the code external retrieval corpus [here](https://huggingface.co/datasets/krarit/retrivcod)

If you would like to cite the paper, here is a bibtex file:
```
@article{hosain2025xolver,
      title={𝕏olver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team}, 
      author={Md Tanzib Hosain and Salman Rahman and Md Kishor Morol and Md Rizwan Parvez},
      journal={arXiv preprint},
      year={2025}
}
```
