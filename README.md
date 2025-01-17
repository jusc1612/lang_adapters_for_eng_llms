# lang_adapters_for_eng_llms

This repository contains all code for my master's thesis entitled 'On the Efficacy of Language Adapters for Cross-lingual Transfer in English-centric LLMs'.

## Setup Instructions

Follow these steps to set up the project before running any scripts:

1. **Clone the repository**  
   Use the following command to clone the project repository:
   ```bash
   git clone https://github.com/jusc1612/lang_adapters_for_eng_llms.git
   cd lang_adapters_for_eng_llms

2. **Create and activate a Conda environment**
   - Depending on the adapter architecture you're using, different Conda environments are required. Stick to the environment names as stated below to avoid errors. 
   - For adapters using the bottleneck architecture, the [Adapters](https://github.com/adapter-hub/adapters) library is used. Use the following command to create an environment with all required dependencies:
     ```bash
     conda create --name adapters python=3.12 -y
     conda activate adapters
     pip install -r requirements_adapters.txt

   - For adapters using a different architecture, the [PEFT](https://github.com/huggingface/peft) library is used. Use the following command to create an environment with all required dependencies:
     ```bash
     conda create --name peft python=3.12 -y
     conda activate peft
     pip install -r requirements_peft.txt

## Training and Evaluation

- To train language adapters, run:
   ```bash
  ./train_la.sh

- To train task adapters, run:
   ```bash
  ./train_ta.sh

- To evaluate cross-lingual transfer using the previously trained task adapters, run:
   ```bash
  ./eval_ta.sh

- To evaluate cross-lingual transfer using in-context learning, run:
   ```bash
  ./eval_icl.sh

Before executing these commands, **check the corresponding bash script** for required customizations and optional training and evaluation modifications. 

## Issues

If you encounter any issues or have questions, please feel free to reach out:

📧 Email: [jusc00031@stud.uni-saarland.de](mailto:usc00031@stud.uni-saarland.de)

 
