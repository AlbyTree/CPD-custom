**IMPORTANT**: This is a modified version of [this](https://github.com/wuzhe71/CPD) repository. I modified the code for my thesis. Read the instructions of the original authors first.  
I used this code for my thesis.
**IMPORTANT**: If you use my script to download the VGG16 model from pytorch check the default value of the pytorch environment variable `TORCH_MODEL_ZOO`: I used this code on Ubuntu 18.04 LTS and `~/.torch/models` was the default value; if the value is different, modify the last instruction of the script file.

* For my thesis I trained the CPD model with these options
  *  NVIDIA GeForce 750 Ti 2 GB VRAM
  *  training dataset = MSRA-5K
  *  backbone model = VGG16
  *  batch size = 2
  *  learning rate = 0.0001
  *  epoch = 10
  *  no validation set  

  I ended up with an average loss of 0.2954(more info [here](https://app.wandb.ai/albytree/cpd-train/runs/fanryq6k/overview?workspace=user-albytree)).
* I added an unique naming of trained models to keep track of my runs.
* I used wandb wrapper code to monitor the training.
* I added more printed info while training and testing.
* I tested my trained model on the datasets PASCAL-S, ECSSD, HKU-IS, DUTS Test Set and DUT-OMRON(you can see the saliency maps in the **'results'** folder).
* You can see quantitative results in this [images](https://github.com/AlbyTree/sod-evaluation-code-custom/tree/master/FIGURES)(there are other models results too as I evaluated and compared multiple models).
