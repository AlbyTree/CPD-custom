from torchvision.models import vgg16
import os

# It will automatically download the pre-trained model into the current directory.
# IMPORTANT: the model must be present in the working directory
# as the 'torch.load()' function expect to find files in the working directory.
os.environ["TORCH_MODEL_ZOO"] = "./"
vgg16(pretrained=True)
# Reset variable value to default
os.environ["TORCH_MODEL_ZOO"] = "~/.torch/models"
