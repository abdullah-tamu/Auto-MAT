# AutoMAT
Automatic mask estimation method for image inpainting
# Requirements
Same requirements as Mask Aware Transformors (MAT). Please refer to 
[https://github.com/fenglinglwb/MAT
](https://github.com/fenglinglwb/MAT) for the requrements.
Also, please install opencv

# Usage:
Downlowd our the FFHQ pretrained model and place it in the main project directory using this link:
[https://drive.google.com/file/d/1D3Q5fdNjWCeLr76bkhBpwmw9XRcvp_Kb/view?usp=drive_link
](https://drive.google.com/file/d/1D3Q5fdNjWCeLr76bkhBpwmw9XRcvp_Kb/view?usp=drive_link)```
```
python AutoMAT_v1.0.py --network pretrained/FFHQ_512.pkl --dpath ./examples --mpath masks/mask.png
# Citation
```
If you find this implementation helpful in your research, please also consider citing:
```
@article{hayajneh2023unsupervised,
  title={Unsupervised anomaly appraisal of cleft faces using a StyleGAN2-based model adaptation technique},
  author={Hayajneh, Abdullah and Shaqfeh, Mohammad and Serpedin, Erchin and Stotland, Mitchell A},
  journal={Plos one},
  volume={18},
  number={8},
  pages={e0288228},
  year={2023},
  publisher={Public Library of Science San Francisco, CA USA}
}
