# Semantic Encoder Guided Generative Adversarial Network For Face Image Super-Resolution



## Proposed Method ##

We propose a novel super-resolution model named Semantic Encoder guided Generative Adversarial Network (SEGAN). Our SEGAN has three key components: Sementic Encoder (as shown in Fig. 1), new architecture for generator(as shown in Fig. 2, 3), and least squares discriminator. In this section, we first discuss the benefits of sementic encoder, then describe the proposed generator architecture. Finally, we illustrate the least squares discriminator used to optimize the adversarial process. The entire architecture of SEGAN is illustrated in Fig. 1.



![new_overview_921](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Network-For-Face-Image-Super-Resolution/blob/master/918_overview%20(1).png)

**Figure 1. Proposed SEGAN and its subnetworks: semantic Encoder $$E$$, Generator $$G$$, Discriminator $$D$$ and Feature Extractor $$\phi$$.** DCB describes the dense connection block. $$I^{HR}$$ and $$I^{LR}$$ denote HR face images and LR face images respectively. $$I^{SR}$$ is super-resolved images from $$G$$. Futhermore, $$E(\cdot)$$ denotes global semantic embedded obtained from $$E$$. Morever $$D(\cdot)$$ represents the output probability of $$D$$. $\phi(I^{HR})$ and $$\phi(I^{SR})$$ describes the features learned by $$\phi$$.



#### Generator

In our generator, a first convolutional layer extracts the low-level feature information of LR face images and passes it to the upsampling block, and then the Multiple Residual Dense Block (MRDB), as Fig. 3 shown, performs high-level feature extraction and transmission. Finally through non-linear mapping, three upsampling layers are used to increase the feature size, and the generator reconstructs the super-resolved face images while improves the resolution by training these three upsampling layers.

In the specific design of the generator architecture, we incorporate residual learning and dense connections to propose the novel block as basic architecture of the generator named the Residual in Internel Dense Block (RIDB) as shown in Fig. 2. 



![Dense Block-RIDB_formal](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Network-For-Face-Image-Super-Resolution/blob/master/Dense%20Block-RIDB_formal.png)

**Figure 2. The basic block used in SEGAN called Residual in Internel Dense Block(RIDB)**. The block of middle describes performing addition operation.

Furthermore, Dense Connection Block (DCB), as illustirted in Fig. 3, is built up by 3 RIDBs and a residual scaling layer behind the last RIDB.



![Dense Block-Dense Block Formal](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Network-For-Face-Image-Super-Resolution/blob/master/Dense%20Block-Dense%20Block%20Formal.png)

**Figure 3. Multiple Residual Dense Block**. RIDB denotes Residual in Internel Dense Block



## Overall objective function

The total loss function $$L_{perceptual}$$ for generator can be represented as weighted combination of two parts: content loss $$L_{content}$$ and adversarial loss $$L_{G}^{adv}$$, the formula are described as follow:

$$ L_{perceptual} = \lambda _{con}L_{Content} + \lambda_{adv}L_{G}^{adv}$$ 																													(4)

where $$\lambda_{con}$$ , $$\lambda_{adv}$$ are the trade-off weights for the $$L_{content}$$ and the $$L_{Adversarial}$$ . We set $$\lambda_{con}$$ , $$\lambda_{adv}$$ empirically to 1 and $$10^{-3}$$ respectively. As a result of the $$L_{perceptual}$$ , Our generator can be updated effectively so that it is able to learn the LR-HR mappling.



## Experiments

#### Datasets

We conducted experiments on public large-scale face image datasets, CelebA and Helen dataset, most of which are frontal facial images. It consists of 202,599 face images of 10,177 celebrities. In our experiments, 162,048 HR face images are randomly selected as the training set, and the next 40,511 images were used as the testing set. 

#### Comparisons with state-of-the-Art Methods

The quantitative comparison among state-of-the-art methods were demonstrated in Table 1. It shown the average PSNR and SSIM criterion of these methods on the CelebA dataset for scale factor 4x, 8x.

##### Quantitative Comparison

![comparsons](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Network-For-Face-Image-Super-Resolution/blob/master/comparsons.png)

**Table1. Quantitative comparison on CelebA dataset for scale factor x4,x8 in terms of average PSNR(db) and SSIM. Numerical  in bold are indicated the best evaluation results among state-of-the-art methods.**

 

##### Qualitative Comparison

Qualitative results were depicted in Fig5. and Fig6. 

![Dense Block-celeba 4x_result (1)](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Network-For-Face-Image-Super-Resolution/blob/master/Dense%20Block-celeba%204x_result%20(1).png)

**Fig5. Qualitative comparison of 4x super-resolved face images on CelebA dataset. Column: (a) HR ground truth; (b) Input LR; (c) Bicubic (d) SRGAN (e) ESRGAN (f) Our method.** 



![Dense Block-celeba_8x_result (1)](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Network-For-Face-Image-Super-Resolution/blob/master/Dense%20Block-celeba_8x_result%20(1).png)

**Fig6. Qualitative comparison of 8x super-resolved face images on CelebA dataset. Column: (a) HR ground truth; (b) Input LR; (c) Bicubic (d) SRGAN (e) ESRGAN (f) Our method.** 

![Helen_4x_result](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Network-For-Face-Image-Super-Resolution/blob/master/Helen_4x_result.png)

**Fig6. Qualitative comparison of 4x super-resolved face images on Helen dataset. Column: (a) HR ground truth; (b) Input LR; (c) Bicubic (d) SRGAN (e) ESRGAN (f) Our method.** 

![Helen_8x_result](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Network-For-Face-Image-Super-Resolution/blob/master/Helen_8x_result.png)

**Fig6. Qualitative comparison of 8x super-resolved face images on Helen dataset. Column: (a) HR ground truth; (b) Input LR; (c) Bicubic (d) SRGAN (e) ESRGAN (f) Our method.** 

## References

[1] Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017.

[2] Wang, Xintao, et al. "Esrgan: Enhanced super-resolution generative adversarial networks." *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018.

[3] Chen, Yu, et al. "Fsrnet: End-to-end learning face super-resolution with facial priors." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018.

[4] S.KolouriandG.K.Rohde.Transport-basedsingleframesuperresolutionofverylowresolutionfaceimages.InProc.IEEEConf.Comp.Vis.Patt.Recogn.,2015.

[5] C.-Y.Yang,S.Liu,andM.-H.Yang.Structuredfacehalluci-nation.InProc.IEEEConf.Comp.Vis.Patt.Recogn.,2013.

[6] Lim, Bee, et al. "Enhanced deep residual networks for single image super-resolution." *Proceedings of the IEEE conference on computer vision and pattern recognition workshops*. 2017.

[7] Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." *European conference on computer vision*. Springer, Cham, 2016.

[8]Yu, Xin, et al. "Super-resolving very low-resolution face images with supplementary attributes." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

[20] Lim, B., Son, S., Kim, H., Nah, S., & Mu Lee, K. (2017). Enhanced deep residual networks for single image super-resolution. In *Proceedings of the IEEE conference on computer vision and pattern recognition workshops* (pp. 136-144).

[11] Zhang, Y., Tian, Y., Kong, Y., Zhong, B., & Fu, Y. (2018). Residual dense network for image super-resolution. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 2472-2481).

[12] Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., & Fu, Y. (2018). Image super-resolution using very deep residual channel attention networks. In *Proceedings of the European Conference on Computer Vision (ECCV)* (pp. 286-301).

[28]Ahn, N., Kang, B., & Sohn, K. A. (2018). Fast, accurate, and lightweight super-resolution with cascading residual network. In *Proceedings of the European Conference on Computer Vision (ECCV)* (pp. 252-268).

[31] Lim, B., Son, S., Kim, H., Nah, S., & Mu Lee, K. (2017). Enhanced deep residual networks for single image super-resolution. In *Proceedings of the IEEE conference on computer vision and pattern recognition workshops* (pp. 136-144).



Method reference :

[1] Yang, J., Wright, J., Huang, T. S., & Ma, Y. (2010). Image super-resolution via sparse representation. *IEEE transactions on image processing*, *19*(11), 2861-2873.

[2] Dong, C., Loy, C. C., He, K., & Tang, X. (2015). Image super-resolution using deep convolutional networks. *IEEE transactions on pattern analysis and machine intelligence*, *38*(2), 295-307.

[3] Ma, X., Zhang, J., & Qi, C. (2010). Hallucinating face by position-patch. *Pattern Recognition*, *43*(6), 2224-2236.

[4] Bora, A., Jalal, A., Price, E., & Dimakis, A. G. (2017). Compressed sensing using generative models. *arXiv preprint arXiv:1703.03208*.

[5] Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). Deep image prior. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 9446-9454).

[6] Yang, C. Y., Liu, S., & Yang, M. H. (2013). Structured face hallucination. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 1099-1106).Yang, C. Y., Liu, S., & Yang, M. H. (2013). Structured face hallucination. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 1099-1106).

[7] Liu, Ce, Heung-Yeung Shum, and William T. Freeman. "Face hallucination: Theory and practice." *International Journal of Computer Vision* 75.1 (2007): 115-134.

[8] Hussein, S. A., Tirer, T., & Giryes, R. (2019). Image-adaptive GAN based reconstruction. *arXiv preprint arXiv:1906.05284*.

[9] Kim, J., Kwon Lee, J., & Mu Lee, K. (2016). Accurate image super-resolution using very deep convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 1646-1654).

[10] Yu, X., & Porikli, F. (2017, February). Face hallucination with tiny unaligned images by transformative discriminative neural networks. In *Thirty-First AAAI conference on artificial intelligence*.

[11] Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., ... & Shi, W. (2017). Photo-realistic single image super-resolution using a generative adversarial network. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4681-4690).

[12] Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., ... & Change Loy, C. (2018). Esrgan: Enhanced super-resolution generative adversarial networks. In *Proceedings of the European Conference on Computer Vision (ECCV)* (pp. 0-0).

[13] Zhu, S., Liu, S., Loy, C. C., & Tang, X. (2016, October). Deep cascaded bi-network for face hallucination. In *European conference on computer vision* (pp. 614-630). Springer, Cham.

[14] Yu, X., & Porikli, F. (2017). Hallucinating very low-resolution unaligned and noisy face images by transformative discriminative autoencoders. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 3760-3768).

[15] Yu, X., Fernando, B., Hartley, R., & Porikli, F. (2018). Super-resolving very low-resolution face images with supplementary attributes. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 908-917).





