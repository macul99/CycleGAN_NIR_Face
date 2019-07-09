import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Nir2VisModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='resnet_9blocks', max_dataset_size=100000, load_size=112, crop_size=112, no_flip=False, batch_size=32, \
                            serial_batches=False, preprocess='resize_and_crop', num_threads=0) # for both aligned and unaligned
        if is_train:
            parser.set_defaults(gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN_P', 'G_GAN_U', 'G_L1', 'D_P_real', 'D_P_fake', 'D_U_real', 'D_U_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A_pair', 'fake_B_pair', 'real_B_pair', 'real_A_unpair', 'fake_B_unpair', 'real_B_unpair']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D_P', 'D_U']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD_P = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # for paired data
            self.netD_U = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # for unpaired data

        if self.isTrain:
            # define loss functions
            self.fake_B_unpair_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_U = torch.optim.Adam(self.netD_U.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_P)
            self.optimizers.append(self.optimizer_D_U)

    def set_input(self, input_pair, input_unpair):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.set_input_pair(input_pair)
        self.set_input_unpair(input_unpair)

    def set_input_pair(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A_pair = input['A'].to(self.device)
        self.real_B_pair = input['B'].to(self.device)
        self.path_rA_pair = input['A_paths']
        self.path_rB_pair = input['B_paths']

    def set_input_unpair(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A_unpair = input['A'].to(self.device)
        self.real_B_unpair = input['B'].to(self.device)
        self.path_rA_unpair = input['A_paths']
        self.path_rB_unpair = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B_pair = self.netG(self.real_A_pair)  # G(A)
        self.fake_B_unpair = self.netG(self.real_A_unpair)  # G(A)

    def backward_D_P(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A_pair, self.fake_B_pair), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD_P(fake_AB.detach())
        self.loss_D_P_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A_pair, self.real_B_pair), 1)
        pred_real = self.netD_P(real_AB)
        self.loss_D_P_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_P = (self.loss_D_P_fake + self.loss_D_P_real) * 0.5
        self.loss_D_P.backward()

    def backward_D_U(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B_unpair_pool.query(self.fake_B_unpair)
        pred_fake = self.netD_U(fake_B.detach())
        self.loss_D_U_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD_U(self.real_A_unpair)
        self.loss_D_U_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_U = (self.loss_D_U_fake + self.loss_D_U_real) * 0.5
        self.loss_D_U.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A_pair, self.fake_B_pair), 1)
        pred_fake_P = self.netD_P(fake_AB)
        self.loss_G_GAN_P = self.criterionGAN(pred_fake_P, True)
        pred_fake_U = self.netD_U(self.fake_B_unpair)
        self.loss_G_GAN_U = self.criterionGAN(pred_fake_U, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B_pair, self.real_B_pair) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN_P + self.loss_G_GAN_U + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD_P, True)  # enable backprop for D
        self.optimizer_D_P.zero_grad()     # set D's gradients to zero
        self.backward_D_P()                # calculate gradients for D
        self.optimizer_D_P.step()          # update D's weights

        self.set_requires_grad(self.netD_U, True)  # enable backprop for D
        self.optimizer_D_U.zero_grad()     # set D's gradients to zero
        self.backward_D_U()                # calculate gradients for D
        self.optimizer_D_U.step()          # update D's weights

        # update G
        self.set_requires_grad(self.netD_P, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_U, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
