import numpy as np
import os
import time
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import utils
# from utils import *
from Gen_Discr_models import Generator, Discriminator
from molecular_dataset import *
from molecular_dataset import MolecularDataset
import matplotlib.pyplot as plt
# from utils import classification_report, mols2grid_image, reconstructions

class Trainer(object):

    def __init__(self, args, data, idxs, mol_data_dir='data_smiles/qm8-diabetes-drugbank.pkl.dataset'):
        """Initialize configurations."""

        print( "type(data)", type(data) )
        self.args = args

        # Data loader.
        self.data = MolecularDataset()
        self.data.loadNoRandom(mol_data_dir)
        print( "type(self.data)", type(self.data) )

        # Model configurations.
        self.z_dim = args.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = args.g_conv_dim
        self.d_conv_dim = args.d_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.d_repeat_num = args.d_repeat_num
        self.lambda_cls = args.lambda_cls
        self.lambda_rec = args.lambda_rec
        self.lambda_gp = args.lambda_gp
        self.post_method = args.post_method

        # Training configurations.
        self.batch_size = args.batch_size
        self.num_iters_local = args.num_iters_local
        self.num_iters_decay = args.num_iters_decay
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.dropout = args.dropout
        self.n_critic = args.n_critic
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.resume_iters = args.resume_iters
        self.epochs_global = args.epochs_global
        self.frac = args.frac
        self.num_users = args.num_users

        # Test configurations.
        self.test_iters = args.test_iters

        # Miscellaneous.
        self.use_tensorboard = args.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.model_save_dir = args.model_save_dir
        self.result_dir = args.result_dir

        # Step size.
        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step
        self.lr_update_step = args.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
            
    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.z_dim,
                           self.data.vertexes,
                           self.data.bond_num_types,
                           self.data.atom_num_types,
                           self.dropout)
        num_types_delta = self.data.atom_num_types -5
        self.D = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout, num_types_delta=num_types_delta)
        self.V = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout, num_types_delta=num_types_delta)

        #global model with both generator & discriminator
        # self.g_global_model = self.G
        # self.d_global_model = self.D

        self.g_optimizer = torch.optim.Adam(list(self.G.parameters())+list(self.V.parameters()),
                                            self.g_lr, [self.beta1, self.beta2])
        self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, 'min')
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.d_optimizer, 'min')
        # self.print_network(self.G, 'G')
        # self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)
        self.V.to(self.device)

        return self.G, self.D
    
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  #get no. of params iteratively
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size())+[dim]).to(self.device)
        out.scatter_(len(out.size())-1,labels.unsqueeze(-1),1.)
        return out

    def sample_z(self, batch_size):   #draw random samples from normal Gaussian distr.
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    def postprocess(self, inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    

    def tnr(self, modeld, modelg, modelv, global_round, progressDictLocal={}):

        global z
        global edges_hard, nodes_hard
        global edges_hat, nodes_hat

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        self.V.load_state_dict(modelv.state_dict())
        self.D.load_state_dict(modeld.state_dict())
        self.G.load_state_dict(modelg.state_dict())
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        g_epoch_loss = []
        d_epoch_loss = []
        # Start training.
        print('Start training...')
        start_time = time.time()
        g_batch_loss = []
        d_batch_loss = []
        g_valid_loss = []
        d_valid_loss = []
        isValid,opName = False, ""
        n_epoch = 0
        for i in range(start_iters, self.num_iters_local):
            progressDictLocal[i] = {}
            if (i+1) % self.log_step == 0:
                mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                z = self.sample_z(a.shape[0])
                print('[Valid]', '')
                # progressDictLocal[i]["a.shape"] = ( str(a.shape), str( type(a.shape) ) )
            else:
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
                z = self.sample_z(self.batch_size)
                # progressDictLocal[i]["a.shape"] = ( str(a.shape), str( type(a.shape) ) )

            # print( "local_model.data.train_counter", self.data.train_counter )
            # print( "local_model.data.train_count", self.data.train_count )
            if self.data.train_counter + 16 >= self.data.train_count:
              # progressDictLocal[i]["self.data.train_counter"]  = self.data.train_counter
              # progressDictLocal[i]["self.data.train_count"] = self.data.train_count
              # progressDictLocal[i]["self.data.validation_counter"]  = self.data.validation_counter
              # progressDictLocal[i]["self.data.validation_count"] = self.data.validation_count
              if len(g_valid_loss) > 0:
                self.g_scheduler.step(g_valid_loss[-1])
                self.d_scheduler.step(d_valid_loss[-1])
                progressDictLocal[i]['glast_lr'] = ( self.g_scheduler.get_last_lr() , g_valid_loss[-1], len(g_valid_loss) )
                progressDictLocal[i]['dlast_lr'] = ( self.d_scheduler.get_last_lr() , d_valid_loss[-1], len(d_valid_loss) )
              n_epoch += 1

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
            x = torch.from_numpy(x).to(self.device).long()            # Nodes.
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = self.label2onehot(x, self.m_dim)
            z = torch.from_numpy(z).to(self.device).float()
            
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # d_batch_loss = []
            
            # Compute loss with real images.
            logits_real, features_real = self.D(a_tensor, None, x_tensor)
            d_loss_real = - torch.mean(logits_real)

            # Compute loss with fake images.
            edges_logits, nodes_logits = self.G(z)
            
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            d_loss_fake = torch.mean(logits_fake)

            # Compute loss for gradient penalty.
            eps = torch.rand(logits_real.size(0),1,1,1).to(self.device)
            x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
            grad0, grad1 = self.D(x_int0, None, x_int1)
            d_loss_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

            # Backward and optimize.
            d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
            self.reset_grad()
            if (i+1) % self.log_step != 0:
              d_loss.backward()
              self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            # g_batch_loss = []
            if (i+1) % self.n_critic == 0:
                
                # Z-to-target
                edges_logits, nodes_logits = self.G(z)

                # Postprocess with Gumbel softmax
                (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
                logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
                g_loss_fake = - torch.mean(logits_fake)

                (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
                edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
                # print(edges_hard)
                # print(nodes_hard)
                # print(edges_hard.shape)
                # print(nodes_hard.shape)
                mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                        for e_, n_ in zip(edges_hard, nodes_hard)]

                # Value loss
                value_logit_real,_ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
                value_logit_fake,_ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)
                g_loss_value = torch.mean((value_logit_real) ** 2 + (value_logit_fake) ** 2)

                # Backward and optimize.
                g_loss = g_loss_fake + g_loss_value
                self.reset_grad()
                if (i+1) % self.log_step != 0:
                  g_loss.backward()
                  self.g_optimizer.step()
                else:
                  g_valid_loss.append(g_loss.item())
                  d_valid_loss.append(d_loss.item())
                  # self.g_scheduler.step(g_loss.item())
                  # self.d_scheduler.step(d_loss.item())
                  # progressDictLocal[i]['glast_lr'] = self.g_scheduler.get_last_lr()
                  # progressDictLocal[i]['dlast_lr'] = self.d_scheduler.get_last_lr()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_value'] = g_loss_value.item()

                d_batch_loss.append(d_loss.item())
                g_batch_loss.append(g_loss.item())

                # print(d_batch_loss, g_batch_loss)

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(timedelta(seconds=et))[:-7]
                log = "Global Round [{}], Elapsed [{}], Iteration [{}/{}]".format(global_round, et, i+1, self.num_iters_local)

                # Log update
                m0, m1 = utils.all_scores(mols, self.data, sample=True, norm=True)     #'mols' is output of Fake Reward
                m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
                m0.update(m1)
                loss.update(m0)
                # for tag, value in loss.items():
                #     log += ", {}: {:.4f}".format(tag, value)
                # print(log)
                # progressDictLocal[i]['log'] = log

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)
   

            # Save model checkpoints.
            # if (i+1) % self.model_save_step == 0:
            #     G_path = os.path.join(self.model_save_dir, '{}-{}-G.ckpt'.format(global_round, i+1))
            #     D_path = os.path.join(self.model_save_dir, '{}-{}-D.ckpt'.format(global_round, i+1))
            #     V_path = os.path.join(self.model_save_dir, '{}-{}-V.ckpt'.format(global_round, i+1))
            #     torch.save(self.G.state_dict(), G_path)
            #     torch.save(self.D.state_dict(), D_path)
            #     torch.save(self.V.state_dict(), V_path)
            #     print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # progressDictLocal[i]["loss"] = str( loss )
            # progressDictLocal[i]["type(loss)"] = str( type(loss) )
            progressDictLocal[i]["loss-items"] = {}
            for tag, value in loss.items():
              progressDictLocal[i]["loss-items"][tag] = str(value)
            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters_local - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
                progressDictLocal[i][("Decayed learning rates", "g_lr, d_lr")] = (g_lr, d_lr)

        d_epoch_loss.append(sum(d_batch_loss[-2:])/len(d_batch_loss[-2:]))
        g_epoch_loss.append(sum(g_batch_loss[-2:])/len(g_batch_loss[-2:]))
        progressDictLocal["len(d_batch_loss)"] = len(d_batch_loss)
        progressDictLocal["sum(d_batch_loss)"] = sum(d_batch_loss)
        progressDictLocal["sum(d_batch_loss[-2:])/len(d_batch_loss[-2:])"] = sum(d_batch_loss[-2:])/len(d_batch_loss[-2:])
        progressDictLocal["sum(g_batch_loss[-2:])/len(g_batch_loss[-2:])"] = sum(g_batch_loss[-2:])/len(g_batch_loss[-2:])
        progressDictLocal["sum(d_batch_loss)/len(d_batch_loss)"] = sum(d_batch_loss)/len(d_batch_loss)
        progressDictLocal["sum(g_batch_loss)/len(g_batch_loss)"] = sum(g_batch_loss)/len(g_batch_loss)
        progressDictLocal["d_valid_loss"] = d_valid_loss
        progressDictLocal["g_valid_loss"] = g_valid_loss
        progressDictLocal["d_batch_loss"] = d_batch_loss
        progressDictLocal["g_batch_loss"] = g_batch_loss
        progressDictLocal["n_epoch"] = n_epoch

        # report = utils.report(self.data)
        # print(report) 

        # recons_mols = utils.reconstructed_mols(self.data, sample=True)
        # print(recons_mols)

        # return self.G.state_dict(), self.D.state_dict(), sum(g_epoch_loss) / len(g_epoch_loss), sum(d_epoch_loss) / len(d_epoch_loss)
        return self.G.state_dict(), self.D.state_dict(), self.V.state_dict(), g_epoch_loss, d_epoch_loss

    # def misc_step(self, mols, loss, start_time):
    def misc_step(self, mols, loss):
      # et = time.time() - start_time
      # et = str(timedelta(seconds=et))[:-7]
      # log = "Global Round [{}], Elapsed [{}], Iteration [{}/{}]".format(global_round, et, i+1, self.num_iters_local)

      # Log update
      m0, m1 = utils.all_scores(mols, self.data, sample=True, norm=True)     #'mols' is output of Fake Reward
      m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
      m0.update(m1)
      loss.update(m0)

    def lr_step(self, progressDictLocal, i, g_valid_loss, d_valid_loss):
      # if self.data.train_counter + self.batch_size >= self.data.train_count:
        if len(g_valid_loss) > 0:
          self.g_scheduler.step(g_valid_loss[-1])
          self.d_scheduler.step(d_valid_loss[-1])
          progressDictLocal[i]['glast_lr'] = ( self.g_scheduler.get_last_lr() , g_valid_loss[-1], len(g_valid_loss) )
          progressDictLocal[i]['dlast_lr'] = ( self.d_scheduler.get_last_lr() , d_valid_loss[-1], len(d_valid_loss) )
        # n_epoch += 1
        pass

    def prep_step(self, islog_step=False):
      if islog_step:
        mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
        z = self.sample_z(a.shape[0])
        print('[Valid]', '')
      else:
        mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
        z = self.sample_z(self.batch_size)

      a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
      x = torch.from_numpy(x).to(self.device).long()            # Nodes.
      a_tensor = self.label2onehot(a, self.b_dim)
      x_tensor = self.label2onehot(x, self.m_dim)
      z = torch.from_numpy(z).to(self.device).float()
      return a, x, a_tensor, x_tensor, z

    def d_step(self, d_batch_loss, d_valid_loss, a_tensor, x_tensor, z, loss, islog_step=False):
      # if islog_step == True:
      #   d_valid_loss.append(d_loss.item())
      # d_batch_loss.append(d_loss.item())

      logits_real, features_real = self.D(a_tensor, None, x_tensor)
      d_loss_real = - torch.mean(logits_real)

      # Compute loss with fake images.
      edges_logits, nodes_logits = self.G(z)
      
      # Postprocess with Gumbel softmax
      (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
      logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
      d_loss_fake = torch.mean(logits_fake)

      # Compute loss for gradient penalty.
      eps = torch.rand(logits_real.size(0),1,1,1).to(self.device)
      x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
      x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
      grad0, grad1 = self.D(x_int0, None, x_int1)
      d_loss_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

      # Backward and optimize.
      d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
      self.reset_grad()
      # if (i+1) % self.log_step != 0:
      if islog_step != True:
        d_loss.backward()
        self.d_optimizer.step()

      # Logging.
      # loss = {}
      # loss['D/loss_real'] = d_loss_real.item()
      # loss['D/loss_fake'] = d_loss_fake.item()
      # loss['D/loss_gp'] = d_loss_gp.item()

      if islog_step == True:
        d_valid_loss.append(d_loss.item())
        loss['D/loss_real'] = d_loss_real.item()
        loss['D/loss_fake'] = d_loss_fake.item()
        loss['D/loss_gp'] = d_loss_gp.item()
      else:
        d_batch_loss.append(d_loss.item())

    def g_step(self, g_batch_loss, g_valid_loss, z, loss, a_tensor, x_tensor, islog_step=False):
      mols = None
      # Z-to-target
      edges_logits, nodes_logits = self.G(z)

      # Postprocess with Gumbel softmax
      (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
      logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
      g_loss_fake = - torch.mean(logits_fake)

      (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
      edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]

      if islog_step == True:
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]

      # Value loss
      value_logit_real,_ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
      value_logit_fake,_ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)
      g_loss_value = torch.mean((value_logit_real) ** 2 + (value_logit_fake) ** 2)

      # Backward and optimize.
      g_loss = g_loss_fake + g_loss_value
      self.reset_grad()
      # if (i+1) % self.log_step != 0:
      if islog_step != True:
        g_loss.backward()
        self.g_optimizer.step()
        g_batch_loss.append(g_loss.item())
      else:
        g_valid_loss.append(g_loss.item())
        # d_valid_loss.append(d_loss.item())
        loss['G/loss_fake'] = g_loss_fake.item()
        loss['G/loss_value'] = g_loss_value.item()

      # loss['G/loss_fake'] = g_loss_fake.item()
      # loss['G/loss_value'] = g_loss_value.item()

      # d_batch_loss.append(d_loss.item())
      # g_batch_loss.append(g_loss.item())
      return mols

    def tnr_sequence_gan(self, modeld, modelg, modelv, global_round, d_steps, g_steps, progressDictLocal={}):
      #  = d_steps / g_steps
      # Learning rate cache for decaying.
      g_lr = self.g_lr
      d_lr = self.d_lr

      # Start training from scratch or resume training.
      self.V.load_state_dict(modelv.state_dict())
      self.D.load_state_dict(modeld.state_dict())
      self.G.load_state_dict(modelg.state_dict())
      # start_iters = 0
      i = 0 # start_iters = 0

      g_epoch_loss = []
      d_epoch_loss = []
      print('Start training...')
      # start_time = time.time()
      g_batch_loss = []
      d_batch_loss = []
      g_valid_loss = []
      d_valid_loss = []
      isValid,opName = False, ""
      n_epoch = 0

      # print("i, d_steps, g_steps, self.num_iters_local", i, d_steps, g_steps, self.num_iters_local)
      # print("i > self.num_iters_local", i > self.num_iters_local)
      # for i in range(start_iters, self.num_iters_local):
      while i < self.num_iters_local:
        # print(i, "i > self.num_iters_local")
        progressDictLocal[i] = {}
        loss = {}
        for i_d in range(i, i+d_steps): progressDictLocal[i_d] = {}
        for i_g in range(i, i+g_steps): progressDictLocal[i_g] = {}
        # for _ in range(g_steps):
        # for _ in range(g_steps):
        for i_d in range(i, i+d_steps):
          a, x, a_tensor, x_tensor, z = self.prep_step(islog_step=False)
          self.d_step(d_batch_loss, d_valid_loss, a_tensor, x_tensor, z, loss, islog_step=False)
          if self.data.train_counter + self.batch_size >= self.data.train_count:
            self.lr_step(progressDictLocal, i_d, g_valid_loss, d_valid_loss)
            n_epoch += 1

        for i_g in range(i, i+g_steps):
          _ = self.g_step(g_batch_loss, g_valid_loss, z, loss, a_tensor, x_tensor, islog_step=False)
          pass

        # if (i+1) % self.log_step == 0:
        #   pass

        if d_steps >= g_steps:
          i_new = i + d_steps
        else:
          i_new = i + g_steps

        for i_islog_step in range(i,i_new):
          loss = {}
          if ( i_islog_step +1 ) % self.log_step == 0:
            a, x, a_tensor, x_tensor, z = self.prep_step(islog_step=True)
            self.d_step(d_batch_loss, d_valid_loss, a_tensor, x_tensor, z, loss, islog_step=True)
            mols = self.g_step(g_batch_loss, g_valid_loss, z, loss, a_tensor, x_tensor, islog_step=True)
            self.misc_step(mols, loss)
            pass
            progressDictLocal[i_islog_step]["loss-items"] = {}
            for tag, value in loss.items():
              progressDictLocal[i_islog_step]["loss-items"][tag] = str(value)
        i = i_new

      d_epoch_loss.append(sum(d_batch_loss[-2:])/len(d_batch_loss[-2:]))
      g_epoch_loss.append(sum(g_batch_loss[-2:])/len(g_batch_loss[-2:]))
      progressDictLocal["len(d_batch_loss)"] = len(d_batch_loss)
      progressDictLocal["sum(d_batch_loss)"] = sum(d_batch_loss)
      progressDictLocal["sum(d_batch_loss[-2:])/len(d_batch_loss[-2:])"] = sum(d_batch_loss[-2:])/len(d_batch_loss[-2:])
      progressDictLocal["sum(g_batch_loss[-2:])/len(g_batch_loss[-2:])"] = sum(g_batch_loss[-2:])/len(g_batch_loss[-2:])
      progressDictLocal["sum(d_batch_loss)/len(d_batch_loss)"] = sum(d_batch_loss)/len(d_batch_loss)
      progressDictLocal["sum(g_batch_loss)/len(g_batch_loss)"] = sum(g_batch_loss)/len(g_batch_loss)
      progressDictLocal["d_valid_loss"] = d_valid_loss
      progressDictLocal["g_valid_loss"] = g_valid_loss
      progressDictLocal["d_batch_loss"] = d_batch_loss
      progressDictLocal["g_batch_loss"] = g_batch_loss
      progressDictLocal["d_steps"] = d_steps
      progressDictLocal["g_steps"] = g_steps
      progressDictLocal["n_epoch"] = n_epoch
      return self.G.state_dict(), self.D.state_dict(), self.V.state_dict(), g_epoch_loss, d_epoch_loss

# def tnr(self, model, global_round):

#         global z
#         global edges_hard, nodes_hard
#         global edges_hat, nodes_hat

#         # Learning rate cache for decaying.
#         g_lr = self.g_lr
#         d_lr = self.d_lr

#         # Start training from scratch or resume training.
#         start_iters = 0
#         if self.resume_iters:
#             start_iters = self.resume_iters
#             self.restore_model(self.resume_iters)

#         g_epoch_loss = []
#         d_epoch_loss = []
#         # Start training.
#         print('Start training...')
#         start_time = time.time()
#         for i in range(start_iters, self.num_iters_local):
#             if (i+1) % self.log_step == 0:
#                 mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
#                 z = self.sample_z(a.shape[0])
#                 print('[Valid]', '')
#             else:
#                 mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
#                 z = self.sample_z(self.batch_size)

#             # =================================================================================== #
#             #                             1. Preprocess input data                                #
#             # =================================================================================== #

#             a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
#             x = torch.from_numpy(x).to(self.device).long()            # Nodes.
#             a_tensor = self.label2onehot(a, self.b_dim)
#             x_tensor = self.label2onehot(x, self.m_dim)
#             z = torch.from_numpy(z).to(self.device).float()
            
#             # =================================================================================== #
#             #                             2. Train the discriminator                              #
#             # =================================================================================== #
#             d_batch_loss = []
            
#             # Compute loss with real images.
#             logits_real, features_real = self.D(a_tensor, None, x_tensor)
#             d_loss_real = - torch.mean(logits_real)

#             # Compute loss with fake images.
#             edges_logits, nodes_logits = self.G(z)
            
#             # Postprocess with Gumbel softmax
#             (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
#             logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
#             d_loss_fake = torch.mean(logits_fake)

#             # Compute loss for gradient penalty.
#             eps = torch.rand(logits_real.size(0),1,1,1).to(self.device)
#             x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
#             x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
#             grad0, grad1 = self.D(x_int0, None, x_int1)
#             d_loss_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

#             # Backward and optimize.
#             d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
#             self.reset_grad()
#             d_loss.backward()
#             self.d_optimizer.step()

#             # Logging.
#             loss = {}
#             loss['D/loss_real'] = d_loss_real.item()
#             loss['D/loss_fake'] = d_loss_fake.item()
#             loss['D/loss_gp'] = d_loss_gp.item()

#             # =================================================================================== #
#             #                               3. Train the generator                                #
#             # =================================================================================== #
#             g_batch_loss = []
#             if (i+1) % self.n_critic == 0:
                
#                 # Z-to-target
#                 edges_logits, nodes_logits = self.G(z)

#                 # Postprocess with Gumbel softmax
#                 (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
#                 logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
#                 g_loss_fake = - torch.mean(logits_fake)

#                 (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
#                 edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
#                 # print(edges_hard)
#                 # print(nodes_hard)
#                 # print(edges_hard.shape)
#                 # print(nodes_hard.shape)
#                 mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
#                         for e_, n_ in zip(edges_hard, nodes_hard)]

#                 # Value loss
#                 value_logit_real,_ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
#                 value_logit_fake,_ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)
#                 g_loss_value = torch.mean((value_logit_real) ** 2 + (value_logit_fake) ** 2)

#                 # Backward and optimize.
#                 g_loss = g_loss_fake + g_loss_value
#                 self.reset_grad()
#                 g_loss.backward()
#                 self.g_optimizer.step()

#                 # Logging.
#                 loss['G/loss_fake'] = g_loss_fake.item()
#                 loss['G/loss_value'] = g_loss_value.item()

#                 d_batch_loss.append(d_loss.item())
#                 g_batch_loss.append(g_loss.item())

#                 # print(d_batch_loss, g_batch_loss)

#             # =================================================================================== #
#             #                                 4. Miscellaneous                                    #
#             # =================================================================================== #

#             # Print out training information.
#             if (i+1) % self.log_step == 0:
#                 et = time.time() - start_time
#                 et = str(timedelta(seconds=et))[:-7]
#                 log = "Global Round [{}], Elapsed [{}], Iteration [{}/{}]".format(global_round, et, i+1, self.num_iters_local)

#                 # Log update
#                 m0, m1 = utils.all_scores(mols, self.data, sample=True, norm=True)     #'mols' is output of Fake Reward
#                 m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
#                 m0.update(m1)
#                 loss.update(m0)
#                 for tag, value in loss.items():
#                     log += ", {}: {:.4f}".format(tag, value)
#                 print(log)

#                 if self.use_tensorboard:
#                     for tag, value in loss.items():
#                         self.logger.scalar_summary(tag, value, i+1)
   

#             # Save model checkpoints.
#             if (i+1) % self.model_save_step == 0:
#                 G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
#                 torch.save(self.G.state_dict(), G_path)
#                 print('Saved model checkpoints into {}...'.format(self.model_save_dir))

#             # Decay learning rates.
#             if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters_local - self.num_iters_decay):
#                 g_lr -= (self.g_lr / float(self.num_iters_decay))
#                 self.update_lr(g_lr)
#                 print ('Decayed learning rates, g_lr: {}'.format(g_lr))
                
#         g_epoch_loss.append(sum(g_batch_loss)/len(g_batch_loss))

#         # report = utils.report(self.data)
#         # print(report) 

#         # recons_mols = utils.reconstructed_mols(self.data, sample=True)
#         # print(recons_mols)

#         # return self.G.state_dict(), self.D.state_dict(), sum(g_epoch_loss) / len(g_epoch_loss), sum(d_epoch_loss) / len(d_epoch_loss)
#         return self.G.state_dict(), self.D.state_dict(), g_epoch_loss, d_epoch_loss



    def test(self):
        # Load the trained generator.
        global edges_hard, nodes_hard
        global edges_hat, nodes_hat
        self.restore_model(self.test_iters)

        reportDict = {}
        reportObjDict = {}
        with torch.no_grad():
            mols, _, _, a, x, _, _, _, _ = self.data.next_test_batch()
            reportDict["a (next_test_batch) ; type"]  = str( type(a) )
            reportDict["a (next_test_batch) ; shape"] = str( a.shape )
            reportObjDict["a (next_test_batch)"] = a
            z = self.sample_z(a.shape[0])
            reportDict["z (sample_z)"] = str( type(z) )
            reportDict["z (sample_z) ; shape"] = str( z.shape )
            reportObjDict["z (sample_z)"] = z
            z = torch.from_numpy(z)
            reportDict["z (from_numpy)"] = str( type(z) )
            reportDict["z (from_numpy) ; shape"] = str( z.shape )
            reportObjDict["z (from_numpy)"] = z

            # Z-to-target
            edges_logits, nodes_logits = self.G(z.float().to(self.device))
            reportDict["edges_logits G"] = str( type(edges_logits) )
            reportDict["edges_logits G ; shape"] = str( edges_logits.shape )
            reportObjDict["edges_logits G"] = edges_logits
            reportDict["nodes_logits G"] = str( type(nodes_logits) )
            reportDict["nodes_logits G ; shape"] = str( nodes_logits.shape )
            reportObjDict["nodes_logits G"] = nodes_logits
            
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            reportObjDict["edges_hat"] = edges_hat
            reportObjDict["nodes_hat"] = nodes_hat

            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            g_loss_fake = - torch.mean(logits_fake)
            reportObjDict["logits_fake D"] = logits_fake
            reportObjDict["features_fake D"] = features_fake

            # Preprocess with hard Gumbel
            (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
            edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
            reportObjDict["edges_hard"] = edges_hard
            reportObjDict["nodes_hard"] = nodes_hard
            reportDict["nodes_hard[0].data type"] = str( type( nodes_hard[0].data ) )
            reportDict["nodes_hard[0].data shape"] = str( nodes_hard[0].data.shape )
            reportDict["edges_hard[0].data type"] = str( type( edges_hard[0].data ) )
            reportDict["edges_hard[0].data shape"] = str( edges_hard[0].data.shape )
            mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                    for e_, n_ in zip(edges_hard, nodes_hard)]

            # Print out testing information.
            start_time = time.time()
            start_iters = self.test_iters
            num_iters_local = 1010
            reportDict["start_iters"] = start_iters
            reportDict["num_iters_local"] = num_iters_local
            for i in range(start_iters, num_iters_local):
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, num_iters_local)
                    # Log update
                    m0, m1 = utils.all_scores(mols, self.data, sample=True, norm=True)     # 'mols' is output of Fake Reward
                    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
                    m0.update(m1)
                    
                    for tag, value in m0.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            # report = utils.report(self.data)
            # print(report) 

            recons_mols = utils.reconstructed_mols(self.data, sample=True)
            
            #print the reconstructed image
            recons_image = utils.mols2grid_image(recons_mols[:30], molsPerRow = 5)
            recons_image.save("fedgan5/recons_image.png", dpi=(1000, 1000))
            recons_image.show()

            #print generated image
            # img = utils.mols2grid_image(mols[:5], molsPerRow = 5) 
            # img.show()
            # img.save("/Users/daniel/Desktop/PhD materials/Fed-GNN-GAN/fedgan/mols_img/mols_grid.png")

        import pprint
        print( "reportDict", reportDict )
        print( pprint.pformat(reportDict, indent=2, sort_dicts=False) )
        for i,kv in enumerate( reportObjDict.items()):
          k,v = kv
          print(i, k, type(v), v.shape)
