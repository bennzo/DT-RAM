import os
import time
import shutil
import pickle

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

from model import RecurrentAttentionDynamic
from utils import AverageMeter


class TrainerDynamic:
    """A Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        self.config = config

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # reinforce DT-RAM params
        self.gamma = config.gamma

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = config.num_classes
        self.num_channels = config.num_channels

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "test_dtram_{}_{}x{}_{}".format(
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )

        self.plot_dir = "./plots/" + self.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttentionDynamic(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
        )
        self.model.to(self.device)

        # freeze location and recurrent network parameters
        for child_module in self.model.children():
            if child_module._get_name() in ['GlimpseNetwork', 'CoreNetwork', 'LocationNetwork']:
                for parameters in child_module.parameters():
                    parameters.requires_grad = False

        # initialize optimizer and scheduler
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=self.config.init_lr
        # )
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.config.init_lr, momentum=self.momentum
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=self.lr_patience
        )

    def reset(self):
        h_t = torch.zeros(
            self.batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True

        return h_t, l_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False, ckpt_name=self.config.ckpt_name, fresh=self.config.resume_fresh)

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
                )
            )

            # train for 1 epoch
            train_loss, train_acc, train_glimpses = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc, valid_glimpses = self.validate(epoch)

            # # reduce lr if validation loss plateaus
            self.scheduler.step(-valid_acc)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} - train glimpses: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f} - val glimpses: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_acc, train_glimpses, valid_loss, valid_acc, 100 - valid_acc, valid_glimpses
                )
            )

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                },
                is_best,
            )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        glimpses = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x, y = x.to(self.device), y.to(self.device)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()

                # save images
                imgs = []
                imgs.append(x[0:9])

                # >>>>>>>>>>>>>>>> DT-RAM ADDITION >>>>>>>>>>>>>>>>
                # extract the glimpses
                locs = []                   # list: locations actions - (x, y)
                log_pi_loc = []             # list: log probability for the location
                baselines = []              # list: baselines for REINFORCE variance reduction
                stops = []                  # list: stopping actions - {0,1}
                log_pi_stops = []           # list: log probability for the stops
                log_probs_class = []        # list: log probability for classification

                # loop over glimpses
                for t in range(self.num_glimpses):
                    # forward pass through model
                    h_t, l_t, b_t, log_probas, p, a_t, log_pi_stop = self.model(x, l_t, h_t)

                    # store
                    locs.append(l_t[0:9])
                    log_pi_loc.append(p)

                    stops.append(a_t)
                    log_pi_stops.append(log_pi_stop)

                    baselines.append(b_t)
                    log_probs_class.append(log_probas)

                # convert list to tensors and reshape to put Batch in the first axis
                baselines = torch.stack(baselines).transpose(1, 0)              # (N, steps)
                stops = torch.stack(stops).transpose(1, 0)                      # (N, steps)
                log_pi_loc = torch.stack(log_pi_loc).transpose(1, 0)            # (N, steps)
                log_pi_stops = torch.stack(log_pi_stops).transpose(1, 0)        # (N, steps)
                log_probs_class = torch.stack(log_probs_class).transpose(1, 0)  # (N, steps, classes)

                # calculate reward
                # create mask for the classification tensor:
                # 1. find the # of step (T(n)) with correct classification
                # 2. zero-out gradients for step n>T(n)
                predictions_per_step = torch.max(log_probs_class, dim=2)[1]                     # (N, steps)
                reward_per_step = (predictions_per_step.detach() == y.unsqueeze(1)).float()     # (N, steps)
                # R, first_correct_step = torch.max(reward_per_step, dim=1)                       # (N)
                first_correct_stop = torch.max(stops[:, 1:] == reward_per_step[:, 1:], dim=1)[1]  # TEST
                first_correct_step = first_correct_stop + 1
                R = reward_per_step[range(self.batch_size), first_correct_step]         # TEST
                R = R * torch.pow(self.gamma, first_correct_step.detach() + 1)

                # compute losses for differentiable module
                loss_classification = F.nll_loss(log_probs_class[:, 1, :], y)
                for glimpse in range(2, self.num_glimpses):
                    loss_classification = loss_classification + F.nll_loss(log_probs_class[:, glimpse, :], y)
                loss_classification = loss_classification / self.num_glimpses

                # compute reinforce loss
                # compute loss only for the first step that classified correctly and average across batch
                loss_reinforce = -log_pi_loc[range(self.batch_size), first_correct_step] * log_pi_stops[range(self.batch_size), first_correct_step] * R
                loss_reinforce = torch.mean(loss_reinforce)

                # sum up into a hybrid loss
                loss = loss_classification + loss_reinforce * 0.01

                # compute accuracy
                correct = (predictions_per_step[range(self.batch_size), first_correct_step] == y).float()
                acc = 100 * (correct.sum() / len(y))
                # <<<<<<<<<<<<<<<< DT-RAM ADDITION <<<<<<<<<<<<<<<<

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])
                glimpses.update(torch.sum(first_correct_step + 1)/torch.sum(R), torch.sum(R))

                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

                # dump the glimpses and locs
                if plot:
                    imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    locs = [l.cpu().data.numpy() for l in locs]
                    pickle.dump(
                        imgs, open(self.plot_dir + "g_{}.p".format(epoch + 1), "wb")
                    )
                    pickle.dump(
                        locs, open(self.plot_dir + "l_{}.p".format(epoch + 1), "wb")
                    )

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("train_loss", losses.avg, iteration)
                    log_value("train_acc", accs.avg, iteration)

            return losses.avg, accs.avg, glimpses.avg

    @torch.no_grad()
    def validate(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        glimpses = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # >>>>>>>>>>>>>>>> DT-RAM ADDITION >>>>>>>>>>>>>>>>
            # extract the glimpses
            locs = []  # list: locations actions - (x, y)
            log_pi_loc = []  # list: log probability for the location
            baselines = []  # list: baselines for REINFORCE variance reduction
            stops = []  # list: stopping actions - {0,1}
            log_pi_stops = []  # list: log probability for the stops
            log_probs_class = []  # list: log probability for classification

            # loop over glimpses
            for t in range(self.num_glimpses):
                # forward pass through model
                h_t, l_t, b_t, log_probas, p, a_t, log_pi_stop = self.model(x, l_t, h_t)

                # store
                locs.append(l_t[0:9])
                log_pi_loc.append(p)

                stops.append(a_t)
                log_pi_stops.append(log_pi_stop)

                baselines.append(b_t)
                log_probs_class.append(log_probas)

            # convert list to tensors and reshape to put Batch in the first axis
            baselines = torch.stack(baselines).transpose(1, 0)  # (N, steps)
            stops = torch.stack(stops).transpose(1, 0)  # (N, steps)
            log_pi_loc = torch.stack(log_pi_loc).transpose(1, 0)  # (N, steps)
            log_pi_stops = torch.stack(log_pi_stops).transpose(1, 0)  # (N, steps)
            log_probs_class = torch.stack(log_probs_class).transpose(1, 0)  # (N, steps, classes)

            # calculate reward
            # create mask for the classification tensor:
            # 1. find the # of step (T(n)) with correct classification
            # 2. zero-out gradients for step n>T(n)
            predictions_per_step = torch.max(log_probs_class, dim=2)[1]                     # (N, steps)
            reward_per_step = (predictions_per_step.detach() == y.unsqueeze(1)).float()     # (N, steps)
            # R, first_correct_step = torch.max(reward_per_step, dim=1)                     # (N)
            first_correct_stop = torch.max(stops[:, 1:] == reward_per_step[:, 1:], dim=1)[1]  # TEST
            first_correct_step = first_correct_stop + 1
            R = reward_per_step[range(self.batch_size), first_correct_step]  # TEST
            R = R * torch.pow(self.gamma, first_correct_step.detach() + 1)

            # compute losses for differentiable module
            loss_classification = F.nll_loss(log_probs_class[:, 0, :], y)
            for glimpse in range(1, self.num_glimpses):
                loss_classification = loss_classification + F.nll_loss(log_probs_class[:, glimpse, :], y)
            loss_classification = loss_classification / self.num_glimpses

            # compute reinforce loss
            # compute loss only for the first step that classified correctly and average across batch
            loss_reinforce = -(log_pi_loc[range(self.batch_size), first_correct_step] + log_pi_stops[range(self.batch_size), first_correct_step]) * R
            loss_reinforce = torch.mean(loss_reinforce)

            # sum up into a hybrid loss
            loss = loss_classification + loss_reinforce * 0.01

            # compute accuracy
            correct = (predictions_per_step[range(self.batch_size), first_correct_step] == y).float()
            acc = 100 * (correct.sum() / len(y))
            # <<<<<<<<<<<<<<<< DT-RAM ADDITION <<<<<<<<<<<<<<<<

            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])
            glimpses.update(torch.sum(first_correct_step + 1)/torch.sum(R), torch.sum(R))

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                log_value("valid_loss", losses.avg, iteration)
                log_value("valid_acc", accs.avg, iteration)

        return losses.avg, accs.avg, glimpses.avg

    @torch.no_grad()
    def test(self):
        """Test the DT-RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0
        glimpses = 0

        # load the best checkpoint
        self.load_checkpoint(best=True, ckpt_name=self.config.ckpt_name, fresh=self.config.resume_fresh)

        for i, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            log_probs_class = []  # list: log probability for classification
            stops = []  # list: stopping actions - {0,1}
            # extract the glimpses
            for t in range(self.num_glimpses):
                # forward pass through model
                h_t, l_t, b_t, log_probas, p, a_t, log_pi_stop = self.model(x, l_t, h_t)

                # store
                log_probs_class.append(log_probas)
                stops.append(a_t)

            log_probs_class = torch.stack(log_probs_class).transpose(1, 0)  # (N, steps, classes)
            stops = torch.stack(stops).transpose(1, 0)  # (N, steps)

            # calculate reward
            # create mask for the classification tensor:
            # 1. find the # of step (T(n)) with correct classification
            # 2. zero-out gradients for step n>T(n)
            predictions_per_step = torch.max(log_probs_class, dim=2)[1]  # (N, steps)
            reward_per_step = (predictions_per_step.detach() == y.unsqueeze(1)).float()  # (N, steps)
            # R, first_correct_step = torch.max(reward_per_step, dim=1)  # (N)
            first_stop = torch.max(stops[:, 1:], dim=1)[1] + 1  # TEST

            log_probas = log_probs_class[range(len(log_probs_class)), first_stop, :]
            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            glimpses += (first_stop + 1).sum()

        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc
        avg_glimpses = int(glimpses) / int(correct)
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%), Avg Glimpses: {:.2f}".format(
                correct, self.num_test, perc, error, avg_glimpses
            )
        )

    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, best=False, fresh=False, ckpt_name=''):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        if len(ckpt_name) > 0:
            filename = ckpt_name
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.model.load_state_dict(ckpt["model_state"], strict=False)   # strict=False for loading RAM into DT-RAM
        if not fresh:
            self.start_epoch = ckpt["epoch"]
            self.best_valid_acc = ckpt["best_valid_acc"]
            self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
