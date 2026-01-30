import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
from tqdm import tqdm

from data.LEVIR_CC import LEVIRCCDataset
from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *


class Trainer(object):
    """Caption-only trainer (RSCC on LEVIR-CC).

    Fine-grained recognition / change detection branch has been removed.
    This trainer only optimizes the captioning loss.
    """

    def __init__(self, args):
        self.args = args
        if args.train_goal != 1:
            raise ValueError("This repo has been modified to caption-only (RSCC). "
                             "Please set --train_goal 1.")

        random_str = str(random.randint(10, 100))
        name = 'rscc_levircc_' + time_file_str() + f'_train_goal_{args.train_goal}_' + random_str
        self.args.savepath = os.path.join(args.savepath, name)
        os.makedirs(self.args.savepath, exist_ok=True)

        self.log = open(os.path.join(self.args.savepath, f'{name}.log'), 'w')
        print_log(f'=>dataset: {args.data_name}', self.log)
        print_log(f'=>network: {args.network}', self.log)
        print_log(f'=>encoder_lr: {args.encoder_lr}', self.log)
        print_log(f'=>decoder_lr: {args.decoder_lr}', self.log)
        print_log(f'=>num_epochs: {args.num_epochs}', self.log)
        print_log(f'=>train_batchsize: {args.train_batchsize}', self.log)

        self.best_bleu4 = 0.0

        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)

        self.build_model()

        # Captioning loss
        self.criterion_cap = torch.nn.CrossEntropyLoss().cuda()

        # Dataloaders (seg_label is ignored / dummy)
        self.train_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file,
                           args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)

        self.val_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file,
                           args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

        # Stats: [batch_time, cap_loss, top5_acc]
        self.index_i = 0
        self.hist = np.zeros((args.num_epochs * len(self.train_loader), 3))

    def build_model(self):
        args = self.args

        self.encoder = Encoder(args.network)
        self.encoder.fine_tune(args.fine_tune_encoder)

        self.encoder_trans = AttentiveEncoder(
            train_stage=args.train_stage,
            n_layers=args.n_layers,
            feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
            heads=args.n_heads,
            dropout=args.dropout
        )
        self.decoder = DecoderTransformer(
            encoder_dim=args.encoder_dim,
            feature_dim=args.feature_dim,
            vocab_size=len(self.word_vocab),
            max_lengths=args.max_length,
            word_vocab=self.word_vocab,
            n_head=args.n_heads,
            n_layers=args.decoder_n_layers,
            dropout=args.dropout
        )

        # Optionally load checkpoint
        if args.train_stage == 's2':
            if args.checkpoint is None:
                raise ValueError('Error: checkpoint is None for train_stage=s2.')
            checkpoint = torch.load(args.checkpoint)
            print('Load Model from {}'.format(args.checkpoint))
            self.decoder.load_state_dict(checkpoint['decoder_dict'])
            self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
            self.encoder.load_state_dict(checkpoint['encoder_dict'])

            args.fine_tune_encoder = False
            self.encoder.fine_tune(args.fine_tune_encoder)
            self.encoder_trans.fine_tune(args.train_goal)
            self.decoder.fine_tune(True)

        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            params=self.encoder.parameters(),
            lr=args.encoder_lr
        ) if args.fine_tune_encoder else None

        self.encoder_trans_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
            lr=args.encoder_lr
        )
        self.decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=args.decoder_lr
        )

        # Move to GPU
        self.encoder = self.encoder.cuda()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder = self.decoder.cuda()

        # LR schedulers
        self.encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.encoder_optimizer, step_size=5, gamma=1.0
        ) if args.fine_tune_encoder else None
        self.encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.encoder_trans_optimizer, step_size=5, gamma=1.0
        )
        self.decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.decoder_optimizer, step_size=5, gamma=1.0
        )

    def training(self, args, epoch):
        self.encoder.train() if self.encoder is not None else None
        self.encoder_trans.train()
        self.decoder.train()

        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad()
        self.encoder_trans_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()

        for batch_id, (imgA, imgB, _seg_label, _token_all, _token_all_len, token, token_len, _name) in enumerate(self.train_loader):
            start_time = time.time()
            accum_steps = max(1, 64 // args.train_batchsize)

            imgA = imgA.cuda()
            imgB = imgB.cuda()
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()

            feat1, feat2 = self.encoder(imgA, imgB)
            feat1, feat2 = self.encoder_trans(feat1, feat2)

            scores, caps_sorted, decode_lengths, sort_ind = self.decoder(feat1, feat2, token, token_len)

            targets = caps_sorted[:, 1:]
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            cap_loss = self.criterion_cap(scores_packed, targets_packed.to(torch.int64))

            (cap_loss / accum_steps).backward()

            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(self.decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(self.encoder_trans.parameters(), args.grad_clip)
                if self.encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.encoder.parameters(), args.grad_clip)

            if (batch_id + 1) % accum_steps == 0 or (batch_id + 1) == len(self.train_loader):
                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.step()
                self.encoder_trans_optimizer.step()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.step()

                if self.decoder_lr_scheduler is not None:
                    self.decoder_lr_scheduler.step()
                self.encoder_trans_lr_scheduler.step()
                if self.encoder_lr_scheduler is not None:
                    self.encoder_lr_scheduler.step()

                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.zero_grad()
                self.encoder_trans_optimizer.zero_grad()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()

            # logging stats
            self.hist[self.index_i, 0] = time.time() - start_time
            self.hist[self.index_i, 1] = cap_loss.item()
            self.hist[self.index_i, 2] = accuracy(scores_packed, targets_packed, 5)
            self.index_i += 1

            if (batch_id + 1) % args.print_freq == 0:
                bt = np.mean(self.hist[self.index_i - args.print_freq:self.index_i - 1, 0]) * args.print_freq
                cl = np.mean(self.hist[self.index_i - args.print_freq:self.index_i - 1, 1])
                top5 = np.mean(self.hist[self.index_i - args.print_freq:self.index_i - 1, 2])
                print_log(f'Epoch: {epoch} | Batch: {batch_id + 1}/{len(self.train_loader)} | '
                          f'BatchTime(sum): {bt:.3f} | CapLoss: {cl:.5f} | Top5: {top5:.5f}', self.log)

    def validation(self, epoch):
        word_vocab = self.word_vocab
        self.decoder.eval()
        self.encoder_trans.eval()
        if self.encoder is not None:
            self.encoder.eval()

        references = []
        hypotheses = []

        val_start_time = time.time()
        with torch.no_grad():
            for ind, (imgA, imgB, _seg_label, token_all, _token_all_len, _token, _token_len, _name) in enumerate(
                    tqdm(self.val_loader, desc='val_' + "EVALUATING AT BEAM SIZE " + str(1))):
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                token_all = token_all.squeeze(0).cuda()

                feat1, feat2 = self.encoder(imgA, imgB)
                feat1, feat2 = self.encoder_trans(feat1, feat2)

                seq = self.decoder.sample(feat1, feat2, k=1)

                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                                      img_token))
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)

        val_time = time.time() - val_start_time
        score_dict = get_eval_score(references, hypotheses)
        Bleu_4 = score_dict['Bleu_4']

        print_log('Captioning_Validation:\n'
                  f'Time: {val_time:.3f}\t'
                  f"BLEU-1: {score_dict['Bleu_1']:.5f}\t"
                  f"BLEU-2: {score_dict['Bleu_2']:.5f}\t"
                  f"BLEU-3: {score_dict['Bleu_3']:.5f}\t"
                  f"BLEU-4: {score_dict['Bleu_4']:.5f}\t"
                  f"METEOR: {score_dict['METEOR']:.5f}\t"
                  f"ROUGE_L: {score_dict['ROUGE_L']:.5f}\t"
                  f"CIDEr: {score_dict['CIDEr']:.5f}\t",
                  self.log)

        if Bleu_4 > self.best_bleu4:
            self.best_bleu4 = Bleu_4
            print('Save Model (best BLEU-4)')
            state = {
                'encoder_dict': self.encoder.state_dict(),
                'encoder_trans_dict': self.encoder_trans.state_dict(),
                'decoder_dict': self.decoder.state_dict()
            }
            metric = f'Bleu4_{round(100000 * self.best_bleu4)}'
            model_name = f'{args.data_name}_bts_{args.train_batchsize}_{args.network}_epo_{epoch}_{metric}.pth'
            if epoch > 1:
                torch.save(state, os.path.join(args.savepath, model_name))


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args)
    for epoch in range(args.num_epochs):
        trainer.training(args, epoch)
        trainer.validation(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning (RSCC) on LEVIR-CC')

    # Data parameters
    parser.add_argument('--data_folder', default='D:\Dataset\LEVIR-CC\images',
                        help='folder with data files (contains train/val/test)')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='vocab filename (without .json)')
    parser.add_argument('--max_length', type=int, default=41, help='max caption length')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC", help='dataset name')

    # Train
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint for train_stage=s2')
    parser.add_argument('--print_freq', type=int, default=100, help='print stats every __ batches')
    parser.add_argument('--savepath', default='./models_ckpt/', help='path to save checkpoints/logs')

    # Training parameters
    parser.add_argument('--train_goal', type=int, default=1, help='caption-only, must be 1')
    parser.add_argument('--train_stage', default='s1', help='s1: train from scratch; s2: resume from checkpoint')
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--num_epochs', type=int, default=250, help='number of epochs to train')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value')
    parser.add_argument('--seed', type=int, default=1234)

    # Backbone parameters
    parser.add_argument('--network', default='segformer-mit_b1', help='backbone encoder')
    parser.add_argument('--encoder_dim', type=int, default=512, help='feature dim')
    parser.add_argument('--feat_size', type=int, default=16, help='feature map size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentiveEncoder')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')

    args = parser.parse_args()
    main(args)
