import argparse
import json
import os

import cv2
import numpy as np
import torch
from imageio.v2 import imread

from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer


class Change_Perception(object):
    """Inference helper for RSCC (caption-only) on LEVIR-CC style inputs."""

    def define_args(self):
        parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning (RSCC)')

        parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
        parser.add_argument('--vocab_file', default='vocab')
        parser.add_argument('--max_length', type=int, default=41)

        parser.add_argument('--checkpoint', default='./models_ckpt/MCI_model.pth', help='path to checkpoint')

        parser.add_argument('--network', default='segformer-mit_b1')
        parser.add_argument('--encoder_dim', type=int, default=512)
        parser.add_argument('--feat_size', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)

        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--decoder_n_layers', type=int, default=1)
        parser.add_argument('--feature_dim', type=int, default=512)

        return parser.parse_args(args=[])

    def __init__(self):
        args = self.define_args()
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)

        checkpoint = torch.load(args.checkpoint)

        self.encoder = Encoder(args.network)
        self.encoder_trans = AttentiveEncoder(train_stage=None, n_layers=args.n_layers,
                                              feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                              heads=args.n_heads, dropout=args.dropout)
        self.decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                                          vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                                          word_vocab=self.word_vocab, n_head=args.n_heads,
                                          n_layers=args.decoder_n_layers, dropout=args.dropout)

        self.encoder.load_state_dict(checkpoint['encoder_dict'])
        self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
        self.decoder.load_state_dict(checkpoint['decoder_dict'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.eval().to(self.device)
        self.encoder_trans.eval().to(self.device)
        self.decoder.eval().to(self.device)

    def preprocess(self, path_A, path_B):
        imgA = imread(path_A)
        imgB = imread(path_B)
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)

        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        for i in range(len(self.mean)):
            imgA[i, :, :] = (imgA[i, :, :] - self.mean[i]) / self.std[i]
            imgB[i, :, :] = (imgB[i, :, :] - self.mean[i]) / self.std[i]

        if imgA.shape[1] != 256 or imgA.shape[2] != 256:
            imgA = cv2.resize(imgA, (256, 256))
            imgB = cv2.resize(imgB, (256, 256))

        imgA = torch.FloatTensor(imgA).unsqueeze(0)
        imgB = torch.FloatTensor(imgB).unsqueeze(0)
        return imgA, imgB

    def generate_change_caption(self, path_A, path_B):
        imgA, imgB = self.preprocess(path_A, path_B)
        imgA = imgA.to(self.device)
        imgB = imgB.to(self.device)

        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2 = self.encoder_trans(feat1, feat2)

        seq = self.decoder.sample(feat1, feat2, k=1)
        pred_seq = [w for w in seq if w not in {self.word_vocab['<START>'], self.word_vocab['<END>'], self.word_vocab['<NULL>']}]
        pred_caption = " ".join([list(self.word_vocab.keys())[i] for i in pred_seq])
        return pred_caption.strip()


if __name__ == '__main__':
    imgA_path = '/path/to/A.png'
    imgB_path = '/path/to/B.png'
    cp = Change_Perception()
    print(cp.generate_change_caption(imgA_path, imgB_path))
