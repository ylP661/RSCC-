import argparse
import json
import os
import time

import torch
from torch.utils import data
from tqdm import tqdm

from data.LEVIR_CC import LEVIRCCDataset
from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *


def main(args):
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)

    checkpoint = torch.load(args.checkpoint)

    args.result_path = os.path.join(args.result_path, os.path.basename(args.checkpoint).replace('.pth', ''))
    os.makedirs(args.result_path, exist_ok=True)

    encoder = Encoder(args.network)
    encoder_trans = AttentiveEncoder(train_stage=None, n_layers=args.n_layers,
                                     feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                     heads=args.n_heads, dropout=args.dropout)
    decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                                 vocab_size=len(word_vocab), max_lengths=args.max_length,
                                 word_vocab=word_vocab, n_head=args.n_heads,
                                 n_layers=args.decoder_n_layers, dropout=args.dropout)

    encoder.load_state_dict(checkpoint['encoder_dict'])
    encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
    decoder.load_state_dict(checkpoint['decoder_dict'])

    encoder.eval().cuda()
    encoder_trans.eval().cuda()
    decoder.eval().cuda()

    test_loader = data.DataLoader(
        LEVIRCCDataset(args.data_folder, args.list_path, 'test', args.token_folder, args.vocab_file,
                       args.max_length, args.allow_unk),
        batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    references, hypotheses = [], []

    test_start_time = time.time()
    with torch.no_grad():
        for _, (imgA, imgB, _seg_label, token_all, _token_all_len, _token, _token_len, _name) in enumerate(
                tqdm(test_loader, desc='test_' + " EVALUATING AT BEAM SIZE 1")):
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            token_all = token_all.squeeze(0).cuda()

            feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2 = encoder_trans(feat1, feat2)

            seq = decoder.sample(feat1, feat2, k=1)

            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'],
                                                                       word_vocab['<NULL>']}],
                                  img_token))
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)

    test_time = time.time() - test_start_time
    score_dict = get_eval_score(references, hypotheses)
    print('Test of Captioning:\n'
          'Time: {0:.3f}\t'
          'BLEU-1: {1:.5f}\t'
          'BLEU-2: {2:.5f}\t'
          'BLEU-3: {3:.5f}\t'
          'BLEU-4: {4:.5f}\t'
          'Meteor: {5:.5f}\t'
          'Rouge: {6:.5f}\t'
          'Cider: {7:.5f}\t'
          .format(test_time, score_dict['Bleu_1'], score_dict['Bleu_2'], score_dict['Bleu_3'], score_dict['Bleu_4'],
                  score_dict['METEOR'], score_dict['ROUGE_L'], score_dict['CIDEr']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RSCC Test on LEVIR-CC')

    parser.add_argument('--data_folder', default='D:\Dataset\LEVIR-CC\images', help='folder with image files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41)
    parser.add_argument('--allow_unk', type=int, default=1)
    parser.add_argument('--data_name', default="LEVIR_CC")

    parser.add_argument('--checkpoint', default='./models_ckpt/MCI_model.pth', help='path to checkpoint')
    parser.add_argument('--test_batchsize', default=1, type=int)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--result_path', default="./predict_result/", help='path to save results')

    parser.add_argument('--network', default='segformer-mit_b1')
    parser.add_argument('--encoder_dim', type=int, default=512)
    parser.add_argument('--feat_size', type=int, default=16)

    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512)

    args = parser.parse_args()
    main(args)
