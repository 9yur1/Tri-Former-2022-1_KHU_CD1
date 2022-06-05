import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Triformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, tri_decomp
import math
import numpy as np

import os
class Model(nn.Module):
    """
    Tri-former is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = tri_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def decomp_print(self,trend,trend_true, seasonal,seasonal_true, noise, noise_true, pred, truth, pred_len):
         # result save
        
        folder_path = './graphs/exchange_96_96_truth validation/' 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        trends =[]
        trends.append(trend[:, -pred_len:, :].detach().cpu().numpy())
        trends_true =[]
        trends_true.append(trend_true[:, -pred_len:, :].detach().cpu().numpy())
        seasonals =[]
        seasonals.append(seasonal[:, -pred_len:, :].detach().cpu().numpy())
        seasonals_true =[]
        seasonals_true.append(seasonal_true[:, -pred_len:, :].detach().cpu().numpy())
        noises =[]
        noises.append(noise[:, -pred_len:, :].detach().cpu().numpy())
        noises_true =[]
        noises_true.append(noise_true[:, -pred_len:, :].detach().cpu().numpy())
        preds =[]
        preds.append(pred[:, -pred_len:, :].detach().cpu().numpy())
        truths =[]
        truths.append(truth[:, -pred_len:, :].detach().cpu().numpy())

        np.save(folder_path + 'trend.npy', trends)
        np.save(folder_path + 'trend_true.npy', trends_true)
        np.save(folder_path + 'seasonal.npy', seasonals)
        np.save(folder_path + 'seasonal_true.npy', seasonals_true)
        np.save(folder_path + 'noise.npy', noises)
        np.save(folder_path + 'noise_true.npy', noises_true)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'truth.npy', truths)

    def export_truth_trend(self,x_dec):
        seasonal, trend, noise = self.decomp(x_dec)

        return seasonal, trend, noise

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)      
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device) 
        seasonal_init, trend_init, noise_init = self.decomp(x_enc)
        
        # truth 
        seasonal_true, trend_true, noise_true = self.decomp(x_dec)
        noise_true = torch.cat([noise_init[:, -self.label_len:, :], noise_true[:, -self.pred_len:, :]], dim=1)

        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        noise_init = torch.cat([noise_init[:, -self.label_len:, :], zeros], dim=1)
        # enc 
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)  
        #noise_init = self.dec_embedding(noise_init, x_mark_dec) #decoder에 noise 추가  
        seasonal_part, trend_part, noise_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init, noise=noise_init) 
        # final
        dec_out = seasonal_part + trend_part

        #data save
        #self.decomp_print(trend_part, trend_true, seasonal_part, seasonal_true, noise_init, noise_true, dec_out, x_dec, self.pred_len)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        """
        trend_init = self.dec_embedding(trend_init, x_mark_dec)
        """
