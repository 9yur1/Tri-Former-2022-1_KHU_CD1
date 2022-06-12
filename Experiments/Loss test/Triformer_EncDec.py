import torch
import torch.nn as nn
import torch.nn.functional as F

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)    
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class seasonal_decomp(nn.Module):
    """
    seasonal_decomp
    """
    def __init__(self, kernel_size, period):
        super(seasonal_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.period = period

    def forward(self, x):   
        #batch_size= len(x) 
        seq_len = len(x[0]) 
        #enc_in = len(x[0][0]) 

        x = x.permute(0, 2, 1)  # batch*seq*enc

        mean =[[0] for i in range(self.period)]
        for i in range(self.period):
            mean[i] = torch.mean(x[:,:,i::self.period],dim=2) # batch*enc

        x = torch.stack(mean, dim=1) # batch*period*enc
        x = x.permute(0, 2, 1) # batch*enc*period

        w = torch.mean(x,dim=2) # batch*enc
        mean_w =[[0] for i in range(seq_len)]
        for i in range(seq_len):
            mean_w[i] = w # batch*enc
        w = torch.stack(mean_w, dim=1) # batch*seq*enc

        x = x.repeat(1,1,seq_len//self.period) # batch*enc*seq
        x = x.permute(0, 2, 1) # batch*seq*enc
        x = x - w

        return x

class STL_decomp(nn.Module):
    """
    STL_decomp
    """
    def __init__(self, kernel_size, period):
        super(STL_decomp, self).__init__()
        self.kernel_size = kernel_size
        self.period = period

    def forward(self, x):   
        from statsmodels.tsa.seasonal import seasonal_decompose
        import numpy as np

        with torch.cuda.amp.autocast():
            enc_in = len(x[0][0]) 

            series = x.detach().cpu().numpy() # batch*seq*enc
            #series = x

            trend = [[0] for i in range(enc_in)]
            seasonal = [[0] for i in range(enc_in)]
            noise = [[0] for i in range(enc_in)]

            for i in range(enc_in):
                result = seasonal_decompose(series[:,:,i], model = 'additive', period =self.period, extrapolate_trend=self.period) # batch*seq
                trend[i] = result.trend
                seasonal[i] = result.seasonal
                noise[i] = result.resid
            
            trend = torch.Tensor(trend) # enc*batch*seq
            trend = trend.to("cuda:0")

            seasonal = torch.Tensor(seasonal) # enc*batch*seq
            seasonal = seasonal.to("cuda:0")

            noise = torch.Tensor(noise) # enc*batch*seq
            noise = noise.to("cuda:0")

            trend = trend.permute(1, 2, 0) # batch*seq*enc
            seasonal = seasonal.permute(1, 2, 0)
            noise = noise.permute(1, 2, 0) 

        return trend, seasonal, noise

class tri_decomp(nn.Module):
    """
    Tri Series decomposition block
    """
    def __init__(self, kernel_size):
        super(tri_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.seasonal_decomp = seasonal_decomp(kernel_size, period=8)  
        self.STL_decomp = STL_decomp(kernel_size, period=8) 


    def forward(self, x):
        moving_mean = self.moving_avg(x)
        seasonal = self.seasonal_decomp(x - moving_mean)
        #moving_mean, seasonal, noise = self.STL_decomp(x)
        noise = x - moving_mean - seasonal

        return seasonal, moving_mean, noise

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        noise = res-res
        return res, moving_mean, noise


class EncoderLayer(nn.Module):
    """
    Tri-former encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Tri-former encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Tri-former decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None, noise=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1, noise1 = self.decomp1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2, noise2 = self.decomp2(x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        x, trend3, noise3 = self.decomp3(x + y)
        
        residual_trend = trend1 + trend2 + trend3 
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        residual_noise = noise1 + noise2 + noise3
        residual_noise = self.projection(residual_noise.permute(0, 2, 1)).transpose(1, 2)

        return x, residual_trend, residual_noise


class Decoder(nn.Module):
    """
    Tri-former encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None, noise=None):
        
        for layer in self.layers:
            x, residual_trend, residual_noise = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, trend=trend, noise=noise) 
            """
            noise = self.norm(noise)
            noise = self.projection(noise)
            """     
            trend = trend + residual_trend
            noise = noise + residual_noise

        if self.norm is not None:
            x = self.norm(x)
            
        if self.projection is not None:
            x = self.projection(x)        

        return x, trend, noise
   