import torch
from torch import nn
from torch import optim
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SAT(nn.Module):
    def __init__(self, vocab_size, context_dim=512, emb_dim=512, lstm_dim=512, attn_dim=512):
        super(SAT, self).__init__()
        self.context_dim = context_dim
        self.emb_dim = emb_dim
        self.lstm_dim = lstm_dim
        self.attn_dim = attn_dim
        self.vocab_size = vocab_size
        self.encoder = Encoder()
        self.decoder = Decoder(context_dim, emb_dim, lstm_dim,
                               attn_dim, vocab_size)
        self.opt = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        self.loss_func = nn.CrossEntropyLoss(ignore_index = 2)

    def forward(self, x, y):
        enc_output = self.encoder(x)
        dec_output = self.decoder(enc_output, y)
        return dec_output

    def train_batch(self, xb, yb):
        xb = torch.stack(xb).to(device=device, dtype=torch.float)
        yb = torch.stack(yb).to(device=device, dtype=torch.long)
        pred = self.forward(xb, yb)[:,:,:-1] #[B, V, L-1]
        ground_truth = yb[:,1:] #[B, L-1]
        loss = self.loss_func(pred, ground_truth)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss

    def valid_batch(self, xb, yb):
        xb = torch.stack(xb).to(device=device, dtype=torch.float)
        yb = torch.stack(yb).to(device=device, dtype=torch.long)
        pred = self.forward(xb, yb)[:,:,:-1]
        ground_truth = yb[:,1:]
        loss = self.loss_func(pred, ground_truth)
        return loss

    def save(self, file_name, num_epoch):
        save_state = {
            'num_epoch': num_epoch,
            'weights': self.state_dict(),
            'optim': self.opt.state_dict()
        }
        torch.save(save_state, file_name)

    def load(self, file_name):
        load_state = torch.load(file_name, map_location=device)
        self.load_state_dict(load_state['weights'])
        self.opt.load_state_dict(load_state['optim'])

    def inference(self, x, beam_size=5, max_len=36):
        self.eval()
        with torch.no_grad():
            x = torch.stack(x).to(device=device, dtype=torch.float) #[B,]
            x = x[::5,] #[B/5, ]
            batch_size = x.size()[0]
            x = x.repeat_interleave(beam_size, dim=0)
            probs = torch.ones([batch_size, beam_size]).to(device)
            y = torch.zeros([batch_size, beam_size, 1], dtype=torch.long).to(device) #[B, 1]
            for i in range(1, max_len):
                pred = self.forward(x, y.reshape(batch_size*beam_size, -1)).transpose(1,2)[:,-1]
                pred = torch.softmax(pred, dim=1) #[batch*beam, vocab]
                pred = pred.reshape(batch_size, beam_size, self.vocab_size)
                pred = probs.unsqueeze(2) * pred
                pred = pred.reshape(batch_size, -1)

                probs, indices = torch.topk(pred, beam_size) #[batch_size, beam_size]

                beam_indices = indices // self.vocab_size
                beam_indices = beam_indices.unsqueeze(2).repeat_interleave(i, dim=2) #[batch_size, beam_size, L]
                word_indices = indices % self.vocab_size

                y = torch.gather(y, 1, beam_indices)
                new_col = word_indices.unsqueeze(2)
                y = torch.cat([y, new_col], dim=2)

            best_idx = torch.argmax(probs, dim=1)
            y = y[torch.arange(batch_size), best_idx]

            y = nn.functional.pad(y, [0,1])
            y[:, -1] = 1
            pad_mask = nn.functional.pad((y == 1)[:,:-1], [1,0])
            y[pad_mask] = 2
            return y.cpu().numpy().tolist()







class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.vgg = models.vgg16_bn(pretrained=True)
        self.modules = list(self.vgg.features.children())[:34]
        self.subnet = nn.Sequential(*self.modules)
        # for p in self.vgg.parameters():
        #    p.requires_grad = False

    def forward(self, img):
        batch_size = img.size()[0]
        out = self.subnet(img)
        # print(out.size()) #[B, 512, 14, 14]
        # print(self.subnet)
        return out.reshape(batch_size, 512, -1).transpose(1, 2)
        # [B, 512, 14, 14]->[B, 512, 196]->[B, 196, 512]


class Decoder(nn.Module):
    def __init__(self, context_dim, emb_dim, lstm_dim, attn_dim, vocab_size):
        super(Decoder, self).__init__()
        self.context_dim = context_dim
        self.enc_dim = context_dim
        self.emb_dim = emb_dim
        self.lstm_dim = lstm_dim
        self.attn_dim = attn_dim
        self.vocab_size = vocab_size

        self.init_h_mlp = nn.Linear(self.enc_dim, lstm_dim)
        self.init_c_mlp = nn.Linear(self.enc_dim, lstm_dim)
        self.vocab_distribution = nn.Linear(self.lstm_dim, self.vocab_size)
        self.emb_layer = nn.Embedding(vocab_size, emb_dim, padding_idx=2)
        self.attention = Attention(context_dim, lstm_dim, attn_dim)
        self.lstm_cell = nn.LSTMCell(context_dim + emb_dim, lstm_dim)

    def init_hc(self, enc_output):  # enc_output: [B, L, 512]
        mean_enc_output = enc_output.mean(dim=1)  # [B, 512]
        init_h = self.init_h_mlp(mean_enc_output)  # [B, lstm_dim]
        init_c = self.init_c_mlp(mean_enc_output)
        return init_h, init_c

    def forward_step(self, y, h, c, z):
        '''
        :param y: [B] -> embed_y: [B, M] M: embedding_dim
        :param h: [B, N] N: lstm_dim
        :param c: [B, N] N: lstm_dim
        :param z: [B, D] D: context_dim
        :return: h: [B, lstm_dim] c: [B, lstm_dim]
        '''
        y = self.emb_layer(y)
        h, c = self.lstm_cell(torch.cat([y, z], dim=-1), (h, c))
        return h, c

    def forward(self, enc_output, y):
        '''
        :param enc_output: [B, 196, 512]
        :param y: [B, L]
        :return:
        '''
        batch_size = y.size()[0]
        caption_length = y.size()[1]
        h, c = self.init_hc(enc_output)
        z = self.attention(enc_output, h)

        P = []
        for i in range(caption_length):
            h, c = self.forward_step(y[:, i], h, c, z)
            z = self.attention(enc_output, h)
            preds = self.vocab_distribution(h)
            P.append(preds)

        P = torch.stack(P).to(device).transpose(0, 1)  # [B, L, vocab_size]
        return P.transpose(1, 2)  # [B, vocab_size, L]


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        '''
        :param encoder_out: [B, L, enc_dim]
        :param decoder_hidden: [B, 1, hid_dim]
        :return: attention [B, enc_dim]
        '''
        att1 = self.encoder_att(encoder_out)  # [b, L, attD]
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # [b, 1, attD]

        energy = torch.bmm(att2, att1.transpose(1, 2))  # [b, 1, L]
        attn_score = self.softmax(energy)
        context = torch.bmm(attn_score, encoder_out).squeeze(1)  # [b, enc_dim]
        return context
