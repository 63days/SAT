# Show, Attend and Tell: Pytorch Implementation
## Objective
This repo is a Image Captioning Model, which is in the **"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (ICML'15)"** paper.

## Results
| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| ------ | ------ | ------ | ------ |
| 61.09  | 36.54  | 22.64  | 14.21  |

BLEU(Bilingual Evaluation Understudy)is a score for comparing a candidate translation of text to one or more reference translations.  
![image](https://user-images.githubusercontent.com/37788686/100361207-35484880-303d-11eb-96a2-cae554881be1.png)

![image](https://user-images.githubusercontent.com/37788686/88451050-bf658200-ce8e-11ea-8875-5d6f5a46b104.png)
## Used In
**DATASET:** Flickr8k  
**ENCODER LAYER:** pretrained Vgg16Net

## To train
`python3 main.py`

## To test
`python3 main.py --test`
## Code Explanation
* model.py
```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vgg = models.vgg16_bn(pretrained=True)
        self.modules = list(self.vgg.features.children())[:34]
        self.subnet = nn.Sequential(*self.modules)
```        
SAT is the encoder-decoder framework. Encoder takes a single raw image and generates a caption y encoded as a sequence of 1-of-K encoded words. Encoder uses a convolution neural network in order to extract a set of feature vectors which we refer to as annotation vectors. 
```python        
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
```
Decoder consists of LSTM cells. 

<img src="https://user-images.githubusercontent.com/37788686/100361456-85bfa600-303d-11eb-9799-d02db89cabb3.png" width="40%"><img src="https://user-images.githubusercontent.com/37788686/98816467-0ee6b280-246c-11eb-9a26-d78118201fd3.png" width="40%">

```
i: input
f: forget
c: memory
o: output
h: hidden state
```

```python
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
 ```
 forward_step is one step in LSTM cells. In forward function, LSTM cell outputs h, c vectors and makes z which is maded by attention method w.r.t encoder output and h. 
```python
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
```
I implemented attention as Bahdanau attention. This uses two fc. fc outputs are used to calculate energy and attention score.
```python
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
```
At inferencing, I uses beam search. Beam search is similar with winner_takes_all function. In tree, beam search prunes only k largest probabilities at every step.

<img src="https://user-images.githubusercontent.com/37788686/98817788-042d1d00-246e-11eb-96f5-36ad403870b1.png" width="70%">

## Reference

[Show Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)
