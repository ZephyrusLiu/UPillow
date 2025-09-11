"""
Realtime EEG sliding-window inference demo.
- Simulates BLE/serial stream input (replace generator with actual pyserial/ble library).
- Maintains a ring buffer of length >= window.
- Applies majority vote over last N predictions.
"""
import numpy as np, torch, time, collections
from src.models.cnn_multich import MultiChannelCNN

class RingBuffer:
    def __init__(self, size, n_ch):
        self.size = size
        self.n_ch = n_ch
        self.data = np.zeros((size,n_ch),dtype=np.float32)
        self.idx = 0
        self.full = False
    def append(self, sample):
        self.data[self.idx] = sample
        self.idx = (self.idx+1)%self.size
        if self.idx==0: self.full=True
    def get(self):
        if not self.full:
            return self.data[:self.idx].copy()
        # return in chronological order
        return np.concatenate([self.data[self.idx:], self.data[:self.idx]], axis=0)

def fake_stream(n_samples=2000,n_ch=1,fs=200):
    """Simulate BLE stream: yields random samples"""
    for _ in range(n_samples):
        yield np.random.randn(n_ch).astype(np.float32)
        time.sleep(1.0/fs)

def realtime_inference(model, fs=200, window_sec=2, vote_k=5, n_ch=1):
    win_len = window_sec*fs
    buf = RingBuffer(win_len, n_ch)
    votes = collections.deque(maxlen=vote_k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    for sample in fake_stream(n_samples=2000,n_ch=n_ch,fs=fs):
        buf.append(sample)
        cur = buf.get()
        if len(cur)>=win_len:
            x = cur[-win_len:,:].T[None,:,:]  # shape (1,n_ch,T)
            x = (x - x.mean())/(x.std()+1e-8)
            xt = torch.from_numpy(x.astype("float32")).to(device)
            with torch.no_grad():
                logits = model(xt)
                pred = logits.argmax(1).item()
            votes.append(pred)
            maj = int(np.round(np.mean(votes))) if votes else pred
            print(f"Pred={pred}, MajVote={maj}")
            # Here you could send maj via BLE/serial, or trigger alarm

if __name__=="__main__":
    model = MultiChannelCNN(n_ch=1, n_classes=2, in_len=400)
    realtime_inference(model,n_ch=1)
