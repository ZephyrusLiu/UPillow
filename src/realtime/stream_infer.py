import argparse, time, threading, queue, numpy as np, torch
from collections import deque
from src.models.cnn_1d_multi import TinySleepCNNMulti

class RingBuffer:
    def __init__(self, channels, window_len):
        self.C = channels; self.T = window_len
        self.buf = deque(maxlen=window_len)  # store arrays of shape (C,)
        for _ in range(window_len):
            self.buf.append(np.zeros((channels,), dtype=np.float32))
    def append(self, sample_vec):
        self.buf.append(sample_vec.astype(np.float32))
    def get_window(self):
        arr = np.stack(self.buf, axis=1)  # (T, C) -> transpose
        return arr.T  # (C, T)

def majority_vote(preds, k=5):
    if len(preds) == 0:
        return None
    last = list(preds)[-k:]
    vals, counts = np.unique(last, return_counts=True)
    return int(vals[np.argmax(counts)])

def preprocess(seg):
    # z-score per channel
    seg = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True)+1e-8)
    return seg.astype(np.float32)

def load_model(ckpt, n_classes, in_ch, in_len, device):
    model = TinySleepCNNMulti(n_classes=n_classes, in_len=in_len, in_ch=in_ch).to(device)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model

def infer_stream(model, device, ring: RingBuffer, step, label_names=None, vote_k=5):
    preds = deque([], maxlen=1000)
    while True:
        time.sleep(step)  # step seconds per hop
        seg = ring.get_window()
        seg = preprocess(seg)[None, ...]  # (1,C,T)
        with torch.no_grad():
            xb = torch.from_numpy(seg).to(device)
            logits = model(xb)
            pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        preds.append(pred)
        voted = majority_vote(preds, k=vote_k)
        label = label_names[pred] if label_names else str(pred)
        vlabel = label_names[voted] if (label_names and voted is not None) else str(voted)
        print(f"[step] raw={label}  voted={vlabel}")

# --- Backends: Serial & BLE ---
def serial_reader(port, baud, channels, out_q):
    import serial
    ser = serial.Serial(port, baudrate=baud)
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        # Expect CSV of C values per sample
        parts = line.split(",")
        if len(parts) >= channels:
            try:
                vec = np.array([float(x) for x in parts[:channels]], dtype=np.float32)
                out_q.put(vec)
            except:
                continue

async def ble_reader(address, char_uuid, channels, out_q):
    from bleak import BleakClient
    def handle(_, data: bytearray):
        # Expect bytes for C floats per sample
        try:
            arr = np.frombuffer(data, dtype=np.float32)
            if arr.size >= channels:
                out_q.put(arr[:channels])
        except:
            pass
    async with BleakClient(address) as client:
        await client.start_notify(char_uuid, handle)
        while True:
            await asyncio.sleep(0.01)

def run_serial(args):
    rb = RingBuffer(args.channels, args.window_len)
    q = queue.Queue()
    t = threading.Thread(target=serial_reader, args=(args.port, args.baud, args.channels, q), daemon=True)
    t.start()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, args.n_classes, args.channels, args.window_len, device)
    # feeder
    def feeder():
        while True:
            vec = q.get()
            rb.append(vec)
    threading.Thread(target=feeder, daemon=True).start()
    labels = args.labels.split(",") if args.labels else None
    infer_stream(model, device, rb, args.step, labels, vote_k=args.vote_k)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["serial","ble"], default="serial")
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--ble_address", default="")
    ap.add_argument("--ble_char", default="")
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--window_len", type=int, default=3000)  # e.g., 30s@100Hz or 2s@1500Hz
    ap.add_argument("--step", type=float, default=0.5)       # hop seconds
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_classes", type=int, default=2)
    ap.add_argument("--labels", type=str, default="")        # comma-separated
    ap.add_argument("--vote_k", type=int, default=5)
    args = ap.parse_args()

    if args.mode == "serial":
        run_serial(args)
    else:
        import asyncio, queue
        rb = RingBuffer(args.channels, args.window_len)
        q = queue.Queue()
        labels = args.labels.split(",") if args.labels else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args.ckpt, args.n_classes, args.channels, args.window_len, device)
        # Start BLE task
        import asyncio
        async def main():
            await ble_reader(args.ble_address, args.ble_char, args.channels, q)
        def feeder():
            while True:
                vec = q.get()
                rb.append(vec)
        threading.Thread(target=feeder, daemon=True).start()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
