# train.py
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset   # <--- from HuggingFace
from dataset import HuggingFaceMRIDataset  # <--- from our file
from model import get_resnet18
from utils import save_label_map, plot_confusion_matrix, plot_multiclass_roc
from sklearn.metrics import classification_report
from tqdm import tqdm


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x,y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
    return total_loss/len(loader.dataset)

def eval_model(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            p = F.softmax(out, dim=1).cpu().numpy()
            pred = p.argmax(axis=1)
            ys.append(y.numpy())
            preds.append(pred)
            probs.append(p)
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    return ys, preds, probs

def main(args):
    # load dataset from huggingface
    dataset = load_dataset("Falah/Alzheimer_MRI")

    classes = dataset["train"].features["label"].names
    print("Classes:", classes)

    # transforms
    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    train_ds = HuggingFaceMRIDataset(dataset["train"], transform=transform_train)
    test_ds = HuggingFaceMRIDataset(dataset["test"], transform=transform_eval)

    # split test into val/test
    val_size = int(0.2*len(test_ds))
    test_size = len(test_ds)-val_size
    val_ds, test_ds = torch.utils.data.random_split(test_ds, [val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18(num_classes=len(classes), pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.outdir, exist_ok=True)
    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        ys_val, preds_val, probs_val = eval_model(model, val_loader, device)
        val_acc = (preds_val == ys_val).mean()
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'label_map': {c:i for i,c in enumerate(classes)}
            }
            torch.save(checkpoint, os.path.join(args.outdir, "best_model.pt"))
            print("Saved best model.")

    # test eval
    ckpt = torch.load(os.path.join(args.outdir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    ys_test, preds_test, probs_test = eval_model(model, test_loader, device)
    print("Classification report (test):")
    print(classification_report(ys_test, preds_test, target_names=classes))
    # save artifacts
    save_label_map({c:i for i,c in enumerate(classes)}, os.path.join(args.outdir, "label_map.json"))
    plot_confusion_matrix(ys_test, preds_test, classes, os.path.join(args.outdir, "confusion.png"))
    plot_multiclass_roc(ys_test, probs_test, classes, os.path.join(args.outdir, "roc.png"))
    print(f"Artifacts saved to {args.outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
