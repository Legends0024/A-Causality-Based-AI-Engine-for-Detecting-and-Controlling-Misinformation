import pickle
import torch
from model import RumourGAT
from inference import greedy_intervene
from utils import graph_to_json

DEVICE = torch.device("cpu")
model = RumourGAT(in_channels=5, hidden=32, heads=4, dropout=0.3)

with open("all_graphs.pkl", "rb") as f:
    all_graphs = pickle.load(f)

G, root, label = all_graphs[0]
print(f"Graph nodes: {len(G.nodes)}")

try:
    result = greedy_intervene(G, root, model, DEVICE, K=5)
    print("Optimization success!")
    print(result.keys())
except Exception as e:
    import traceback
    traceback.print_exc()
