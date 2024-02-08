import faiss
import numpy as np
import sqlite3
from tqdm import tqdm
import os

# Connect to the database
# id64, name, mainStar, x, y, z, updateTime
conn1 = sqlite3.connect("systems.db")
c1 = conn1.cursor()
q = faiss.IndexFlatL2(3)
index = faiss.IndexIVFFlat(q, 3, 10000)
index.nprobe = 10
# load the data
c1.execute("SELECT * FROM systems")

if os.path.exists("systems.npy"):
    all_data = np.load("systems.npy")
    ids = np.load("ids.npy")
else:
    all_data = []
    ids = []
    print("Loading data...")
    for row in tqdm(c1):
        x = np.array([row[3], row[4], row[5]], dtype=np.float32)
        all_data.append(x)
        ids.append(row[0])
    all_data = np.array(all_data)
    ids = np.array(ids)
    np.save("systems.npy", all_data)
    np.save("ids.npy", ids)

print("Training index...")
index.train(all_data)
print("Adding data to index...")
index.add_with_ids(all_data, np.array(ids))

# save the index
faiss.write_index(index, "systems.faiss")

conn1.close()
