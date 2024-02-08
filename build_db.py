import json
import sqlite3
import tqdm

conn = sqlite3.connect("systems.db")
# {'id64': int, 'name': str, 'mainStar': str, 'coords': {'x': float, 'y': float, 'z': float}, 'updateTime': str}
# id64, name, mainStar, x, y, z, updateTime
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS systems (id64 INTEGER PRIMARY KEY, name TEXT, mainStar TEXT, x REAL, y REAL, z REAL, updateTime TEXT)")
with open("systems.json", "r") as f:
    for line in tqdm.tqdm(f):
        try:
            if line.strip() == "[" or line.strip() == "]":
                continue
            data = json.loads(line.strip().strip(","))
            try:
                c.execute("INSERT INTO systems (id64, name, mainStar, x, y, z, updateTime) VALUES (?, ?, ?, ?, ?, ?, ?)", (data['id64'], data['name'], data.get(
                    'mainStar', ""), data['coords']['x'], data['coords']['y'], data['coords']['z'], data['updateTime']))
            except sqlite3.IntegrityError:
                print(f"Duplicate entry: {data['id64']}")
                continue
        except json.JSONDecodeError:
            print("Error decoding JSON: ", line)
            continue

c.execute("CREATE INDEX id_index ON systems (id64);")
c.execute("CREATE INDEX name_index ON systems (name);")
c.execute("CREATE INDEX mainStar_index ON systems (mainStar);")

conn.commit()
conn.close()
