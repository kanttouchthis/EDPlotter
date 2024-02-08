import sqlite3
import faiss
import numpy as np
from time import perf_counter
from functools import lru_cache


class Timer:
    def __init__(self):
        self.start = perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end = perf_counter()
        self.interval = self.end - self.start


conn = sqlite3.connect("systems.db")
c = conn.cursor()
index = faiss.read_index("systems.faiss")
index.nprobe = 1


def search(coords, n_samples):
    D, I = index.search(coords.reshape(1, 3), n_samples)
    ids = [int(I[0][i]) for i in range(n_samples)]
    ds = [np.sqrt(D[0][i]) for i in range(n_samples)]
    return {id: d for id, d in zip(ids, ds)}


class System:
    def __init__(self, id64: int, name: str, main_star: str, x: float, y: float, z: float, update_time: str):
        self.id64 = id64
        self.name = name
        self.main_star = main_star
        self.x = x
        self.y = y
        self.z = z
        self.update_time = update_time
        self.coords = np.array([x, y, z], dtype=np.float32)

    def __str__(self):
        return f"System {self.name} ({self.id64}) at ({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def get_by_name(cls, name: str) -> 'System':
        c.execute("SELECT * FROM systems WHERE name = ?", (name,))
        return cls(*c.fetchone())

    @classmethod
    @lru_cache(maxsize=1000)
    def get_by_id(cls, id64: int) -> 'System':
        c.execute("SELECT * FROM systems WHERE id64 = ?", (id64,))
        return cls(*c.fetchone())

    @classmethod
    def get_by_xyz(cls, x: float, y: float, z: float) -> 'System':
        nearest = search(np.array([x, y, z], dtype=np.float32), 1)
        id64 = list(nearest.keys())[0]
        return cls.get_by_id(id64)

    @classmethod
    def get_by_coords(cls, coords: np.ndarray) -> 'System':
        nearest = search(coords, 1)
        id64 = list(nearest.keys())[0]
        return cls.get_by_id(id64)

    def distance(self, other: 'System') -> float:
        return np.linalg.norm(self.coords - other.coords, ord=2)

    def _nearest(self, n_samples: int) -> dict:
        result = search(self.coords, n_samples)
        result.pop(self.id64, None)
        return result

    def nearest(self, n_samples: int) -> list:
        nearest = self._nearest(n_samples)
        return [System.get_by_id(id64) for id64 in nearest.keys()]


class SearchTree:
    def __init__(self, start: System, end: System, max_distance: float, n_samples: int = 5, validate: bool = True):
        self.start = start
        self.end = end
        if max_distance < 2:
            raise ValueError("Max distance must be at least 2 LY")
        if max_distance > start.distance(end):
            self.path = [start, end]
            return
        self.max_distance = max_distance
        self.n_samples = n_samples
        self.visited = {start}
        self.path = [start]
        self._search()
        if validate:
            if not self.validate():
                raise ValueError("Path is invalid")

    def _search(self):
        current = self.start
        while current.distance(self.end) > self.max_distance:
            direction = self.end.coords - current.coords
            distance = np.linalg.norm(direction, ord=2)
            normed_direction = direction / distance
            target_coords = current.coords + normed_direction * self.max_distance - 1
            targets = search(target_coords, self.n_samples)
            targets = [System.get_by_id(id64) for id64 in targets.keys()]
            targets = [t for t in targets if t not in self.visited]
            targets = [t for t in targets if current.distance(t) < self.max_distance]
            targets.sort(key=lambda t: t.distance(self.end))
            if len(targets) == 0:
                if len(self.path) == 1:
                    print(
                        f"No path found, increasing n_samples to {self.n_samples + 1}")
                    self.n_samples += 1
                    continue
                self.path.pop()
                current = self.path[-1]
                continue

            self.visited.add(targets[0])
            self.path.append(targets[0])
            current = targets[0]
            if current.distance(self.end) < self.max_distance:
                break
        self.path.append(self.end)
        self.visited.add(self.end)
        return self.path

    def __str__(self):
        return f"SearchTree from {self.start.name} to {self.end.name} with max distance of {self.max_distance} LY"

    def validate(self):
        for i in range(len(self.path) - 1):
            if self.path[i].distance(self.path[i+1]) > self.max_distance:
                return False
        return True


if __name__ == "__main__":
    s = System.get_by_name("Sol")
    print(s)
    print(s.nearest(5))
    print(s.distance(System.get_by_name("Barnard's Star")))
    with Timer() as t:
        pathsearch = SearchTree(s, System.get_by_name("Maia"), 65)
    print(f"Search took {t.interval:.6f} seconds")
    print(pathsearch.path)
    print(len(pathsearch.path))
