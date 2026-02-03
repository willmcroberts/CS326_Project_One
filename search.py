import random
import heapq
import json
import time
from collections import deque

class Grid:
    def __init__(self, m, n, start, goal, min_cost, max_cost):
        self.m = m
        self.n = n
        self.start = start
        self.goal = goal
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.costs = {}
        self._assign_costs()

    def _assign_costs(self):
        for r in range(self.m):
            for c in range(self.n):
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.m and 0 <= nc < self.n:
                        self.costs[((r,c),(nr,nc))] = random.randint(self.min_cost, self.max_cost)

    def neighbors(self, node):
        r, c = node
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.m and 0 <= nc < self.n:
                yield nr, nc

    def cost(self, a, b):
        return self.costs[(a, b)]

def reconstruct_path(parent, start, goal):
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    return list(reversed(path))

# ---------------- BFS ----------------
def bfs(grid):
    start_time = time.time()

    frontier = deque([grid.start])
    explored = set()
    parent = {}
    generated = 1
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        node = frontier.popleft()

        if node == grid.goal:
            end_time = time.time()
            path = reconstruct_path(parent, grid.start, grid.goal)
            cost = sum(grid.cost(path[i], path[i+1]) for i in range(len(path)-1))
            return {
                "path": path,
                "steps": len(path) - 1,
                "total_cost": cost,
                "expanded": len(explored),
                "generated": generated,
                "max_frontier": max_frontier,
                "runtime_ms": (end_time - start_time) * 1000,
                "status": "success"
            }

        explored.add(node)

        for nbr in grid.neighbors(node):
            if nbr not in explored and nbr not in frontier:
                parent[nbr] = node
                frontier.append(nbr)
                generated += 1

    end_time = time.time()
    return {
        "path": [],
        "steps": 0,
        "total_cost": 0,
        "expanded": len(explored),
        "generated": generated,
        "max_frontier": max_frontier,
        "runtime_ms": (end_time - start_time) * 1000,
        "status": "failure"
    }

# ---------------- DFS ----------------
def dfs(grid):
    start_time = time.time()

    frontier = [grid.start]
    explored = set()
    parent = {}
    generated = 1
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        node = frontier.pop()

        if node == grid.goal:
            end_time = time.time()
            path = reconstruct_path(parent, grid.start, grid.goal)
            cost = sum(grid.cost(path[i], path[i+1]) for i in range(len(path)-1))
            return {
                "path": path,
                "steps": len(path) - 1,
                "total_cost": cost,
                "expanded": len(explored),
                "generated": generated,
                "max_frontier": max_frontier,
                "runtime_ms": (end_time - start_time) * 1000,
                "status": "success"
            }

        explored.add(node)

        for nbr in grid.neighbors(node):
            if nbr not in explored and nbr not in frontier:
                parent[nbr] = node
                frontier.append(nbr)
                generated += 1

    end_time = time.time()
    return {
        "path": [],
        "steps": 0,
        "total_cost": 0,
        "expanded": len(explored),
        "generated": generated,
        "max_frontier": max_frontier,
        "runtime_ms": (end_time - start_time) * 1000,
        "status": "failure"
    }

# ---------------- UCS ----------------
def ucs(grid):
    start_time = time.time()

    frontier = [(0, grid.start)]
    explored = set()
    parent = {}
    cost_so_far = {grid.start: 0}
    generated = 1
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        curr_cost, node = heapq.heappop(frontier)

        if node == grid.goal:
            end_time = time.time()
            path = reconstruct_path(parent, grid.start, grid.goal)
            return {
                "path": path,
                "steps": len(path) - 1,
                "total_cost": curr_cost,
                "expanded": len(explored),
                "generated": generated,
                "max_frontier": max_frontier,
                "runtime_ms": (end_time - start_time) * 1000,
                "status": "success"
            }

        if node in explored:
            continue

        explored.add(node)

        for nbr in grid.neighbors(node):
            new_cost = curr_cost + grid.cost(node, nbr)
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                parent[nbr] = node
                heapq.heappush(frontier, (new_cost, nbr))
                generated += 1

    end_time = time.time()
    return {
        "path": [],
        "steps": 0,
        "total_cost": 0,
        "expanded": len(explored),
        "generated": generated,
        "max_frontier": max_frontier,
        "runtime_ms": (end_time - start_time) * 1000,
        "status": "failure"
    }

# ---------------- MAIN ----------------
def main():
    print("\n--- Grid Search Program ---\n")

    m = int(input("Enter grid rows (m): "))
    n = int(input("Enter grid columns (n): "))

    rs = int(input("Enter start row: "))
    cs = int(input("Enter start column: "))
    rg = int(input("Enter goal row: "))
    cg = int(input("Enter goal column: "))

    min_cost = int(input("Enter minimum move cost: "))
    max_cost = int(input("Enter maximum move cost: "))

    seed = int(input("Enter random seed: "))
    random.seed(seed)

    algorithm = input("Enter algorithm (bfs, dfs, ucs): ").lower()

    grid = Grid(m, n, (rs, cs), (rg, cg), min_cost, max_cost)

    if algorithm == "bfs":
        result = bfs(grid)
    elif algorithm == "dfs":
        result = dfs(grid)
    elif algorithm == "ucs":
        result = ucs(grid)
    else:
        print("Invalid algorithm.")
        return

    # Print results
    print("\nAlgorithm:", algorithm.upper())
    print("Path:", result["path"])
    print("Steps:", result["steps"])
    print("Total cost:", result["total_cost"])
    print("Expanded states:", result["expanded"])
    print("Generated nodes:", result["generated"])
    print("Max frontier size:", result["max_frontier"])
    print("Runtime (ms):", result["runtime_ms"])
    print("Status:", result["status"])

    # Build JSON
    output = {
        "algorithm": algorithm,
        "m": m,
        "n": n,
        "start": [rs, cs],
        "goal": [rg, cg],
        "min_cost": min_cost,
        "max_cost": max_cost,
        "seed": seed,
        "path": result["path"],
        "steps": result["steps"],
        "total_cost": result["total_cost"],
        "expanded": result["expanded"],
        "generated": result["generated"],
        "max_frontier_size": result["max_frontier"],
        "runtime_ms": result["runtime_ms"],
        "status": result["status"]
    }

    # A2 formatting: compact arrays + clean indentation
    with open("results.json", "w") as f:
        json.dump(output, f, indent=4, separators=(",", ": "))

if __name__ == "__main__":
    main()