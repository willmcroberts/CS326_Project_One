import random
import heapq
import json
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

def bfs(grid):
    frontier = deque([grid.start])
    explored = set()
    parent = {}

    while frontier:
        node = frontier.popleft()
        if node == grid.goal:
            path = reconstruct_path(parent, grid.start, grid.goal)
            cost = sum(grid.cost(path[i], path[i+1]) for i in range(len(path)-1))
            return {"path": path, "cost": cost, "expanded": len(explored)}

        explored.add(node)

        for nbr in grid.neighbors(node):
            if nbr not in explored and nbr not in frontier:
                parent[nbr] = node
                frontier.append(nbr)

    return None

def dfs(grid):
    frontier = [grid.start]
    explored = set()
    parent = {}

    while frontier:
        node = frontier.pop()
        if node == grid.goal:
            path = reconstruct_path(parent, grid.start, grid.goal)
            cost = sum(grid.cost(path[i], path[i+1]) for i in range(len(path)-1))
            return {"path": path, "cost": cost, "expanded": len(explored)}

        explored.add(node)

        for nbr in grid.neighbors(node):
            if nbr not in explored and nbr not in frontier:
                parent[nbr] = node
                frontier.append(nbr)

    return None

def ucs(grid):
    frontier = [(0, grid.start)]
    explored = set()
    parent = {}
    cost_so_far = {grid.start: 0}

    while frontier:
        curr_cost, node = heapq.heappop(frontier)

        if node == grid.goal:
            path = reconstruct_path(parent, grid.start, grid.goal)
            return {"path": path, "cost": curr_cost, "expanded": len(explored)}

        if node in explored:
            continue

        explored.add(node)

        for nbr in grid.neighbors(node):
            new_cost = curr_cost + grid.cost(node, nbr)
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                parent[nbr] = node
                heapq.heappush(frontier, (new_cost, nbr))

    return None

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

    print("\nAlgorithm:", algorithm.upper())
    print("Path:", result["path"])
    print("Cost:", result["cost"])
    print("Nodes expanded:", result["expanded"])

    results = {
        "seed": seed,
        "algorithm": algorithm,
        "path": result["path"],
        "cost": result["cost"],
        "expanded": result["expanded"]
    }

    with open("results.json", "w") as f:
        compact = json.dumps(results, separators=(",", ":"))
        pretty = compact.replace("{", "{\n\n    ").replace("}", "\n\n}").replace(",", ", ").replace("]],", "]],\n    ").replace("},","},\n")
        f.write(pretty)

if __name__ == "__main__":
    main()