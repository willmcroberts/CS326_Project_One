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


# TESTS
def is_legal_move_sequence(grid, path):
    if not path:
        return False
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        if dr + dc != 1:
            return False
        if not (0 <= r2 < grid.m and 0 <= c2 < grid.n):
            return False
    return True


def test_path_start_end():
    m, n = 5, 5
    start = (0, 0)
    goal = (4, 4)
    min_cost, max_cost = 1, 5
    seed = 42
    random.seed(seed)
    grid = Grid(m, n, start, goal, min_cost, max_cost)

    for algo in (bfs, dfs, ucs):
        result = algo(grid)
        path = result["path"]
        assert path[0] == start, f"{algo.__name__}: path does not start at S"
        assert path[-1] == goal, f"{algo.__name__}: path does not end at G"


def test_legal_moves():
    m, n = 5, 5
    start = (0, 0)
    goal = (4, 4)
    min_cost, max_cost = 1, 5
    seed = 123
    random.seed(seed)
    grid = Grid(m, n, start, goal, min_cost, max_cost)

    for algo in (bfs, dfs, ucs):
        result = algo(grid)
        path = result["path"]
        assert is_legal_move_sequence(grid, path), f"{algo.__name__}: illegal move in path"


def test_ucs_cost_matches():
    m, n = 5, 5
    start = (0, 0)
    goal = (4, 4)
    min_cost, max_cost = 1, 10
    seed = 999
    random.seed(seed)
    grid = Grid(m, n, start, goal, min_cost, max_cost)

    result = ucs(grid)
    path = result["path"]
    reported_cost = result["total_cost"]
    recomputed_cost = sum(grid.cost(path[i], path[i + 1]) for i in range(len(path) - 1))
    assert (
            reported_cost == recomputed_cost
    ), f"UCS cost mismatch: reported {reported_cost}, recomputed {recomputed_cost}"


def run_tests():
    print("Running tests...")
    test_path_start_end()
    print("path starts at S and ends at G")
    test_legal_moves()
    print("every move in path is legal")
    test_ucs_cost_matches()
    print("UCS total cost matches recomputed path cost")
    print("All tests passed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Grid Navigation Search Agent (BFS/DFS/UCS)")
    parser.add_argument("--m", type=int, help="Grid rows")
    parser.add_argument("--n", type=int, help="Grid columns")
    parser.add_argument("--rs", type=int, help="Start row")
    parser.add_argument("--cs", type=int, help="Start column")
    parser.add_argument("--rg", type=int, help="Goal row")
    parser.add_argument("--cg", type=int, help="Goal column")
    parser.add_argument("--min_cost", type=int, help="Minimum move cost")
    parser.add_argument("--max_cost", type=int, help="Maximum move cost")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--algorithm", type=str, choices=["bfs", "dfs", "ucs"], help="Search algorithm")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests and exit")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON filename")
    return parser.parse_args()


def interactive_inputs():
    print("\n--- Grid Search Program (Interactive Mode) ---\n")

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