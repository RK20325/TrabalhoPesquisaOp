# # Caminho Minimo na malha viaria de Curitiba
# Dijkstra, Bellman-Ford e A* implementados do zero (sem caixa-preta),
# validados contra o networkx.

# %% Instalacao (rodar so no Colab)
# !pip install osmnx scikit-learn

# %% Imports
import heapq
import math
import time

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

# %% 1. Dados: malha viaria (regiao do Batel/Centro, raio de 1,5 km)
CENTRO = (-25.4446, -49.2986)  # (lat, lon) -- ajuste para o bairro do grupo
G = ox.graph_from_point(CENTRO, dist=7000, network_type="drive")
print(f"Nos: {G.number_of_nodes()} | Arestas: {G.number_of_edges()}")

# Lista de adjacencia simples: adj[u][v] = menor comprimento (m) entre u e v
adj = {u: {} for u in G.nodes}
for u, v, d in G.edges(data=True):
    w = d["length"]
    if v not in adj[u] or w < adj[u][v]:
        adj[u][v] = w

coords = {n: (G.nodes[n]["y"], G.nodes[n]["x"]) for n in G.nodes}  # (lat, lon)

# %% 2. Origem e destino (ajuste as coordenadas)
ORIGEM_LATLON = (-25.446250812740498, -49.35879349832706)
DESTINO_LATLON = (-25.4408863501242, -49.237954202124364)
origem = ox.distance.nearest_nodes(G, X=ORIGEM_LATLON[1], Y=ORIGEM_LATLON[0])
destino = ox.distance.nearest_nodes(G, X=DESTINO_LATLON[1], Y=DESTINO_LATLON[0])


# %% 3. Algoritmos
def reconstruir(prev, s, t):
    if t not in prev and t != s:
        return []
    caminho = [t]
    while caminho[-1] != s:
        caminho.append(prev[caminho[-1]])
    return caminho[::-1]


def dijkstra(adj, s, t):
    dist = {s: 0.0}
    prev = {}
    visitados = set()
    fila = [(0.0, s)]
    while fila:
        d, u = heapq.heappop(fila)
        if u in visitados:
            continue
        visitados.add(u)
        if u == t:
            break
        for v, w in adj[u].items():
            nd = d + w  # relaxamento: d[v] = min(d[v], d[u] + w)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(fila, (nd, v))
    return dist.get(t, math.inf), reconstruir(prev, s, t), visitados


def bellman_ford(adj, s, t):
    dist = {n: math.inf for n in adj}
    dist[s] = 0.0
    prev = {}
    rodadas = 0
    for _ in range(len(adj) - 1):
        mudou = False
        rodadas += 1
        for u in adj:
            if dist[u] == math.inf:
                continue
            for v, w in adj[u].items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    mudou = True
        if not mudou:  # parada antecipada
            break
    return dist[t], reconstruir(prev, s, t), rodadas


def haversine(a, b):
    """Distancia em linha reta (m) entre dois pontos (lat, lon)."""
    R = 6371000
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def a_estrela(adj, s, t):
    h = lambda n: haversine(coords[n], coords[t])  # heurística admissível
    g = {s: 0.0}
    prev = {}
    visitados = set()
    fila = [(h(s), s)]  # f = g + h
    while fila:
        _, u = heapq.heappop(fila)
        if u in visitados:
            continue
        visitados.add(u)
        if u == t:
            break
        for v, w in adj[u].items():
            ng = g[u] + w
            if ng < g.get(v, math.inf):
                g[v] = ng
                prev[v] = u
                heapq.heappush(fila, (ng + h(v), v))
    return g.get(t, math.inf), reconstruir(prev, s, t), visitados


# %% 4. Execucao, tempo e validacao
resultados = {}
for nome, fn in [("Dijkstra", dijkstra), ("Bellman-Ford", bellman_ford), ("A*", a_estrela)]:
    t0 = time.perf_counter()
    custo, caminho, extra = fn(adj, origem, destino)
    dt = time.perf_counter() - t0
    resultados[nome] = (custo, caminho, extra, dt)

ref = nx.shortest_path_length(G, origem, destino, weight="length")
print(f"Referencia networkx: {ref:.1f} m\n")
print(f"{'Algoritmo':<14}{'Custo (m)':>12}{'Tempo (ms)':>13}{'Nos explorados / rodadas':>28}")
for nome, (custo, caminho, extra, dt) in resultados.items():
    expl = len(extra) if isinstance(extra, set) else f"{extra} rodadas"
    print(f"{nome:<14}{custo:>12.1f}{dt*1000:>13.2f}{str(expl):>28}")

# %% 5. Visualizacao: nos explorados (Dijkstra x A*) + rota otima
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, nome in zip(axes, ["Dijkstra", "A*"]):
    custo, caminho, visitados, _ = resultados[nome]
    ox.plot_graph(G, ax=ax, show=False, close=False, node_size=0,
                  edge_color="#cccccc", edge_linewidth=0.5)
    ax.scatter([coords[n][1] for n in visitados], [coords[n][0] for n in visitados],
               s=6, c="orange", label=f"explorados ({len(visitados)})", zorder=2)
    ax.plot([coords[n][1] for n in caminho], [coords[n][0] for n in caminho],
            c="red", linewidth=2.5, label=f"rota otima ({custo:.0f} m)", zorder=3)
    ax.scatter(*coords[origem][::-1], c="green", s=80, zorder=4, label="origem")
    ax.scatter(*coords[destino][::-1], c="blue", s=80, zorder=4, label="destino")
    ax.set_title(nome)
    ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
