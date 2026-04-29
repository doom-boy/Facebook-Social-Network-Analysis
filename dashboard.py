import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import random

st.set_page_config(
    page_title="CS4230 Facebook Page Networks Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hardcoded precomputed stats
# May change in future for full workflow
STATS = {
    "artist":       {"nodes": 50515, "edges": 819090, "avg_degree": 32.433, "median_degree": 13, "max_degree": 1469, "density": 0.000642, "clustering": 0.1381, "approx_path_length": 3.68},
    "new_sites":    {"nodes": 27917, "edges": 206259, "avg_degree": 14.771, "median_degree":  8, "max_degree":  709, "density": 0.000529, "clustering": 0.2343, "approx_path_length": 4.41},
    "athletes":     {"nodes": 13866, "edges":  86858, "avg_degree": 12.527, "median_degree":  7, "max_degree":  418, "density": 0.000904, "clustering": 0.2781, "approx_path_length": 4.82},
    "company":      {"nodes": 14113, "edges":  52310, "avg_degree":  7.413, "median_degree":  4, "max_degree":  326, "density": 0.000525, "clustering": 0.2392, "approx_path_length": 5.31},
    "public_figure":{"nodes": 11565, "edges":  67114, "avg_degree": 11.600, "median_degree":  6, "max_degree":  590, "density": 0.001003, "clustering": 0.1793, "approx_path_length": 4.91},
    "politician":   {"nodes":  5908, "edges":  41729, "avg_degree": 14.124, "median_degree":  8, "max_degree":  348, "density": 0.002393, "clustering": 0.3851, "approx_path_length": 4.28},
    "government":   {"nodes":  7057, "edges":  89455, "avg_degree": 25.357, "median_degree": 15, "max_degree":  697, "density": 0.003592, "clustering": 0.4109, "approx_path_length": 3.77},
    "tvshow":       {"nodes":  3892, "edges":  17262, "avg_degree":  8.876, "median_degree":  5, "max_degree":  218, "density": 0.002281, "clustering": 0.3737, "approx_path_length": 6.42},
}

COMMUNITY = {
    "artist":        {"num_communities": 614,  "largest_pct": 31.2},
    "new_sites":     {"num_communities": 945,  "largest_pct": 72.3},
    "athletes":      {"num_communities": 287,  "largest_pct": 22.4},
    "company":       {"num_communities": 1272, "largest_pct": 15.8},
    "public_figure": {"num_communities": 418,  "largest_pct": 19.6},
    "politician":    {"num_communities": 273,  "largest_pct":  8.6},
    "government":    {"num_communities": 112,  "largest_pct": 42.5},
    "tvshow":        {"num_communities": 399,  "largest_pct":  4.5},
}

LABELS = {
    "artist": "Artist", "new_sites": "News Sites", "athletes": "Athletes",
    "company": "Company", "public_figure": "Public Figure",
    "politician": "Politician", "government": "Government", "tvshow": "TV Show",
}

CATEGORY_ORDER = ["artist", "new_sites", "athletes", "company",
                  "public_figure", "politician", "government", "tvshow"]

POLITICAL = {"government", "politician", "public_figure"}
ENTERTAINMENT = {"artist", "athletes", "tvshow"}

DATA_DIR = "facebook_clean_data"

@st.cache_data(show_spinner=False)
def load_graph(category):
    path = os.path.join(DATA_DIR, f"{category}_edges.csv")
    with open(path) as f:
        next(f)
        G = nx.read_edgelist(f, delimiter=",", nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

@st.cache_data(show_spinner=False)
def build_sample_subgraph(category, n_hubs=8, neighbors_per_hub=20, seed=42):
    random.seed(seed)
    G = load_graph(category)
    degrees = dict(G.degree())
    hubs = sorted(degrees, key=degrees.get, reverse=True)[:n_hubs]
    nodes_to_include = set(hubs)
    for hub in hubs:
        nbrs = list(G.neighbors(hub))
        nodes_to_include.update(random.sample(nbrs, min(neighbors_per_hub, len(nbrs))))
    sub = G.subgraph(nodes_to_include).copy()
    pos = nx.spring_layout(sub, seed=seed, k=1.2)
    return G, sub, pos

def build_plotly_network(G_full, sub, pos, highlight_hubs=True):
    degrees_full = dict(G_full.degree())
    hubs_set = set(sorted(degrees_full, key=degrees_full.get, reverse=True)[:8])

    edge_x, edge_y = [], []
    for u, v in sub.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.6, color="#aaa"), hoverinfo="none")

    nodes = list(sub.nodes())
    deg_vals = [degrees_full.get(n, 1) for n in nodes]
    d_min, d_max = min(deg_vals), max(deg_vals)
    sizes = [14 + 32 * (d - d_min) / max(d_max - d_min, 1) for d in deg_vals]
    colors = ["#e63946" if n in hubs_set else "#457b9d" for n in nodes]
    labels = [f"Node {n}<br>Degree: {degrees_full.get(n, 0)}" for n in nodes]

    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers", hovertext=labels, hoverinfo="text",
        marker=dict(size=sizes, color=colors, line=dict(width=0.8, color="white")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, height=480, margin=dict(l=5, r=5, t=10, b=5),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# build summary dataframe
def make_summary_df():
    rows = []
    for k in CATEGORY_ORDER:
        s = STATS[k]; c = COMMUNITY[k]
        rows.append({
            "Category": LABELS[k],
            "Nodes": f"{s['nodes']:,}",
            "Edges": f"{s['edges']:,}",
            "Avg Degree": s["avg_degree"],
            "Density": s["density"],
            "Clustering": s["clustering"],
            "Avg Path Length": s["approx_path_length"],
            "# Communities": c["num_communities"],
            "Largest Comm %": c["largest_pct"],
        })
    return pd.DataFrame(rows).set_index("Category")

#########################################################################################################

st.title("Analyzing Facebook Page Networks")
st.markdown(
    "Analyzing 8 categories of verified Facebook pages from 2017 (**artist, athletes, company, government, news sites, politician, public figure, TV show**) connected by mutual likes."
    "Questions we asked were how do clustering, centrality, and community fragmentation differ across categories, and are political pages more polarized than entertainment pages?"
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Network Explorer", "Degree Distribution",
    "Community Structure", "Political vs. Entertainment",
    "Centrality Analysis", "Findings & Limitations"
])

#########################################################################################################
# Tab 1 - overview

with tab1:
    st.header("Dataset Overview")
    st.markdown(
        "Each category is a separate network of verified Facebook pages. Edges represent **mutual likes** or when two pages liked each other's page. "
        "Networks are **undirected, unweighted, and fully connected**."
    )

    cols = st.columns(4)
    total_nodes = sum(STATS[k]["nodes"] for k in CATEGORY_ORDER)
    total_edges = sum(STATS[k]["edges"] for k in CATEGORY_ORDER)
    cols[0].metric("Total Pages (Nodes)", f"{total_nodes:,}")
    cols[1].metric("Total Connections (Edges)", f"{total_edges:,}")
    cols[2].metric("Categories", "8")
    cols[3].metric("Data Year", "2017")

    st.subheader("Summary Statistics by Category")
    df_sum = make_summary_df()
    st.dataframe(df_sum, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Network Size (Nodes)")
        df_nodes = pd.DataFrame({
            "Category": [LABELS[k] for k in CATEGORY_ORDER],
            "Nodes": [STATS[k]["nodes"] for k in CATEGORY_ORDER],
        }).sort_values("Nodes", ascending=True)
        fig_nodes = px.bar(df_nodes, x="Nodes", y="Category", orientation="h",
                           color="Nodes", color_continuous_scale="Blues",
                           title="Number of Pages per Category")
        fig_nodes.update_layout(coloraxis_showscale=False, height=340)
        st.plotly_chart(fig_nodes, use_container_width=True)

    with col_b:
        st.subheader("Clustering vs. Density")
        df_scatter = pd.DataFrame({
            "Category": [LABELS[k] for k in CATEGORY_ORDER],
            "Density": [STATS[k]["density"] for k in CATEGORY_ORDER],
            "Clustering": [STATS[k]["clustering"] for k in CATEGORY_ORDER],
            "Nodes": [STATS[k]["nodes"] for k in CATEGORY_ORDER],
        })
        fig_sc = px.scatter(df_scatter, x="Density", y="Clustering", text="Category",
                            size="Nodes", size_max=40,
                            title="Clustering Coefficient vs. Network Density",
                            labels={"Density": "Network Density", "Clustering": "Avg Clustering Coeff"})
        fig_sc.update_traces(textposition="top center", textfont_size=10)
        fig_sc.update_layout(height=340)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.info(
        "**Government** stands out with the highest density (0.0036) and clustering (0.41), suggesting official agency pages form very tight mutual-like clusters. "
        "**Artist** is the largest network but sparsely connected."
    )

#########################################################################################################
# TAB 2 - network explor

with tab2:
    st.header("Network Explorer")
    st.markdown(
        "Select a category to visualize a **sampled subgraph** built around the 8 most-connected hub pages plus a random sample of their neighbors. "
    )
    st.markdown(
        "Red nodes = top-8 hubs by degree. Blue nodes = sampled neighbors. Node size scales w/ degree in the full network."
    )

    selected_cat = st.selectbox(
        "Choose a category", [LABELS[k] for k in CATEGORY_ORDER],
        key="net_cat_select"
    )
    cat_key = {v: k for k, v in LABELS.items()}[selected_cat]

    with st.spinner(f"Loading {selected_cat} network…"):
        G_full, sub, pos = build_sample_subgraph(cat_key)

    fig_net = build_plotly_network(G_full, sub, pos)
    st.plotly_chart(fig_net, use_container_width=True)

    s = STATS[cat_key]
    st.caption(
        f"Subgraph shown: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges "
        f"(sampled from {s['nodes']:,} total nodes)."
    )

    st.markdown("---")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Nodes", f"{s['nodes']:,}")
    m2.metric("Edges", f"{s['edges']:,}")
    m3.metric("Avg Degree", s["avg_degree"])
    m4.metric("Max Degree", s["max_degree"])
    m5.metric("Clustering", s["clustering"])
    m6.metric("Avg Path Length", s["approx_path_length"])
    m7.metric("Density", s["density"])

#########################################################################################################
#tab 3 - degree dist

with tab3:
    st.header("Degree Distribution")
    st.markdown(
        "Degree is the number of mutual-like connections a page has. Note the long-tailed distribution: most pages have few connections, and a small number of \"hub pages\" accumulate far more. "
    )
    st.markdown(
        "Compare any two categories to see how the spread differs."
    )

    c1, c2 = st.columns(2)
    cat_a = c1.selectbox("Category A", [LABELS[k] for k in CATEGORY_ORDER], index=6, key="da")
    cat_b = c2.selectbox("Category B", [LABELS[k] for k in CATEGORY_ORDER], index=5, key="db")

    label_to_key = {v: k for k, v in LABELS.items()}
    deg_data = []
    for label in [cat_a, cat_b]:
        ci = label_to_key[label]
        s = STATS[ci]
        deg_data.append({"Category": label, "Metric": "Avg Degree",    "Value": s["avg_degree"]})
        deg_data.append({"Category": label, "Metric": "Median Degree", "Value": s["median_degree"]})
        deg_data.append({"Category": label, "Metric": "Max Degree",    "Value": s["max_degree"]})

    df_deg = pd.DataFrame(deg_data)
    fig_deg = px.bar(
        df_deg, x="Metric", y="Value", color="Category", barmode="group",
        title=f"Degree Stats: {cat_a} vs {cat_b}",
        color_discrete_sequence=["#e63946", "#457b9d"],
    )
    fig_deg.update_layout(height=360, yaxis_title="Degree")
    st.plotly_chart(fig_deg, use_container_width=True)

    st.markdown(
        "The gap between median and max degree is pronounced in every category. "
        "A handful of pages hold connections far exceeding the typical page, kind of like a  power-law distribution."
    )

    ############# MAYBE SHOULD REMOVE MAX DEGREE FROM THIS VIEW? <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    st.markdown("---")
    st.subheader("All Categories at a Glance")
    df_all = pd.DataFrame({
        "Category":      [LABELS[k] for k in CATEGORY_ORDER],
        "Avg Degree":    [STATS[k]["avg_degree"]    for k in CATEGORY_ORDER],
        "Median Degree": [STATS[k]["median_degree"] for k in CATEGORY_ORDER],
        "Max Degree":    [STATS[k]["max_degree"]    for k in CATEGORY_ORDER],
    })
    fig_all = px.bar(
        df_all.melt(id_vars="Category", var_name="Metric", value_name="Degree"),
        x="Category", y="Degree", color="Metric", barmode="group",
        title="Avg / Median / Max Degree Across All Categories",
        color_discrete_sequence=["#1d3557", "#457b9d", "#a8dadc"],
    )
    fig_all.update_layout(height=360)
    st.plotly_chart(fig_all, use_container_width=True)

#########################################################################################################
# Tab 4 - community structure

with tab4:
    st.header("Community Structure")
    st.markdown(
        "Community detection groups pages that are densely interconnected. "
    )
    st.markdown(
        "A **high largest-community %** means most pages belong to one cohesive group. A **low %** means pages fragment into many small isolated clusters."
    )

    df_comm = pd.DataFrame({
        "Category": [LABELS[k] for k in CATEGORY_ORDER],
        "Largest Community %": [COMMUNITY[k]["largest_pct"] for k in CATEGORY_ORDER],
        "# Communities": [COMMUNITY[k]["num_communities"] for k in CATEGORY_ORDER],
        "Group": ["Political" if k in POLITICAL else ("Entertainment" if k in ENTERTAINMENT else "Other")
                  for k in CATEGORY_ORDER],
    }).sort_values("Largest Community %", ascending=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_lc = px.bar(df_comm, x="Largest Community %", y="Category", orientation="h",
                        color="Group",
                        color_discrete_map={"Political": "#e63946", "Entertainment": "#457b9d", "Other": "#a8dadc"},
                        title="% of Pages in the Largest Community",
                        labels={"Largest Community %": "% of All Pages"})
        fig_lc.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50%")
        fig_lc.update_layout(height=380)
        st.plotly_chart(fig_lc, use_container_width=True)

    with col2:
        df_nc = pd.DataFrame({
            "Category": [LABELS[k] for k in CATEGORY_ORDER],
            "# Communities": [COMMUNITY[k]["num_communities"] for k in CATEGORY_ORDER],
            "Group": ["Political" if k in POLITICAL else ("Entertainment" if k in ENTERTAINMENT else "Other")
                      for k in CATEGORY_ORDER],
        }).sort_values("# Communities", ascending=True)
        fig_nc = px.bar(df_nc, x="# Communities", y="Category", orientation="h",
                        color="Group",
                        color_discrete_map={"Political": "#e63946", "Entertainment": "#457b9d", "Other": "#a8dadc"},
                        title="Number of Communities Detected per Category")
        fig_nc.update_layout(height=380)
        st.plotly_chart(fig_nc, use_container_width=True)

    st.info(
        "**News Sites** is the most cohesive: 72.3% of pages in one community. "
        "**Politician** (8.6%) and **TV Show** (4.5%) are the most fragmented. TV Show fragmentation, though, could probably be based on genre/regional clustering rather than ideological beliefs."
    )

    st.markdown("---")
    st.subheader("Clustering & Path Length by Category")
    df_cp = pd.DataFrame({
        "Category": [LABELS[k] for k in CATEGORY_ORDER],
        "Clustering": [STATS[k]["clustering"] for k in CATEGORY_ORDER],
        "Avg Path Length": [STATS[k]["approx_path_length"] for k in CATEGORY_ORDER],
        "Group": ["Political" if k in POLITICAL else ("Entertainment" if k in ENTERTAINMENT else "Other")
                  for k in CATEGORY_ORDER],
    })

    c3, c4 = st.columns(2)
    with c3:
        fig_cl = px.bar(df_cp.sort_values("Clustering"), x="Clustering", y="Category",
                        orientation="h", color="Group",
                        color_discrete_map={"Political": "#e63946", "Entertainment": "#457b9d", "Other": "#a8dadc"},
                        title="Avg Clustering Coefficient")
        fig_cl.update_layout(height=340)
        st.plotly_chart(fig_cl, use_container_width=True)

    with c4:
        fig_pl = px.bar(df_cp.sort_values("Avg Path Length"), x="Avg Path Length", y="Category",
                        orientation="h", color="Group",
                        color_discrete_map={"Political": "#e63946", "Entertainment": "#457b9d", "Other": "#a8dadc"},
                        title="Avg Path Length (approx 200-node sample)")
        fig_pl.update_layout(height=340)
        st.plotly_chart(fig_pl, use_container_width=True)




#########################################################################################################
#TAB 5 — political VS entertainment (mocked)


with tab5:
    st.header("Political vs. Entertainment Pages")

    st.info(
        "**This tab is a mockup for future implementation.** "
        "Placeholders below are for our intended layout of charts and tables. "
        "Final analysis will use computed community fragmentation and centrality scores."
    )

    st.markdown(
        "This section will directly compare structural properties of political categories "
        "(Government, Public Figure, Politician) against entertainment categories "
        "(Artist, Athletes, TV Show)."
    )

    st.markdown("---")

    # Mockup table
    st.subheader("Group Comparison Table")
    st.caption("Placeholder values; NOT REAL")
    mock_table = pd.DataFrame({
        "Metric":              ["Avg Clustering", "Avg Path Length", "Avg Density",
                                "Avg Largest Comm %", "Avg # Communities", "Avg Betweenness (top node)"],
        "Political (mock)":   [0.32, 4.32, 0.0024, 23.6, 268, "—"],
        "Entertainment (mock)":[0.20, 4.97, 0.0008, 19.4, 433, "—"],
    }).set_index("Metric")
    st.dataframe(mock_table, use_container_width=True)

    st.markdown("---")
    st.subheader("Planned Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Chart 1 — Largest Community % by Group**")
        st.caption("Bar chart comparing fragmentation across all 6 categories, colored by group.")
        

    with col2:
        st.markdown("**Chart 2 — Clustering vs. Fragmentation Scatter**")
        st.caption(
            "Each point is a category. X = clustering coefficient, "
            "Y = fragmentation index (100 / largest community %). "
            "Bubble size will encode average path length. "
            "betweenness centrality for diff colors?"
        )


#########################################################################################################
# Tab 6 — centrality analysis (mocked)

with tab6:
    st.header("Centrality Analysis")

    st.info(
        "**This tab is a mockup for future implementation.** "
        "Placeholders below are for our intended layout of charts and tables. "
        "Final analysis will use computed community fragmentation and centrality scores."
    )

    st.markdown(
        "Centrality measures are more than just degree to characterize different kinds of structural importance. "
        "A page can be highly connected (degree), act as a bridge between communities (betweenness), or sit close to all others on average (closeness). "
        "This tab will let you compare those roles across all 8 categories."
    )

    st.markdown("---")

    col_ctrl1, col_ctrl2, _ = st.columns([1, 1, 2])
    mock_cat   = col_ctrl1.selectbox("Category", [LABELS[k] for k in CATEGORY_ORDER], index=6, key="cent_cat", disabled=True)
    mock_metric = col_ctrl2.selectbox("Centrality metric", ["Degree", "Betweenness", "Closeness", "Eigenvector"], disabled=True)
    st.caption("Controls disabled")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Chart 1 — Top 10 Nodes by Centrality (per category)**")
        st.caption(
            "Bar chart of the 20 highest-ranked nodes by the selected metric. "
            "Color will encode community membership so hub-bridge roles can be read at a glance."
        )
        mock_top = pd.DataFrame({
            "Rank":        [f"#{i}" for i in range(1, 11)],
            "Betweenness": [0.41, 0.38, 0.33, 0.29, 0.25, 0.21, 0.18, 0.15, 0.12, 0.09],
            "Community":   [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
        })
        fig_top = px.bar(
            mock_top, x="Betweenness", y="Rank", orientation="h",
            color="Community", color_continuous_scale="Viridis",
            title="Top 10 Nodes by Betweenness in Government (not real)",
        )
        fig_top.update_layout(height=340, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.markdown("**Chart 2 — Degree vs. Betweenness Scatterplot**")
        st.caption(
            "Nodes in the upper-left (high betweenness, moderate degree) are bridge nodes — "
            "they connect otherwise separate communities without being the most-liked pages. "
            "These are the structurally interesting hubs for RQ3."
        )
        rng = np.random.default_rng(0)
        mock_sc = pd.DataFrame({
            "Degree":      rng.integers(5, 700, 80),
            "Betweenness": rng.uniform(0, 0.45, 80),
            "Community":   rng.integers(0, 5, 80),
        })
        fig_sc2 = px.scatter(
            mock_sc, x="Degree", y="Betweenness", color="Community",
            color_continuous_scale="Viridis",
            title="Degree vs. Betweenness Government (not real)",
            opacity=0.7,
        )
        fig_sc2.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_sc2, use_container_width=True)

    st.markdown("---")
    st.subheader("Hub Candidates Table")
    st.caption("Top nodes across all four metrics side-by-side, sortable by column.")
    mock_hub = pd.DataFrame({
        "Node":        ["Node A", "Node B", "Node C", "Node D", "Node E"],
        "Degree":      [697, 580, 412, 388, 301],
        "Betweenness": ["—", "—", "—", "—", "—"],
        "Closeness":   ["—", "—", "—", "—", "—"],
        "Eigenvector": ["—", "—", "—", "—", "—"],
        "Community":   [0, 0, 1, 2, 1],
    })
    st.dataframe(mock_hub, use_container_width=True, hide_index=True)


#########################################################################################################
#Tab 7 — findings & limitations
with tab7:
    st.header("Findings & Limitations")

    st.subheader("Key Findings")
    st.markdown("""
[Placeholder - findings will be written here.]

- RQ1: ...
- RQ2: ...
- RQ3: ...
    """)

    st.markdown("---")
    st.subheader("Limitations")
    st.markdown("""
[Placeholder - limitations will be written here.]

- ...
- ...
- ...
    """)