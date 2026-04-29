import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Network Dashboard Demo",
    layout="wide"
)


@st.cache_data

def load_network_data():
    """Load the Karate Club network and precompute node-level metrics."""
    G = nx.karate_club_graph()

    degree_dict = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

    communities = list(nx.community.greedy_modularity_communities(G))
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i

    positions = nx.spring_layout(G, seed=42)

    node_df = pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [degree_dict[n] for n in G.nodes()],
        "degree_centrality": [degree_centrality[n] for n in G.nodes()],
        "betweenness": [betweenness[n] for n in G.nodes()],
        "closeness": [closeness[n] for n in G.nodes()],
        "eigenvector": [eigenvector[n] for n in G.nodes()],
        "community": [community_map[n] for n in G.nodes()],
        "x": [positions[n][0] for n in G.nodes()],
        "y": [positions[n][1] for n in G.nodes()],
        "club": [G.nodes[n]["club"] for n in G.nodes()],
    })

    edge_df = pd.DataFrame(list(G.edges()), columns=["source", "target"])
    return G, node_df, edge_df


def build_network_figure(plot_nodes: pd.DataFrame, plot_edges: pd.DataFrame, size_metric: str):
    """Create a Plotly network figure from node and edge tables."""
    edge_x = []
    edge_y = []

    node_lookup = plot_nodes.set_index("node")

    for _, row in plot_edges.iterrows():
        source = row["source"]
        target = row["target"]

        if source not in node_lookup.index or target not in node_lookup.index:
            continue

        x0, y0 = node_lookup.loc[source, ["x", "y"]]
        x1, y1 = node_lookup.loc[target, ["x", "y"]]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        hoverinfo="none",
        line=dict(width=0.8, color="lightgray")
    )

    metric_values = plot_nodes[size_metric]
    metric_min = metric_values.min()
    metric_max = metric_values.max()

    if metric_max == metric_min:
        scaled_sizes = [20] * len(plot_nodes)
    else:
        scaled_sizes = 12 + 36 * (metric_values - metric_min) / (metric_max - metric_min)

    node_trace = go.Scatter(
        x=plot_nodes["x"],
        y=plot_nodes["y"],
        mode="markers+text",
        text=plot_nodes["node"],
        textposition="top center",
        customdata=plot_nodes[["node", "community", "degree", "betweenness", "club"]].values,
        hovertemplate=(
            "Node: %{customdata[0]}<br>"
            "Community: %{customdata[1]}<br>"
            "Degree: %{customdata[2]}<br>"
            "Betweenness: %{customdata[3]:.3f}<br>"
            "Club: %{customdata[4]}<extra></extra>"
        ),
        marker=dict(
            size=scaled_sizes,
            color=plot_nodes["community"],
            colorscale="Viridis",
            line=dict(width=1, color="black"),
            showscale=True,
            colorbar=dict(title="Community")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=10, r=10, t=50, b=10),
        height=650
    )
    return fig


def build_ego_figure(G: nx.Graph, node_df: pd.DataFrame, selected_node: int):
    """Create a small ego-network figure for the selected node."""
    ego_G = nx.ego_graph(G, selected_node)
    ego_pos = nx.spring_layout(ego_G, seed=42)

    ego_nodes = list(ego_G.nodes())
    ego_edges = list(ego_G.edges())

    ego_node_df = node_df[node_df["node"].isin(ego_nodes)].copy()
    ego_node_df["ego_x"] = ego_node_df["node"].map(lambda n: ego_pos[n][0])
    ego_node_df["ego_y"] = ego_node_df["node"].map(lambda n: ego_pos[n][1])

    edge_x = []
    edge_y = []
    for source, target in ego_edges:
        x0, y0 = ego_pos[source]
        x1, y1 = ego_pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        hoverinfo="none",
        line=dict(width=1, color="lightgray")
    )

    node_trace = go.Scatter(
        x=ego_node_df["ego_x"],
        y=ego_node_df["ego_y"],
        mode="markers+text",
        text=ego_node_df["node"],
        textposition="top center",
        customdata=ego_node_df[["node", "community", "degree"]].values,
        hovertemplate=(
            "Node: %{customdata[0]}<br>"
            "Community: %{customdata[1]}<br>"
            "Degree: %{customdata[2]}<extra></extra>"
        ),
        marker=dict(
            size=[28 if n == selected_node else 18 for n in ego_node_df["node"]],
            color=ego_node_df["community"],
            colorscale="Viridis",
            line=dict(width=1, color="black"),
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=10, r=10, t=50, b=10),
        height=500,
        title=f"Ego Network for Node {selected_node}"
    )
    return fig


G, node_df, edge_df = load_network_data()

st.title("From Network Analysis to Dashboard")
st.write(
    "An effective dashboard should be grounded in your guiding/research questions."
    "A useful structure is to go from context/overview to findings to exploration to limitations."
    "Design lesson: decide the dashboard story before adding charts."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Centrality",
    "Communities",
    "Node Explorer",
    "Interpretation & Limitations"
])

with tab1:
    st.header("Overview")

    st.markdown("""
    ### Guiding Question
    In this network, **who appears structurally central**, and **how does that relate to community structure**?
    """)

    st.markdown("""
    ### What this network represents
    - **Nodes** are members of a karate club.
    - **Edges** represent social ties between members.
    - This is a small teaching dataset, but it helps us practice the logic of dashboard design.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", G.number_of_nodes())
    c2.metric("Edges", G.number_of_edges())
    c3.metric("Density", round(nx.density(G), 3))
    c4.metric("Communities", node_df["community"].nunique())

    st.subheader("Why start here?")
    st.write(
        "A good dashboard does not begin with a complicated graph. "
        "It first tells the user what the dataset is, what the nodes and edges mean, "
        "and what question the analysis is trying to answer."
    )

    with st.expander("Show node metrics table"):
        st.dataframe(
            node_df[[
                "node", "degree", "degree_centrality", "betweenness",
                "closeness", "eigenvector", "community", "club"
            ]],
            use_container_width=True
        )

    st.success(
        "Interpretation: Before showing visualizations, this tab orients the user. "
        "Students should learn that context is part of the dashboard, not something left for the report."
    )

with tab2:
    st.header("Who is Central?")

    st.markdown("""
    This section examines **structural prominence**.
    The key lesson is that **importance depends on how we define it**.
    """)

    metric = st.selectbox(
        "Choose a centrality measure",
        ["degree", "degree_centrality", "betweenness", "closeness", "eigenvector"]
    )

    metric_explanations = {
        "degree": "Degree counts direct ties. It highlights nodes with many immediate connections.",
        "degree_centrality": "Degree centrality rescales degree relative to network size.",
        "betweenness": "Betweenness highlights nodes that lie on many shortest paths and may bridge groups.",
        "closeness": "Closeness highlights nodes that are, on average, close to others in the network.",
        "eigenvector": "Eigenvector centrality rewards nodes connected to other well-connected nodes."
    }
    st.caption(metric_explanations[metric])

    top_k = st.slider("How many top nodes should be shown?", min_value=5, max_value=15, value=10)

    top_nodes = node_df.sort_values(metric, ascending=False).head(top_k)

    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("Top-ranked nodes")
        st.dataframe(
            top_nodes[["node", "community", "club", metric]],
            use_container_width=True
        )

    with right:
        fig_bar = px.bar(
            top_nodes,
            x="node",
            y=metric,
            color="community",
            hover_data=["club", "degree"],
            title=f"Top {top_k} Nodes by {metric}"
        )
        fig_bar.update_layout(xaxis_title="Node", yaxis_title=metric)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.write("### What should the user conclude here?")
    st.write(
        "This view supports comparison. If the top-ranked nodes change when the metric changes, "
        "that tells us centrality is not a single idea. A node can be highly connected, highly bridging, "
        "or well-positioned in different ways."
    )

    st.success(
        f"Interpretation: Under **{metric}**, certain nodes stand out as structurally prominent. "
        "Your dashboard should focus on highlighting the main takeaways for each graph/tab and interpreting the findings based on known context."
    )

with tab3:
    st.header("Communities and Network Structure")

    st.markdown("""
    This section asks whether the network breaks into **meaningful clusters**.
    The goal is not only to detect communities, but to help users compare them.
    """)

    community_sizes = (
        node_df.groupby("community")
        .size()
        .reset_index(name="num_nodes")
        .sort_values("community")
    )

    selected_community = st.selectbox(
        "Focus on a community",
        ["All"] + sorted(node_df["community"].unique().tolist())
    )

    size_metric = st.selectbox(
        "Size nodes in the network view by",
        ["degree", "degree_centrality", "betweenness", "closeness", "eigenvector"],
        key="community_size_metric"
    )

    if selected_community == "All":
        plot_nodes = node_df.copy()
        plot_edges = edge_df.copy()
        summary_df = node_df.copy()
        title_suffix = "All Communities"
    else:
        plot_nodes = node_df[node_df["community"] == selected_community].copy()
        community_nodes = set(plot_nodes["node"])
        plot_edges = edge_df[
            edge_df["source"].isin(community_nodes) &
            edge_df["target"].isin(community_nodes)
        ].copy()
        summary_df = plot_nodes.copy()
        title_suffix = f"Community {selected_community}"

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Community sizes")
        fig_comm = px.bar(
            community_sizes,
            x="community",
            y="num_nodes",
            title="Detected Community Sizes"
        )
        fig_comm.update_layout(xaxis_title="Community", yaxis_title="Number of Nodes")
        st.plotly_chart(fig_comm, use_container_width=True)

        st.subheader("Selected community summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nodes", len(summary_df))
        m2.metric("Avg degree", round(summary_df["degree"].mean(), 2))
        m3.metric("Avg betweenness", round(summary_df["betweenness"].mean(), 3))

        st.dataframe(
            summary_df[["node", "degree", "betweenness", "closeness", "club"]]
            .sort_values("degree", ascending=False),
            use_container_width=True
        )

    with col2:
        st.subheader(f"Network view: {title_suffix}")
        fig_network = build_network_figure(plot_nodes, plot_edges, size_metric=size_metric)
        fig_network.update_layout(title=f"{title_suffix} (node size = {size_metric}, color = community)")
        st.plotly_chart(fig_network, use_container_width=True)

    st.write("### What should the user conclude here?")
    st.write(
        "Communities help break a complex network into smaller structural units. "
        "A strong dashboard uses those units for comparison, rather than just reporting that an algorithm found them."
    )

    st.info(
        "Interpretation: Community structure suggests that the network is not homogeneous. "
        "But detected communities still require explanation. Students should not treat algorithmic output as self-explanatory social truth."
    )

with tab4:
    st.header("Node Explorer")

    st.markdown("""
    This section supports **drill-down**.
    After seeing global structure, the user can inspect one node in detail.
    """)

    selected_node = st.selectbox(
        "Choose a node to inspect",
        sorted(node_df["node"].tolist())
    )

    node_row = node_df[node_df["node"] == selected_node].iloc[0]
    neighbors = list(G.neighbors(selected_node))
    neighbor_df = node_df[node_df["node"].isin(neighbors)].copy().sort_values("degree", ascending=False)

    a, b, c, d = st.columns(4)
    a.metric("Degree", int(node_row["degree"]))
    b.metric("Community", int(node_row["community"]))
    c.metric("Betweenness", round(node_row["betweenness"], 3))
    d.metric("Closeness", round(node_row["closeness"], 3))

    st.markdown(
        f"""
        **Node {selected_node}** is in **community {int(node_row['community'])}** and is affiliated with **{node_row['club']}**.
        It has **{int(node_row['degree'])} direct ties** in the network.
        """
    )

    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("Immediate neighbors")
        st.dataframe(
            neighbor_df[["node", "degree", "betweenness", "community", "club"]],
            use_container_width=True
        )

    with right:
        st.subheader("Local ego network")
        fig_ego = build_ego_figure(G, node_df, selected_node)
        st.plotly_chart(fig_ego, use_container_width=True)

    st.write("### What should the user conclude here?")
    st.write(
        "Node-level exploration helps connect the big picture to a specific case. "
        "This is often more informative than looking only at the full graph, especially in larger networks."
    )

    st.success(
        "Interpretation: A dashboard should support both overview and drill-down. "
        "This tab shows students that a graph can become much more meaningful when it is tied to a specific analytical question."
    )

with tab5:
    st.header("Interpretation and Limitations")

    st.markdown("""
    ### Discussion:
    blah blah blah
    """)

    st.markdown("""
    ### Limitations:
    blah blah blah
    """)

    st.markdown("""
    ### Why this matters for your final project
    A strong dashboard does more than display outputs. It should help the user understand:
    1. what was measured,
    2. what the evidence suggests,
    3. what the user can explore,
    4. and where possible ethical issues and limitations are.
    """)


