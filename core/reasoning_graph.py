from streamlit_agraph import agraph, Node, Edge, Config
import streamlit as st

def render_reasoning_graph(result: dict, cot_parsed: list) -> None:
    nodes = []
    edges = []

    # 1. Query Node
    query_text = st.session_state.get("last_query", "Query")
    if len(query_text) > 60:
        query_text = query_text[:57] + "..."
    
    nodes.append(
        Node(id="query", label=f"Query\n{query_text}", size=30, color="#6b7280", shape="box")
    )

    # 2. Run Nodes and Reasoning
    for i, cot in enumerate(cot_parsed):
        run_id = f"run_{i}"
        
        # Get first 50 chars of the answer
        answer_preview = cot.get("answer", "")
        if len(answer_preview) > 50:
            answer_preview = answer_preview[:47] + "..."
            
        nodes.append(
            Node(
                id=run_id, 
                label=f"Run {i+1}\n{answer_preview}", 
                size=25, 
                color="#3b82f6", # blue
                shape="box"
            )
        )
        edges.append(Edge(source="query", target=run_id, color="#9ca3af"))
        
        # Parse reasoning
        reasoning_text = cot.get("reasoning", "")
        lines = [line.strip() for line in reasoning_text.split("\n") if line.strip()]
        
        # Create reasoning nodes
        for j, line in enumerate(lines):
            if line.startswith("-"):
                line = line[1:].strip()
            
            reason_id = f"reason_{i}_{j}"
            
            # Determine color based on reasoning type
            color = "#8b5cf6"  # purple default
            if "Based on:" in line:
                color = "#10b981"  # green
                line = line.replace("Based on:", "Based on:\n")
            elif "Uncertain about:" in line:
                color = "#f59e0b"  # amber
                line = line.replace("Uncertain about:", "Uncertain about:\n")
            elif "User should:" in line:
                color = "#0ea5e9"  # light blue
                line = line.replace("User should:", "User should:\n")
                
            # Truncate to 50 chars max
            if len(line) > 50:
                line = line[:47] + "..."
                
            nodes.append(
                Node(
                    id=reason_id,
                    label=line,
                    size=20,
                    color=color,
                    shape="box"
                )
            )
            edges.append(Edge(source=run_id, target=reason_id, color="#d1d5db", width=1))

    # 3. Trust Score Node
    score = result.get("trust_score", 0)
    tier = result.get("tier", "LOW")
    
    score_color = "#ef4444" # red for LOW
    if tier == "HIGH":
        score_color = "#10b981" # green
    elif tier == "MODERATE":
        score_color = "#f59e0b" # amber
        
    nodes.append(
        Node(id="trust_score", label=f"Trust Score\n{score:.1f}/100 - {tier}", size=35, color=score_color, shape="ellipse")
    )
    
    for i in range(len(cot_parsed)):
        edges.append(Edge(source=f"run_{i}", target="trust_score", color=score_color, width=2))

    # Configure graph layout - Use Left to Right (LR) for readability
    config = Config(
        width="100%",
        height=600,
        directed=True, 
        physics=False, 
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True,
        layout={
            "hierarchical": {
                "enabled": True, 
                "direction": "LR", 
                "sortMethod": "directed", 
                "nodeSpacing": 80, 
                "levelSeparation": 300
            }
        }
    )

    agraph(nodes=nodes, edges=edges, config=config)
