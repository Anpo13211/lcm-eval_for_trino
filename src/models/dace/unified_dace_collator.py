"""
Unified DACE collator

DACEはシンプル:
1. プランツリーを深さ優先でトラバース
2. 各ノードから3つの特徴量を抽出: [operator_type, estimated_cost, estimated_cardinality]
3. シーケンスとしてエンコード
4. Transformerに渡す

既存コードに依存せず、論理名のみを使用。
"""

import torch
import numpy as np
from typing import Tuple, List
from core.features.mapper import FeatureMapper


def unified_dace_collator(
    batch: Tuple,
    dbms_name: str,
    feature_statistics: dict,
    config
):
    """
    Unified DACE collator for all DBMS.
    
    Args:
        batch: List of (sample_idx, plan) tuples
        dbms_name: DBMS name
        feature_statistics: Feature statistics
        config: DACE model configuration
    
    Returns:
        tuple: (seq_encodings, attention_masks, loss_masks, run_times, labels, sample_idxs)
    """
    seq_encodings = []
    attention_masks = []
    loss_masks = []
    run_times_list = []
    labels = []
    sample_idxs = []
    
    for sample_idx, plan in batch:
        sample_idxs.append(sample_idx)
        # plan_runtime is already in milliseconds, convert to seconds
        labels.append(plan.plan_runtime / 1000)
        
        # Extract sequence with masks (4つの値を返す)
        seq, attn_mask, loss_mask, run_times = plan_to_sequence(
            plan,
            dbms_name,
            feature_statistics,
            pad_length=config.pad_length,
            node_length=config.node_length
        )
        
        seq_encodings.append(seq)
        attention_masks.append(attn_mask)
        loss_masks.append(loss_mask)
        run_times_list.append(run_times)
    
    # Stack to tensors
    seq_encodings = torch.stack(seq_encodings)
    # Attention masks: Stack all masks as 3D tensor (batch, seq_len, seq_len)
    # Each sample has its own attention mask based on its tree structure
    attention_mask = torch.stack(attention_masks) if attention_masks else None
    loss_masks = torch.stack(loss_masks)
    run_times_tensor = torch.stack(run_times_list)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return seq_encodings, attention_mask, loss_masks, run_times_tensor, labels, sample_idxs


def plan_to_sequence(
    plan,
    dbms_name: str,
    feature_statistics: dict,
    pad_length: int,
    node_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert plan tree to sequence with proper DFS ordering and masks.
    
    完全実装:
    1. DFS順序でノード収集（既存と同じ）
    2. 隣接行列構築
    3. 高さ計算（rootからの距離）
    4. Attention mask（到達可能性ベース）
    5. Loss mask（高さベースの重み付け）
    
    Returns:
        tuple: (flat_sequence, attention_mask, loss_mask, runtimes)
    """
    mapper = FeatureMapper(dbms_name)
    op_name_dict = feature_statistics['op_name']['value_dict']
    num_operators = len(op_name_dict)
    
    # Step 1: DFS traversal to collect nodes, adjacency, and compute heights
    nodes = []
    adjacency_matrix = []  # (parent_id, child_id)
    run_times = []
    
    def depth_first_search(node, parent_id=None):
        """Depth-first search to collect nodes."""
        if len(nodes) >= pad_length:
            return
        
        current_id = len(nodes)
        nodes.append(node)
        
        # Get runtime (act_time is in milliseconds, convert to seconds for consistency)
        runtime = getattr(node.plan_parameters, 'act_time', 0.0) if hasattr(node.plan_parameters, 'act_time') else 0.0
        run_times.append(runtime / 1000.0 if runtime else 0.0)
        
        # Add edge
        if parent_id is not None:
            adjacency_matrix.append((parent_id, current_id))
        
        # Recursively process children
        for child in node.children:
            depth_first_search(child, current_id)
    
    depth_first_search(plan)
    
    num_real_nodes = len(nodes)
    
    # Step 2: Calculate heights (distance from root)
    node_heights = calculate_heights(adjacency_matrix, num_real_nodes)
    
    # Step 3: Encode each node
    flat_sequence = []
    
    for i, node in enumerate(nodes):
        op_type = mapper.get_feature('operator_type', node.plan_parameters)
        est_cost = mapper.get_feature('estimated_cost', node.plan_parameters) or 0.0
        est_card = mapper.get_feature('estimated_cardinality', node.plan_parameters) or 1.0
        
        op_id = op_name_dict.get(str(op_type), 0)
        
        # Create node encoding: [op_id (normalized), cost, cardinality, ...]
        node_encoding = [0.0] * node_length
        node_encoding[0] = float(op_id) / max(num_operators, 1)
        
        if node_length > 1:
            node_encoding[1] = normalize_feature(est_cost, feature_statistics.get('estimated_cost', {}))
        if node_length > 2:
            node_encoding[2] = normalize_feature(est_card, feature_statistics.get('estimated_cardinality', {}))
        
        flat_sequence.extend(node_encoding)
    
    # Step 4: Pad to pad_length nodes
    while len(run_times) < pad_length:
        flat_sequence.extend([0.0] * node_length)
        run_times.append(0.0)
    
    # Pad node_heights
    while len(node_heights) < pad_length:
        node_heights.append(0)
    
    # Truncate if too long
    if len(run_times) > pad_length:
        flat_sequence = flat_sequence[:pad_length * node_length]
        run_times = run_times[:pad_length]
        node_heights = node_heights[:pad_length]
        num_real_nodes = min(num_real_nodes, pad_length)
    
    # Step 5: Create attention mask (到達可能性ベース)
    attention_mask = create_attention_mask(adjacency_matrix, num_real_nodes, pad_length)
    
    # Step 6: Create loss mask (高さベースの重み付け)
    loss_mask = create_loss_mask(num_real_nodes, pad_length, node_heights, loss_weight=0.5)
    
    # Convert to tensors
    seq_tensor = torch.tensor(flat_sequence, dtype=torch.float32)
    run_times_tensor = torch.tensor(run_times, dtype=torch.float32)
    
    return seq_tensor, attention_mask, loss_mask, run_times_tensor


def create_attention_mask(adjacency_matrix: list, num_nodes: int, pad_length: int) -> torch.Tensor:
    """
    Create attention mask based on tree structure.
    
    Args:
        adjacency_matrix: List of (parent, child) tuples
        num_nodes: Number of real nodes
        pad_length: Padded length
    
    Returns:
        Attention mask (pad_length, pad_length) - 0=attend, 1=mask
    """
    # Initialize mask (all ones = all masked)
    mask = np.ones((pad_length, pad_length))
    
    # Node can attend to itself
    for i in range(pad_length):
        mask[i, i] = 0
    
    # Node can attend to reachable nodes
    for parent, child in adjacency_matrix:
        if parent < pad_length and child < pad_length:
            mask[parent, child] = 0
    
    # Compute transitive closure (reachability)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if mask[i, j] == 0:
                for k in range(num_nodes):
                    if mask[j, k] == 0:
                        mask[i, k] = 0
    
    return torch.tensor(mask, dtype=torch.bool)


def create_loss_mask(num_nodes: int, pad_length: int, heights: list, loss_weight: float = 0.5) -> torch.Tensor:
    """
    Create loss mask with height-based weighting.
    
    Args:
        num_nodes: Number of real nodes
        pad_length: Padded length
        heights: Node heights in tree
        loss_weight: Weight decay factor
    
    Returns:
        Loss mask (pad_length,)
    """
    loss_mask = np.zeros(pad_length)
    
    # Apply exponential weighting by height
    for i in range(min(num_nodes, len(heights))):
        loss_mask[i] = loss_weight ** heights[i]
    
    return torch.tensor(loss_mask, dtype=torch.float32)


def calculate_heights(adjacency_matrix: list, num_nodes: int) -> list:
    """
    Calculate height of each node (distance from root).
    
    Height = 0 for root, increases as we go down the tree.
    
    Args:
        adjacency_matrix: List of (parent, child) tuples
        num_nodes: Number of nodes
    
    Returns:
        List of heights
    """
    if num_nodes == 0:
        return []
    
    if num_nodes == 1:
        return [0]  # Root has height 0
    
    # Build adjacency list
    children = {i: [] for i in range(num_nodes)}
    for parent, child in adjacency_matrix:
        children[parent].append(child)
    
    # BFS to compute heights
    heights = [-1] * num_nodes
    heights[0] = 0  # Root
    
    queue = [0]
    while queue:
        node_id = queue.pop(0)
        for child_id in children[node_id]:
            if child_id < num_nodes and heights[child_id] == -1:
                heights[child_id] = heights[node_id] + 1
                queue.append(child_id)
    
    # Fill unvisited nodes with 0
    heights = [h if h >= 0 else 0 for h in heights]
    
    return heights


def normalize_feature(value, stats: dict) -> float:
    """Normalize numeric feature using scaler if available."""
    if value is None:
        return 0.0
    
    try:
        value = float(value)
    except:
        return 0.0
    
    # Use scaler if available (like original DACE)
    if 'scaler' in stats:
        import numpy as np
        scaled = stats['scaler'].transform(np.array([[value]]))
        return float(scaled[0, 0])
    
    # Fallback to manual scaling
    center = stats.get('center', 0.0)
    scale = stats.get('scale', 1.0)
    
    return (value - center) / scale if scale > 0 else value

