import numpy as np
from sklearn.preprocessing import RobustScaler
from training.preprocessing.feature_statistics import FeatureType

def encode(column, plan_params, feature_statistics):
    """
    Encode a feature value using feature statistics.
    
    Handles both numeric features (scaling) and categorical features (one-hot/embedding index).
    
    Args:
        column: Feature name
        plan_params: Dictionary or object containing feature values
        feature_statistics: Dictionary containing feature statistics (scalers, value dicts)
        
    Returns:
        Encoded value (float or int)
    """
    # SimpleNamespace unification: support both dict and SimpleNamespace
    use_dict = isinstance(plan_params, dict)
    
    # fallback in case actual cardinality is not in plan parameters
    if column == 'act_card' or column == 'act_output_rows':
        if use_dict:
            has_value = column in plan_params
        else:
            has_value = hasattr(plan_params, column)
        
        if not has_value:
            value = 0
        else:
            value = plan_params[column] if use_dict else getattr(plan_params, column, 0)
    else:
        value = plan_params.get(column, 0) if use_dict else getattr(plan_params, column, 0)
    
    if column not in feature_statistics:
        # If feature not in statistics, return 0 (or handle gracefully)
        return 0.0
        
    stats = feature_statistics[column]
    
    if stats.get('type') == str(FeatureType.numeric):
        if 'scaler' in stats:
            # Use pre-fitted scaler
            enc_value = stats['scaler'].transform(np.array([[value]])).item()
        else:
            # Fallback if scaler not attached (should not happen if add_numerical_scalers called)
            enc_value = value
            
    elif stats.get('type') == str(FeatureType.categorical):
        value_dict = stats['value_dict']
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        
        # Handle unknown values
        if str(value) not in value_dict:
            # Use default value (0 or min index)
            if 0 in value_dict.values():
                enc_value = 0
            else:
                min_index = min(value_dict.values()) if value_dict else 0
                enc_value = min_index
        else:
            enc_value = value_dict[str(value)]
    else:
        raise NotImplementedError(f"Unknown feature type for {column}")
        
    return enc_value


def add_numerical_scalers(feature_statistics):
    """
    Add Scikit-Learn RobustScaler objects to feature statistics.
    
    Args:
        feature_statistics: Dictionary of feature statistics
    """
    if feature_statistics is None:
        return

    for k, v in feature_statistics.items():
        if v.get('type') == str(FeatureType.numeric):
            # Reconstruct scaler from saved statistics
            scaler = RobustScaler()
            scaler.center_ = v['center']
            scaler.scale_ = v['scale']
            feature_statistics[k]['scaler'] = scaler

