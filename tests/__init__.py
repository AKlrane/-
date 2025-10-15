"""
Test suite for Industry Simulation environment.
"""


def create_single_action(op, invest_dict, create_dict, max_actions=10):
    """
    Helper function to create a properly formatted single action for the multi-action system.
    
    Args:
        op: 0 for invest, 1 for create
        invest_dict: Dict with "firm_id" and "amount" keys
        create_dict: Dict with "initial_capital", "sector", and "location" keys
        max_actions: Maximum number of actions per step (default 10)
        
    Returns:
        Dict in the format expected by the multi-action system
    """
    # Create the main action
    main_action = {
        "op": op,
        "invest": invest_dict,
        "create": create_dict
    }
    
    # Fill remaining slots with dummy actions
    dummy_action = {
        "op": 0,
        "invest": {"firm_id": 0, "amount": [0.0]},
        "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
    }
    
    return {
        "num_actions": 1,
        "actions": [main_action] + [dummy_action for _ in range(max_actions - 1)]
    }
