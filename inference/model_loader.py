"""Load both the frozen Stage-1 branch and the adapted Stage-2 transformer."""
def load_model(state):
    from engram.core import build_backbone
    model = build_backbone('reference8', False, .30)
    model.side_branch.load_state_dict(state['side_branch'], strict=True)
    if state['config'].get('version') == 'V22-stage2-bbfull':
        # Stage 2 trains the whole transformer. Never silently skip its weights.
        model.transformer.load_state_dict(state['backbone_blocks'], strict=True)
    else:
        raise ValueError('Final inference requires a Stage-2 checkpoint')
    return model
